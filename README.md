
### Speculative Inference to speed up LLM inference

<figure align="center">
    <img src="assets/llama_cute.jpg" width="300" height="300"/>
    <figcaption style="color: grey;">
        The cutest image ever from karpathy's repo. I couldn't not have included this.
    </figcaption>
</figure>

This is my attempt to do speculative inference, purely in C.   
Built on top of karpathy's [llama2.c]( https://github.com/karpathy/llama2.c )   
(not making a fork because i don't need everything from that repo)


### Running
Download the model weights from [llama2.c]( https://github.com/karpathy/llama2.c ). put them in `models/`  
Use the same commands as in `llama2.c`:  

`make run`  
to compile

`./run <modelpath> -i "<input prompt>"`  
to generate output  
and similarly other commands


### Problems
Can't do speculative inference because llama2.c does all operations in mat muls sequentially in a single thread (can't parallelise computation for tokens in sentence simultaneously)
To get output embeddings for each token in sentence:
<ol>
    <li>
        run generate() function for each token in Tokens[1..T]
    </li>
    <li>
		obtain k,v values for this token, and append in k_cache, v_cache
    </li>
    <li>
		cache the generated 'out' tensor
    </li>
</ol>
steps 1 and 2 are sequential in llama2.c => step 3 can't be done parallely.  

As far as I understand, llama.cpp uses cuda/metal to do step 1 parallelly (by using matrices of dimension 1 more than those in llama2.c for the T dimension, doing the operations for all tokens in a sentence simultaneously)

### What now:
<ol>
    <li>
        [karpathy says]( https://twitter.com/karpathy/status/1697318534555336961 ) speculative inference works because loading weights from VRAM into on-chip cache is much slower than actually doing the operations.
        <ul>
            <li>
                if this is true, i can change llama2.c to compute the matrices for all Timesteps serially and then try doing speculative inference.
            </li>
            <li>
                also, if this is true, llama2.c will benefit from speculative inference(even the smaller models) because the smaller model has simply less params to shuttle in/out from the cache.
            </li>
        </ul>
    </li>
    <li>
	Write it in python
        <ul>
            <li>
                leverage cuda/metal to do matrix multiplications parallelly
            </li>
            <li>
                Would be muuuuch easier than soln 1.
            </li>
            <li>
                would be much faster too because:
                <ol>
                    <li>
                        same benefits as in 1.
                    </li>
                    <li>
                        cuda does matrix multiplications parallelly, so all operations will take the same time
                    </li>
                </ol>
            </li>
        </ul>
    </li>
</ol>


### Welp
going ahead with 1, cuz 2 feels too easy. let's see if it improves at all.



### License
MIT