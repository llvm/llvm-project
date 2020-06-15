__kernel void decode_gpu(__global const char *in, __global char *out, __global size_t *strlength_ptr) {
    size_t strlength = *strlength_ptr;
	int num = get_global_id(0);
    if(num < strlength) 
       out[num] = in[num];
}
