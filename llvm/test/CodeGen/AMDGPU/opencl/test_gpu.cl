// test_gpu.cl - Simple OpenCL kernel for GPU backend testing
__kernel void vector_add(__global const int* a, __global const int* b, __global int* c, int n) {
    int idx = get_global_id(0);
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}