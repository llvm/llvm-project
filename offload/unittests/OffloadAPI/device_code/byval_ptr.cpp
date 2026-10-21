#include <gpuintrin.h>
#include <stdint.h>

// The output pointer is only reachable through a by-value struct, like a
// captured SYCL kernel lambda, so the device accesses it indirectly.
struct Wrapper {
  uint32_t *Out;
};

extern "C" __gpu_kernel void byval_ptr(Wrapper W) {
  W.Out[__gpu_thread_id(0)] = __gpu_thread_id(0);
}
