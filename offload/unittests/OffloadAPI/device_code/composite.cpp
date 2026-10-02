#include <gpuintrin.h>

struct Foo {
  uint32_t a;
  uint32_t b;
};

extern "C" __gpu_kernel void composite(uint8_t N, Foo F, uint32_t *Out) {
  Out[__gpu_thread_id(0)] = N + F.a + F.b + __gpu_thread_id(0);
}
