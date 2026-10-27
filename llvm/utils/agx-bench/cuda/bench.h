// Minimal CUDA device declarations for clang's NVPTX target without the CUDA
// toolkit: clang -x cuda --cuda-device-only -nocudainc -nocudalib.
#pragma once
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
#define __forceinline__ __attribute__((always_inline)) inline
#include "__clang_cuda_builtin_vars.h"

typedef unsigned int uint;
typedef unsigned char uchar;
// f16 as storage, converted with cvt (as cuda_fp16.hpp does).
typedef unsigned short half;
struct __attribute__((aligned(4))) half2 { half x, y; };
struct __attribute__((aligned(16))) f4 { float x, y, z, w; };
struct __attribute__((aligned(4))) uc4 { uchar x, y, z, w; };

static __device__ __forceinline__ float h2f(half h) {
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}
static __device__ __forceinline__ half f2h(float f) {
  half h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
  return h;
}
static __device__ __forceinline__ int min(int a, int b) { return a < b ? a : b; }
static __device__ __forceinline__ int max(int a, int b) { return a > b ? a : b; }

#define FULL 0xffffffffu
static __device__ __forceinline__ void sync() { __syncthreads(); }
static __device__ __forceinline__ float shfl_down(float v, int d) {
  return __nvvm_shfl_sync_down_f32(FULL, v, d, 0x1f);
}
static __device__ __forceinline__ float shfl_xor(float v, int d) {
  return __nvvm_shfl_sync_bfly_f32(FULL, v, d, 0x1f);
}
static __device__ __forceinline__ int shfl_xor_i(int v, int d) {
  return __nvvm_shfl_sync_bfly_i32(FULL, v, d, 0x1f);
}
static __device__ __forceinline__ int shfl_up_i(int v, int d) {
  return __nvvm_shfl_sync_up_i32(FULL, v, d, 0);
}
static __device__ __forceinline__ double shfl_xor_d(double v, int d) {
  long long b = __builtin_bit_cast(long long, v);
  int lo = __nvvm_shfl_sync_bfly_i32(FULL, (int)b, d, 0x1f);
  int hi = __nvvm_shfl_sync_bfly_i32(FULL, (int)(b >> 32), d, 0x1f);
  return __builtin_bit_cast(double, ((long long)hi << 32) | (unsigned)lo);
}
static __device__ __forceinline__ uint ballot(bool p) {
  return __nvvm_vote_ballot_sync(FULL, p);
}
// --use_fast_math forms
static __device__ __forceinline__ float fexp(float x) {
  return __nvvm_ex2_approx_f(x * 1.4426950408889634f);
}
static __device__ __forceinline__ float frsqrt(float x) { return __nvvm_rsqrt_approx_f(x); }
static __device__ __forceinline__ float frcp(float x) { return __nvvm_rcp_approx_ftz_f(x); }
static __device__ __forceinline__ float fsin(float x) { return __nvvm_sin_approx_f(x); }
static __device__ __forceinline__ float fcos(float x) { return __nvvm_cos_approx_f(x); }
static __device__ __forceinline__ float ftanh(float x) {
  float r;
  asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
static __device__ __forceinline__ float atomic_add(float *p, float v) {
  return __nvvm_atom_add_gen_f(p, v);
}
static __device__ __forceinline__ int atomic_add_i(int *p, int v) {
  return __nvvm_atom_add_gen_i(p, v);
}
static __device__ __forceinline__ float warp_sum(float v) {
  for (int d = 16; d > 0; d >>= 1)
    v += shfl_xor(v, d);
  return v;
}
static __device__ __forceinline__ float warp_max(float v) {
  for (int d = 16; d > 0; d >>= 1)
    v = __builtin_fmaxf(v, shfl_xor(v, d));
  return v;
}
// Sum over a threadgroup of blockDim.x (a multiple of 32, at most 1024).
static __device__ __forceinline__ float block_sum(float v, float *tmp) {
  v = warp_sum(v);
  int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  if (l == 0)
    tmp[w] = v;
  sync();
  v = l < (blockDim.x >> 5) ? tmp[l] : 0.0f;
  return warp_sum(v);
}
