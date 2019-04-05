// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==--------------- half_type.cpp - SYCL half type test --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

using namespace cl::sycl;

constexpr float FLT_EPSILON = 9.77e-4;

constexpr size_t N = 100;

template <typename T> void assert_close(const T &C, const float ref) {
  for (size_t i = 0; i < N; i++) {
    float diff = C[i] - ref;
    assert(std::fabs(diff) < FLT_EPSILON);
  }
}

void verify_add(queue &q, buffer<half, 1> &a, buffer<half, 1> &b, range<1> &r,
                const float ref) {
  buffer<half, 1> c{r};

  q.submit([&](handler &cgh) {
    auto A = a.get_access<access::mode::read>(cgh);
    auto B = b.get_access<access::mode::read>(cgh);
    auto C = c.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class calc_add>(
        r, [=](id<1> index) { C[index] = A[index] + B[index]; });
  });

  assert_close(c.get_access<access::mode::read>(), ref);
}

void verify_min(queue &q, buffer<half, 1> &a, buffer<half, 1> &b, range<1> &r,
                const float ref) {
  buffer<half, 1> c{r};

  q.submit([&](handler &cgh) {
    auto A = a.get_access<access::mode::read>(cgh);
    auto B = b.get_access<access::mode::read>(cgh);
    auto C = c.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class calc_min>(
        r, [=](id<1> index) { C[index] = A[index] - B[index]; });
  });

  assert_close(c.get_access<access::mode::read>(), ref);
}

void verify_mul(queue &q, buffer<half, 1> &a, buffer<half, 1> &b, range<1> &r,
                const float ref) {
  buffer<half, 1> c{r};

  q.submit([&](handler &cgh) {
    auto A = a.get_access<access::mode::read>(cgh);
    auto B = b.get_access<access::mode::read>(cgh);
    auto C = c.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class calc_mul>(
        r, [=](id<1> index) { C[index] = A[index] * B[index]; });
  });

  assert_close(c.get_access<access::mode::read>(), ref);
}

void verify_div(queue &q, buffer<half, 1> &a, buffer<half, 1> &b, range<1> &r,
                const float ref) {
  buffer<half, 1> c{r};

  q.submit([&](handler &cgh) {
    auto A = a.get_access<access::mode::read>(cgh);
    auto B = b.get_access<access::mode::read>(cgh);
    auto C = c.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class calc_div>(
        r, [=](id<1> index) { C[index] = A[index] / B[index]; });
  });

  assert_close(c.get_access<access::mode::read>(), ref);
}

inline bool bitwise_comparison_fp16(const half val, const uint16_t exp) {
  return reinterpret_cast<const uint16_t&>(val) == exp;
}

inline bool bitwise_comparison_fp32(const half val, const uint32_t exp) {
  const float fp32 = static_cast<float>(val);
  return reinterpret_cast<const uint32_t&>(fp32) == exp;
}

int main() {
  // We assert that the length is 1 because we use macro to select the device
  assert(device::get_devices().size() == 1);

  auto dev = device::get_devices()[0];
  if (!dev.is_host() && !dev.has_extension("cl_khr_fp16")) {
    std::cout << "This device doesn't support the extension cl_khr_fp16"
              << std::endl;
    return 0;
  }

  std::vector<half> vec_a(N, 5.0);
  std::vector<half> vec_b(N, 2.0);

  range<1> r(N);
  buffer<half, 1> a{vec_a.data(), r};
  buffer<half, 1> b{vec_b.data(), r};

  queue q;

  verify_add(q, a, b, r, 7.0);
  verify_min(q, a, b, r, 3.0);
  verify_mul(q, a, b, r, 10.0);
  verify_div(q, a, b, r, 2.5);

  if (!dev.is_host()) {
    return 0;
  }

  // Basic tests: fp32->fp16
  // The following references are from `_cvtss_sh` with truncate mode.
  // +inf
  assert(bitwise_comparison_fp16(75514, 31744));
  // -inf
  assert(bitwise_comparison_fp16(-75514, 64512));
  // +0
  assert(bitwise_comparison_fp16(0.0, 0));
  // -0
  assert(bitwise_comparison_fp16(-0.0, 32768));
  // nan
  assert(bitwise_comparison_fp16(0.0 / 0.0, 32256));
  assert(bitwise_comparison_fp16(-0.0 / 0.0, 32256));
  // special nan
  uint32_t special_nan = 0x7f800001;
  assert(
      bitwise_comparison_fp16(reinterpret_cast<float &>(special_nan), 32256));
  special_nan = 0xff800001;
  assert(
      bitwise_comparison_fp16(reinterpret_cast<float &>(special_nan), 65024));
  // subnormal
  assert(bitwise_comparison_fp16(9.8E-45, 0));
  assert(bitwise_comparison_fp16(-9.8E-45, 32768));
  // overflow
  assert(bitwise_comparison_fp16(half(55504) * 3, 31744));
  assert(bitwise_comparison_fp16(half(-55504) * 3, 64512));
  // underflow
  assert(bitwise_comparison_fp16(half(8.1035e-05) / half(3), 453));

  // Basic tests: fp16->fp32
  // The following references are from `_cvtsh_ss`.
  // +inf
  const uint16_t pinf = 0x7a00;
  assert(bitwise_comparison_fp32(reinterpret_cast<const half &>(pinf),
                                 1195376640));
  // -inf
  const uint16_t ninf = 0xfa00;
  assert(bitwise_comparison_fp32(reinterpret_cast<const half &>(ninf),
                                 3342860288));
  // +0
  const uint16_t p0 = 0x0;
  assert(bitwise_comparison_fp32(reinterpret_cast<const half &>(p0), 0));
  // -0
  const uint16_t n0 = 0x8000;
  assert(
      bitwise_comparison_fp32(reinterpret_cast<const half &>(n0), 2147483648));
  // nan
  const uint16_t nan16 = 0x7a03;
  assert(bitwise_comparison_fp32(reinterpret_cast<const half &>(nan16),
                                 1195401216));
  // subnormal
  const uint16_t subnormal = 0x0005;
  assert(bitwise_comparison_fp32(reinterpret_cast<const half &>(subnormal),
                                 882900992));

  return 0;
}
