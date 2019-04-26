//===-- platform_util.cpp - Platform utilities implementation --*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/detail/platform_util.hpp>
#include <CL/sycl/exception.hpp>

#if defined(SYCL_RT_OS_LINUX)
#include <cpuid.h>
#elif defined(SYCL_RT_OS_WINDOWS)
#include <intrin.h>
#endif

namespace cl {
namespace sycl {
namespace detail {

// Used by methods that duplicate OpenCL behaviour in order to get CPU info
static void cpuid(uint32_t *CPUInfo, uint32_t Type, uint32_t SubType = 0) {
#if defined(SYCL_RT_OS_LINUX)
  __cpuid_count(Type, SubType, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
#elif defined(SYCL_RT_OS_WINDOWS)
  __cpuidex(reinterpret_cast<int *>(CPUInfo), Type, SubType);
#endif
}

uint32_t PlatformUtil::getMaxClockFrequency() {
  throw runtime_error(
      "max_clock_frequency parameter is not supported for host device");
  uint32_t CPUInfo[4];
  string_class Buff(sizeof(CPUInfo) * 3 + 1, 0);
  size_t Offset = 0;

  for (uint32_t i = 0x80000002; i <= 0x80000004; i++) {
    cpuid(CPUInfo, i);
    std::copy(reinterpret_cast<char *>(CPUInfo),
              reinterpret_cast<char *>(CPUInfo) + sizeof(CPUInfo),
              Buff.begin() + Offset);
    Offset += sizeof(CPUInfo);
  }
  std::size_t Found = Buff.rfind("Hz");
  // Bail out if frequency is not found in CPUID string
  if (Found == std::string::npos)
    return 0;

  Buff = Buff.substr(0, Found);
  uint32_t Freq = 0;
  switch (Buff[Buff.size() - 1]) {
  case 'M':
    Freq = 1;
    break;
  case 'G':
    Freq = 1000;
    break;
  }
  Buff = Buff.substr(Buff.rfind(' '), Buff.length());
  Freq *= std::stod(Buff);
  return Freq;
}

uint32_t PlatformUtil::getMemCacheLineSize() {
  uint32_t CPUInfo[4];
  cpuid(CPUInfo, 0x80000006);
  return CPUInfo[2] & 0xff;
}

uint64_t PlatformUtil::getMemCacheSize() {
  uint32_t CPUInfo[4];
  cpuid(CPUInfo, 0x80000006);
  return static_cast<uint64_t>(CPUInfo[2] >> 16) * 1024;
}

uint32_t PlatformUtil::getNativeVectorWidth(PlatformUtil::TypeIndex TIndex) {
  // SSE4.2 has 16 byte (XMM) registers
  static constexpr uint32_t VECTOR_WIDTH_SSE42[] = {16, 8, 4, 2, 4, 2, 0};
  // AVX supports 32 byte (YMM) registers only for floats and doubles
  static constexpr uint32_t VECTOR_WIDTH_AVX[] = {16, 8, 4, 2, 8, 4, 0};
  // AVX2 has a full set of 32 byte (YMM) registers
  static constexpr uint32_t VECTOR_WIDTH_AVX2[] = {32, 16, 8, 4, 8, 4, 0};
  // AVX512 has 64 byte (ZMM) registers
  static constexpr uint32_t VECTOR_WIDTH_AVX512[] = {64, 32, 16, 8, 16, 8, 0};

  uint32_t Index = static_cast<uint32_t>(TIndex);

#if defined(SYCL_RT_OS_LINUX)
  if (__builtin_cpu_supports("avx512f"))
    return VECTOR_WIDTH_AVX512[Index];
  if (__builtin_cpu_supports("avx2"))
    return VECTOR_WIDTH_AVX2[Index];
  if (__builtin_cpu_supports("avx"))
    return VECTOR_WIDTH_AVX[Index];
#elif defined(SYCL_RT_OS_WINDOWS)

  uint32_t Info[4];

  // Check that CPUID func number 7 is available.
  cpuid(Info, 0);
  if (Info[0] >= 7) {
    // avx512f = CPUID.7.EBX[16]
    cpuid(Info, 7);
    if (Info[1] & (1 << 16))
      return VECTOR_WIDTH_AVX512[Index];

    // avx2 = CPUID.7.EBX[5]
    if (Info[1] & (1 << 5))
      return VECTOR_WIDTH_AVX2[Index];
  }
  // It is assumed that CPUID func number 1 is always available.
  // avx = CPUID.1.ECX[28]
  cpuid(Info, 1);
  if (Info[2] & (1 << 28))
    return VECTOR_WIDTH_AVX[Index];
#endif

  return VECTOR_WIDTH_SSE42[Index];
}

} // namespace detail
} // namespace sycl
} // namespace cl
