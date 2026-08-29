//===- CPUFeaturesDarwin.cpp - Darwin CPU feature detection ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "orc-rt/bedrock/ExecutorProcessInfo.h"

#include "orc-rt-internal/bedrock/TargetDetails.h"

#include <cstdint>
#include <sys/sysctl.h>
#include <sys/types.h>

namespace orc_rt {

namespace {

/// Reads an integer hw.optional.* flag. Absent flags are reported as false,
bool sysctlFlag(const char *Name) noexcept {
  int32_t V = 0;
  size_t S = sizeof(V);
  return sysctlbyname(Name, &V, &S, nullptr, 0) == 0 && V != 0;
}

} // namespace
std::vector<std::string_view> ExecutorProcessInfo::detectTargetCPUFeatures() {
  using namespace orc_rt::target_detail;
  std::vector<std::string_view> Features;

#if defined(__x86_64__) || defined(__i386__)
  // The hw.optional flags already account for kernel support of the extended
  // register state, so no OSXSAVE / XCR0 check is required here unlike linux
  // this is why we diverge from using the compiler intrinsics here.
  if (sysctlFlag("hw.optional.sse4_1"))
    Features.push_back(feature::x86::sse4_1);
  if (sysctlFlag("hw.optional.sse4_2"))
    Features.push_back(feature::x86::sse4_2);
  if (sysctlFlag("hw.optional.avx1_0"))
    Features.push_back(feature::x86::avx);
  if (sysctlFlag("hw.optional.avx2_0"))
    Features.push_back(feature::x86::avx2);

#elif defined(__arm64__) || defined(__aarch64__)
  // NEON is mandatory on all Apple AArch64 hardware.
  Features.push_back(feature::aarch64::neon);

  if (sysctlFlag("hw.optional.arm.FEAT_DotProd"))
    Features.push_back(feature::aarch64::dotprod);
  if (sysctlFlag("hw.optional.arm.FEAT_FP16"))
    Features.push_back(feature::aarch64::fullfp16);
  if (sysctlFlag("hw.optional.arm.FEAT_SHA3"))
    Features.push_back(feature::aarch64::sha3);
#endif

  return Features;
}

} // namespace orc_rt
