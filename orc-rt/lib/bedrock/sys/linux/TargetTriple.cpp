//===- TargetTriple.cpp - Linux target triple detection -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "orc-rt-internal/bedrock/sys/TargetTriple.h"

#include "orc-rt-internal/bedrock/TargetDetails.h"
#include "orc-rt-internal/support/StringExtras.h"

namespace orc_rt::sys {

std::string detectTargetTriple() noexcept {
  static const std::string Cache = [] {
    using namespace target_detail;

#if defined(__aarch64__)
    constexpr std::string_view Arch = arch::aarch64;
#elif defined(__x86_64h__)
    constexpr std::string_view Arch = arch::x86_64h;
#elif defined(__x86_64__)
    constexpr std::string_view Arch = arch::x86_64;
#elif defined(__i386__)
    constexpr std::string_view Arch = arch::i386;
#else
#error "Unsupported architecture"
#endif

#if defined(__x86_64__) || defined(__i386__)
    constexpr std::string_view Vendor = vendor::pc;
#else
    constexpr std::string_view Vendor = vendor::unknown;
#endif

#if defined(__GLIBC__)
    return join({Arch, Vendor, "linux", "gnu"}, "-");
#else
    return join({Arch, Vendor, "linux"}, "-");
#endif
  }();

  return Cache;
}

} // namespace orc_rt::sys
