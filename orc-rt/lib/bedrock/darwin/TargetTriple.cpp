//===- TargetTriple.cpp - Darwin target triple detection ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Target triple detection on Darwin.
//
//===----------------------------------------------------------------------===//

#include "orc-rt-internal/bedrock/TargetDetails.h"
#include "orc-rt/bedrock/ExecutorProcessInfo.h"

#include <TargetConditionals.h>
#include <cstdlib>
#include <cstring>
#include <sys/sysctl.h>
#include <sys/types.h>

namespace orc_rt {

namespace {
// FIXME: jared - Add in error handling rather than an empty string.
std::string sysctlString(const char *Name) noexcept {
  size_t S = 0;
  if (sysctlbyname(Name, nullptr, &S, nullptr, 0) != 0)
    return {};
  if (S == 0)
    return {};

  std::string V(S - 1, '\0');
  if (sysctlbyname(Name, V.data(), &S, nullptr, 0) != 0)
    return {};

  return V;
}

} // namespace

std::string ExecutorProcessInfo::detectTargetTriple() noexcept {
  // Detection may involve system calls, so cache the result.
  static const std::string Cache = [] {
    using namespace target_detail;

#if defined(__arm64e__)
    constexpr std::string_view Arch = arch::arm64e;
#elif defined(__arm64__) || defined(__aarch64__)
    constexpr std::string_view Arch = arch::arm64;
#elif defined(__x86_64h__)
    constexpr std::string_view Arch = arch::x86_64h;
#elif defined(__x86_64__)
    constexpr std::string_view Arch = arch::x86_64;
#elif defined(__i386__)
    constexpr std::string_view Arch = arch::i386;
#else
#error "Unsupported architecture"
#endif

    // As I have learned, order is important here.
    // TARGET_OS_MACCATALYST needs to come first.
#if TARGET_OS_MACCATALYST
    constexpr std::string_view Platform = "ios";
#elif TARGET_OS_VISION
    constexpr std::string_view Platform = "xros";
#elif TARGET_OS_TV
    constexpr std::string_view Platform = "tvos";
#elif TARGET_OS_WATCH
    constexpr std::string_view Platform = "watchos";
#elif TARGET_OS_IOS
    constexpr std::string_view Platform = "ios";
#elif TARGET_OS_OSX
    constexpr std::string_view Platform = "macosx";
#else
#error "Unsupported Darwin platform"
#endif

#if TARGET_OS_MACCATALYST
    constexpr std::string_view Environment = "macabi";
#elif TARGET_OS_SIMULATOR
    constexpr std::string_view Environment = "simulator";
#else
    constexpr std::string_view Environment = "";
#endif

#if TARGET_OS_MACCATALYST
    constexpr const char *VersionSysctl = "kern.iossupportversion";
#else
    constexpr const char *VersionSysctl = "kern.osproductversion";
#endif

    std::string Version;
#if TARGET_OS_SIMULATOR
    if (const char *SimVersion = std::getenv("SIMULATOR_RUNTIME_VERSION"))
      Version = SimVersion;
#endif
    if (Version.empty())
      Version = sysctlString(VersionSysctl);

    return makeTargetTriple(
        {Arch, vendor::apple, std::string(Platform) + Version, Environment});
  }();
  return Cache;
}

} // namespace orc_rt
