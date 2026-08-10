//===- TargetDetails.h - Names for target triple components -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_TARGETDETAILS_H
#define ORC_RT_TARGETDETAILS_H

#include <string_view>

namespace orc_rt::target_detail {

/// Architecture names. Darwin spells AArch64 "arm64"; everywhere else it is
/// "aarch64", so both are present.
namespace arch {
inline constexpr std::string_view aarch64 = "aarch64";
inline constexpr std::string_view arm64 = "arm64";
inline constexpr std::string_view arm64e = "arm64e";
inline constexpr std::string_view i386 = "i386";
inline constexpr std::string_view x86_64 = "x86_64";
inline constexpr std::string_view x86_64h = "x86_64h";
} // namespace arch

namespace vendor {
inline constexpr std::string_view apple = "apple";
inline constexpr std::string_view pc = "pc";
inline constexpr std::string_view unknown = "unknown";
} // namespace vendor

/// These should match llvm feature names.
namespace feature {
namespace x86 {
inline constexpr std::string_view sse4_1 = "sse4.1";
inline constexpr std::string_view sse4_2 = "sse4.2";
inline constexpr std::string_view avx = "avx";
inline constexpr std::string_view avx2 = "avx2";
} // namespace x86
namespace aarch64 {
inline constexpr std::string_view neon = "neon";
inline constexpr std::string_view dotprod = "dotprod";
inline constexpr std::string_view fullfp16 = "fullfp16";
inline constexpr std::string_view sha3 = "sha3";
} // namespace aarch64
} // namespace feature

} // namespace orc_rt::target_detail

#endif // ORC_RT_TARGETDETAILS_H
