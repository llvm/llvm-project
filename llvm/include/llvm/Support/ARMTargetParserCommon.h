//===---------------- ARMTargetParserCommon ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Code that is common to ARMTargetParser and AArch64TargetParser.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ARMTARGETPARSERCOMMON_H
#define LLVM_SUPPORT_ARMTARGETPARSERCOMMON_H

#include "llvm/ADT/StringRef.h"

namespace llvm {
namespace ARM {

/// Converts e.g. "armv8" -> "armv8-a"
StringRef getArchSynonym(StringRef Arch);

/// MArch is expected to be of the form (arm|thumb)?(eb)?(v.+)?(eb)?, but
/// (iwmmxt|xscale)(eb)? is also permitted. If the former, return
/// "v.+", if the latter, return unmodified string, minus 'eb'.
/// If invalid, return empty string.
StringRef getCanonicalArchName(StringRef Arch);

} // namespace ARM
} // namespace llvm
#endif
