//=--- AArch64MCExpr.h - AArch64 specific MC expression classes ---*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes AArch64-specific MCExprs, used for modifiers like
// ":lo12:" or ":gottprel_g1:".
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_MCTARGETDESC_AARCH64MCEXPR_H
#define LLVM_LIB_TARGET_AARCH64_MCTARGETDESC_AARCH64MCEXPR_H

#include "Utils/AArch64BaseInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {

namespace AArch64MCExpr {
using Specifier = uint16_t;
} // namespace AArch64MCExpr

} // end namespace llvm

#endif
