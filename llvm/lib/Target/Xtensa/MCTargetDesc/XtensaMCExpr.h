//===-- XtensaMCExpr.h - Xtensa specific MC expression classes --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes Xtensa-specific MCExprs
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_XTENSA_MCTARGETDESC_XTENSAMCEXPR_H
#define LLVM_LIB_TARGET_XTENSA_MCTARGETDESC_XTENSAMCEXPR_H

#include "llvm/MC/MCExpr.h"

namespace llvm {

class StringRef;

namespace Xtensa {
enum Specifier { S_None, S_TPOFF };

uint8_t parseSpecifier(StringRef name);
StringRef getSpecifierName(uint8_t S);
} // namespace Xtensa

} // end namespace llvm.

#endif // LLVM_LIB_TARGET_XTENSA_MCTARGETDESC_XTENSAMCEXPR_H
