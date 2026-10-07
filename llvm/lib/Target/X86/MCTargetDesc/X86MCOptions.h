//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_X86_MCTARGETDESC_X86MCOPTIONS_H
#define LLVM_LIB_TARGET_X86_MCTARGETDESC_X86MCOPTIONS_H

#include <optional>

namespace llvm::X86 {
// The numbering matches the GCC assembler dialects, which inline asm
// alternatives index.
enum AsmWriterFlavorTy { ATT = 0, Intel = 1 };
} // namespace llvm::X86

#define OPTIONS_STRUCT_DECL
#include "X86MCOptions.inc"

#endif // LLVM_LIB_TARGET_X86_MCTARGETDESC_X86MCOPTIONS_H
