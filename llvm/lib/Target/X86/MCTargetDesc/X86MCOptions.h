//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_X86_MCTARGETDESC_X86MCOPTIONS_H
#define LLVM_LIB_TARGET_X86_MCTARGETDESC_X86MCOPTIONS_H

#include "llvm/MC/MCTargetOptions.h"
#include <optional>

namespace llvm::X86 {
// The numbering matches the GCC assembler dialects, which inline asm
// alternatives index.
enum AsmWriterFlavorTy { ATT = 0, Intel = 1 };
} // namespace llvm::X86

#define OPTIONS_STRUCT_DECL
#include "X86MCOptions.inc"

namespace llvm::X86_MC {
inline const X86MCOptions &getCLOpts(const MCTargetOptions &Options) {
  return cl::LibraryOptionsContext::get<X86MCOptions>(Options.LibraryOptions);
}
} // namespace llvm::X86_MC

#endif // LLVM_LIB_TARGET_X86_MCTARGETDESC_X86MCOPTIONS_H
