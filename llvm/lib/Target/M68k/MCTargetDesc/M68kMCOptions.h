//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_M68K_MCTARGETDESC_M68KMCOPTIONS_H
#define LLVM_LIB_TARGET_M68K_MCTARGETDESC_M68KMCOPTIONS_H

#include "llvm/MC/MCTargetOptions.h"

#define OPTIONS_STRUCT_DECL
#include "M68kMCOptions.inc"

namespace llvm::M68k_MC {
inline const M68kMCOptions &getCLOpts(const MCTargetOptions &Options) {
  return cl::LibraryOptionsContext::get<M68kMCOptions>(Options.LibraryOptions);
}
} // namespace llvm::M68k_MC

#endif // LLVM_LIB_TARGET_M68K_MCTARGETDESC_M68KMCOPTIONS_H
