//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_POWERPC_MCTARGETDESC_PPCMCOPTIONS_H
#define LLVM_LIB_TARGET_POWERPC_MCTARGETDESC_PPCMCOPTIONS_H

#include "llvm/MC/MCTargetOptions.h"

#define OPTIONS_STRUCT_DECL
#include "PPCMCOptions.inc"

namespace llvm::PPC_MC {
inline const PPCMCOptions &getCLOpts(const MCTargetOptions &Options) {
  return cl::LibraryOptionsContext::get<PPCMCOptions>(Options.LibraryOptions);
}
} // namespace llvm::PPC_MC

#endif // LLVM_LIB_TARGET_POWERPC_MCTARGETDESC_PPCMCOPTIONS_H
