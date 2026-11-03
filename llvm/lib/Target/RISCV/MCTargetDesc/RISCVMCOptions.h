//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_MCTARGETDESC_RISCVMCOPTIONS_H
#define LLVM_LIB_TARGET_RISCV_MCTARGETDESC_RISCVMCOPTIONS_H

#include "llvm/MC/MCTargetOptions.h"

#define OPTIONS_STRUCT_DECL
#include "RISCVMCOptions.inc"

namespace llvm::RISCV_MC {
inline const RISCVMCOptions &getCLOpts(const MCTargetOptions &Options) {
  return cl::LibraryOptionsContext::get<RISCVMCOptions>(Options.LibraryOptions);
}
} // namespace llvm::RISCV_MC

#endif // LLVM_LIB_TARGET_RISCV_MCTARGETDESC_RISCVMCOPTIONS_H
