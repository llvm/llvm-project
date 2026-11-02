//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MIPS_MCTARGETDESC_MIPSMCOPTIONS_H
#define LLVM_LIB_TARGET_MIPS_MCTARGETDESC_MIPSMCOPTIONS_H

#include "llvm/MC/MCTargetOptions.h"

#define OPTIONS_STRUCT_DECL
#include "MipsMCOptions.inc"

namespace llvm::MIPS_MC {
inline const MipsMCOptions &getCLOpts(const MCTargetOptions &Options) {
  return cl::LibraryOptionsContext::get<MipsMCOptions>(Options.LibraryOptions);
}
} // namespace llvm::MIPS_MC

#endif // LLVM_LIB_TARGET_MIPS_MCTARGETDESC_MIPSMCOPTIONS_H
