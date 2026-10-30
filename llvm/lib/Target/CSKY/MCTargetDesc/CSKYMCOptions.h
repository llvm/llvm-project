//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_CSKY_MCTARGETDESC_CSKYMCOPTIONS_H
#define LLVM_LIB_TARGET_CSKY_MCTARGETDESC_CSKYMCOPTIONS_H

#include "llvm/MC/MCTargetOptions.h"

#define OPTIONS_STRUCT_DECL
#include "CSKYMCOptions.inc"

namespace llvm::CSKY_MC {
inline const CSKYMCOptions &getCLOpts(const MCTargetOptions &Options) {
  return cl::LibraryOptionsContext::get<CSKYMCOptions>(Options.LibraryOptions);
}
} // namespace llvm::CSKY_MC

#endif // LLVM_LIB_TARGET_CSKY_MCTARGETDESC_CSKYMCOPTIONS_H
