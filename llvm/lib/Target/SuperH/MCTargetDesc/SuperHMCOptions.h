//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SUPERH_MCTARGETDESC_SUPERHMCOPTIONS_H
#define LLVM_LIB_TARGET_SUPERH_MCTARGETDESC_SUPERHMCOPTIONS_H

#include "llvm/MC/MCTargetOptions.h"

#define OPTIONS_STRUCT_DECL
#include "SuperHMCOptions.inc"

namespace llvm::SUPERH_MC {
inline const SuperHMCOptions &getCLOpts(const MCTargetOptions &Options) {
  return cl::LibraryOptionsContext::get<SuperHMCOptions>(Options.LibraryOptions);
}
} // namespace llvm::SUPERH_MC

#endif // LLVM_LIB_TARGET_SUPERH_MCTARGETDESC_SUPERHMCOPTIONS_H
