//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ARM_MCTARGETDESC_ARMMCOPTIONS_H
#define LLVM_LIB_TARGET_ARM_MCTARGETDESC_ARMMCOPTIONS_H

#include "llvm/MC/MCTargetOptions.h"

namespace llvm {
enum class ImplicitItModeTy { Always, Never, ARMOnly, ThumbOnly };
} // namespace llvm

#define OPTIONS_STRUCT_DECL
#include "ARMMCOptions.inc"

namespace llvm::ARM_MC {
inline const ARMMCOptions &getCLOpts(const MCTargetOptions &Options) {
  return cl::LibraryOptionsContext::get<ARMMCOptions>(Options.LibraryOptions);
}
} // namespace llvm::ARM_MC

#endif // LLVM_LIB_TARGET_ARM_MCTARGETDESC_ARMMCOPTIONS_H
