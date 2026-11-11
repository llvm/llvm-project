//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_MC_MCCLOPTIONS_H
#define LLVM_LIB_MC_MCCLOPTIONS_H

#include "llvm/MC/MCTargetOptions.h"
#include <optional>

#define OPTIONS_STRUCT_DECL
#include "MCCLOptions.inc"

namespace llvm {
inline const MCCLOptions &getMCCLOptions(const MCTargetOptions &Options) {
  return cl::LibraryOptionsContext::get<MCCLOptions>(Options.LibraryOptions);
}
} // namespace llvm

#endif // LLVM_LIB_MC_MCCLOPTIONS_H
