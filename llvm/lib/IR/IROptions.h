//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_IR_IROPTIONS_H
#define LLVM_LIB_IR_IROPTIONS_H

#include "llvm/Config/llvm-config.h"
#include "llvm/IR/LLVMContext.h"

namespace llvm {
#ifdef LLVM_ENABLE_PROFCHECK
constexpr bool ElideAllZeroBranchWeightsDefault = false;
#else
constexpr bool ElideAllZeroBranchWeightsDefault = true;
#endif
} // namespace llvm

#define OPTIONS_STRUCT_DECL
#include "IROptions.inc"

namespace llvm {
inline const IROptions &getIROptions(const LLVMContext &C) {
  return C.getLibraryOptions().get<IROptions>();
}
} // namespace llvm

#endif // LLVM_LIB_IR_IROPTIONS_H
