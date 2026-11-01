//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_PASSES_PASSESOPTIONS_H
#define LLVM_LIB_PASSES_PASSESOPTIONS_H

#include "llvm/Analysis/InlineAdvisor.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/IPO/Attributor.h"

#define OPTIONS_STRUCT_DECL
#include "PassesOptions.inc"

namespace llvm {

inline const PassesOptions &getPassesOptions(const LLVMContext &C) {
  return C.getLibraryOptions().get<PassesOptions>();
}

} // namespace llvm

#endif // LLVM_LIB_PASSES_PASSESOPTIONS_H
