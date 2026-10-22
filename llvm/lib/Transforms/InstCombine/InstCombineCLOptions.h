//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_INSTCOMBINE_INSTCOMBINECLOPTIONS_H
#define LLVM_LIB_TRANSFORMS_INSTCOMBINE_INSTCOMBINECLOPTIONS_H

#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"

#define OPTIONS_STRUCT_DECL
#include "InstCombineCLOptions.inc"

namespace llvm {

inline const InstCombineCLOptions &
getInstCombineCLOptions(const LLVMContext &C) {
  return C.getLibraryOptions().get<InstCombineCLOptions>();
}

inline const InstCombineCLOptions &getInstCombineCLOptions(const Function &F) {
  return getInstCombineCLOptions(F.getContext());
}

} // namespace llvm

#endif // LLVM_LIB_TRANSFORMS_INSTCOMBINE_INSTCOMBINECLOPTIONS_H
