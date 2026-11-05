//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_IPO_IPOOPTIONS_H
#define LLVM_LIB_TRANSFORMS_IPO_IPOOPTIONS_H

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include <optional>

#define OPTIONS_STRUCT_DECL
#include "IPOOptions.inc"

namespace llvm {

inline const IPOOptions &getIPOOptions(const LLVMContext &C) {
  return C.getLibraryOptions().get<IPOOptions>();
}

inline const IPOOptions &getIPOOptions(const Module &M) {
  return getIPOOptions(M.getContext());
}

} // namespace llvm

#endif // LLVM_LIB_TRANSFORMS_IPO_IPOOPTIONS_H
