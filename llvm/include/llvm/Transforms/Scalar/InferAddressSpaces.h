//===- InferAddressSpace.h - ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_INFERADDRESSSPACES_H
#define LLVM_TRANSFORMS_SCALAR_INFERADDRESSSPACES_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class InferAddressSpacesPass
    : public OptionalPassInfoMixin<InferAddressSpacesPass> {
  unsigned FlatAddrSpace = 0;

  /// The default address space is assumed as the flat address space. This is
  /// mainly for test purpose.
  const bool AssumeDefaultIsFlatAddressSpace;

public:
  LLVM_ABI InferAddressSpacesPass(bool AssumeDefaultIsFlatAddressSpace = false);
  LLVM_ABI InferAddressSpacesPass(unsigned AddressSpace,
                                  bool AssumeDefaultIsFlatAddressSpace = false);
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  LLVM_ABI void
  printPipeline(raw_ostream &OS,
                function_ref<StringRef(StringRef)> MapClassName2PassName);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_INFERADDRESSSPACES_H
