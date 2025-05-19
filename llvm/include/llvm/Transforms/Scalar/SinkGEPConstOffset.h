//===- SinkGEPConstOffset.h -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_SINKGEPCONSTOFFSET_H
#define LLVM_TRANSFORMS_SCALAR_SINKGEPCONSTOFFSET_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class SinkGEPConstOffsetPass
    : public PassInfoMixin<SinkGEPConstOffsetPass> {
public:
  SinkGEPConstOffsetPass() {}
  void printPipeline(raw_ostream &OS,
                     function_ref<StringRef(StringRef)> MapClassName2PassName);
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_SINKGEPCONSTOFFSET_H
