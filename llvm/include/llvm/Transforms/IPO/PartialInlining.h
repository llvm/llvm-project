//===- PartialInlining.h - Inline parts of functions --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs partial inlining, typically by inlining an if statement
// that surrounds the body of the function.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_PARTIALINLINING_H
#define LLVM_TRANSFORMS_IPO_PARTIALINLINING_H

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// Pass to remove unused function declarations.
class PartialInlinerPass : public PassInfoMixin<PartialInlinerPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};
}
#endif // LLVM_TRANSFORMS_IPO_PARTIALINLINING_H
