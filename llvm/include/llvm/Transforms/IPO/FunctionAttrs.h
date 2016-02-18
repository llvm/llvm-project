//===-- FunctionAttrs.h - Compute function attrs --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// Provides passes for computing function attributes based on interprocedural
/// analyses.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_FUNCTIONATTRS_H
#define LLVM_TRANSFORMS_IPO_FUNCTIONATTRS_H

#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// Computes function attributes in post-order over the call graph.
///
/// By operating in post-order, this pass computes precise attributes for
/// called functions prior to processsing their callers. This "bottom-up"
/// approach allows powerful interprocedural inference of function attributes
/// like memory access patterns, etc. It can discover functions that do not
/// access memory, or only read memory, and give them the readnone/readonly
/// attribute. It also discovers function arguments that are not captured by
/// the function and marks them with the nocapture attribute.
class PostOrderFunctionAttrsPass {
public:
  static StringRef name() { return "PostOrderFunctionAttrsPass"; }

  PreservedAnalyses run(LazyCallGraph::SCC &C, CGSCCAnalysisManager *AM);
};

/// Create a legacy pass manager instance of a pass to compute function attrs
/// in post-order.
Pass *createPostOrderFunctionAttrsLegacyPass();

}

#endif // LLVM_TRANSFORMS_IPO_FUNCTIONATTRS_H
