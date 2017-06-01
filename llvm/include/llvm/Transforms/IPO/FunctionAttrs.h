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
struct PostOrderFunctionAttrsPass : PassInfoMixin<PostOrderFunctionAttrsPass> {
  PreservedAnalyses run(LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM,
                        LazyCallGraph &CG, CGSCCUpdateResult &UR);
};

/// Create a legacy pass manager instance of a pass to compute function attrs
/// in post-order.
Pass *createPostOrderFunctionAttrsLegacyPass();

/// A pass to do RPO deduction and propagation of function attributes.
///
/// This pass provides a general RPO or "top down" propagation of
/// function attributes. For a few (rare) cases, we can deduce significantly
/// more about function attributes by working in RPO, so this pass
/// provides the compliment to the post-order pass above where the majority of
/// deduction is performed.
// FIXME: Currently there is no RPO CGSCC pass structure to slide into and so
// this is a boring module pass, but eventually it should be an RPO CGSCC pass
// when such infrastructure is available.
class ReversePostOrderFunctionAttrsPass
    : public PassInfoMixin<ReversePostOrderFunctionAttrsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};
}

#endif // LLVM_TRANSFORMS_IPO_FUNCTIONATTRS_H
