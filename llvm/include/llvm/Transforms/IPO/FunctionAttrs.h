//===- FunctionAttrs.h - Compute function attributes ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Provides passes for computing function attributes based on interprocedural
/// analyses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_FUNCTIONATTRS_H
#define LLVM_TRANSFORMS_IPO_FUNCTIONATTRS_H

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class AAResults;
class Function;
class Module;
class Pass;

/// The three kinds of memory access relevant to 'readonly' and
/// 'readnone' attributes.
enum MemoryAccessKind {
  MAK_ReadNone = 0,
  MAK_ReadOnly = 1,
  MAK_MayWrite = 2,
  MAK_WriteOnly = 3
};

/// Returns the memory access properties of this copy of the function.
MemoryAccessKind computeFunctionBodyMemoryAccess(Function &F, AAResults &AAR);

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
/// provides the complement to the post-order pass above where the majority of
/// deduction is performed.
// FIXME: Currently there is no RPO CGSCC pass structure to slide into and so
// this is a boring module pass, but eventually it should be an RPO CGSCC pass
// when such infrastructure is available.
class ReversePostOrderFunctionAttrsPass
    : public PassInfoMixin<ReversePostOrderFunctionAttrsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

// Note: AttributeInferer was been moved from FunctionAttrs.cpp to make it
// accessible to other passes.
//
/// Collects a set of attribute inference requests and performs them all in one
/// go on a single SCC Node. Inference involves scanning function bodies
/// looking for instructions that violate attribute assumptions.
/// As soon as all the bodies are fine we are free to set the attribute.
/// Customization of inference for individual attributes is performed by
/// providing a handful of predicates for each attribute.
class AttributeInferer {
  using SCCNodeSet = SmallSetVector<Function *, 8>;
public:
  /// Describes a request for inference of a single attribute.
  struct InferenceDescriptor {

    /// Returns true if this function does not have to be handled.
    /// General intent for this predicate is to provide an optimization
    /// for functions that do not need this attribute inference at all
    /// (say, for functions that already have the attribute).
    std::function<bool(const Function &)> SkipFunction;

    /// Returns true if this instruction violates attribute assumptions.
    std::function<bool(Instruction &)> InstrBreaksAttribute;

    /// Sets the inferred attribute for this function.
    std::function<void(Function &)> SetAttribute;

    /// Attribute we derive.
    Attribute::AttrKind AKind;

    /// If true, only "exact" definitions can be used to infer this attribute.
    /// See GlobalValue::isDefinitionExact.
    bool RequiresExactDefinition;

    InferenceDescriptor(Attribute::AttrKind AK,
                        std::function<bool(const Function &)> SkipFunc,
                        std::function<bool(Instruction &)> InstrScan,
                        std::function<void(Function &)> SetAttr,
                        bool ReqExactDef)
        : SkipFunction(SkipFunc), InstrBreaksAttribute(InstrScan),
          SetAttribute(SetAttr), AKind(AK),
          RequiresExactDefinition(ReqExactDef) {}
  };

private:
  SmallVector<InferenceDescriptor, 4> InferenceDescriptors;

public:
  void registerAttrInference(InferenceDescriptor AttrInference) {
    InferenceDescriptors.push_back(AttrInference);
  }

  bool run(const SCCNodeSet &SCCNodes);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_FUNCTIONATTRS_H
