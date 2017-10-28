//===- SimplifyCFG.h - Simplify and canonicalize the CFG --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file provides the interface for the pass responsible for both
/// simplifying and canonicalizing the CFG.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_SIMPLIFYCFG_H
#define LLVM_TRANSFORMS_SCALAR_SIMPLIFYCFG_H

#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Utils/Local.h"

namespace llvm {

/// A pass to simplify and canonicalize the CFG of a function.
///
/// This pass iteratively simplifies the entire CFG of a function. It may change
/// or remove control flow to put the CFG into a canonical form expected by
/// other passes of the mid-level optimizer. Depending on the specified options,
/// it may further optimize control-flow to create non-canonical forms.
class SimplifyCFGPass : public PassInfoMixin<SimplifyCFGPass> {
  SimplifyCFGOptions Options;

public:
  /// The default constructor sets the pass options to create optimal IR,
  /// rather than canonical IR. That is, by default we do transformations that
  /// are likely to improve performance but make analysis more difficult.
  /// FIXME: This is inverted from what most instantiations of the pass should
  /// be.
  SimplifyCFGPass()
      : SimplifyCFGPass(SimplifyCFGOptions()
                            .forwardSwitchCondToPhi(true)
                            .convertSwitchToLookupTable(true)
                            .needCanonicalLoops(false)) {}

  /// Construct a pass with optional optimizations.
  SimplifyCFGPass(const SimplifyCFGOptions &PassOptions);

  /// \brief Run the pass over the function.
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

}

#endif
