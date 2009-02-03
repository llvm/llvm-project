//===- CallGraphSCCPass.h - Pass that operates BU on call graph -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the CallGraphSCCPass class, which is used for passes which
// are implemented as bottom-up traversals on the call graph.  Because there may
// be cycles in the call graph, passes of this type operate on the call-graph in
// SCC order: that is, they process function bottom-up, except for recursive
// functions, which they process all at once.
//
// These passes are inherently interprocedural, and are required to keep the
// call graph up-to-date if they do anything which could modify it.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CALL_GRAPH_SCC_PASS_H
#define LLVM_CALL_GRAPH_SCC_PASS_H

#include "llvm/Pass.h"

namespace llvm {

class CallGraphNode;
class CallGraph;
class PMStack;

struct CallGraphSCCPass : public Pass {

  explicit CallGraphSCCPass(intptr_t pid) : Pass(pid) {}
  explicit CallGraphSCCPass(void *pid) : Pass(pid) {}

  /// doInitialization - This method is called before the SCC's of the program
  /// has been processed, allowing the pass to do initialization as necessary.
  virtual bool doInitialization(CallGraph &CG) {
    return false;
  }

  /// runOnSCC - This method should be implemented by the subclass to perform
  /// whatever action is necessary for the specified SCC.  Note that
  /// non-recursive (or only self-recursive) functions will have an SCC size of
  /// 1, where recursive portions of the call graph will have SCC size > 1.
  ///
  virtual bool runOnSCC(const std::vector<CallGraphNode *> &SCC) = 0;

  /// doFinalization - This method is called after the SCC's of the program has
  /// been processed, allowing the pass to do final cleanup as necessary.
  virtual bool doFinalization(CallGraph &CG) {
    return false;
  }

  /// Assign pass manager to manager this pass
  virtual void assignPassManager(PMStack &PMS,
                                 PassManagerType PMT = PMT_CallGraphPassManager);

  ///  Return what kind of Pass Manager can manage this pass.
  virtual PassManagerType getPotentialPassManagerType() const {
    return PMT_CallGraphPassManager;
  }

  /// getAnalysisUsage - For this class, we declare that we require and preserve
  /// the call graph.  If the derived class implements this method, it should
  /// always explicitly call the implementation here.
  virtual void getAnalysisUsage(AnalysisUsage &Info) const;
};

} // End llvm namespace

#endif
