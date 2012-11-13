//===- DeadCodeElimination.cpp - Eliminate dead iteration  ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// If values calculated within an iteration are not used later on the iteration
// can be removed entirely. This pass removes such iterations.
//===----------------------------------------------------------------------===//

#include "polly/Dependences.h"
#include "polly/LinkAllPasses.h"
#include "polly/ScopInfo.h"

#include "isl/union_map.h"

using namespace llvm;
using namespace polly;

namespace {

  class DeadCodeElim : public ScopPass {

  public:
    static char ID;
    explicit DeadCodeElim() : ScopPass(ID) {}

    virtual bool runOnScop(Scop &S);
    void printScop(llvm::raw_ostream &OS) const;
    void getAnalysisUsage(AnalysisUsage &AU) const;
  };
}

char DeadCodeElim::ID = 0;

bool DeadCodeElim::runOnScop(Scop &S) {
  Dependences *D = &getAnalysis<Dependences>();

  int dependencyKinds = Dependences::TYPE_RAW
                          | Dependences::TYPE_WAR
                          | Dependences::TYPE_WAW;

  isl_union_map *dependences = D->getDependences(dependencyKinds);

  // The idea of this pass is to loop over all statments and remove statement
  // iterations that do not calculate any value that is read later on. We need
  // to make sure to forward RAR and WAR dependences.
  //
  // A case where this pass might be useful is
  // http://llvm.org/bugs/show_bug.cgi?id=5117
  isl_union_map_free(dependences);
  return false;
}

void DeadCodeElim::printScop(raw_ostream &OS) const {
}

void DeadCodeElim::getAnalysisUsage(AnalysisUsage &AU) const {
  ScopPass::getAnalysisUsage(AU);
  AU.addRequired<Dependences>();
}

INITIALIZE_PASS_BEGIN(DeadCodeElim, "polly-dce",
                      "Polly - Remove dead iterations", false, false)
INITIALIZE_PASS_DEPENDENCY(Dependences)
INITIALIZE_PASS_DEPENDENCY(ScopInfo)
INITIALIZE_PASS_END(DeadCodeElim, "polly-dce",
                      "Polly - Remove dead iterations", false, false)

Pass* polly::createDeadCodeElimPass() {
  return new DeadCodeElim();
}
