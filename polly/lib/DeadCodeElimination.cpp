//===- DeadCodeElimination.cpp - Eliminate dead iteration  ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a skeleton that is meant to contain a dead code elimination pass
// later on.
//
// The idea of this pass is to loop over all statements and to remove statement
// iterations that do not calculate any value that is read later on. We need to
// make sure to forward RAR and WAR dependences.
//
// A case where this pass might be useful is
// http://llvm.org/bugs/show_bug.cgi?id=5117
//
//===----------------------------------------------------------------------===//

#include "polly/Dependences.h"
#include "isl/aff_type.h"
#include "isl/union_map.h"
#include "polly/LinkAllPasses.h"
#include "polly/ScopInfo.h"

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

  int Kinds =
      Dependences::TYPE_RAW | Dependences::TYPE_WAR | Dependences::TYPE_WAW;

  isl_union_map *Deps = D->getDependences(Kinds);

  isl_union_map_free(Deps);
  return false;
}

void DeadCodeElim::printScop(raw_ostream &OS) const {}

void DeadCodeElim::getAnalysisUsage(AnalysisUsage &AU) const {
  ScopPass::getAnalysisUsage(AU);
  AU.addRequired<Dependences>();
}

Pass *polly::createDeadCodeElimPass() { return new DeadCodeElim(); }

INITIALIZE_PASS_BEGIN(DeadCodeElim, "polly-dce",
                      "Polly - Remove dead iterations", false, false)
INITIALIZE_PASS_DEPENDENCY(Dependences)
INITIALIZE_PASS_DEPENDENCY(ScopInfo)
INITIALIZE_PASS_END(DeadCodeElim, "polly-dce", "Polly - Remove dead iterations",
                    false, false)
