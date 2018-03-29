//===---- Canonicalization.cpp - Run canonicalization passes --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Run the set of default canonicalization passes.
//
// This pass is mainly used for debugging.
//
//===----------------------------------------------------------------------===//

#include "polly/Canonicalization.h"
#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils.h"

using namespace llvm;
using namespace polly;

static cl::opt<bool>
    PollyInliner("polly-run-inliner",
                 cl::desc("Run an early inliner pass before Polly"), cl::Hidden,
                 cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

void polly::registerCanonicalicationPasses(llvm::legacy::PassManagerBase &PM) {
  bool UseMemSSA = true;
  PM.add(polly::createRewriteByrefParamsPass());
  PM.add(llvm::createPromoteMemoryToRegisterPass());
  PM.add(llvm::createEarlyCSEPass(UseMemSSA));
  PM.add(llvm::createInstructionCombiningPass());
  PM.add(llvm::createCFGSimplificationPass());
  PM.add(llvm::createTailCallEliminationPass());
  PM.add(llvm::createCFGSimplificationPass());
  PM.add(llvm::createReassociatePass());
  PM.add(llvm::createLoopRotatePass());
  if (PollyInliner) {
    PM.add(llvm::createFunctionInliningPass(200));
    PM.add(llvm::createPromoteMemoryToRegisterPass());
    PM.add(llvm::createCFGSimplificationPass());
    PM.add(llvm::createInstructionCombiningPass());
    PM.add(createBarrierNoopPass());
  }
  PM.add(llvm::createInstructionCombiningPass());
  PM.add(llvm::createIndVarSimplifyPass());
  PM.add(polly::createCodePreparationPass());
}

namespace {
class PollyCanonicalize : public ModulePass {
  PollyCanonicalize(const PollyCanonicalize &) = delete;
  const PollyCanonicalize &operator=(const PollyCanonicalize &) = delete;

public:
  static char ID;

  explicit PollyCanonicalize() : ModulePass(ID) {}
  ~PollyCanonicalize();

  /// @name FunctionPass interface.
  //@{
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;
  virtual void releaseMemory();
  virtual bool runOnModule(Module &M);
  virtual void print(raw_ostream &OS, const Module *) const;
  //@}
};
} // namespace

PollyCanonicalize::~PollyCanonicalize() {}

void PollyCanonicalize::getAnalysisUsage(AnalysisUsage &AU) const {}

void PollyCanonicalize::releaseMemory() {}

bool PollyCanonicalize::runOnModule(Module &M) {
  legacy::PassManager PM;
  registerCanonicalicationPasses(PM);
  PM.run(M);

  return true;
}

void PollyCanonicalize::print(raw_ostream &OS, const Module *) const {}

char PollyCanonicalize::ID = 0;

Pass *polly::createPollyCanonicalizePass() { return new PollyCanonicalize(); }

INITIALIZE_PASS_BEGIN(PollyCanonicalize, "polly-canonicalize",
                      "Polly - Run canonicalization passes", false, false)
INITIALIZE_PASS_END(PollyCanonicalize, "polly-canonicalize",
                    "Polly - Run canonicalization passes", false, false)
