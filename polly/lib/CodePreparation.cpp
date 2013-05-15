//===---- CodePreparation.cpp - Code preparation for Scop Detection -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The Polly code preparation pass is executed before SCoP detection. Its only
// use is to translate all PHI nodes that can not be expressed by the code
// generator into explicit memory dependences. Depending of the code generation
// strategy different PHI nodes are translated:
//
// - indvars based code generation:
//
// The indvars based code generation requires explicit canonical induction
// variables. Such variables are generated before scop detection and
// also before the code preparation pass. All PHI nodes that are not canonical
// induction variables are not supported by the indvars based code generation
// and are consequently translated into explict memory accesses.
//
// - scev based code generation:
//
// The scev based code generation can code generate all PHI nodes that do not
// reference parameters within the scop. As the code preparation pass is run
// before scop detection, we can not check this condition, because without
// a detected scop, we do not know SCEVUnknowns that appear in the SCEV of
// a PHI node may later be within or outside of the SCoP. Hence, we follow a
// heuristic and translate all PHI nodes that are either directly SCEVUnknown
// or SCEVCouldNotCompute. This will hopefully get most of the PHI nodes that
// are introduced due to conditional control flow, but not the ones that are
// referencing loop counters.
//
// XXX: In the future, we should remove the need for this pass entirely and
// instead add support for scalar dependences to ScopInfo and code generation.
//
//===----------------------------------------------------------------------===//

#include "polly/LinkAllPasses.h"
#include "polly/CodeGen/BlockGenerators.h"
#include "polly/Support/ScopHelper.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;
using namespace polly;

namespace {
/// @brief Prepare the IR for the scop detection.
///
class CodePreparation : public FunctionPass {
  // DO NOT IMPLEMENT.
  CodePreparation(const CodePreparation &);
  // DO NOT IMPLEMENT.
  const CodePreparation &operator=(const CodePreparation &);

  LoopInfo *LI;
  ScalarEvolution *SE;

  void clear();

  bool eliminatePHINodes(Function &F);

public:
  static char ID;

  explicit CodePreparation() : FunctionPass(ID) {}
  ~CodePreparation();

  /// @name FunctionPass interface.
  //@{
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;
  virtual void releaseMemory();
  virtual bool runOnFunction(Function &F);
  virtual void print(raw_ostream &OS, const Module *) const;
  //@}

};
}

void CodePreparation::clear() {}

CodePreparation::~CodePreparation() { clear(); }

bool CodePreparation::eliminatePHINodes(Function &F) {
  // The PHINodes that will be deleted.
  std::vector<PHINode *> PNtoDelete;
  // The PHINodes that will be preserved.
  std::vector<PHINode *> PreservedPNs;

  // Scan the PHINodes in this function.
  for (Function::iterator BI = F.begin(), BE = F.end(); BI != BE; ++BI)
    for (BasicBlock::iterator II = BI->begin(), IE = BI->getFirstNonPHI();
         II != IE; ++II) {
      PHINode *PN = cast<PHINode>(II);
      if (SCEVCodegen) {
        if (SE->isSCEVable(PN->getType())) {
          const SCEV *S = SE->getSCEV(PN);
          if (!isa<SCEVUnknown>(S) && !isa<SCEVCouldNotCompute>(S)) {
            PreservedPNs.push_back(PN);
            continue;
          }
        }
      } else {
        if (Loop *L = LI->getLoopFor(BI)) {
          // Induction variables will be preserved.
          if (L->getCanonicalInductionVariable() == PN) {
            PreservedPNs.push_back(PN);
            continue;
          }
        }
      }

      // As DemotePHIToStack does not support invoke edges, we preserve
      // PHINodes that have invoke edges.
      if (hasInvokeEdge(PN))
        PreservedPNs.push_back(PN);
      else
        PNtoDelete.push_back(PN);
    }

  if (PNtoDelete.empty())
    return false;

  while (!PNtoDelete.empty()) {
    PHINode *PN = PNtoDelete.back();
    PNtoDelete.pop_back();

    DemotePHIToStack(PN);
  }

  // Move preserved PHINodes to the beginning of the BasicBlock.
  while (!PreservedPNs.empty()) {
    PHINode *PN = PreservedPNs.back();
    PreservedPNs.pop_back();

    BasicBlock *BB = PN->getParent();
    if (PN == BB->begin())
      continue;

    PN->moveBefore(BB->begin());
  }

  return true;
}

void CodePreparation::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfo>();
  AU.addRequired<ScalarEvolution>();

  AU.addPreserved<LoopInfo>();
  AU.addPreserved<RegionInfo>();
  AU.addPreserved<DominatorTree>();
  AU.addPreserved<DominanceFrontier>();
}

bool CodePreparation::runOnFunction(Function &F) {
  LI = &getAnalysis<LoopInfo>();
  SE = &getAnalysis<ScalarEvolution>();

  splitEntryBlockForAlloca(&F.getEntryBlock(), this);

  eliminatePHINodes(F);

  return false;
}

void CodePreparation::releaseMemory() { clear(); }

void CodePreparation::print(raw_ostream &OS, const Module *) const {}

char CodePreparation::ID = 0;
char &polly::CodePreparationID = CodePreparation::ID;

Pass *polly::createCodePreparationPass() { return new CodePreparation(); }

INITIALIZE_PASS_BEGIN(CodePreparation, "polly-prepare",
                      "Polly - Prepare code for polly", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfo)
INITIALIZE_PASS_END(CodePreparation, "polly-prepare",
                    "Polly - Prepare code for polly", false, false)
