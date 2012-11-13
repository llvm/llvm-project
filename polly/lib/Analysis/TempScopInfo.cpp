//===---------- TempScopInfo.cpp  - Extract TempScops ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Collect information about the control flow regions detected by the Scop
// detection, such that this information can be translated info its polyhedral
// representation.
//
//===----------------------------------------------------------------------===//

#include "polly/TempScopInfo.h"

#include "polly/LinkAllPasses.h"
#include "polly/Support/GICHelper.h"
#include "polly/Support/ScopHelper.h"
#include "polly/Support/SCEVValidator.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/RegionIterator.h"
#include "llvm/DataLayout.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "polly-analyze-ir"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace polly;

//===----------------------------------------------------------------------===//
/// Helper Class

void Comparison::print(raw_ostream &OS) const {
  // Not yet implemented.
}

/// Helper function to print the condition
static void printBBCond(raw_ostream &OS, const BBCond &Cond) {
  assert(!Cond.empty() && "Unexpected empty condition!");
  Cond[0].print(OS);
  for (unsigned i = 1, e = Cond.size(); i != e; ++i) {
    OS << " && ";
    Cond[i].print(OS);
  }
}

inline raw_ostream &operator<<(raw_ostream &OS, const BBCond &Cond) {
  printBBCond(OS, Cond);
  return OS;
}

//===----------------------------------------------------------------------===//
// TempScop implementation
TempScop::~TempScop() {
  if (MayASInfo) delete MayASInfo;
}

void TempScop::print(raw_ostream &OS, ScalarEvolution *SE, LoopInfo *LI) const {
  OS << "Scop: " << R.getNameStr() << ", Max Loop Depth: "<< MaxLoopDepth
    << "\n";

  printDetail(OS, SE, LI, &R, 0);
}

void TempScop::printDetail(llvm::raw_ostream &OS, ScalarEvolution *SE,
                           LoopInfo *LI, const Region *CurR,
                           unsigned ind) const {
}

void TempScopInfo::buildAccessFunctions(Region &R, BasicBlock &BB) {
  AccFuncSetType Functions;

  for (BasicBlock::iterator I = BB.begin(), E = --BB.end(); I != E; ++I) {
    Instruction &Inst = *I;
    if (isa<LoadInst>(&Inst) || isa<StoreInst>(&Inst)) {
      unsigned Size;
      enum IRAccess::TypeKind Type;

      if (LoadInst *Load = dyn_cast<LoadInst>(&Inst)) {
        Size = TD->getTypeStoreSize(Load->getType());
        Type = IRAccess::READ;
      } else {
        StoreInst *Store = cast<StoreInst>(&Inst);
        Size = TD->getTypeStoreSize(Store->getValueOperand()->getType());
        Type = IRAccess::WRITE;
      }

      const SCEV *AccessFunction = SE->getSCEV(getPointerOperand(Inst));
      const SCEVUnknown *BasePointer =
        dyn_cast<SCEVUnknown>(SE->getPointerBase(AccessFunction));

      assert(BasePointer && "Could not find base pointer");
      AccessFunction = SE->getMinusSCEV(AccessFunction, BasePointer);

      bool IsAffine = isAffineExpr(&R, AccessFunction, *SE,
                                   BasePointer->getValue());

      Functions.push_back(std::make_pair(IRAccess(Type,
                                                  BasePointer->getValue(),
                                                  AccessFunction, Size,
                                                  IsAffine),
                                         &Inst));
    }
  }

  if (Functions.empty())
    return;

  AccFuncSetType &Accs = AccFuncMap[&BB];
  Accs.insert(Accs.end(), Functions.begin(), Functions.end());
}

void TempScopInfo::buildLoopBounds(TempScop &Scop) {
  Region &R = Scop.getMaxRegion();
  unsigned MaxLoopDepth = 0;

  for (Region::block_iterator I = R.block_begin(), E = R.block_end();
       I != E; ++I) {
    Loop *L = LI->getLoopFor(*I);

    if (!L || !R.contains(L))
      continue;

    if (LoopBounds.find(L) != LoopBounds.end())
      continue;

    const SCEV *BackedgeTakenCount = SE->getBackedgeTakenCount(L);
    LoopBounds[L] = BackedgeTakenCount;

    Loop *OL = R.outermostLoopInRegion(L);
    unsigned LoopDepth = L->getLoopDepth() - OL->getLoopDepth() + 1;

    if (LoopDepth > MaxLoopDepth)
      MaxLoopDepth = LoopDepth;
  }

  Scop.MaxLoopDepth = MaxLoopDepth;
}

void TempScopInfo::buildAffineCondition(Value &V, bool inverted,
                                         Comparison **Comp,
                                         TempScop &Scop) const {
  if (ConstantInt *C = dyn_cast<ConstantInt>(&V)) {
    // If this is always true condition, we will create 1 >= 0,
    // otherwise we will create 1 == 0.
    const SCEV *LHS = SE->getConstant(C->getType(), 0);
    const SCEV *RHS = SE->getConstant(C->getType(), 1);

    if (C->isOne() == inverted)
      *Comp = new Comparison(RHS, LHS, ICmpInst::ICMP_NE);
    else
      *Comp = new Comparison(LHS, LHS, ICmpInst::ICMP_EQ);

    return;
  }

  ICmpInst *ICmp = dyn_cast<ICmpInst>(&V);
  assert(ICmp && "Only ICmpInst of constant as condition supported!");

  const SCEV *LHS = SE->getSCEV(ICmp->getOperand(0)),
             *RHS = SE->getSCEV(ICmp->getOperand(1));

  ICmpInst::Predicate Pred = ICmp->getPredicate();

  // Invert the predicate if needed.
  if (inverted)
    Pred = ICmpInst::getInversePredicate(Pred);

  switch (Pred) {
  case ICmpInst::ICMP_UGT:
  case ICmpInst::ICMP_UGE:
  case ICmpInst::ICMP_ULT:
  case ICmpInst::ICMP_ULE:
    // TODO: At the moment we need to see everything as signed. This is an
    //       correctness issue that needs to be solved.
    //AffLHS->setUnsigned();
    //AffRHS->setUnsigned();
    break;
  default:
    break;
  }

  *Comp = new Comparison(LHS, RHS, Pred);
}

void TempScopInfo::buildCondition(BasicBlock *BB, BasicBlock *RegionEntry,
                                  TempScop &Scop) {
  BBCond Cond;

  DomTreeNode *BBNode = DT->getNode(BB), *EntryNode = DT->getNode(RegionEntry);
  assert(BBNode && EntryNode && "Get null node while building condition!");

  // Walk up the dominance tree until reaching the entry node. Add all
  // conditions on the path to BB except if BB postdominates the block
  // containing the condition.
  while (BBNode != EntryNode) {
    BasicBlock *CurBB = BBNode->getBlock();
    BBNode = BBNode->getIDom();
    assert(BBNode && "BBNode should not reach the root node!");

    if (PDT->dominates(CurBB, BBNode->getBlock()))
      continue;

    BranchInst *Br = dyn_cast<BranchInst>(BBNode->getBlock()->getTerminator());
    assert(Br && "A Valid Scop should only contain branch instruction");

    if (Br->isUnconditional())
      continue;

    // Is BB on the ELSE side of the branch?
    bool inverted = DT->dominates(Br->getSuccessor(1), BB);

    Comparison *Cmp;
    buildAffineCondition(*(Br->getCondition()), inverted, &Cmp, Scop);
    Cond.push_back(*Cmp);
  }

  if (!Cond.empty())
    BBConds[BB] = Cond;
}

TempScop *TempScopInfo::buildTempScop(Region &R) {
  TempScop *TScop = new TempScop(R, LoopBounds, BBConds, AccFuncMap);

  for (Region::block_iterator I = R.block_begin(), E = R.block_end();
       I != E; ++I) {
    buildAccessFunctions(R, **I);
    buildCondition(*I, R.getEntry(), *TScop);
  }

  buildLoopBounds(*TScop);

  // Build the MayAliasSets.
  TScop->MayASInfo->buildMayAliasSets(*TScop, *AA);
  return TScop;
}

TempScop *TempScopInfo::getTempScop(const Region *R) const {
  TempScopMapType::const_iterator at = TempScops.find(R);
  return at == TempScops.end() ? 0 : at->second;
}

void TempScopInfo::print(raw_ostream &OS, const Module *) const {
  for (TempScopMapType::const_iterator I = TempScops.begin(),
       E = TempScops.end(); I != E; ++I)
    I->second->print(OS, SE, LI);
}

bool TempScopInfo::runOnFunction(Function &F) {
  DT = &getAnalysis<DominatorTree>();
  PDT = &getAnalysis<PostDominatorTree>();
  SE = &getAnalysis<ScalarEvolution>();
  LI = &getAnalysis<LoopInfo>();
  SD = &getAnalysis<ScopDetection>();
  AA = &getAnalysis<AliasAnalysis>();
  TD = &getAnalysis<DataLayout>();

  for (ScopDetection::iterator I = SD->begin(), E = SD->end(); I != E; ++I) {
    Region *R = const_cast<Region*>(*I);
    TempScops.insert(std::make_pair(R, buildTempScop(*R)));
  }

  return false;
}

void TempScopInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<DataLayout>();
  AU.addRequiredTransitive<DominatorTree>();
  AU.addRequiredTransitive<PostDominatorTree>();
  AU.addRequiredTransitive<LoopInfo>();
  AU.addRequiredTransitive<ScalarEvolution>();
  AU.addRequiredTransitive<ScopDetection>();
  AU.addRequiredID(IndependentBlocksID);
  AU.addRequired<AliasAnalysis>();
  AU.setPreservesAll();
}

TempScopInfo::~TempScopInfo() {
  clear();
}

void TempScopInfo::clear() {
  BBConds.clear();
  LoopBounds.clear();
  AccFuncMap.clear();
  DeleteContainerSeconds(TempScops);
  TempScops.clear();
}

//===----------------------------------------------------------------------===//
// TempScop information extraction pass implement
char TempScopInfo::ID = 0;

INITIALIZE_PASS_BEGIN(TempScopInfo, "polly-analyze-ir",
                      "Polly - Analyse the LLVM-IR in the detected regions",
                      false, false)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_DEPENDENCY(LoopInfo)
INITIALIZE_PASS_DEPENDENCY(PostDominatorTree)
INITIALIZE_PASS_DEPENDENCY(RegionInfo)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_DEPENDENCY(DataLayout)
INITIALIZE_PASS_END(TempScopInfo, "polly-analyze-ir",
                    "Polly - Analyse the LLVM-IR in the detected regions",
                    false, false)

Pass *polly::createTempScopInfoPass() {
  return new TempScopInfo();
}
