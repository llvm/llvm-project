//===-------- PrecomputeLoop.cpp - Precompute expressions in a loop -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass detects and evaluates expressions based on loop induction
// variables. Loops and induction variables will need to have compile-time known
// trip count and increments. Then this pass will determine if the detected
// expressions can benefit from being replaced with loads to precomputed table.
// The precomputed table is initialized with values computed based on
// expressions and iteration space.
//
// For example:
//  int N = 36, sum;
//  for (int p=0; p<N; p++){
//    sum = 0;
//    for (int m=0; m < N/2; m++)
//      sum += cos_table[((2*p+1+N/2)*(2*m+1))%144];
//    out[p] = sum;
//  }
//
// Expression "(...*(2*m+1))%144" is detected and to be replaced with a single
// load to precomputed table.
// The precomputed table is created as a ConstantArray of size
// [36 x [18 x i32]], and use the expression and iteration space to initialize.
// E.g., array element at (p=0,m=2) is 95.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APInt.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/PrecomputeLoopExpressions.h"

#include <atomic>
#include <deque>
#include <map>
#include <set>
#include <vector>

#define DEBUG_TYPE "pcle"

using namespace llvm;

static cl::opt<unsigned> MinCostThreshold("pcle-min-cost", cl::Hidden,
                                          cl::init(8));

static const int kByte = 1024;
static const int MByte = 1024 * 1024;

static cl::opt<unsigned> MaxSizeThreshold("pcle-max-size", cl::Hidden,
                                          cl::init(512 * kByte));

static cl::opt<unsigned> MaxTotalSizeThreshold("pcle-max-total-size",
                                               cl::Hidden, cl::init(2 * MByte));

namespace llvm {
cl::opt<bool> DisablePCLE("disable-pcle", cl::Hidden, cl::init(false),
                          cl::desc("Disable Precomputing Loop Expressions"));
}

namespace {
typedef int32_t Integer;
#define BitSize(T) (8 * sizeof(T))

struct IVInfo {
  IVInfo() : L(0) {}
  IVInfo(Loop *Lp) : L(Lp) {}
  Integer Start, End, Bump;
  Loop *L;

  bool EqualIterSpace(const IVInfo &I) const {
    return Start == I.Start && End == I.End && Bump == I.Bump;
  }
};
#ifndef NDEBUG
raw_ostream &operator<<(raw_ostream &OS, const IVInfo &II) {
  if (II.L)
    OS << "Loop header: " << II.L->getHeader()->getName();
  else
    OS << "No loop";
  OS << "   Start:" << II.Start << "  End:" << II.End << "  Bump:" << II.Bump;
  return OS;
}
#endif

typedef std::vector<Value *> ValueVect;
typedef std::map<Value *, IVInfo> IVInfoMap;

#ifndef NDEBUG
raw_ostream &operator<<(raw_ostream &OS, const IVInfoMap &M) {
  for (auto &I : M)
    OS << I.first->getName() << " -> " << I.second << '\n';
  return OS;
}
#endif

class InitDescKey {
public:
  ArrayType *ATy;
  ValueVect IVs;

  InitDescKey() : ATy(0), IVs(), IVInfos(0) {}
  InitDescKey(ArrayType *T, ValueVect &Vs, IVInfoMap &IVM)
      : ATy(T), IVs(Vs), IVInfos(&IVM) {}

  bool operator==(const InitDescKey &K) const {
    if (ATy != K.ATy)
      return false;

    unsigned Dims = IVs.size();
    if (Dims != K.IVs.size())
      return false;

    for (unsigned i = 0; i < Dims; ++i) {
      IVInfo &I = (*IVInfos)[IVs[i]];
      IVInfo &KI = (*K.IVInfos)[K.IVs[i]];
      if (!I.EqualIterSpace(KI))
        return false;
    }
    return true;
  }
  bool operator<(const InitDescKey &K) const {
    unsigned Dims = IVs.size();
    if (Dims != K.IVs.size())
      return Dims < K.IVs.size();
    // Dims are equal here.
    if (ATy != K.ATy)
      return uintptr_t(ATy) < uintptr_t(K.ATy);
    // Types are the same.
    for (unsigned i = 0; i < Dims; ++i) {
      IVInfo &I = (*IVInfos)[IVs[i]];
      IVInfo &KI = (*K.IVInfos)[K.IVs[i]];
      if (I.Start != KI.Start)
        return I.Start < KI.Start;
      if (I.End != KI.End)
        return I.End < KI.End;
      if (I.Bump != KI.Bump)
        return I.Bump < KI.Bump;
    }
    return false;
  }

private:
  IVInfoMap *IVInfos;
};

class InitDescVal {
public:
  Value *Ex;
  GlobalVariable *GV;
  Constant *Init;
  unsigned Seq;

  InitDescVal() : Ex(0), GV(0), Init(0), Seq(0) {}
  InitDescVal(Value *E, GlobalVariable *G, Constant *I)
      : Ex(E), GV(G), Init(I), Seq(std::atomic_fetch_add(&SeqCounter, 1U)) {}

  static std::atomic<unsigned> SeqCounter;
};

typedef std::vector<Integer> IntVect;
typedef std::deque<Value *> ValueQueue;
typedef std::set<Value *> ValueSet;
typedef std::pair<GlobalVariable *, Value *> AdjustedInit;
typedef std::multimap<InitDescKey, InitDescVal> InitializerCache;

struct OrderMap {
  OrderMap() {}
  typedef std::map<Instruction *, unsigned> MapType;
  MapType::mapped_type operator[](Instruction *In) {
    if (Map.find(In) == Map.end())
      recalculate(*In->getParent()->getParent());
    assert(Map.find(In) != Map.end());
    return Map[In];
  }

  void recalculate(Function &F);
  MapType Map;
};

void OrderMap::recalculate(Function &F) {
  Map.clear();
  unsigned Ord = 0;
  for (auto &B : F)
    for (auto &I : B)
      Map.insert(std::make_pair(&I, ++Ord));
}

#ifndef NDEBUG
raw_ostream &operator<<(raw_ostream &OS, const ValueSet &S) {
  OS << '{';
  for (auto &I : S)
    OS << ' ' << *I;
  OS << " }";
  return OS;
}
#endif

class PrecomputeLoopExpressions {
public:
  PrecomputeLoopExpressions(DominatorTree *DT, LoopInfo *LI,
                            ScalarEvolution *SE, TargetLibraryInfo *TLI,
                            unsigned TotalInitSize)
      : DT(DT), LI(LI), SE(SE), TLI(TLI), TotalInitSize(TotalInitSize) {};

  bool run(Function &Fn);

private:
  bool isLoopValid(Loop *L);
  bool processLatchForIV(Instruction *TrIn, Value *&IV, IVInfo &IVI);
  bool processPHIForIV(Instruction *PIn, Value *IV, IVInfo &IVI);
  void collectInductionVariables();

  bool isAllowedOpcode(unsigned Opc);
  bool verifyExpressionNode(Value *Ex, ValueSet &Valid);
  bool verifyExpression(Value *Ex, ValueSet &Valid);
  void extendExpression(Value *Ex, ValueSet &Valid, ValueSet &New);
  unsigned computeInitializerSize(Value *V);
  unsigned computeExpressionCost(Value *V, ValueSet &Vs, unsigned ExLoopDepth);
  void collectCandidateExpressions();

  void extractInductionVariables(Value *Ex, ValueVect &IVs);
  ArrayType *createTypeForArray(Type *ETy, ValueVect &IVs);
  Integer evaluateExpression(Value *Ex, ValueVect &IVs, IntVect &C);
  Constant *createInitializerForSlice(Value *Ex, unsigned Dim, ArrayType *ATy,
                                      ValueVect &IVs, bool Zero, IntVect &C,
                                      IntVect &Starts, IntVect &Ends,
                                      IntVect &Bumps);
  Constant *createInitializerForArray(Value *Ex, ArrayType *ATy,
                                      ValueVect &IVs);
  AdjustedInit getInitializerForArray(Value *Ex, ArrayType *ATy,
                                      ValueVect &IVs);
  Value *computeDifference(Value *A, Value *B);
  bool rewriteExpression(Value *Ex, Value *Adj, ArrayType *ATy, ValueVect &IVs,
                         GlobalVariable *GV);
  bool processCandidateExpressions();

  Function *F;
  DominatorTree *DT;
  LoopInfo *LI;
  ScalarEvolution *SE;
  TargetLibraryInfo *TLI;
  OrderMap Order;

  IVInfoMap IVInfos;
  ValueSet IVEs;
  InitializerCache InitCache;
  unsigned TotalInitSize;

  static std::atomic<unsigned> Counter;
};
} // namespace

std::atomic<unsigned> InitDescVal::SeqCounter(0);
std::atomic<unsigned> PrecomputeLoopExpressions::Counter(0);

static unsigned Log2p(unsigned A) {
  if (A == 0)
    return 1;

  unsigned L = 1;
  while (A >>= 1)
    L++;

  return L;
}

bool PrecomputeLoopExpressions::isLoopValid(Loop *L) {
  BasicBlock *H = L->getHeader();
  if (!H)
    return false;
  BasicBlock *PH = L->getLoopPreheader();
  if (!PH)
    return false;
  BasicBlock *EB = L->getExitingBlock();
  if (!EB)
    return false;

  if (std::distance(pred_begin(H), pred_end(H)) != 2)
    return false;

  unsigned TC = SE->getSmallConstantTripCount(L, EB);
  if (TC == 0)
    return false;

  return true;
}

bool PrecomputeLoopExpressions::processLatchForIV(Instruction *TrIn, Value *&IV,
                                                  IVInfo &IVI) {
  // Need a conditional branch.
  BranchInst *Br = dyn_cast<BranchInst>(TrIn);
  if (!Br || !Br->isConditional())
    return false;

  // The branch condition needs to be an integer compare.
  Value *CV = Br->getCondition();
  Instruction *CIn = dyn_cast<Instruction>(CV);
  if (!CIn || CIn->getOpcode() != Instruction::ICmp)
    return false;

  // The comparison has to be less-than.
  ICmpInst *ICIn = cast<ICmpInst>(CIn);
  CmpInst::Predicate P = ICIn->getPredicate();
  if (P != CmpInst::ICMP_ULT && P != CmpInst::ICMP_SLT)
    return false;

  // Less-than a constant int to be exact.
  Value *CR = ICIn->getOperand(1);
  if (!isa<ConstantInt>(CR))
    return false;

  // The int has to fit in 32 bits.
  const APInt &U = cast<ConstantInt>(CR)->getValue();
  if (!U.isSignedIntN(BitSize(Integer)))
    return false;

  // The value that is less-than the int needs to be an add.
  Value *VC = ICIn->getOperand(0);
  Instruction *VCIn = dyn_cast<Instruction>(VC);
  if (!VCIn || VCIn->getOpcode() != Instruction::Add)
    return false;

  // An add of a constant int.
  Value *ValA, *ValI;
  if (isa<ConstantInt>(VCIn->getOperand(1))) {
    ValA = VCIn->getOperand(0);
    ValI = VCIn->getOperand(1);
  } else {
    ValA = VCIn->getOperand(1);
    ValI = VCIn->getOperand(0);
  }
  if (!isa<ConstantInt>(ValI))
    return false;

  // The added int has to fit in 32 bits.
  const APInt &B = cast<ConstantInt>(ValI)->getValue();
  if (!B.isSignedIntN(BitSize(Integer)))
    return false;

  // Done...
  IV = ValA;
  IVI.End = (Integer)U.getSExtValue();
  IVI.Bump = (Integer)B.getSExtValue();
  return true;
}

bool PrecomputeLoopExpressions::processPHIForIV(Instruction *PIn, Value *IV,
                                                IVInfo &IVI) {
  if (IV != PIn)
    return false;

  // The PHI must only have two incoming blocks.
  PHINode *P = cast<PHINode>(PIn);
  if (P->getNumIncomingValues() != 2)
    return false;

  // The blocks have to be preheader and loop latch.
  BasicBlock *PH = IVI.L->getLoopPreheader();
  BasicBlock *LT = IVI.L->getLoopLatch();

  if (P->getIncomingBlock(0) == PH) {
    if (P->getIncomingBlock(1) != LT)
      return false;
  } else if (P->getIncomingBlock(1) == PH) {
    if (P->getIncomingBlock(0) != LT)
      return false;
  } else {
    return false;
  }

  // The value coming from the preheader needs to be a constant int.
  Value *VPH = P->getIncomingValueForBlock(PH);
  if (!isa<ConstantInt>(VPH))
    return false;

  // That int has to fit in 32 bits.
  const APInt &S = cast<ConstantInt>(VPH)->getValue();
  if (!S.isSignedIntN(BitSize(Integer)))
    return false;

  // All checks passed.
  IVI.Start = static_cast<Integer>(S.getSExtValue());
  return true;
}

void PrecomputeLoopExpressions::collectInductionVariables() {
  IVInfos.clear();

  typedef std::deque<Loop *> LoopQueue;
  LoopQueue Work;

  for (LoopInfo::iterator I = LI->begin(), E = LI->end(); I != E; ++I) {
    Work.push_back(*I);
  }

  while (!Work.empty()) {
    Loop *L = Work.front();
    Work.pop_front();

    for (Loop::iterator I = L->begin(), E = L->end(); I != E; ++I) {
      Work.push_back(*I);
    }
    if (!isLoopValid(L))
      continue;

    Value *IV;
    IVInfo IVI(L);
    Instruction *TrIn = L->getLoopLatch()->getTerminator();

    bool LatchOk = processLatchForIV(TrIn, IV, IVI);
    if (!LatchOk)
      continue;

    BasicBlock *H = L->getHeader();
    for (BasicBlock::iterator PI = H->begin(); isa<PHINode>(PI); ++PI) {
      Instruction *I = &*PI;
      if (I == IV) {
        bool PHIOk = processPHIForIV(I, IV, IVI);
        if (PHIOk) {
          IVInfos.insert(std::make_pair(IV, IVI));
        }
        break;
      }
    }
  }
}

bool PrecomputeLoopExpressions::isAllowedOpcode(unsigned Opc) {
  switch (Opc) {
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::URem:
  case Instruction::SRem:
  case Instruction::Shl:
  case Instruction::AShr:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    return true;
  }
  return false;
}

bool PrecomputeLoopExpressions::verifyExpressionNode(Value *Ex,
                                                     ValueSet &Valid) {
  Type *T = Ex->getType();
  if (!T->isIntegerTy())
    return false;
  if (cast<IntegerType>(T)->getBitWidth() > BitSize(Integer))
    return false;

  Instruction *In = dyn_cast<Instruction>(Ex);
  if (!In)
    return false;
  if (!isAllowedOpcode(In->getOpcode()))
    return false;

  return true;
}

bool PrecomputeLoopExpressions::verifyExpression(Value *Ex, ValueSet &Valid) {
  if (Valid.count(Ex))
    return true;
  if (isa<ConstantInt>(Ex))
    return true;

  if (!verifyExpressionNode(Ex, Valid))
    return false;

  assert(isa<Instruction>(Ex) && "Should have checked for instruction");
  Instruction *In = cast<Instruction>(Ex);
  for (unsigned i = 0, n = In->getNumOperands(); i < n; ++i) {
    bool ValidOp = verifyExpression(In->getOperand(i), Valid);
    if (!ValidOp)
      return false;
  }
  return true;
}

void PrecomputeLoopExpressions::extendExpression(Value *Ex, ValueSet &Valid,
                                                 ValueSet &New) {
  for (Value::user_iterator I = Ex->user_begin(), E = Ex->user_end(); I != E;
       ++I) {
    Value *U = *I;
    if (Valid.count(U))
      continue;
    if (U->getType()->isVoidTy())
      continue;

    bool BadUser = false;

    if (Instruction *In = dyn_cast<Instruction>(U)) {
      if (In->getOpcode() == Instruction::PHI)
        continue;
      if (!verifyExpressionNode(U, Valid))
        continue;

      for (unsigned i = 0, n = In->getNumOperands(); i < n; ++i) {
        Value *Op = In->getOperand(i);
        if (Op != Ex && !verifyExpression(Op, Valid)) {
          BadUser = true;
          break;
        }
      }
    } else {
      BadUser = true;
    }
    if (BadUser)
      continue;

    New.insert(U);
  }
}

unsigned
PrecomputeLoopExpressions::computeExpressionCost(Value *V, ValueSet &Vs,
                                                 unsigned ExLoopDepth) {
  if (Vs.count(V))
    return 0;
  Vs.insert(V);

  unsigned C = 0;
  if (Instruction *In = dyn_cast<Instruction>(V)) {
    switch (In->getOpcode()) {
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::Shl:
    case Instruction::AShr:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor:
      C = 1;
      break;
    case Instruction::Mul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::URem:
    case Instruction::SRem:
      C = 3;
      break;
    case Instruction::PHI:
      return 0;
    }

    for (unsigned i = 0, n = In->getNumOperands(); i < n; ++i) {
      Value *Op = In->getOperand(i);
      unsigned OpLoopDepth =
          isa<Instruction>(Op)
              ? LI->getLoopDepth(cast<Instruction>(Op)->getParent())
              : 0;
      if (OpLoopDepth != ExLoopDepth)
        continue;
      C += computeExpressionCost(In->getOperand(i), Vs, ExLoopDepth);
    }
  }

  return C;
}

unsigned PrecomputeLoopExpressions::computeInitializerSize(Value *V) {
  ValueVect IVs;

  extractInductionVariables(V, IVs);

  Type *T = V->getType();
  assert(T->isIntegerTy());
  unsigned Total = (cast<IntegerType>(T)->getBitWidth()) / 8;

  for (unsigned i = 0, Dims = IVs.size(); i < Dims; ++i) {
    IVInfo &IVI = IVInfos[IVs[i]];
    unsigned D = std::abs(IVI.End - IVI.Start);
    if (Log2p(D) + Log2p(Total) > 8 * sizeof(Integer))
      return UINT_MAX;
    Total *= D;
  }

  return Total;
}

void PrecomputeLoopExpressions::collectCandidateExpressions() {
  ValueQueue Work;

  IVEs.clear();

  for (auto &KV : IVInfos) {
    IVEs.insert(KV.first);
    Work.push_back(KV.first);
  }

  ValueSet NewIVEs;
  while (!Work.empty()) {
    Value *V = Work.front();
    Work.pop_front();
    NewIVEs.clear();

    extendExpression(V, IVEs, NewIVEs);
    IVEs.insert(NewIVEs.begin(), NewIVEs.end());
    Work.insert(Work.end(), NewIVEs.begin(), NewIVEs.end());
  }

  auto remove_if = [](auto &set, auto cond) {
    for (auto it = set.begin(); it != set.end();) {
      if (cond(*it))
        it = set.erase(it);
      else
        ++it;
    }
  };

  // Prune all IV expressions that would require oversize initializers.
  remove_if(IVEs, [this](Value *V) {
    return computeInitializerSize(V) >= MaxSizeThreshold;
  });

  // Remove IV expressions that are always subexpressions of another
  // IV expression.
  remove_if(IVEs, [this](Value *V) {
    return llvm::all_of(V->users(), [this](Value *U) { return IVEs.count(U); });
  });

  // Remove IV expressions that are not complex enough. Always remove the
  // IV themselves.
  ValueSet Tmp;
  remove_if(IVEs, [&](Value *V) {
    Tmp.clear();
    unsigned LoopDepth =
        isa<Instruction>(V)
            ? LI->getLoopDepth(cast<Instruction>(V)->getParent())
            : 0;
    return IVInfos.count(V) ||
           computeExpressionCost(V, Tmp, LoopDepth) < MinCostThreshold;
  });

  // Remove IV expression where its subexpression is used in higher order
  // instructions.
  remove_if(IVEs, [this](Value *V) {
    Instruction *Ex = cast<Instruction>(V);
    for (unsigned i = 0; i < Ex->getNumOperands(); ++i) {
      Instruction *Op = dyn_cast<Instruction>(Ex->getOperand(i));
      if (Op && llvm::any_of(Op->users(), [&](Value *U) {
            return Order[cast<Instruction>(U)] > Order[Ex];
          }))
        return true;
      else
        continue;
    }
    return false;
  });
}

void PrecomputeLoopExpressions::extractInductionVariables(Value *Ex,
                                                          ValueVect &IVs) {
  ValueQueue Work;
  Work.push_back(Ex);

  ValueSet Memo;
  Memo.insert(Ex);

  while (!Work.empty()) {
    Value *V = Work.front();
    Work.pop_front();

    if (IVInfos.count(V)) {
      IVs.push_back(V);
    } else if (Instruction *In = dyn_cast<Instruction>(V)) {
      for (unsigned i = 0, n = In->getNumOperands(); i < n; ++i) {
        Value *Op = In->getOperand(i);
        if (!Memo.count(Op)) {
          Memo.insert(Op);
          Work.push_back(Op);
        }
      }
    }
  }
}

ArrayType *PrecomputeLoopExpressions::createTypeForArray(Type *ETy,
                                                         ValueVect &IVs) {
  ArrayType *ATy = 0;

  for (ValueVect::iterator I = IVs.begin(), E = IVs.end(); I != E; ++I) {
    Value *V = *I;
    IVInfo &IVI = IVInfos[V];
    assert((IVI.Start < IVI.End) && "Backward loop?");

    if (ATy) {
      ATy = ArrayType::get(ATy, IVI.End - IVI.Start);
    } else {
      ATy = ArrayType::get(ETy, IVI.End - IVI.Start);
    }
  }

  return ATy;
}

Integer PrecomputeLoopExpressions::evaluateExpression(Value *Ex, ValueVect &IVs,
                                                      IntVect &C) {

  if (ConstantInt *CI = dyn_cast<ConstantInt>(Ex)) {
    const APInt &A = CI->getValue();
    return A.getSExtValue();
  }

  if (Instruction *In = dyn_cast<Instruction>(Ex)) {
    switch (In->getOpcode()) {
    case Instruction::Add:
      return evaluateExpression(In->getOperand(0), IVs, C) +
             evaluateExpression(In->getOperand(1), IVs, C);
    case Instruction::Sub:
      return evaluateExpression(In->getOperand(0), IVs, C) -
             evaluateExpression(In->getOperand(1), IVs, C);
    case Instruction::Shl:
      return evaluateExpression(In->getOperand(0), IVs, C)
             << evaluateExpression(In->getOperand(1), IVs, C);
    case Instruction::AShr:
      return evaluateExpression(In->getOperand(0), IVs, C) >>
             evaluateExpression(In->getOperand(1), IVs, C);
    case Instruction::Mul:
      return evaluateExpression(In->getOperand(0), IVs, C) *
             evaluateExpression(In->getOperand(1), IVs, C);
    case Instruction::UDiv:
    case Instruction::SDiv:
      return evaluateExpression(In->getOperand(0), IVs, C) /
             evaluateExpression(In->getOperand(1), IVs, C);
    case Instruction::URem:
    case Instruction::SRem:
      return evaluateExpression(In->getOperand(0), IVs, C) %
             evaluateExpression(In->getOperand(1), IVs, C);
    case Instruction::And:
      return evaluateExpression(In->getOperand(0), IVs, C) &
             evaluateExpression(In->getOperand(1), IVs, C);
    case Instruction::Or:
      return evaluateExpression(In->getOperand(0), IVs, C) |
             evaluateExpression(In->getOperand(1), IVs, C);
    case Instruction::Xor:
      return evaluateExpression(In->getOperand(0), IVs, C) ^
             evaluateExpression(In->getOperand(1), IVs, C);
    default:
      break;
    }
  }

  for (unsigned i = 0, n = IVs.size(); i < n; ++i) {
    if (Ex == IVs[i])
      return C[i];
  }

  dbgs() << *Ex << "\n";
  llvm_unreachable("Unexpected expression");
  return 0;
}

Constant *PrecomputeLoopExpressions::createInitializerForSlice(
    Value *Ex, unsigned Dim, ArrayType *ATy, ValueVect &IVs, bool Zero,
    IntVect &C, IntVect &Starts, IntVect &Ends, IntVect &Bumps) {
  Integer S = Starts[Dim];
  Integer E = Ends[Dim];
  Integer B = Bumps[Dim];
  std::vector<Constant *> Init(E - S);

  Type *ETy = ATy->getElementType();

  for (unsigned i = 0; i <= Dim; ++i) {
    C[i] = Starts[i];
  }

  if (Dim > 0) {
    assert(ETy->isArrayTy() && "Expecting array type");
    ArrayType *AETy = cast<ArrayType>(ETy);

    Integer i = S;
    while (i < E) {
      Init[i - S] = createInitializerForSlice(Ex, Dim - 1, AETy, IVs, Zero, C,
                                              Starts, Ends, Bumps);
      i++;
      for (Integer j = 0; j < B - 1; ++j) {
        if (i >= E)
          break;
        Init[i - S] = createInitializerForSlice(Ex, Dim - 1, AETy, IVs, true, C,
                                                Starts, Ends, Bumps);
        i++;
      }
      C[Dim] += B;
    }
  } else {
    assert(ETy->isIntegerTy() && "Expecting integer type");
    IntegerType *IETy = cast<IntegerType>(ETy);

    Integer i = S;
    while (i < E) {
      Integer A = Zero ? 0 : evaluateExpression(Ex, IVs, C);
      Init[i - S] = ConstantInt::getSigned(IETy, A);
      i++;
      for (Integer j = 0; j < B - 1; ++j) {
        if (i >= E)
          break;
        Init[i - S] = ConstantInt::getSigned(IETy, 0);
        i++;
      }
      C[Dim] += B;
    }
  }

  ArrayRef<Constant *> AR(Init);
  return ConstantArray::get(ATy, AR);
}

Constant *PrecomputeLoopExpressions::createInitializerForArray(Value *Ex,
                                                               ArrayType *ATy,
                                                               ValueVect &IVs) {
  unsigned Dims = IVs.size();
  IntVect C(Dims), Starts(Dims), Ends(Dims), Bumps(Dims);

  for (unsigned i = 0; i < Dims; ++i) {
    IVInfo &IVI = IVInfos[IVs[i]];
    Starts[i] = IVI.Start;
    Ends[i] = IVI.End;
    Bumps[i] = IVI.Bump;
  }

  return createInitializerForSlice(Ex, Dims - 1, ATy, IVs, false, C, Starts,
                                   Ends, Bumps);
}

Value *PrecomputeLoopExpressions::computeDifference(Value *A, Value *B) {
  assert(isa<Instruction>(A));
  assert(isa<Instruction>(B));

  Instruction *InA = cast<Instruction>(A)->clone();
  Instruction *InB = cast<Instruction>(B)->clone();

  ValueVect IVsA, IVsB;
  extractInductionVariables(InA, IVsA);
  extractInductionVariables(InB, IVsB);
  for (unsigned i = 0, Dims = IVsA.size(); i < Dims; ++i) {
    if (IVsA[i] != IVsB[i])
      InA->replaceUsesOfWith(IVsA[i], IVsB[i]);
  }

  const DataLayout &DL = F->getParent()->getDataLayout();

  // InstSimplify does not allow the use of functions that do
  // not belong in a function, so we insert the BinOp temporarily
  // into the function containing the values A/B
  std::optional<BasicBlock::iterator> InsertPos;
  if (DT->dominates(A, cast<Instruction>(B))) {
    InsertPos = cast<Instruction>(B)->getInsertionPointAfterDef();
    InA->insertBefore(*InsertPos);
    InB->insertBefore(*InsertPos);
  } else {
    InsertPos = cast<Instruction>(A)->getInsertionPointAfterDef();
    InB->insertBefore(*InsertPos);
    InA->insertBefore(*InsertPos);
  }

  BinaryOperator *Sub = BinaryOperator::Create(Instruction::Sub, InA, InB);
  Sub->insertBefore(*InsertPos);

  Value *S = simplifyInstruction(Sub, {DL, TLI});

  Sub->eraseFromParent();
  InA->eraseFromParent();
  InB->eraseFromParent();

  return S; // Can return null.
}

bool PrecomputeLoopExpressions::rewriteExpression(Value *Ex, Value *Adj,
                                                  ArrayType *ATy,
                                                  ValueVect &IVs,
                                                  GlobalVariable *GV) {
  unsigned Dims = IVs.size();

  Instruction *In = dyn_cast<Instruction>(Ex);
  assert(In && "Expecting instruction");
  IRBuilder<> Builder(In);

  ValueVect Ops(Dims + 1);

  Ops[0] = ConstantInt::get(IVs[0]->getType(), 0);

  for (unsigned i = 0; i < Dims; ++i) {
    unsigned IVx = Dims - i - 1;
    IVInfo &IVI = IVInfos[IVs[IVx]];
    if (IVI.Start != 0) {
      Value *StartV = ConstantInt::get(Ex->getType(), IVI.Start);
      Ops[i + 1] = Builder.CreateSub(IVs[IVx], StartV);
    } else {
      Ops[i + 1] = IVs[IVx];
    }
  }

  ArrayRef<Value *> Idx(Ops);
  Value *GEP = Builder.CreateGEP(GV->getValueType(), GV, Idx, "txgep");
  Type *ResultTy = cast<GEPOperator>(GEP)->getResultElementType();
  Value *Load = Builder.CreateLoad(ResultTy, GEP, "txld");
  Value *LoadAdj = Adj ? Builder.CreateAdd(Load, Adj, "txa") : Load;

  Ex->replaceAllUsesWith(LoadAdj);
  return true;
}

struct LoopCompare {
  LoopCompare(IVInfoMap &M) : IVMap(M) {}
  bool operator()(Value *V, Value *W) {
    Loop *LV = IVMap[V].L;
    Loop *LW = IVMap[W].L;
    return LW->contains(LV);
  }

private:
  IVInfoMap &IVMap;
};

static bool IsSubexpression(Value *A, Value *B) {
  if (A == B)
    return true;
  if (!isa<Instruction>(B))
    return false;

  Instruction *InB = cast<Instruction>(B);

  if (InB->getOpcode() == Instruction::PHI)
    return false;

  for (unsigned i = 0, n = InB->getNumOperands(); i < n; ++i) {
    if (A == InB->getOperand(i))
      return true;
  }
  for (unsigned i = 0, n = InB->getNumOperands(); i < n; ++i) {
    if (IsSubexpression(A, InB->getOperand(i)))
      return true;
  }
  return false;
}

bool PrecomputeLoopExpressions::processCandidateExpressions() {
  ValueVect IVs;
  LoopCompare LC(IVInfos);
  InitCache.clear();
  bool Changed = false;

  auto ExprLess = [this](Value *A, Value *B) {
    if (A == B)
      return false;

    // First, sort by subexpression. We want the largest expression
    // to be the max element.
    if (IsSubexpression(A, B))
      return true;

    if (IsSubexpression(B, A))
      return false;

    Instruction *InA = cast<Instruction>(A);
    Instruction *InB = cast<Instruction>(B);
    // Next, sort by dominance. If B dominates A, assume A < B.
    if (DT->dominates(InB, InA))
      return true;

    // Finally, resolve the tie by the order in the function.
    // Expressions appearing first will be considered "greater".
    return Order[InB] < Order[InA];
  };

  struct RewriteRec {
    RewriteRec(Value *E, ArrayType *T, AdjustedInit I, const ValueVect &V)
        : Ex(E), ATy(T), AI(I), IVs(V) {}
    Value *Ex;
    ArrayType *ATy;
    AdjustedInit AI;
    ValueVect IVs;
  };

  std::vector<RewriteRec> RewriteList;

  while (!IVEs.empty()) {
    ValueSet::iterator M = std::max_element(IVEs.begin(), IVEs.end(), ExprLess);
    Value *Ex = *M;
    IVEs.erase(Ex);
    LLVM_DEBUG(dbgs() << "Processing: " << *Ex << '\n');

    // Some expressions can have enough of their subexpressions precomputed,
    // that they are no longer expensive.
    ValueSet Tmp;
    unsigned LoopDepth =
        isa<Instruction>(Ex)
            ? LI->getLoopDepth(cast<Instruction>(Ex)->getParent())
            : 0;
    if (computeExpressionCost(Ex, Tmp, LoopDepth) < MinCostThreshold)
      continue;

    IVs.clear();
    extractInductionVariables(Ex, IVs);
    // Sort from innermost to outermost.
    std::sort(IVs.begin(), IVs.end(), LC);

    ArrayType *ATy = createTypeForArray(Ex->getType(), IVs);
    AdjustedInit AI = getInitializerForArray(Ex, ATy, IVs);

    if (AI.first)
      RewriteList.push_back(RewriteRec(Ex, ATy, AI, IVs));
  }

  for (auto &R : RewriteList)
    Changed |= rewriteExpression(R.Ex, R.AI.second, R.ATy, R.IVs, R.AI.first);

  return Changed;
}

bool PrecomputeLoopExpressions::run(Function &Fn) {

  LLVM_DEBUG(dbgs() << "Before PCLE\n" << Fn);

  F = &Fn;

  Order.recalculate(Fn);

  collectInductionVariables();
  if (IVInfos.empty())
    return false;
  LLVM_DEBUG(dbgs() << "IV Infos:\n" << IVInfos);

  collectCandidateExpressions();
  if (IVEs.empty())
    return false;
  LLVM_DEBUG(dbgs() << "IVEs:\n" << IVEs << '\n');

  bool Changed = processCandidateExpressions();
  if (Changed)
    LLVM_DEBUG(dbgs() << "After PCLE\n" << Fn);

  return Changed;
}

AdjustedInit PrecomputeLoopExpressions::getInitializerForArray(Value *Ex,
                                                               ArrayType *ATy,
                                                               ValueVect &IVs) {

  typedef InitializerCache::iterator ICIterator;
  InitDescKey DK(ATy, IVs, IVInfos);

  std::pair<ICIterator, ICIterator> P = InitCache.equal_range(DK);

  unsigned MinCost = UINT_MAX;
  unsigned MinSeq = UINT_MAX;
  Value *MinDiff = 0;
  GlobalVariable *MinGV = 0;
  ValueSet Tmp;

  for (ICIterator I = P.first; I != P.second; ++I) {
    Tmp.clear();
    InitDescVal &D = I->second;
    Value *Diff = computeDifference(Ex, D.Ex);
    if (!Diff)
      continue;

    unsigned LoopDepth =
        isa<Instruction>(Diff)
            ? LI->getLoopDepth(cast<Instruction>(Diff)->getParent())
            : 0;
    unsigned C = computeExpressionCost(Diff, Tmp, LoopDepth);
    if (C < MinCost || (C == MinCost && D.Seq < MinSeq)) {
      MinCost = C;
      MinSeq = D.Seq;
      // The result of "computeDifference" can be a constant.  In such
      // case we wouldn't want to delete it, since it may be used by
      // other instructions.  Only delete it if it has no uses.
      // Don't delete constants because have no SSA uses.
      //
      // FIXME: Prevent potential memory leaks.
      if (MinDiff && MinDiff->use_empty() && !isa<Constant>(MinDiff))
        MinDiff->deleteValue();
      MinDiff = Diff;
      MinGV = D.GV;
    } else {
      if (Diff && Diff->use_empty() && !isa<Constant>(Diff))
        Diff->deleteValue();
    }
  }

  if (MinCost < MinCostThreshold)
    return std::make_pair(MinGV, MinDiff);

  unsigned S = computeInitializerSize(Ex);
  unsigned TS = TotalInitSize + S;
  // Check if there is an overflow in the addition or total size threshold
  // was exceeded.
  if (TS < TotalInitSize || TS < S || TS > MaxTotalSizeThreshold)
    return std::make_pair<GlobalVariable *, Value *>(0, 0);

  TotalInitSize = TS;

  // No candidates to reuse.  Create a new initializer.
  unsigned VarN = std::atomic_fetch_add(&Counter, 1U);
  GlobalVariable *GV = new GlobalVariable(*F->getParent(), ATy, true,
                                          GlobalValue::PrivateLinkage, 0,
                                          Twine("tx") + Twine(VarN));
  Constant *Init = createInitializerForArray(Ex, ATy, IVs);
  GV->setInitializer(Init);

  InitDescVal DV(Ex, GV, Init);
  InitCache.insert(std::make_pair(DK, DV));

  return std::pair<GlobalVariable *, Value *>(GV, 0);
}

PreservedAnalyses
PrecomputeLoopExpressionsPass::run(Function &F, FunctionAnalysisManager &AM) {
  // FIXME: skipFunction

  auto *DT = &AM.getResult<DominatorTreeAnalysis>(F);
  auto *LI = &AM.getResult<LoopAnalysis>(F);
  auto *SE = &AM.getResult<ScalarEvolutionAnalysis>(F);
  auto *TLI = &AM.getResult<TargetLibraryAnalysis>(F);

  PrecomputeLoopExpressions PCLE(DT, LI, SE, TLI, TotalInitSize);

  if (PCLE.run(F))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
