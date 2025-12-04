//===---------------------- HexagonOptShuffleVector.cpp -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Optimize vector shuffles by postponing them as late as possible. The intent
// here is to remove uncessary shuffles and also increases the oportunities for
// adjacent shuffles to be merged together.
//
//===----------------------------------------------------------------------===//

#include "HexagonTargetMachine.h"
#include "llvm/ADT/APInt.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsHexagon.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "hex-shuff-vec"
/// A command line argument to limit the search space along def chain.
static cl::opt<int> MaxDefSearchCount(
    "shuffvec-max-search-count",
    cl::desc("Maximum number of instructions traversed along def chain."),
    cl::Hidden, cl::init(15));

#ifndef NDEBUG
static cl::opt<int>
    ShuffVecLimit("shuff-vec-max",
                  cl::desc("Maximum number of shuffles to be relocated."),
                  cl::Hidden, cl::init(-1));
#endif

namespace llvm {
void initializeHexagonOptShuffleVectorPass(PassRegistry &);
FunctionPass *createHexagonOptShuffleVector(const HexagonTargetMachine &);
} // end namespace llvm

namespace {

class HexagonOptShuffleVector : public FunctionPass {
public:
  static char ID;
#ifndef NDEBUG
  static int NumRelocated;
#endif
  HexagonOptShuffleVector() : FunctionPass(ID) {
    initializeHexagonOptShuffleVectorPass(*PassRegistry::getPassRegistry());
  }

  HexagonOptShuffleVector(const HexagonTargetMachine *TM)
      : FunctionPass(ID), TM(TM) {
    initializeHexagonOptShuffleVectorPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "Hexagon Optimize Vector Shuffles";
  }

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    FunctionPass::getAnalysisUsage(AU);
  }

private:
  using ValueVector = SmallVector<Value *, 8>;
  const HexagonTargetMachine *TM = nullptr;
  const HexagonSubtarget *HST = nullptr;
  SmallPtrSet<Instruction *, 8> Visited;
  using ShuffUseList =
      SmallDenseMap<Instruction *, SmallVector<Instruction *, 2>>;
  ShuffUseList ShuffUses;
  int DefSearchCount;

  bool visitBlock(BasicBlock *B);
  bool findNewShuffLoc(Instruction *I, ArrayRef<int> &ShuffMask,
                       Value *&NewLoc);
  bool isValidIntrinsic(IntrinsicInst *I);
  bool relocateShuffVec(Instruction *I, ArrayRef<int> &M, Value *NewLoc,
                        std::list<Instruction *> &WorkList);
  bool getUseList(Instruction *I, ValueVector &UseList);
  bool analyzeHiLoUse(Instruction *HI, Instruction *LO,
                      ArrayRef<int> &ShuffMask, Value *&NewLoc,
                      ShuffUseList &CurShuffUses);
  bool isHILo(Value *V, bool IsHI);
  bool hasDefWithSameShuffMask(Value *V, SmallVector<Instruction *, 2> &ImmUse,
                               ArrayRef<int> &ShuffMask,
                               ShuffUseList &CurShuffUses);
  void FindHiLoUse(ValueVector &UseList, Instruction *&HI, Instruction *&LO);
  bool isConcatMask(ArrayRef<int> &Mask, Instruction *ShuffInst);
  bool isValidUseInstr(ValueVector &UseList, Instruction *&UI);
  bool areAllOperandsValid(Instruction *I, Instruction *UI,
                           ArrayRef<int> &ShuffMask,
                           ShuffUseList &CurShuffUses);
  Value *getOperand(Instruction *I, unsigned i);
  static iterator_range<User::op_iterator> getArgOperands(User *U);
  static std::pair<Value *, Value *> stripCasts(Value *V);
  static bool isConstantVectorSplat(Value *V);
};

} // end anonymous namespace

#ifndef NDEBUG
int HexagonOptShuffleVector::NumRelocated = 0;
#endif
char HexagonOptShuffleVector::ID = 0;

INITIALIZE_PASS_BEGIN(HexagonOptShuffleVector, "shuff-vec",
                      "Hexagon Optimize Shuffle Vector", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(HexagonOptShuffleVector, "shuff-vec",
                    "Hexagon Optimize Shuffle Vector", false, false)

bool HexagonOptShuffleVector::isConcatMask(ArrayRef<int> &Mask,
                                           Instruction *ShuffInst) {
  Type *ShuffTy = ShuffInst->getType();
  int NumElts = cast<FixedVectorType>(ShuffTy)->getNumElements();
  for (int i = 0; i < NumElts; i++) {
    if (Mask[i] != i)
      return false;
  }
  return true;
}

bool HexagonOptShuffleVector::isValidIntrinsic(IntrinsicInst *I) {
  switch (I->getIntrinsicID()) {
  default:
    return false;
  case Intrinsic::hexagon_V6_vaddubh_128B:
  case Intrinsic::hexagon_V6_vadduhw_128B:
  case Intrinsic::hexagon_V6_vaddhw_128B:
  case Intrinsic::hexagon_V6_vaddh_dv_128B:
  case Intrinsic::hexagon_V6_vsububh_128B:
  case Intrinsic::hexagon_V6_vsubuhw_128B:
  case Intrinsic::hexagon_V6_vsubhw_128B:
  case Intrinsic::hexagon_V6_vsubh_dv_128B:
  case Intrinsic::hexagon_V6_vmpyubv_128B:
  case Intrinsic::hexagon_V6_vmpybv_128B:
  case Intrinsic::hexagon_V6_vmpyuhv_128B:
  case Intrinsic::hexagon_V6_vmpyhv_128B:
  case Intrinsic::hexagon_V6_vmpybusv_128B:
  case Intrinsic::hexagon_V6_vmpyhus_128B:
  case Intrinsic::hexagon_V6_vavgb_128B:
  case Intrinsic::hexagon_V6_vavgub_128B:
  case Intrinsic::hexagon_V6_vavgh_128B:
  case Intrinsic::hexagon_V6_vavguh_128B:
  case Intrinsic::hexagon_V6_vavgw_128B:
  case Intrinsic::hexagon_V6_vavguw_128B:
  case Intrinsic::hexagon_V6_hi_128B:
  case Intrinsic::hexagon_V6_lo_128B:
  case Intrinsic::sadd_sat:
  case Intrinsic::uadd_sat:
  // Generic hexagon vector intrinsics
  case Intrinsic::hexagon_vadd_su:
  case Intrinsic::hexagon_vadd_uu:
  case Intrinsic::hexagon_vadd_ss:
  case Intrinsic::hexagon_vadd_us:
  case Intrinsic::hexagon_vsub_su:
  case Intrinsic::hexagon_vsub_uu:
  case Intrinsic::hexagon_vsub_ss:
  case Intrinsic::hexagon_vsub_us:
  case Intrinsic::hexagon_vmpy_su:
  case Intrinsic::hexagon_vmpy_uu:
  case Intrinsic::hexagon_vmpy_ss:
  case Intrinsic::hexagon_vmpy_us:
  case Intrinsic::hexagon_vavgu:
  case Intrinsic::hexagon_vavgs:
  case Intrinsic::hexagon_vmpy_ub_b:
  case Intrinsic::hexagon_vmpy_ub_ub:
  case Intrinsic::hexagon_vmpy_uh_uh:
  case Intrinsic::hexagon_vmpy_h_h:
    return true;
  }
  llvm_unreachable("Unsupported instruction!");
}

bool HexagonOptShuffleVector::getUseList(Instruction *I, ValueVector &UseList) {
  for (auto UI = I->user_begin(), UE = I->user_end(); UI != UE;) {
    Instruction *J = dyn_cast<Instruction>(*UI);
    if (!J)
      return false;
    if (auto *C = dyn_cast<CastInst>(*UI)) {
      if (!getUseList(C, UseList))
        return false;
    } else
      UseList.push_back(*UI);
    ++UI;
  }
  return true;
}

bool HexagonOptShuffleVector::isHILo(Value *V, bool IsHI) {
  if (!(dyn_cast<Instruction>(V)))
    return false;
  Instruction *I = dyn_cast<Instruction>(V);
  if (!isa<CallInst>(I))
    return false;
  IntrinsicInst *II = dyn_cast<IntrinsicInst>(I);
  if (!II)
    return false;
  if ((II->getIntrinsicID() == Intrinsic::hexagon_V6_hi_128B && IsHI) ||
      (II->getIntrinsicID() == Intrinsic::hexagon_V6_lo_128B && !IsHI))
    return true;
  return false;
}

Value *HexagonOptShuffleVector::getOperand(Instruction *I, unsigned i) {
  Value *V = I->getOperand(i);
  if (auto *C = dyn_cast<CastInst>(V))
    return C->getOperand(0);
  return V;
}

iterator_range<User::op_iterator>
HexagonOptShuffleVector::getArgOperands(User *U) {
  if (auto *CB = dyn_cast<CallBase>(U))
    return CB->args();
  return U->operands();
}

// Strip out all the cast operations to find the first non-cast definition of a
// value. The function also returns the last cast operation in the def-chain.
std::pair<Value *, Value *> HexagonOptShuffleVector::stripCasts(Value *V) {
  Value *LastCast = nullptr;
  while (auto *C = dyn_cast<CastInst>(V)) {
    LastCast = V;
    V = C->getOperand(0);
  }
  return std::make_pair(V, LastCast);
}

bool HexagonOptShuffleVector::isConstantVectorSplat(Value *V) {
  if (auto *CV = dyn_cast<ConstantVector>(V))
    return CV->getSplatValue();
  if (auto *CV = dyn_cast<ConstantDataVector>(V))
    return CV->isSplat();
  return false;
}

// Make sure all the operations on HI and LO counterparts are identical
// until both halves are merged together. When a merge point (concat)
// is found, set it as 'NewLoc' and return.
bool HexagonOptShuffleVector::analyzeHiLoUse(Instruction *HI, Instruction *LO,
                                             ArrayRef<int> &ShuffMask,
                                             Value *&NewLoc,
                                             ShuffUseList &CurShuffUses) {
  ValueVector HiUseList, LoUseList;
  getUseList(HI, HiUseList);
  getUseList(LO, LoUseList);

  // To keep the analsis simple, only handle Hi and Lo with a single use. Also,
  // not even sure at this point if it will be profitable due to multiple
  // merge points.
  if (HiUseList.size() != 1 || LoUseList.size() != 1)
    return false;

  Instruction *HiUse = dyn_cast<Instruction>(HiUseList[0]);
  Instruction *LoUse = dyn_cast<Instruction>(LoUseList[0]);
  if (!HiUse || !LoUse)
    return false;

  bool IsUseIntrinsic = false;
  if (isa<CallInst>(HiUse)) {
    if (!isa<CallInst>(LoUse))
      return false;
    // Continue only if both Hi and Lo uses are calls to the same intrinsic.
    IntrinsicInst *HiUseII = dyn_cast<IntrinsicInst>(HiUse);
    IntrinsicInst *LoUseII = dyn_cast<IntrinsicInst>(LoUse);
    if (!HiUseII || !LoUseII ||
        HiUseII->getIntrinsicID() != LoUseII->getIntrinsicID() ||
        !isValidIntrinsic(HiUseII))
      return false;
    IsUseIntrinsic = true;
    HiUse = HiUseII;
    LoUse = LoUseII;
  }
  if (HiUse->getOpcode() != LoUse->getOpcode())
    return false;

  // If both Hi and Lo use are same and is a concat operation, set it
  // as a 'NewLoc'.
  if (HiUse == LoUse) {
    // Return true if use is a concat of Hi and Lo.
    ArrayRef<int> M;
    if (match(HiUse, (m_Shuffle(m_Value(), m_Value(), m_Mask(M))))) {
      if (isConcatMask(M, HiUse)) {
        NewLoc = HiUse;
        return true;
      }
    }
    return false;
  }

  // Check if HiUse and LoUse are shuffles with the same mask. If so, safe to
  // continue the search.
  ArrayRef<int> M1, M2;
  if (match(HiUse, (m_Shuffle(m_Value(), m_Poison(), m_Mask(M1)))) &&
      match(LoUse, (m_Shuffle(m_Value(), m_Poison(), m_Mask(M2)))) &&
      M1.equals(M2))
    return analyzeHiLoUse(HiUse, LoUse, ShuffMask, NewLoc, CurShuffUses);

  // For now, only handling binary ops and some of the instrinsics
  // which appear to be safe (hardcoded in isValidIntrinsic()).
  if (!HiUse->isBinaryOp() && !IsUseIntrinsic)
    return false;

  ValueVector HiUseOperands, LoUseOperands;
  int HiOpNum = -1, LoOpNum = -1;
  for (unsigned i = 0; i < HiUse->getNumOperands(); i++) {
    Value *V = getOperand(HiUse, i);
    if (V == HI)
      HiOpNum = i;
    else
      HiUseOperands.push_back(V);
  }
  for (unsigned i = 0; i < LoUse->getNumOperands(); i++) {
    Value *V = getOperand(LoUse, i);
    if (V == LO)
      LoOpNum = i;
    else
      LoUseOperands.push_back(V);
  }

  // Enforcing strict ordering which is not necessary in case of
  // commutative operations and may be relaxed in future if needed.
  if (HiOpNum < 0 || HiOpNum != LoOpNum ||
      LoUseOperands.size() != HiUseOperands.size())
    return false;

  unsigned NumOperands = HiUseOperands.size();
  for (unsigned i = 0; i < NumOperands; i++) {
    if (HiUseOperands[i] == LoUseOperands[i])
      continue;
    // Only handle the case where other operands to Hi and Lo uses
    // are comming from another Hi and Lo pair.
    if (!isHILo(HiUseOperands[i], true) || !isHILo(LoUseOperands[i], false))
      return false;

    Value *DefHiUse = dyn_cast<Instruction>(HiUseOperands[i])->getOperand(0);
    Value *DefLoUse = dyn_cast<Instruction>(LoUseOperands[i])->getOperand(0);
    if (!DefHiUse || DefHiUse != DefLoUse)
      return false;
    SmallVector<Instruction *, 2> ImmUseList;
    if (dyn_cast<CastInst>(DefHiUse))
      ImmUseList.push_back(dyn_cast<Instruction>(DefHiUse));
    else {
      ImmUseList.push_back(HiUse);
      ImmUseList.push_back(LoUse);
    }

    // Make sure that the Hi/Lo def has the same shuffle mask.
    if (!hasDefWithSameShuffMask(DefHiUse, ImmUseList, ShuffMask, CurShuffUses))
      return false;
  }

  // Continue the search along Hi/Lo use-chain.
  return analyzeHiLoUse(HiUse, LoUse, ShuffMask, NewLoc, CurShuffUses);
}

bool HexagonOptShuffleVector::hasDefWithSameShuffMask(
    Value *V, SmallVector<Instruction *, 2> &ImmUses, ArrayRef<int> &ShuffMask,
    ShuffUseList &CurShuffUses) {
  // Follow def-chain until we have found a shuffle_vector or have run out
  // of max number of attempts.
  if (DefSearchCount >= MaxDefSearchCount)
    return false;

  ++DefSearchCount;
  V = stripCasts(V).first;
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I)
    return false;
  bool Found = true;
  ArrayRef<int> M;
  if (match(V, (m_Shuffle(m_Value(), m_Value(), m_Mask(M)))) &&
      M.equals(ShuffMask)) {
    CurShuffUses[I] = ImmUses;
    return true;
  }
  if ((match(V, m_Shuffle(m_InsertElt(m_Poison(), m_Value(), m_Zero()),
                          m_Poison(), m_ZeroMask()))))
    return true; // scalar converted to a vector

  auto *II = dyn_cast<IntrinsicInst>(I);
  if (!I->isBinaryOp() && (!II || !isValidIntrinsic(II)))
    return false;

  for (Value *OpV : getArgOperands(I)) {
    std::pair<Value *, Value *> P = stripCasts(OpV);
    OpV = P.first;

    SmallVector<Instruction *, 2> ImmUseList;
    if (P.second)
      ImmUseList.push_back(dyn_cast<Instruction>(P.second));
    else
      ImmUseList.push_back(dyn_cast<Instruction>(I));

    if (isa<PoisonValue>(OpV))
      continue;
    if (isConstantVectorSplat(OpV))
      continue;
    if (!dyn_cast<Instruction>(OpV))
      return false;
    if ((match(OpV, m_Shuffle(m_InsertElt(m_Poison(), m_Value(), m_Zero()),
                              m_Poison(), m_ZeroMask()))))
      continue;
    Found &= hasDefWithSameShuffMask(OpV, ImmUseList, ShuffMask, CurShuffUses);
  }
  return Found;
}

void HexagonOptShuffleVector::FindHiLoUse(ValueVector &UseList,
                                          Instruction *&HI, Instruction *&LO) {

  for (unsigned i = 0; i < UseList.size(); i++) {
    auto *J = dyn_cast<Instruction>(UseList[i]);
    auto *CI = dyn_cast<CallInst>(J);
    if (CI) {
      auto *II = dyn_cast<IntrinsicInst>(CI);
      if (II) {
        Intrinsic::ID IntID = II->getIntrinsicID();
        if (IntID == Intrinsic::hexagon_V6_hi_128B)
          HI = J;
        if (IntID == Intrinsic::hexagon_V6_lo_128B)
          LO = J;
      }
    }
  }
}

bool HexagonOptShuffleVector::isValidUseInstr(ValueVector &UseList,
                                              Instruction *&UI) {
  // Don't allow multiple uses. Only done in case of a Hi/Lo pair.
  if (UseList.size() != 1)
    return false;
  UI = dyn_cast<Instruction>(UseList[0]);
  if (!UI)
    return false;
  // Should be either a binary op or one of the supported instrinsics.
  if (auto *CI = dyn_cast<CallInst>(UI)) {
    auto *II = dyn_cast<IntrinsicInst>(CI);
    if (!II || !isValidIntrinsic(II))
      return false;
    UI = II;
  } else if (!UI->isBinaryOp())
    return false;
  return true;
}

// Check all the operands of 'Use' to make sure that they are either:
// 1) a constant
// 2) a scalar
// 3) a constant vector
// 4) a vector using the same mask as I
bool HexagonOptShuffleVector::areAllOperandsValid(Instruction *I,
                                                  Instruction *Use,
                                                  ArrayRef<int> &ShuffMask,
                                                  ShuffUseList &CurShuffUses) {
  bool AllOperandsOK = true;
  for (Value *OpV : getArgOperands(Use)) {
    bool HasOneUse = OpV->hasOneUse();
    std::pair<Value *, Value *> P = stripCasts(OpV);
    OpV = P.first;

    SmallVector<Instruction *, 2> ImmUseList;
    if (P.second)
      ImmUseList.push_back(dyn_cast<Instruction>(P.second));
    else
      ImmUseList.push_back(dyn_cast<Instruction>(Use));

    if (OpV == I || isa<PoisonValue>(OpV))
      continue;
    if (isConstantVectorSplat(OpV))
      continue;
    if (!dyn_cast<Instruction>(OpV) || !HasOneUse)
      return false;

    if ((match(OpV, m_Shuffle(m_InsertElt(m_Poison(), m_Value(), m_Zero()),
                              m_Poison(), m_ZeroMask()))))
      continue;
    AllOperandsOK &=
        hasDefWithSameShuffMask(OpV, ImmUseList, ShuffMask, CurShuffUses);
  }
  return AllOperandsOK;
}

// Find the new location where it's safe to relocate shuffle instruction 'I'.
bool HexagonOptShuffleVector::findNewShuffLoc(Instruction *I,
                                              ArrayRef<int> &ShuffMask,
                                              Value *&NewLoc) {
  DefSearchCount = 0;
  ValueVector UseList;
  if (!getUseList(I, UseList))
    return false;

  using ShuffUseList =
      SmallDenseMap<Instruction *, SmallVector<Instruction *, 2>>;
  ShuffUseList CurShuffUses;
  // Check for Hi and Lo pair.
  Instruction *HI = nullptr, *LO = nullptr;
  FindHiLoUse(UseList, HI, LO);
  if (UseList.size() == 2 && HI && LO) {
    // If 'I' has Hi and Lo use-pair, then it can be relocated only after Hi/Lo
    // use-chain's merge point, i.e., after a concat vector provided it's safe
    // to do so.
    LLVM_DEBUG({
      dbgs() << "\tFollowing the Hi/LO pair :\n";
      dbgs() << "\t\tHI - ";
      HI->dump();
      dbgs() << "\t\tLO - ";
      LO->dump();
    });
    if (!analyzeHiLoUse(HI, LO, ShuffMask, NewLoc, CurShuffUses))
      return false;
    for (auto &it : CurShuffUses)
      ShuffUses[it.first] = it.second;
    return true;
  } else { // Single use case
    Instruction *UI = nullptr;
    if (!isValidUseInstr(UseList, UI))
      return false;
    assert(UI && "Expected a valid use, but found none!!");

    if (HI || LO) {
      // If the single use case is either Hi or Lo, it is not safe to relocate
      return false;
    }

    LLVM_DEBUG(dbgs() << "\tChecking operands in 'use' : \n\t\t"; UI->dump());
    if (!areAllOperandsValid(I, UI, ShuffMask, CurShuffUses)) {
      LLVM_DEBUG(dbgs() << "\t\tNOT SAFE -- Exiting!!\n");
      return false;
    }
    for (auto &it : CurShuffUses)
      ShuffUses[it.first] = it.second;
    NewLoc = UI;
    // Keep looking for the new location until can't proceed any longer.
    findNewShuffLoc(UI, ShuffMask, NewLoc);
  }
  return true;
}

// Move shuffle instruction 'I' after 'NewLoc'.
bool HexagonOptShuffleVector::relocateShuffVec(
    Instruction *I, ArrayRef<int> &M, Value *NewLoc,
    std::list<Instruction *> &WorkList) {
  // Remove original vector shuffles at the input operands.
  // However, it can be done only if the replacements have the
  // same number of vector elements as the original operands.
  std::map<Instruction *, Value *> InstrMap;
  bool CanReplace = true;
  unsigned ShuffInstCount = ShuffUses.size();
  for (auto &it : ShuffUses) {
    Instruction *J = it.first;
    Visited.insert(J);
    Value *ShuffleOP = nullptr;
    match(J, (m_Shuffle(m_Value(ShuffleOP), m_Poison(), m_Mask(M))));
    VectorType *JTy = cast<FixedVectorType>(J->getType());
    VectorType *ShuffTy = cast<FixedVectorType>(ShuffleOP->getType());
    if (JTy->getElementCount() != ShuffTy->getElementCount())
      CanReplace = false;

    // Relocate shufflevector after a wider instruction only if there are
    // at least two or more shufflevectors being relocated in order for the
    // relocation to be profitable as otherwise it will require more shuffles.
    VectorType *NewShuffTy = cast<FixedVectorType>(NewLoc->getType());
    if (ShuffInstCount == 1 &&
        NewShuffTy->getElementType() > ShuffTy->getElementType())
      CanReplace = false;
    InstrMap[J] = ShuffleOP;
  }
  if (!CanReplace) {
    LLVM_DEBUG(dbgs() << "\tRelocation FAILED!! \n");
    return false;
  }
  for (auto IM : InstrMap) {
    Instruction *J = IM.first;
    assert(ShuffUses.count(J));
    SmallVector<Instruction *, 2> Uses = ShuffUses[J];
    if (Uses.size() > 0) {
      for (auto *U : Uses)
        U->replaceUsesOfWith(IM.first, IM.second);
    } else
      // This is the shuffle we started with, and we have already made sure
      // that it has either single use or a HI/LO use pair. So, it's okay
      // to replace all its uses with the input to the shuffle instruction.
      IM.first->replaceAllUsesWith(IM.second);
  }
  // Shuffle the output of NewLoc based on the original mask.
  Instruction *Pos = dyn_cast<Instruction>(NewLoc);
  assert(Pos);
  Pos = Pos->getNextNode();
  IRBuilder<> IRB(Pos);
  Value *NewShuffV =
      IRB.CreateShuffleVector(NewLoc, PoisonValue::get(NewLoc->getType()), M);
  Instruction *NewInst = dyn_cast<Instruction>(NewShuffV);
  if (!NewInst) {
    LLVM_DEBUG(dbgs() << "\tRelocation FAILED!! \n");
    return false;
  }
  for (auto UI = NewLoc->user_begin(), UE = NewLoc->user_end(); UI != UE;) {
    Use &TheUse = UI.getUse();
    ++UI;
    Instruction *J = dyn_cast<Instruction>(TheUse.getUser());
    if (J && TheUse.getUser() != NewShuffV)
      J->replaceUsesOfWith(NewLoc, NewShuffV);
  }
  WorkList.push_back(NewInst);
  LLVM_DEBUG(dbgs() << "\tRelocation Successfull!! \n");
  LLVM_DEBUG(dbgs() << "\tAdded to Worklist :\n"; NewInst->dump());
  return true;
}

bool HexagonOptShuffleVector::visitBlock(BasicBlock *B) {
  bool Changed = false;
  ArrayRef<int> M;
  std::list<Instruction *> WorkList;
  LLVM_DEBUG(dbgs() << "Preparing worklist for BB:\n");
  LLVM_DEBUG(B->dump());
  for (auto &I : *B) {
    if (match(&I, (m_Shuffle(m_Value(), m_Value(), m_ZeroMask()))))
      continue; // Skip - building vector from a scalar
    if (match(&I, (m_Shuffle(m_Value(), m_Poison(), m_Mask(M))))) {
      WorkList.push_back(&I);
      LLVM_DEBUG(dbgs() << "\tAdded instr - "; I.dump());
    }
  }

  LLVM_DEBUG(dbgs() << "Processing worklist:\n");
  while (!WorkList.empty()) {
#ifndef NDEBUG
    int Limit = ShuffVecLimit;
    if (Limit >= 0) {
      if (NumRelocated >= ShuffVecLimit) {
        LLVM_DEBUG({
          dbgs() << "Reached maximum limit!! \n";
          dbgs() << "Can't process any more shuffles.... \n";
        });
        return Changed;
      }
    }
#endif
    Instruction *I = WorkList.front();
    WorkList.pop_front();
    LLVM_DEBUG(dbgs() << "\tProcessing instr - "; I->dump());
    Value *NewLoc = nullptr;

    // 'ShuffUses' is used to keep track of the vector shuffles that need to
    // be relocated along with their immediate uses that are known to satisfy
    // all the safety requirements of the relocation.
    // NOTE: The shuffle instr 'I', where the analysis starts, doesn't have
    // its immediate uses set in 'ShuffUses'. This can be done but isn't
    // necessary. At this point, only shuffles with single use or a HI/LO pair
    // are allowed. This is done mostly because those with the multiple uses
    // aren't expected to be much profitable and can be extended in the future
    // if necessary. For now, all the uses in such cases can be safely updated
    // when the corresponding vector shuffle is relocated.

    ShuffUses.clear();
    ShuffUses[I] = SmallVector<Instruction *, 2>();
    // Skip if node already visited.
    if (!Visited.insert(I).second) {
      LLVM_DEBUG(dbgs() << "\t\tSKIPPING - Already visited ...\n");
      continue;
    }
    if (!match(I, (m_Shuffle(m_Value(), m_Poison(), m_Mask(M))))) {
      LLVM_DEBUG(dbgs() << "\t\tSKIPPING - Not a vector shuffle ...\n");
      continue;
    }
    if (!findNewShuffLoc(I, M, NewLoc) || !NewLoc) {
      LLVM_DEBUG(dbgs() << "\t\tSKIPPING - NewLoc not found ...\n");
      continue;
    }
    LLVM_DEBUG(dbgs() << "\t\tRelocating after -- "; NewLoc->dump());
    Changed |= relocateShuffVec(I, M, NewLoc, WorkList);
#ifndef NDEBUG
    NumRelocated++;
#endif
  }
  return Changed;
}

bool HexagonOptShuffleVector::runOnFunction(Function &F) {
  HST = TM->getSubtargetImpl(F);
  // Works only for 128B mode but can be extended for 64B if needed.
  if (skipFunction(F) || !HST->useHVX128BOps())
    return false;

  bool Changed = false;
  for (auto &B : F)
    Changed |= visitBlock(&B);

  return Changed;
}

FunctionPass *
llvm::createHexagonOptShuffleVector(const HexagonTargetMachine &TM) {
  return new HexagonOptShuffleVector(&TM);
}
