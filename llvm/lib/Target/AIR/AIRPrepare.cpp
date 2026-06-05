//===- AIRPrepare.cpp - Pre-serialization IR normalization --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AIRPrepare.h"
#include "AIR.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/SCCPSolver.h"
#include <cstdlib>
#include <functional>

using namespace llvm;

#define DEBUG_TYPE "air-prepare"

namespace {
constexpr unsigned ASDevice = 1;
constexpr unsigned ASThreadgroup = 3;
constexpr unsigned PtrPhiLimit = 32;
} // namespace

// IR helpers used by the threadgroup-global GEP rewrite below.

namespace {

static void collectTGByteGlobals(Module &M,
                                 SmallVectorImpl<GlobalVariable *> &Out) {
  for (auto &GV : M.globals()) {
    if (GV.getAddressSpace() != ASThreadgroup)
      continue;
    auto *AT = dyn_cast<ArrayType>(GV.getValueType());
    if (AT && AT->getElementType()->isIntegerTy(8))
      Out.push_back(&GV);
  }
}

static void collectTGTypedGlobals(Module &M,
                                  SmallVectorImpl<GlobalVariable *> &Out) {
  for (auto &GV : M.globals()) {
    if (GV.getAddressSpace() != ASThreadgroup)
      continue;
    auto *AT = dyn_cast<ArrayType>(GV.getValueType());
    if (!AT || AT->getElementType()->isIntegerTy(8))
      continue;
    if (!GV.getName().starts_with("__tg_dot_ab_"))
      continue;
    Out.push_back(&GV);
  }
}

static Type *inferElementType(Value *V) {
  for (auto *U : V->users()) {
    if (auto *SI = dyn_cast<StoreInst>(U))
      if (SI->getPointerOperand() == V)
        return SI->getValueOperand()->getType();
    if (auto *LI = dyn_cast<LoadInst>(U))
      return LI->getType();
    if (isa<GetElementPtrInst>(U) || isa<GEPOperator>(U) || isa<BitCastInst>(U))
      if (Type *T = inferElementType(U))
        return T;
  }
  return nullptr;
}

static void expandConstantExprUsers(GlobalVariable *GV) {
  SmallVector<std::pair<ConstantExpr *, Instruction *>, 4> ToExpand;
  for (auto *U : GV->users()) {
    auto *CE = dyn_cast<ConstantExpr>(U);
    if (!CE)
      continue;
    for (auto *CEU : CE->users())
      if (auto *I = dyn_cast<Instruction>(CEU))
        ToExpand.push_back({CE, I});
  }
  for (auto &[CE, I] : ToExpand) {
    auto *Inst = CE->getAsInstruction();
    Inst->insertBefore(I->getIterator());
    I->replaceUsesOfWith(CE, Inst);
  }
  SmallVector<ConstantExpr *, 4> Dead;
  for (auto *U : GV->users())
    if (auto *CE = dyn_cast<ConstantExpr>(U))
      if (CE->use_empty())
        Dead.push_back(CE);
  for (auto *CE : Dead)
    CE->destroyConstant();
}

static void collectI8Geps(Value *V, SmallVectorImpl<GetElementPtrInst *> &Out) {
  for (auto *U : V->users()) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
      if (GEP->getSourceElementType()->isIntegerTy(8))
        Out.push_back(GEP);
      else
        collectI8Geps(GEP, Out);
    }
  }
}

// Walks loads/stores transitively reached through GEPs of V and inserts an
// identity bitcast immediately before each non-i8 access. Used by
// retypeByteGlobals when the byte global can't be safely retyped (mixed access
// types, or any unaligned dynamic byte GEP) — the writer still emits a single
// typed global, but per-site bitcasts preserve scalar/sub-element semantics.
static bool insertIdentityBitcastsAtNonByteAccesses(Value *Root) {
  bool Changed = false;
  std::function<void(Value *)> Walk = [&](Value *V) {
    for (auto *U : make_early_inc_range(V->users())) {
      if (auto *SI = dyn_cast<StoreInst>(U)) {
        if (SI->getPointerOperand() == V &&
            !SI->getValueOperand()->getType()->isIntegerTy(8)) {
          auto *BC = new BitCastInst(V, V->getType(), "");
          BC->insertBefore(SI->getIterator());
          SI->setOperand(1, BC);
          Changed = true;
        }
      } else if (auto *LI = dyn_cast<LoadInst>(U)) {
        if (!LI->getType()->isIntegerTy(8)) {
          auto *BC = new BitCastInst(V, V->getType(), "");
          BC->insertBefore(LI->getIterator());
          LI->setOperand(0, BC);
          Changed = true;
        }
      } else if (isa<GetElementPtrInst>(U)) {
        Walk(U);
      }
    }
  };
  Walk(Root);
  return Changed;
}

static bool scalarizeVec1Users(Value *V, Type *I32Ty) {
  bool Changed = false;
  SmallVector<Instruction *, 8> Vec1Users;
  std::function<void(Value *)> FindVec1 = [&](Value *V) {
    for (auto *U : V->users()) {
      if (auto *SI = dyn_cast<StoreInst>(U)) {
        if (SI->getPointerOperand() == V) {
          auto *VT =
              dyn_cast<FixedVectorType>(SI->getValueOperand()->getType());
          if (VT && VT->getNumElements() == 1)
            Vec1Users.push_back(SI);
        }
      } else if (auto *LI = dyn_cast<LoadInst>(U)) {
        auto *VT = dyn_cast<FixedVectorType>(LI->getType());
        if (VT && VT->getNumElements() == 1)
          Vec1Users.push_back(LI);
      } else if (isa<GetElementPtrInst>(U)) {
        FindVec1(U);
      }
    }
  };
  FindVec1(V);
  for (auto *I : Vec1Users) {
    if (auto *SI = dyn_cast<StoreInst>(I)) {
      IRBuilder<> B(SI);
      Value *Scalar = B.CreateExtractElement(SI->getValueOperand(),
                                             ConstantInt::get(I32Ty, 0));
      B.CreateAlignedStore(Scalar, SI->getPointerOperand(), SI->getAlign(),
                           SI->isVolatile());
      SI->eraseFromParent();
      Changed = true;
    } else if (auto *LI = dyn_cast<LoadInst>(I)) {
      IRBuilder<> B(LI);
      auto *VT = cast<FixedVectorType>(LI->getType());
      auto *Scalar =
          B.CreateAlignedLoad(VT->getElementType(), LI->getPointerOperand(),
                              LI->getAlign(), LI->isVolatile());
      Value *Vec = B.CreateInsertElement(UndefValue::get(VT), Scalar,
                                         ConstantInt::get(I32Ty, 0));
      LI->replaceAllUsesWith(Vec);
      LI->eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}

static bool foldExtractInsert(Function &F) {
  bool Changed = false;
  for (auto &BB : F) {
    for (auto It = BB.begin(); It != BB.end();) {
      Instruction &I = *It++;
      if (auto *EE = dyn_cast<ExtractElementInst>(&I)) {
        if (auto *IE = dyn_cast<InsertElementInst>(EE->getVectorOperand())) {
          auto *VT = dyn_cast<FixedVectorType>(IE->getType());
          if (VT && VT->getNumElements() == 1) {
            EE->replaceAllUsesWith(IE->getOperand(1));
            EE->eraseFromParent();
            if (IE->use_empty())
              IE->eraseFromParent();
            Changed = true;
          }
        }
      }
    }
  }
  return Changed;
}

static bool foldExtractInsert(Module &M) {
  bool Changed = false;
  for (auto &F : M)
    Changed |= foldExtractInsert(F);
  return Changed;
}

} // namespace

// TG global GEP rewrite.
// Retypes [N x i8] threadgroup globals into typed arrays based on usage
// inference, and rewrites byte-offset GEPs into element-index GEPs. Six
// sub-stages run in sequence (see comments below). Must run BEFORE the
// other AIRPrepare behaviors since retyping TG globals + their GEPs may
// produce new patterns the later stages need to normalize.

namespace {

// Shared by MergeMMA, Retype, and Strategy C. Rewrites GEPs on OldGV to use
// NewGV with element type ElemTy.
static bool rewriteByteGEPs(GlobalVariable *OldGV, GlobalVariable *NewGV,
                            ArrayType *OldAT, ArrayType *NewAT, Type *ElemTy,
                            unsigned ElemSize, LLVMContext &Ctx,
                            uint64_t ExtraElemOffset = 0) {
  bool Changed = false;
  auto addOff = [&](IRBuilder<> &B, Value *Idx) -> Value * {
    if (ExtraElemOffset == 0)
      return Idx;
    Value *Off = ConstantInt::get(Idx->getType(), ExtraElemOffset);
    if (auto *CI = dyn_cast<ConstantInt>(Idx))
      return ConstantInt::get(Idx->getType(),
                              CI->getZExtValue() + ExtraElemOffset);
    return B.CreateAdd(Idx, Off);
  };
  SmallVector<GetElementPtrInst *, 16> Users;
  for (auto *U : OldGV->users())
    if (auto *GEP = dyn_cast<GetElementPtrInst>(U))
      Users.push_back(GEP);

  for (auto *GEP : Users) {
    if (GEP->getPointerOperand() != OldGV)
      continue;
    IRBuilder<> B(GEP);
    Type *SrcTy = GEP->getSourceElementType();

    if (SrcTy == OldAT) {
      Value *ByteIdx = GEP->getNumIndices() >= 2
                           ? GEP->getOperand(2)
                           : ConstantInt::get(Type::getInt64Ty(Ctx), 0);
      Value *ElemIdx;
      if (auto *CI = dyn_cast<ConstantInt>(ByteIdx))
        ElemIdx =
            ConstantInt::get(CI->getType(), CI->getZExtValue() / ElemSize);
      else
        ElemIdx = B.CreateUDiv(ByteIdx,
                               ConstantInt::get(ByteIdx->getType(), ElemSize));
      ElemIdx = addOff(B, ElemIdx);
      auto *NewGEP = GetElementPtrInst::CreateInBounds(
          NewAT, NewGV, {ConstantInt::get(Type::getInt64Ty(Ctx), 0), ElemIdx},
          GEP->getName());
      NewGEP->insertBefore(B.GetInsertPoint());
      GEP->replaceAllUsesWith(NewGEP);
      GEP->eraseFromParent();
    } else if (SrcTy->isIntegerTy(8)) {
      Value *ByteIdx = GEP->getOperand(1);
      Value *ElemIdx;
      if (auto *CI = dyn_cast<ConstantInt>(ByteIdx))
        ElemIdx =
            ConstantInt::get(CI->getType(), CI->getZExtValue() / ElemSize);
      else
        ElemIdx = B.CreateUDiv(ByteIdx,
                               ConstantInt::get(ByteIdx->getType(), ElemSize));
      ElemIdx = addOff(B, ElemIdx);
      auto *NewGEP = GetElementPtrInst::CreateInBounds(NewAT, NewGV,
          {ConstantInt::get(Type::getInt64Ty(Ctx), 0), ElemIdx},
          GEP->getName());
      NewGEP->insertBefore(B.GetInsertPoint());
      GEP->replaceAllUsesWith(NewGEP);
      GEP->eraseFromParent();
    } else {
      if (ExtraElemOffset == 0) {
        GEP->setOperand(0, NewGV);
      } else {
        IRBuilder<> B2(GEP);
        Value *Base = B2.CreateInBoundsGEP(
            NewAT, NewGV,
            {ConstantInt::get(Type::getInt64Ty(Ctx), 0),
             ConstantInt::get(Type::getInt64Ty(Ctx), ExtraElemOffset)});
        GEP->setOperand(0, Base);
      }
    }
    Changed = true;
  }

  // Redirect remaining direct (non-GEP) instruction users.
  SmallVector<Instruction *, 4> DirectUsers;
  for (auto *U : OldGV->users()) {
    auto *I = dyn_cast<Instruction>(U);
    if (!I || isa<GetElementPtrInst>(I))
      continue;
    DirectUsers.push_back(I);
  }
  for (auto *I : DirectUsers) {
    Value *Base = NewGV;
    if (ExtraElemOffset != 0) {
      IRBuilder<> B2(I);
      Base = B2.CreateInBoundsGEP(
          NewAT, NewGV,
          {ConstantInt::get(Type::getInt64Ty(Ctx), 0),
           ConstantInt::get(Type::getInt64Ty(Ctx), ExtraElemOffset)});
    }
    for (unsigned Op = 0; Op < I->getNumOperands(); Op++)
      if (I->getOperand(Op) == OldGV)
        I->setOperand(Op, Base);
    Changed = true;
  }
  return Changed;
}

// 14a: Split mixed-type byte globals at constant offsets.
static bool
splitMixedByteGlobals(Module &M,
                      SmallVectorImpl<GlobalVariable *> &ByteGlobals) {
  bool Changed = false;
  auto &Ctx = M.getContext();
  auto &DL = M.getDataLayout();

  for (size_t Gi = 0; Gi < ByteGlobals.size(); Gi++) {
    auto *GV = ByteGlobals[Gi];
    expandConstantExprUsers(GV);
    auto *OldAT = cast<ArrayType>(GV->getValueType());
    uint64_t TotalBytes = OldAT->getNumElements();

    SmallPtrSet<Type *, 4> AllScalarTypes;
    SmallVector<int64_t, 4> ConstOffsets;
    bool WideRuntimeBaseBuffer = false;
    std::function<void(Value *, int64_t)> CollectTypes = [&](Value *V,
                                                             int64_t BaseOff) {
      for (auto *U : V->users()) {
        if (auto *SI = dyn_cast<StoreInst>(U)) {
          if (SI->getPointerOperand() == V) {
            Type *T = SI->getValueOperand()->getType();
            if (T->isIntegerTy() || T->isFloatingPointTy())
              AllScalarTypes.insert(T);
          }
        } else if (auto *LI = dyn_cast<LoadInst>(U)) {
          Type *T = LI->getType();
          if (T->isIntegerTy() || T->isFloatingPointTy())
            AllScalarTypes.insert(T);
        } else if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
          APInt Off(64, 0);
          if (GEP->accumulateConstantOffset(DL, Off)) {
            int64_t ByteOff = Off.getSExtValue();
            if (ByteOff != 0)
              ConstOffsets.push_back(ByteOff);
            CollectTypes(GEP, BaseOff + ByteOff);
          } else {
            Type *ST = GEP->getSourceElementType();
            if (BaseOff == 0 &&
                (ST->isIntegerTy() || ST->isFloatingPointTy()) &&
                DL.getTypeAllocSize(ST) > 1)
              WideRuntimeBaseBuffer = true;
            CollectTypes(GEP, BaseOff);
          }
        } else if (isa<BitCastInst>(U)) {
          CollectTypes(U, BaseOff);
        }
      }
    };
    CollectTypes(GV, 0);

    if (AllScalarTypes.size() <= 1 || ConstOffsets.empty())
      continue;

    llvm::sort(ConstOffsets);
    ConstOffsets.erase(std::unique(ConstOffsets.begin(), ConstOffsets.end()),
                       ConstOffsets.end());

    int64_t BaseRegionEnd =
        WideRuntimeBaseBuffer ? ConstOffsets.back() : ConstOffsets.front();

    DenseMap<int64_t, GlobalVariable *> SplitMap;
    for (int64_t Off : ConstOffsets) {
      uint64_t RegionSize = TotalBytes - Off;
      if (RegionSize == 0)
        continue;
      auto *SplitAT = ArrayType::get(Type::getInt8Ty(Ctx), RegionSize);
      auto *SplitGV = new GlobalVariable(
          M, SplitAT, false, GV->getLinkage(), UndefValue::get(SplitAT),
          GV->getName() + "__off" + Twine(Off), GV,
          GlobalVariable::NotThreadLocal, ASThreadgroup);
      SplitGV->setAlignment(GV->getAlign());
      SplitMap[Off] = SplitGV;
    }

    auto *NewAT = ArrayType::get(Type::getInt8Ty(Ctx), BaseRegionEnd);
    auto *NewGV = new GlobalVariable(
        M, NewAT, false, GV->getLinkage(), UndefValue::get(NewAT),
        GV->getName().str(), GV, GlobalVariable::NotThreadLocal, ASThreadgroup);
    NewGV->setAlignment(GV->getAlign());

    SmallVector<GetElementPtrInst *, 8> Users;
    for (auto *U : GV->users())
      if (auto *GEP = dyn_cast<GetElementPtrInst>(U))
        Users.push_back(GEP);

    for (auto *GEP : Users) {
      if (GEP->getPointerOperand() != GV)
        continue;
      APInt Off(64, 0);
      if (GEP->accumulateConstantOffset(DL, Off)) {
        int64_t ByteOff = Off.getSExtValue();
        if (ByteOff == 0) {
          GEP->setOperand(0, NewGV);
          if (GEP->getSourceElementType() == OldAT)
            GEP->setSourceElementType(NewAT);
        } else {
          auto Sit = SplitMap.find(ByteOff);
          if (Sit == SplitMap.end())
            continue;
          GEP->replaceAllUsesWith(Sit->second);
          GEP->eraseFromParent();
        }
      } else {
        GEP->setOperand(0, NewGV);
        if (GEP->getSourceElementType() == OldAT)
          GEP->setSourceElementType(NewAT);
      }
      Changed = true;
    }

    if (GV->use_empty())
      GV->eraseFromParent();
    ByteGlobals[Gi] = NewGV;
    for (auto &Kv : SplitMap)
      ByteGlobals.push_back(Kv.second);
    Changed = true;
  }
  return Changed;
}

static bool concurrentWithMMAScratch(Value *V, SmallPtrSetImpl<Value *> &Seen) {
  if (!Seen.insert(V).second)
    return false;
  for (User *U : V->users()) {
    if (auto *CI = dyn_cast<CallInst>(U)) {
      if (auto *Callee = CI->getCalledFunction()) {
        StringRef N = Callee->getName();
        if (N.starts_with("air.simdgroup_matrix_8x8_load") ||
            N.starts_with("air.simdgroup_matrix_8x8_store"))
          return true;
        if (N.starts_with("air.simdgroup_async_copy")) {
          if (CI->arg_size() > 2 && CI->getArgOperand(2) == V)
            return true;
        }
      }
    } else if (isa<GetElementPtrInst>(U) || isa<BitCastInst>(U)) {
      if (concurrentWithMMAScratch(U, Seen))
        return true;
    }
  }
  return false;
}

// 14b: Merge byte globals into MMA globals.
static bool mergeByteMMA(Module &M,
                         SmallVectorImpl<GlobalVariable *> &ByteGlobals,
                         SmallVectorImpl<GlobalVariable *> &MMAGlobals) {
  if (ByteGlobals.empty() || MMAGlobals.size() != 1)
    return false;

  for (auto &GV : M.globals()) {
    if (GV.getAddressSpace() != ASThreadgroup)
      continue;
    if (GV.getName().starts_with("__tg_cvt_"))
      return false;
  }

  bool Changed = false;
  auto &Ctx = M.getContext();
  auto &DL = M.getDataLayout();
  Type *I32 = Type::getInt32Ty(Ctx);

  auto *ByteGV = ByteGlobals[0];
  expandConstantExprUsers(ByteGV);

  // Check for wide vector stores.
  bool HasWideVec = false;
  {
    std::function<void(Value *)> Check = [&](Value *V) {
      for (auto *U : V->users()) {
        if (auto *SI = dyn_cast<StoreInst>(U))
          if (SI->getPointerOperand() == V)
            if (auto *VT =
                    dyn_cast<FixedVectorType>(SI->getValueOperand()->getType()))
              if (VT->getNumElements() > 1)
                HasWideVec = true;
        if (isa<GetElementPtrInst>(U))
          Check(U);
      }
    };
    Check(ByteGV);
  }

  Changed |= scalarizeVec1Users(ByteGV, I32);
  Changed |= foldExtractInsert(M);

  if (HasWideVec)
    return Changed;

  auto *MMAGV = MMAGlobals[0];
  auto *MMAAT = cast<ArrayType>(MMAGV->getValueType());
  Type *MMAElemTy = MMAAT->getElementType();

  int BestIdx = -1;
  uint64_t BestBytes = 0;
  for (int I = 0; I < (int)ByteGlobals.size(); I++) {
    auto *BAT = cast<ArrayType>(ByteGlobals[I]->getValueType());
    uint64_t BBytes = BAT->getNumElements();
    Type *Inferred = inferElementType(ByteGlobals[I]);
    bool TypeMatch =
        Inferred && (Inferred == MMAElemTy ||
                     (Inferred->isIntegerTy(32) && MMAElemTy->isFloatTy()) ||
                     (Inferred->isFloatTy() && MMAElemTy->isIntegerTy(32)));
    if (TypeMatch && BBytes > BestBytes) {
      BestIdx = I;
      BestBytes = BBytes;
    }
  }
  if (BestIdx < 0) {
    for (int I = 0; I < (int)ByteGlobals.size(); I++) {
      auto *BAT = cast<ArrayType>(ByteGlobals[I]->getValueType());
      if (BAT->getNumElements() > BestBytes) {
        BestBytes = BAT->getNumElements();
        BestIdx = I;
      }
    }
  }

  ByteGV = ByteGlobals[BestIdx >= 0 ? BestIdx : 0];
  auto *ByteAT = cast<ArrayType>(ByteGV->getValueType());
  uint64_t ByteBytes = ByteAT->getNumElements();
  unsigned MMAElemSize = DL.getTypeAllocSize(MMAAT->getElementType());
  uint64_t MMABytes = MMAAT->getNumElements() * MMAElemSize;

  Type *MergeElemTy = MMAElemTy;
  unsigned MergeElemSize = MMAElemSize;
  if (MergeElemSize == 0) {
    MergeElemTy = Type::getFloatTy(Ctx);
    MergeElemSize = 4;
  }
  SmallPtrSet<Value *, 16> SeenMMA;
  bool ByteIsMMA = concurrentWithMMAScratch(ByteGV, SeenMMA);
  uint64_t ByteElemCount =
      (ByteBytes + MergeElemSize - 1) / MergeElemSize;
  uint64_t MMAElemCount =
      (MMABytes + MergeElemSize - 1) / MergeElemSize;
  uint64_t ByteOffset = ByteIsMMA ? MMAElemCount : 0;
  uint64_t MergedElemCount =
      ByteIsMMA ? (ByteElemCount + MMAElemCount)
                : (std::max(ByteBytes, MMABytes) + MergeElemSize - 1) /
                      MergeElemSize;

  auto *MergedAT = ArrayType::get(MergeElemTy, MergedElemCount);
  expandConstantExprUsers(ByteGV);

  auto *MergedGV =
      new GlobalVariable(M, MergedAT, false, ByteGV->getLinkage(),
                         UndefValue::get(MergedAT), ByteGV->getName().str(),
                         ByteGV, GlobalVariable::NotThreadLocal, ASThreadgroup);
  MergedGV->setAlignment(ByteGV->getAlign());

  Changed |= rewriteByteGEPs(ByteGV, MergedGV, ByteAT, MergedAT, MergeElemTy,
                             MergeElemSize, Ctx, ByteOffset);

  if (ByteGV->use_empty())
    ByteGV->eraseFromParent();
  MMAGV->replaceAllUsesWith(MergedGV);
  MMAGV->eraseFromParent();

  if (BestIdx >= 0)
    ByteGlobals.erase(ByteGlobals.begin() + BestIdx);
  else
    ByteGlobals.clear();

  (void)Changed;
  return true;
}

static Value *stripIdentityIntOps(Value *V) {
  for (;;) {
    auto *BO = dyn_cast<BinaryOperator>(V);
    if (!BO)
      return V;
    unsigned Op = BO->getOpcode();
    if (auto *C = dyn_cast<ConstantInt>(BO->getOperand(1)))
      if (C->isZero() && (Op == Instruction::Xor || Op == Instruction::Add ||
                          Op == Instruction::Or || Op == Instruction::Sub)) {
        V = BO->getOperand(0);
        continue;
      }
    if (auto *C0 = dyn_cast<ConstantInt>(BO->getOperand(0)))
      if (C0->isZero() && (Op == Instruction::Xor || Op == Instruction::Add ||
                           Op == Instruction::Or)) {
        V = BO->getOperand(1);
        continue;
      }
    return V;
  }
}

static unsigned minTrailingZeros(Value *V, unsigned Depth = 0) {
  Type *Ty = V->getType();
  if (!Ty->isIntegerTy())
    return 0;
  unsigned BitW = Ty->getIntegerBitWidth();
  if (auto *CI = dyn_cast<ConstantInt>(V))
    return CI->isZero() ? BitW : CI->getValue().countr_zero();
  if (Depth > 24)
    return 0;
  if (auto *BO = dyn_cast<BinaryOperator>(V)) {
    Value *A = BO->getOperand(0), *B = BO->getOperand(1);
    switch (BO->getOpcode()) {
    case Instruction::Shl:
      if (auto *C = dyn_cast<ConstantInt>(B))
        return std::min(BitW, minTrailingZeros(A, Depth + 1) +
                                  (unsigned)C->getZExtValue());
      return minTrailingZeros(A, Depth + 1);
    case Instruction::LShr:
    case Instruction::AShr:
      if (auto *C = dyn_cast<ConstantInt>(B)) {
        unsigned Sh = (unsigned)C->getZExtValue();
        unsigned TZA = minTrailingZeros(A, Depth + 1);
        return TZA > Sh ? TZA - Sh : 0;
      }
      return 0;
    case Instruction::Mul:
      return std::min(BitW, minTrailingZeros(A, Depth + 1) +
                                minTrailingZeros(B, Depth + 1));
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::Or:
    case Instruction::Xor:
      return std::min(minTrailingZeros(A, Depth + 1),
                      minTrailingZeros(B, Depth + 1));
    case Instruction::And:
      return std::max(minTrailingZeros(A, Depth + 1),
                      minTrailingZeros(B, Depth + 1));
    default:
      return 0;
    }
  }
  if (auto *Sel = dyn_cast<SelectInst>(V))
    return std::min(minTrailingZeros(Sel->getTrueValue(), Depth + 1),
                    minTrailingZeros(Sel->getFalseValue(), Depth + 1));
  if (auto *ZE = dyn_cast<ZExtInst>(V))
    return minTrailingZeros(ZE->getOperand(0), Depth + 1);
  if (auto *TR = dyn_cast<TruncInst>(V))
    return std::min(BitW, minTrailingZeros(TR->getOperand(0), Depth + 1));
  return 0;
}

// 14c: Retype [N x i8] -> [M x T].
static bool retypeByteGlobals(Module &M) {
  bool Changed = false;
  auto &Ctx = M.getContext();
  auto &DL = M.getDataLayout();
  Type *I32 = Type::getInt32Ty(Ctx);

  SmallVector<GlobalVariable *, 4> ByteGlobals;
  collectTGByteGlobals(M, ByteGlobals);

  for (auto *GV : ByteGlobals) {
    expandConstantExprUsers(GV);
    if (GV->use_empty()) {
      GV->eraseFromParent();
      Changed = true;
      continue;
    }
    Type *StoreTy = inferElementType(GV);
    if (!StoreTy)
      continue;

    Changed |= scalarizeVec1Users(GV, I32);
    Changed |= foldExtractInsert(M);

    StoreTy = inferElementType(GV);
    if (!StoreTy)
      continue;

    auto *OldAT = cast<ArrayType>(GV->getValueType());
    uint64_t TotalBytes = OldAT->getNumElements();

    Type *ElemTy = StoreTy;
    if (ElemTy->isBFloatTy())
      ElemTy = Type::getHalfTy(Ctx);

    if (auto *VT = dyn_cast<FixedVectorType>(ElemTy)) {
      Type *ScalarTy = VT->getElementType();
      unsigned ScalarBytes = DL.getTypeAllocSize(ScalarTy);
      Type *SiblingScalar = nullptr;
      std::function<void(Value *)> CheckGEPs = [&](Value *V) {
        for (auto *U : V->users()) {
          if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
            Type *ST = GEP->getSourceElementType();
            if ((ST->isFloatingPointTy() || ST->isIntegerTy()) &&
                DL.getTypeAllocSize(ST) == ScalarBytes) {
              if (ST == ScalarTy)
                SiblingScalar = ScalarTy;
              else if (!SiblingScalar)
                SiblingScalar = ST;
            }
            CheckGEPs(GEP);
          }
        }
      };
      CheckGEPs(GV);
      ElemTy = SiblingScalar ? SiblingScalar : ScalarTy;
    }

    unsigned ElemSize = DL.getTypeAllocSize(ElemTy);
    if (ElemSize == 0)
      continue;
    uint64_t NumElems = TotalBytes / ElemSize;
    if (NumElems == 0)
      continue;

    // Classify byte GEPs: those whose offset is a known multiple of ElemSize
    // can be rewritten to typed GEPs; the rest must access the typed global
    // via an i8* alias so per-lane / sub-element scalar access stays correct.
    auto isAlignedByteGEP = [&](GetElementPtrInst *GEP) -> bool {
      if (ElemSize == 1)
        return true;
      if (GEP->getNumIndices() != 1)
        return false;
      Value *Idx = stripIdentityIntOps(GEP->getOperand(1));
      if (auto *CI = dyn_cast<ConstantInt>(Idx))
        return CI->getZExtValue() % ElemSize == 0;
      unsigned TZ = std::max(minTrailingZeros(Idx),
                             computeKnownBits(Idx, DL).countMinTrailingZeros());
      return (1u << TZ) >= ElemSize;
    };

    // If any byte GEP into GV has a dynamic index whose alignment to ElemSize
    // can't be proved, retyping would collapse distinct lanes onto the same
    // wide element (e.g. chained reductions). Fall back to the mixed-access
    // bitcast path: keep [N x i8], insert per-site bitcasts.
    {
      bool AnyUnaligned = false;
      std::function<bool(Value *)> Walk = [&](Value *V) -> bool {
        for (User *U : V->users()) {
          if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
            if (GEP->getSourceElementType()->isIntegerTy(8) &&
                !isAlignedByteGEP(GEP))
              return true;
            if (Walk(GEP))
              return true;
          }
        }
        return false;
      };
      AnyUnaligned = Walk(GV);
      if (AnyUnaligned) {
        Changed |= insertIdentityBitcastsAtNonByteAccesses(GV);
        continue;
      }
    }

    auto *NewAT = ArrayType::get(ElemTy, NumElems);
    auto *NewGV =
        new GlobalVariable(M, NewAT, GV->isConstant(), GV->getLinkage(),
                           UndefValue::get(NewAT), GV->getName() + ".typed", GV,
                           GlobalVariable::NotThreadLocal, ASThreadgroup);
    NewGV->setAlignment(GV->getAlign());

    Changed |= rewriteByteGEPs(GV, NewGV, OldAT, NewAT, ElemTy, ElemSize, Ctx);

    if (GV->use_empty())
      GV->eraseFromParent();

    // Clean up residual i8 GEPs: aligned ones become typed (index / ElemSize),
    // unaligned ones stay as i8 GEPs but rebase on an i8* alias of NewGV so
    // the AIR bitcode writer still emits a single typed global.
    SmallVector<GetElementPtrInst *, 8> ResidualI8;
    collectI8Geps(NewGV, ResidualI8);
    for (auto *GEP : ResidualI8) {
      IRBuilder<> B(GEP);
      Value *ByteIdx = GEP->getOperand(1);
      if (isAlignedByteGEP(GEP)) {
        Value *NewIdx;
        if (auto *CI = dyn_cast<ConstantInt>(ByteIdx))
          NewIdx =
              ConstantInt::get(CI->getType(), CI->getZExtValue() / ElemSize);
        else
          NewIdx = B.CreateUDiv(ByteIdx,
                                ConstantInt::get(ByteIdx->getType(), ElemSize));
        auto *NewGEP = B.CreateInBoundsGEP(ElemTy, GEP->getPointerOperand(),
                                           NewIdx, GEP->getName());
        GEP->replaceAllUsesWith(NewGEP);
      } else {
        // Keep i8 stride: GEP from an i8* alias of NewGV. The alias lets the
        // writer keep the global typed while preserving byte-granular access.
        auto *NewGEP = B.CreateInBoundsGEP(Type::getInt8Ty(Ctx), NewGV, ByteIdx,
                                           GEP->getName());
        GEP->replaceAllUsesWith(NewGEP);
      }
      GEP->eraseFromParent();
      Changed = true;
    }
  }

  // Strategy C: split remaining byte globals at constant offsets + retype.
  SmallVector<GlobalVariable *, 4> Remaining;
  collectTGByteGlobals(M, Remaining);

  for (auto *GV : Remaining) {
    expandConstantExprUsers(GV);
    auto *OldAT = cast<ArrayType>(GV->getValueType());
    uint64_t TotalBytes = OldAT->getNumElements();

    SmallVector<int64_t, 4> Offsets;
    bool HasDynamic = false;
    for (auto *U : GV->users()) {
      auto *GEP = dyn_cast<GetElementPtrInst>(U);
      if (!GEP)
        continue;
      APInt Off(64, 0);
      if (GEP->accumulateConstantOffset(DL, Off)) {
        int64_t ByteOff = Off.getSExtValue();
        if (ByteOff != 0)
          Offsets.push_back(ByteOff);
      } else {
        HasDynamic = true;
      }
    }

    if (HasDynamic || Offsets.empty())
      continue;

    llvm::sort(Offsets);
    Offsets.erase(std::unique(Offsets.begin(), Offsets.end()), Offsets.end());

    DenseMap<int64_t, GlobalVariable *> SplitMap;
    for (int64_t Off : Offsets) {
      uint64_t RegionSize = TotalBytes - Off;
      if (RegionSize == 0)
        continue;
      auto *SplitAT = ArrayType::get(Type::getInt8Ty(Ctx), RegionSize);
      auto *SplitGV = new GlobalVariable(
          M, SplitAT, false, GV->getLinkage(), UndefValue::get(SplitAT),
          GV->getName() + "__off" + Twine(Off), GV,
          GlobalVariable::NotThreadLocal, ASThreadgroup);
      SplitGV->setAlignment(GV->getAlign());
      SplitMap[Off] = SplitGV;
    }

    auto *NewAT = ArrayType::get(Type::getInt8Ty(Ctx), Offsets[0]);
    auto *NewGV = new GlobalVariable(
        M, NewAT, false, GV->getLinkage(), UndefValue::get(NewAT),
        GV->getName().str(), GV, GlobalVariable::NotThreadLocal, ASThreadgroup);
    NewGV->setAlignment(GV->getAlign());

    SmallVector<GetElementPtrInst *, 8> Users;
    for (auto *U : GV->users())
      if (auto *GEP = dyn_cast<GetElementPtrInst>(U))
        Users.push_back(GEP);

    for (auto *GEP : Users) {
      if (GEP->getPointerOperand() != GV)
        continue;
      APInt Off(64, 0);
      if (!GEP->accumulateConstantOffset(DL, Off))
        continue;
      int64_t ByteOff = Off.getSExtValue();
      if (ByteOff == 0) {
        GEP->setOperand(0, NewGV);
        if (GEP->getSourceElementType() == OldAT)
          GEP->setSourceElementType(NewAT);
        Changed = true;
      } else {
        auto Sit = SplitMap.find(ByteOff);
        if (Sit == SplitMap.end())
          continue;
        GEP->replaceAllUsesWith(Sit->second);
        GEP->eraseFromParent();
        Changed = true;
      }
    }

    if (GV->use_empty())
      GV->eraseFromParent();

    SmallVector<GlobalVariable *, 4> ToRetype;
    ToRetype.push_back(NewGV);
    for (auto &Kv : SplitMap)
      ToRetype.push_back(Kv.second);
    for (auto *SplitGV : ToRetype) {
      Type *ElemTy = inferElementType(SplitGV);
      if (!ElemTy)
        continue;
      if (ElemTy->isBFloatTy())
        ElemTy = Type::getHalfTy(Ctx);
      unsigned ESize = DL.getTypeAllocSize(ElemTy);
      if (ESize == 0)
        continue;
      auto *SplitOldAT = cast<ArrayType>(SplitGV->getValueType());
      uint64_t NElems = SplitOldAT->getNumElements() / ESize;
      if (NElems == 0)
        continue;
      auto *TypedAT = ArrayType::get(ElemTy, NElems);
      auto *TypedGV = new GlobalVariable(
          M, TypedAT, false, SplitGV->getLinkage(), UndefValue::get(TypedAT),
          SplitGV->getName().str() + ".typed", SplitGV,
          GlobalVariable::NotThreadLocal, ASThreadgroup);
      TypedGV->setAlignment(SplitGV->getAlign());

      Changed |= rewriteByteGEPs(SplitGV, TypedGV, SplitOldAT, TypedAT, ElemTy,
                                 ESize, Ctx);
      if (SplitGV->use_empty())
        SplitGV->eraseFromParent();
    }
  }
  return Changed;
}

// 14d: Insert preamble GEPs for array TG globals.
static bool insertPreambleGEPs(Module &M) {
  bool Changed = false;
  auto &Ctx = M.getContext();

  SmallVector<GlobalVariable *, 8> AllTGGlobals;
  for (auto &GV : M.globals())
    if (GV.getAddressSpace() == ASThreadgroup &&
        isa<ArrayType>(GV.getValueType()))
      AllTGGlobals.push_back(&GV);

  for (auto &F : M) {
    if (F.isDeclaration())
      continue;

    SmallPtrSet<GlobalVariable *, 4> UsedGlobals;
    for (auto &BB : F)
      for (auto &I : BB)
        for (auto &Op : I.operands())
          if (auto *GV = dyn_cast<GlobalVariable>(Op))
            if (GV->getAddressSpace() == ASThreadgroup)
              UsedGlobals.insert(GV);

    if (UsedGlobals.empty())
      continue;

    DenseMap<GlobalVariable *, Value *> PreambleMap;
    for (auto *GV : AllTGGlobals) {
      if (!UsedGlobals.count(GV))
        continue;
      if (!isa<ArrayType>(GV->getValueType()))
        continue;

      bool NeedsPreamble = false;
      for (auto *U : GV->users()) {
        auto *I = dyn_cast<Instruction>(U);
        if (!I || I->getFunction() != &F)
          continue;
        if (auto *GEPUser = dyn_cast<GetElementPtrInst>(U)) {
          if (GEPUser->getSourceElementType() != GV->getValueType())
            NeedsPreamble = true;
        } else {
          NeedsPreamble = true;
        }
      }
      if (!NeedsPreamble)
        continue;

      auto *AT = cast<ArrayType>(GV->getValueType());
      auto *GEP = GetElementPtrInst::CreateInBounds(
          AT, GV,
          {ConstantInt::get(Type::getInt64Ty(Ctx), 0),
           ConstantInt::get(Type::getInt64Ty(Ctx), 0)},
          "__base_" + GV->getName());
      GEP->insertBefore(F.getEntryBlock().getFirstInsertionPt());
      PreambleMap[GV] = GEP;
    }

    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
          if (GEP->getName().starts_with("__base_"))
            continue;
        for (unsigned i = 0; i < I.getNumOperands(); i++) {
          auto *GV = dyn_cast<GlobalVariable>(I.getOperand(i));
          if (!GV)
            continue;
          auto Pit = PreambleMap.find(GV);
          if (Pit == PreambleMap.end())
            continue;
          if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
            if (i == 0 && GEP->getSourceElementType() == GV->getValueType())
              continue;
          // Insert identity bitcast so the writer's PTM sees the typed-
          // pointer transition explicitly (AIR v1 typed bitcode req;
          // otherwise PSO load fails on bfloat scan2d kernels). Two
          // shapes mirror the post-Prepare NormalizeAllocas re-run:
          // bfloat/half-source GEP off any typed base (case-5b), and
          // float-source GEP off a bfloat/half typed base (case-5a).
          Value *NewOp = Pit->second;
          if (auto *GEP = dyn_cast<GetElementPtrInst>(&I); GEP && i == 0) {
            Type *SrcTy = GEP->getSourceElementType();
            auto *AT = cast<ArrayType>(GV->getValueType());
            Type *GVElem = AT->getElementType();
            bool NeedsBitcast =
                SrcTy->isBFloatTy() || SrcTy->isHalfTy() ||
                (SrcTy->isFloatTy() &&
                 (GVElem->isBFloatTy() || GVElem->isHalfTy()));
            if (NeedsBitcast) {
              auto *BC = CastInst::Create(Instruction::BitCast, NewOp,
                                          NewOp->getType(), "", I.getIterator());
              NewOp = BC;
            }
          }
          I.setOperand(i, NewOp);
          Changed = true;
        }
      }
    }
  }
  return Changed;
}

// 14e: Fix residual i8 GEPs on typed TG pointers.
// The AIR GPU JIT rejects the type mismatch when an i8-source GEP is derived
// from a non-i8 typed GEP on a threadgroup pointer; rewrite the i8 GEP to use
// the producing GEP's element type.
static bool fixResidualI8GEPs(Module &M) {
  bool Changed = false;
  auto &DL = M.getDataLayout();

  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<GetElementPtrInst *, 16> I8Geps;
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
          if (GEP->getSourceElementType()->isIntegerTy(8) &&
              GEP->getPointerAddressSpace() == ASThreadgroup) {
            auto *SrcGEP =
                dyn_cast<GetElementPtrInst>(GEP->getPointerOperand());
            if (SrcGEP && !SrcGEP->getSourceElementType()->isIntegerTy(8))
              I8Geps.push_back(GEP);
          }

    for (auto *GEP : I8Geps) {
      auto *SrcGEP = cast<GetElementPtrInst>(GEP->getPointerOperand());
      Type *ElemTy = SrcGEP->getSourceElementType();
      if (auto *AT = dyn_cast<ArrayType>(ElemTy))
        ElemTy = AT->getElementType();
      unsigned ElemSize = DL.getTypeAllocSize(ElemTy);
      if (ElemSize == 0 || ElemSize == 1)
        continue;

      IRBuilder<> B(GEP);
      Value *ByteIdx = GEP->getOperand(1);
      Value *ElemIdx;
      if (auto *CI = dyn_cast<ConstantInt>(ByteIdx))
        ElemIdx =
            ConstantInt::get(CI->getType(), CI->getZExtValue() / ElemSize);
      else
        ElemIdx = B.CreateUDiv(ByteIdx,
                               ConstantInt::get(ByteIdx->getType(), ElemSize));
      auto *NewGEP = B.CreateInBoundsGEP(ElemTy, GEP->getPointerOperand(),
                                         ElemIdx, GEP->getName());
      GEP->replaceAllUsesWith(NewGEP);
      GEP->eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}

// 14f: Fix mismatched-type GEPs on TG pointers.
// The AIR GPU JIT crashes on non-float typed TG pointers when float-typed
// MMA globals coexist. Rewrite same-sized-type mismatches (e.g. gep i32
// derived from a float TG pointer) to gep float + bitcast at the leaves.
static bool fixMismatchedTGGEPs(Module &M) {
  bool Changed = false;
  auto &DL = M.getDataLayout();
  auto &Ctx = M.getContext();

  for (auto &F : M) {
    if (F.isDeclaration())
      continue;

    SmallVector<GetElementPtrInst *, 16> MismatchGeps;
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
          if (GEP->getPointerAddressSpace() == ASThreadgroup) {
            Type *GEPSrcTy = GEP->getSourceElementType();
            if (isa<ArrayType>(GEPSrcTy))
              continue;
            if (GEPSrcTy->isFloatTy())
              continue;
            Value *SrcPtr = GEP->getPointerOperand();
            while (auto *BC = dyn_cast<BitCastInst>(SrcPtr))
              SrcPtr = BC->getOperand(0);
            auto *SrcGEP = dyn_cast<GetElementPtrInst>(SrcPtr);
            if (!SrcGEP)
              continue;
            Type *ParentTy = SrcGEP->getSourceElementType();
            if (auto *AT = dyn_cast<ArrayType>(ParentTy))
              ParentTy = AT->getElementType();
            if (!ParentTy->isFloatTy())
              continue;
            unsigned GEPElemSize = DL.getTypeAllocSize(GEPSrcTy);
            unsigned FloatSize = DL.getTypeAllocSize(ParentTy);
            if (GEPElemSize == FloatSize)
              MismatchGeps.push_back(GEP);
          }

    for (auto *GEP : MismatchGeps) {
      IRBuilder<> B(GEP);
      Type *FloatTy = Type::getFloatTy(Ctx);
      Value *Idx = GEP->getOperand(1);
      auto *NewGEP = B.CreateInBoundsGEP(FloatTy, GEP->getPointerOperand(), Idx,
                                         GEP->getName());

      SmallVector<Instruction *, 8> Users;
      for (auto *U : GEP->users())
        if (auto *I = dyn_cast<Instruction>(U))
          Users.push_back(I);

      for (auto *U : Users) {
        if (auto *SI = dyn_cast<StoreInst>(U)) {
          if (SI->getPointerOperand() == GEP) {
            IRBuilder<> SB(SI);
            Value *Val = SI->getValueOperand();
            if (!Val->getType()->isFloatTy())
              Val = SB.CreateBitCast(Val, FloatTy);
            SB.CreateAlignedStore(Val, NewGEP, SI->getAlign(),
                                  SI->isVolatile());
            SI->eraseFromParent();
          }
        } else if (auto *LI = dyn_cast<LoadInst>(U)) {
          if (!LI->getType()->isFloatTy()) {
            IRBuilder<> LB(LI);
            auto *NewLoad = LB.CreateAlignedLoad(
                FloatTy, NewGEP, LI->getAlign(), LI->isVolatile());
            Value *Casted = LB.CreateBitCast(NewLoad, LI->getType());
            LI->replaceAllUsesWith(Casted);
            LI->eraseFromParent();
          } else {
            LI->setOperand(0, NewGEP);
          }
        } else {
          for (unsigned I = 0; I < U->getNumOperands(); I++)
            if (U->getOperand(I) == GEP)
              U->setOperand(I, NewGEP);
        }
      }

      if (GEP->use_empty())
        GEP->eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}

static bool scalarizeMixedWidthTGVecStores(Module &M) {
  bool Changed = false;
  Type *I32 = Type::getInt32Ty(M.getContext());
  const DataLayout &DL = M.getDataLayout();

  auto collect = [&](GlobalVariable &GV, SmallVectorImpl<StoreInst *> &Stores,
                     unsigned &MaxStoreElems, unsigned &MinAccessElems,
                     bool &SawNarrowerAccess) {
    SmallVector<Value *, 16> Work{&GV};
    SmallPtrSet<Value *, 16> Seen;
    while (!Work.empty()) {
      Value *V = Work.pop_back_val();
      if (!Seen.insert(V).second)
        continue;
      for (User *U : V->users()) {
        if (isa<GetElementPtrInst>(U) || isa<BitCastInst>(U)) {
          Work.push_back(U);
          continue;
        }
        Type *AccTy = nullptr;
        StoreInst *SI = dyn_cast<StoreInst>(U);
        if (SI && SI->getPointerOperand() == V)
          AccTy = SI->getValueOperand()->getType();
        else if (auto *LI = dyn_cast<LoadInst>(U))
          AccTy = LI->getType();
        if (!AccTy)
          continue;
        unsigned Elems = 1;
        if (auto *VT = dyn_cast<FixedVectorType>(AccTy))
          Elems = VT->getNumElements();
        MinAccessElems = std::min(MinAccessElems, Elems);
        if (SI && SI->getPointerOperand() == V) {
          if (Elems > 1) {
            Stores.push_back(SI);
            MaxStoreElems = std::max(MaxStoreElems, Elems);
          }
        }
      }
    }
    SawNarrowerAccess = MinAccessElems < MaxStoreElems;
  };

  for (GlobalVariable &GV : M.globals()) {
    if (GV.getAddressSpace() != ASThreadgroup)
      continue;
    SmallVector<StoreInst *, 8> WideStores;
    unsigned MaxStoreElems = 1, MinAccessElems = ~0u;
    bool Mixed = false;
    collect(GV, WideStores, MaxStoreElems, MinAccessElems, Mixed);
    if (!Mixed || WideStores.empty())
      continue;

    for (StoreInst *SI : WideStores) {
      auto *VT = cast<FixedVectorType>(SI->getValueOperand()->getType());
      Type *ElemTy = VT->getElementType();
      IRBuilder<> B(SI);
      Value *Vec = SI->getValueOperand();
      Value *BasePtr = SI->getPointerOperand();
      Align A = SI->getAlign();
      for (unsigned i = 0, e = VT->getNumElements(); i != e; ++i) {
        Value *Elt = B.CreateExtractElement(Vec, ConstantInt::get(I32, i));
        Value *Ptr = i == 0 ? BasePtr
                            : B.CreateInBoundsGEP(ElemTy, BasePtr,
                                                  ConstantInt::get(I32, i));
        Align EltAlign = i == 0 ? A : DL.getABITypeAlign(ElemTy);
        B.CreateAlignedStore(Elt, Ptr, EltAlign, SI->isVolatile());
      }
      SI->eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}

static bool rewriteTGGlobalGEPs(Module &M) {
  // Cheap early-out: nothing to do unless there is an array-typed TG global.
  bool HasArrayTG = false;
  for (auto &GV : M.globals())
    if (GV.getAddressSpace() == ASThreadgroup &&
        isa<ArrayType>(GV.getValueType())) {
      HasArrayTG = true;
      break;
    }
  if (!HasArrayTG)
    return false;

  bool Changed = false;

  SmallVector<GlobalVariable *, 4> ByteGlobals;
  SmallVector<GlobalVariable *, 4> MMAGlobals;
  collectTGByteGlobals(M, ByteGlobals);
  collectTGTypedGlobals(M, MMAGlobals);

  Changed |= splitMixedByteGlobals(M, ByteGlobals);

  // Wide-vector stores on threadgroup byte globals do not need scalarizing on
  // AIR 4 (macOS 26): the toolchain emits `store <N x T>` on threadgroup
  // memory intact.

  Changed |= mergeByteMMA(M, ByteGlobals, MMAGlobals);
  Changed |= retypeByteGlobals(M);
  Changed |= insertPreambleGEPs(M);

  // 14e: iterate; scalarized i8 GEPs form chains that peel one level per pass.
  // The iteration bound is defensive padding over the observed chain depth.
  for (int Iter = 0; Iter < 8; Iter++) {
    if (!fixResidualI8GEPs(M))
      break;
    Changed = true;
  }

  Changed |= fixMismatchedTGGEPs(M);
  Changed |= scalarizeMixedWidthTGVecStores(M);
  return Changed;
}

} // namespace

// ── i1 GEP normalization ────────────────────────────────────────────────────

static bool normalizeI1Pointers(Module &M) {
  bool Changed = false;
  Type *I8 = Type::getInt8Ty(M.getContext());

  SmallVector<GlobalVariable *, 4> I1Globals;
  for (GlobalVariable &GV : M.globals()) {
    Type *VT = GV.getValueType();
    if (VT->isIntegerTy(1))
      I1Globals.push_back(&GV);
    else if (auto *AT = dyn_cast<ArrayType>(VT))
      if (AT->getElementType()->isIntegerTy(1))
        I1Globals.push_back(&GV);
  }
  for (GlobalVariable *GV : I1Globals) {
    Type *VT = GV->getValueType();
    Type *NewVT =
        VT->isIntegerTy(1)
            ? I8
            : ArrayType::get(I8, cast<ArrayType>(VT)->getNumElements());
    auto *NewGV = new GlobalVariable(
        M, NewVT, GV->isConstant(), GV->getLinkage(),
        GV->hasInitializer() ? UndefValue::get(NewVT) : nullptr, "", GV,
        GV->getThreadLocalMode(), GV->getAddressSpace());
    NewGV->setAlignment(GV->getAlign().valueOrOne());
    NewGV->takeName(GV);
    GV->replaceAllUsesWith(NewGV);
    GV->eraseFromParent();
    Changed = true;
  }

  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (Instruction &I : llvm::make_early_inc_range(BB)) {
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
          Type *SrcTy = GEP->getSourceElementType();
          if (SrcTy->isIntegerTy(1)) {
            GEP->setSourceElementType(I8);
            GEP->setResultElementType(I8);
            Changed = true;
          } else if (auto *AT = dyn_cast<ArrayType>(SrcTy);
                     AT && AT->getElementType()->isIntegerTy(1)) {
            auto *NewAT = ArrayType::get(I8, AT->getNumElements());
            GEP->setSourceElementType(NewAT);
            if (GEP->getResultElementType() == AT)
              GEP->setResultElementType(NewAT);
            else if (GEP->getResultElementType()->isIntegerTy(1))
              GEP->setResultElementType(I8);
            Changed = true;
          }
          continue;
        }
        if (auto *SI = dyn_cast<StoreInst>(&I)) {
          Value *V = SI->getValueOperand();
          if (V->getType()->isIntegerTy(1)) {
            IRBuilder<> B(SI);
            SI->setOperand(0, B.CreateZExt(V, I8));
            Changed = true;
          }
          continue;
        }
        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          if (LI->getType()->isIntegerTy(1)) {
            IRBuilder<> B(LI);
            auto *L8 = B.CreateLoad(I8, LI->getPointerOperand());
            L8->setAlignment(LI->getAlign());
            Value *Tr = B.CreateTrunc(L8, LI->getType());
            LI->replaceAllUsesWith(Tr);
            LI->eraseFromParent();
            Changed = true;
          }
          continue;
        }
      }
  return Changed;
}

static Value *traceInsertValueElement(Value *V, unsigned Idx) {
  while (auto *IV = dyn_cast<InsertValueInst>(V)) {
    if (IV->getNumIndices() == 1 && IV->getIndices()[0] == Idx)
      return IV->getInsertedValueOperand();
    V = IV->getAggregateOperand();
  }
  if (isa<UndefValue>(V))
    return UndefValue::get(cast<StructType>(V->getType())->getElementType(Idx));
  if (isa<ConstantAggregateZero>(V))
    return Constant::getNullValue(
        cast<StructType>(V->getType())->getElementType(Idx));
  if (auto *C = dyn_cast<Constant>(V))
    if (auto *ST = dyn_cast<StructType>(V->getType()))
      if (Idx < ST->getNumElements())
        return C->getAggregateElement(Idx);
  return nullptr;
}

static bool decomposeStructPhis(Module &M,
                                SmallPtrSetImpl<Function *> &Decomposed) {
  bool Changed = false;

  for (Function &F : M) {
    SmallVector<PHINode *, 8> StructPhis;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *PN = dyn_cast<PHINode>(&I))
          if (isa<StructType>(PN->getType()))
            StructPhis.push_back(PN);

    if (StructPhis.empty())
      continue;

    Decomposed.insert(&F);

    for (PHINode *PN : StructPhis) {
      auto *ST = cast<StructType>(PN->getType());
      unsigned NumElems = ST->getNumElements();
      IRBuilder<> B(PN);

      SmallVector<PHINode *, 4> ScalarPhis;
      for (unsigned i = 0; i < NumElems; i++)
        ScalarPhis.push_back(B.CreatePHI(ST->getElementType(i),
                                         PN->getNumIncomingValues(),
                                         PN->getName() + "_" + Twine(i)));

      for (unsigned Inc = 0; Inc < PN->getNumIncomingValues(); Inc++) {
        Value *InVal = PN->getIncomingValue(Inc);
        BasicBlock *InBB = PN->getIncomingBlock(Inc);

        for (unsigned i = 0; i < NumElems; i++) {
          Value *Elem = traceInsertValueElement(InVal, i);
          if (!Elem) {
            IRBuilder<> PredB(InBB->getTerminator());
            Elem = PredB.CreateExtractValue(
                InVal, i, InVal->getName() + "_ext" + Twine(i));
          }
          ScalarPhis[i]->addIncoming(Elem, InBB);
        }
      }

      SmallVector<Instruction *, 8> ToRemove;
      for (User *U : PN->users())
        if (auto *EV = dyn_cast<ExtractValueInst>(U))
          if (EV->getNumIndices() == 1) {
            unsigned Idx = EV->getIndices()[0];
            if (Idx < NumElems) {
              EV->replaceAllUsesWith(ScalarPhis[Idx]);
              ToRemove.push_back(EV);
            }
          }
      for (Instruction *I : ToRemove)
        I->eraseFromParent();

      if (!PN->use_empty()) {
        IRBuilder<> AfterB(&*PN->getParent()->getFirstNonPHIIt());
        Value *Agg = UndefValue::get(ST);
        for (unsigned i = 0; i < NumElems; i++)
          Agg = AfterB.CreateInsertValue(Agg, ScalarPhis[i], i,
                                         PN->getName() + "_rebuild");
        PN->replaceAllUsesWith(Agg);
      }

      PN->eraseFromParent();
      Changed = true;
    }

    bool CleanedUp = true;
    while (CleanedUp) {
      CleanedUp = false;
      for (BasicBlock &BB : F)
        for (Instruction &I : llvm::make_early_inc_range(BB))
          if (auto *IV = dyn_cast<InsertValueInst>(&I))
            if (IV->use_empty()) {
              IV->eraseFromParent();
              CleanedUp = true;
            }
    }
  }

  return Changed;
}

// ── Pointer-phi to i64 lowering ─────────────────────────────────────────────

static bool hasUndefIncoming(PHINode *PN) {
  for (unsigned i = 0; i < PN->getNumIncomingValues(); i++)
    if (isa<UndefValue>(PN->getIncomingValue(i)))
      return true;
  return false;
}

static void convertPtrPhiToI64(PHINode *PN, Type *I64) {
  Type *PtrTy = PN->getType();

  PHINode *NewPhi =
      PHINode::Create(I64, PN->getNumIncomingValues(), PN->getName() + "_i64");
  NewPhi->insertBefore(PN->getIterator());

  for (unsigned i = 0; i < PN->getNumIncomingValues(); i++) {
    Value *InVal = PN->getIncomingValue(i);
    BasicBlock *InBB = PN->getIncomingBlock(i);

    Value *AsInt;
    if (isa<UndefValue>(InVal)) {
      AsInt = ConstantInt::get(I64, 0);
    } else if (isa<ConstantPointerNull>(InVal)) {
      AsInt = ConstantInt::get(I64, 0);
    } else {
      IRBuilder<> PredB(InBB->getTerminator());
      AsInt = PredB.CreatePtrToInt(InVal, I64, InVal->getName() + "_p2i");
    }
    NewPhi->addIncoming(AsInt, InBB);
  }

  BasicBlock *BB = PN->getParent();
  IRBuilder<> B(&*BB->getFirstNonPHIIt());
  Value *BackToPtr = B.CreateIntToPtr(NewPhi, PtrTy, PN->getName() + "_ptr");

  PN->replaceAllUsesWith(BackToPtr);
  PN->eraseFromParent();
}

static bool ptrPhiToI64(Module &M,
                        const SmallPtrSetImpl<Function *> &ForceConvert) {
  bool Changed = false;
  Type *I64 = Type::getInt64Ty(M.getContext());

  for (Function &F : M) {
    bool ForceAll = ForceConvert.contains(&F);

    bool FunctionHasUndefPtrPhi = false;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *PN = dyn_cast<PHINode>(&I))
          if (PN->getType()->isPointerTy() && hasUndefIncoming(PN)) {
            FunctionHasUndefPtrPhi = true;
            break;
          }

    for (BasicBlock &BB : F) {
      SmallVector<PHINode *, 16> PtrPhis;
      for (Instruction &I : BB)
        if (auto *PN = dyn_cast<PHINode>(&I))
          if (PN->getType()->isPointerTy())
            PtrPhis.push_back(PN);

      if (!ForceAll && PtrPhis.size() <= PtrPhiLimit && !FunctionHasUndefPtrPhi)
        continue;

      for (PHINode *PN : PtrPhis) {
        convertPtrPhiToI64(PN, I64);
        Changed = true;
      }
    }
  }
  return Changed;
}

// ── Atomic intrinsic typed-pointer transition ───────────────────────────────
//
// The writer needs a fresh SSA pointer value before each `air.atomic.global.*`
// call so its pointee type in the side table can differ from the upstream
// GEP-result type (e.g. an i32 atomic on a float buffer needs an i32-typed
// pointer at the call site even though the GEP is typed float). The
// PointeeTypeMap built at write time picks up the inttoptr result via
// `inferFromUsage` and tags it with the intrinsic's expected pointee type.

static bool atomicTypedPointerFixup(Module &M) {
  bool Changed = false;
  Type *I64 = Type::getInt64Ty(M.getContext());

  SmallVector<CallInst *, 8> Fixups;
  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI || !CI->getCalledFunction())
          continue;
        StringRef Name = CI->getCalledFunction()->getName();
        if (!Name.starts_with("air.atomic.global."))
          continue;
        if (!Name.ends_with(".i32") && !Name.ends_with(".f32"))
          continue;
        Value *PtrArg = CI->getArgOperand(0);
        if (!PtrArg->getType()->isPointerTy())
          continue;
        unsigned AddrSpace = PtrArg->getType()->getPointerAddressSpace();
        if (AddrSpace != ASDevice && AddrSpace != ASThreadgroup)
          continue;
        // Only insert a transition when the pointer source is a GEP — the
        // typed-pointer mismatch this is fixing is exactly that case
        // (otherwise inferFromUsage already sees the atomic call directly
        // and would type the pointer to match).
        if (!isa<GetElementPtrInst>(PtrArg))
          continue;
        Fixups.push_back(CI);
      }

  for (CallInst *CI : Fixups) {
    Value *PtrArg = CI->getArgOperand(0);
    unsigned AddrSpace = PtrArg->getType()->getPointerAddressSpace();
    IRBuilder<> B(CI);
    Value *AsInt = B.CreatePtrToInt(PtrArg, I64);
    Value *NewPtr =
        B.CreateIntToPtr(AsInt, PointerType::get(M.getContext(), AddrSpace));
    CI->setArgOperand(0, NewPtr);
    Changed = true;
  }
  return Changed;
}

static bool feedsThreadgroupGEPIndex(Value *Val) {
  SmallVector<Value *, 16> Work{Val};
  SmallPtrSet<Value *, 16> Seen;
  while (!Work.empty()) {
    Value *V = Work.pop_back_val();
    if (!Seen.insert(V).second)
      continue;
    for (User *U : V->users()) {
      if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
        if (GEP->getPointerAddressSpace() == ASThreadgroup)
          for (auto &Idx : GEP->indices())
            if (Idx.get() == V)
              return true;
        Work.push_back(GEP);
      } else if (isa<BinaryOperator>(U) || isa<CastInst>(U) ||
                 isa<SelectInst>(U) || isa<PHINode>(U)) {
        Work.push_back(U);
      }
    }
  }
  return false;
}

static constexpr uint64_t kMaxAggregateFoldWork = 4096;

static bool foldConditionalConstants(Module &M) {
  bool Changed = false;
  TargetLibraryInfoImpl TLIImpl(Triple(M.getTargetTriple()));
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    uint64_t AggregateWork = 0;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *IVI = dyn_cast<InsertValueInst>(&I))
          if (auto *STy = dyn_cast<StructType>(IVI->getType()))
            AggregateWork += STy->getNumElements();
    if (AggregateWork > kMaxAggregateFoldWork)
      continue;

    SCCPSolver Solver(
        M.getDataLayout(),
        [&](Function &Fn) -> const TargetLibraryInfo & {
          static TargetLibraryInfo TLI(TLIImpl, &Fn);
          return TLI;
        },
        M.getContext());

    Solver.markBlockExecutable(&F.front());
    for (Argument &A : F.args())
      Solver.trackValueOfArgument(&A);

    bool ResolvedUndefs = true;
    while (ResolvedUndefs) {
      Solver.solve();
      ResolvedUndefs = Solver.resolvedUndefsIn(F);
    }

    SmallPtrSet<Value *, 32> ThreadVarying;
    {
      SmallVector<Value *, 32> Work;
      for (Argument &A : F.args()) {
        StringRef N = A.getName();
        if (N.starts_with("tid") || N.starts_with("simdlane"))
          Work.push_back(&A);
      }
      while (!Work.empty()) {
        Value *V = Work.pop_back_val();
        if (!ThreadVarying.insert(V).second)
          continue;
        for (User *U : V->users())
          if (isa<Instruction>(U))
            Work.push_back(U);
      }
    }

    for (BasicBlock &BB : F) {
      if (!Solver.isBlockExecutable(&BB))
        continue;
      for (Instruction &I : llvm::make_early_inc_range(BB)) {
        if (!I.getType()->isIntegerTy() || I.use_empty())
          continue;
        Constant *C = Solver.getConstantOrNull(&I);
        if (!C)
          continue;
        if (ThreadVarying.count(&I) && feedsThreadgroupGEPIndex(&I))
          continue;
        I.replaceAllUsesWith(C);
        if (isInstructionTriviallyDead(&I))
          I.eraseFromParent();
        Changed = true;
      }
    }
  }
  return Changed;
}

static bool metalPrepare(Module &M) {
  bool Changed = false;
  SmallPtrSet<Function *, 4> DecomposedFns;
  Changed |= foldConditionalConstants(M);
  Changed |= decomposeStructPhis(M, DecomposedFns);
  Changed |= rewriteTGGlobalGEPs(M);
  Changed |= normalizeI1Pointers(M);
  Changed |= ptrPhiToI64(M, DecomposedFns);
  Changed |= atomicTypedPointerFixup(M);
  return Changed;
}

PreservedAnalyses AIRPreparePass::run(Module &M, ModuleAnalysisManager &AM) {
  return metalPrepare(M) ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

bool AIRPrepareLegacy::runOnModule(Module &M) { return metalPrepare(M); }

char AIRPrepareLegacy::ID = 0;

INITIALIZE_PASS(AIRPrepareLegacy, DEBUG_TYPE, "AIR Prepare", false, false)

ModulePass *llvm::createAIRPrepareLegacyPass() {
  return new AIRPrepareLegacy();
}
