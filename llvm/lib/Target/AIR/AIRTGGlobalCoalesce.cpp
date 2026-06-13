//===- AIRTGGlobalCoalesce.cpp - Merge cvt/dot TG globals ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AIRTGGlobalCoalesce.h"
#include "AIR.h"
#include "AIRAddressSpaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "air-tg-global-coalesce"

// AIR threadgroup address space.

// AIR MMA intrinsic prefix and the well-known TG global name prefixes used by
// the Triton frontend for layout conversion / dot buffers.
static constexpr StringLiteral kMMAPrefix("air.simdgroup_matrix_8x8_");
static constexpr StringLiteral kCvtPrefix("__tg_cvt_");
static constexpr StringLiteral kDotPrefix("__tg_dot_");
static constexpr StringLiteral kDotAbInfix("_ab_");

static bool moduleHasMMA(Module &M) {
  for (Function &F : M)
    if (F.getName().starts_with(kMMAPrefix))
      return true;
  return false;
}

// Erase threadgroup-address-space globals that have no remaining uses. The
// upstream Triton allocate-shared-memory pass sizes a `global_smem` swizzle
// scratch for every layout conversion (e.g. the #mma->#blocked C-output
// convert), but the Apple backend lowers those conversions in-tree through its
// own __tg_cvt_/__tg_dot_ab buffers, leaving `global_smem` declared but never
// loaded or stored. A dead threadgroup global still consumes the per-
// threadgroup memory budget at AIR layout assignment (the 32KB Metal cap), so
// for a wide f32 C tile the dead 32KB global_smem alone exhausts the budget and
// the kernel will not launch. Removing zero-use threadgroup globals reclaims
// that space; it is always safe (no use can observe the removal) and shrinks
// the footprint of every GEMM whose conversions are handled in-tree.
static bool eraseDeadThreadgroupGlobals(Module &M) {
  SmallVector<GlobalVariable *, 4> Dead;
  for (GlobalVariable &GV : M.globals()) {
    if (GV.getAddressSpace() != metal::AS::Threadgroup)
      continue;
    GV.removeDeadConstantUsers();
    if (GV.use_empty())
      Dead.push_back(&GV);
  }
  for (GlobalVariable *GV : Dead)
    GV->eraseFromParent();
  return !Dead.empty();
}

static void collectUseBlocks(GlobalVariable *GV,
                             SmallPtrSetImpl<BasicBlock *> &Blocks) {
  SmallVector<User *, 16> Work(GV->users());
  SmallPtrSet<User *, 16> Seen;
  while (!Work.empty()) {
    User *U = Work.pop_back_val();
    if (!Seen.insert(U).second)
      continue;
    if (auto *I = dyn_cast<Instruction>(U)) {
      if (isa<GetElementPtrInst>(I) || isa<BitCastInst>(I) ||
          isa<AddrSpaceCastInst>(I)) {
        for (User *UU : I->users())
          Work.push_back(UU);
        continue;
      }
      Blocks.insert(I->getParent());
      continue;
    }
    if (isa<ConstantExpr>(U))
      for (User *UU : U->users())
        Work.push_back(UU);
  }
}

static bool functionHasBarrier(Function &F) {
  for (Instruction &I : instructions(F))
    if (auto *CI = dyn_cast<CallInst>(&I))
      if (Function *Callee = CI->getCalledFunction())
        if (Callee->getName().starts_with("air.wg.barrier") ||
            Callee->getName().starts_with("air.threadgroup.barrier"))
          return true;
  return false;
}

static bool mergeDisjointDotBuffers(Module &M) {
  SmallVector<GlobalVariable *, 4> Dots;
  for (GlobalVariable &GV : M.globals()) {
    if (GV.getAddressSpace() != metal::AS::Threadgroup)
      continue;
    auto *AT = dyn_cast<ArrayType>(GV.getValueType());
    if (!AT || AT->getNumElements() <= 64)
      continue;
    StringRef Name = GV.getName();
    if (Name.starts_with(kDotPrefix) && Name.contains(kDotAbInfix))
      Dots.push_back(&GV);
  }
  if (Dots.size() < 2)
    return false;

  Function *Kernel = nullptr;
  for (Function &F : M)
    if (!F.isDeclaration()) {
      Kernel = &F;
      break;
    }
  if (!Kernel || !functionHasBarrier(*Kernel))
    return false;
  DominatorTree DT(*Kernel);

  SmallVector<Instruction *, 8> Barriers;
  for (Instruction &I : instructions(*Kernel))
    if (auto *CI = dyn_cast<CallInst>(&I))
      if (Function *Callee = CI->getCalledFunction())
        if (Callee->getName().starts_with("air.wg.barrier") ||
            Callee->getName().starts_with("air.threadgroup.barrier"))
          Barriers.push_back(&I);

  struct DotInfo {
    GlobalVariable *GV;
    SmallPtrSet<BasicBlock *, 8> Blocks;
  };
  SmallVector<DotInfo, 4> Infos;
  for (GlobalVariable *GV : Dots) {
    DotInfo DI;
    DI.GV = GV;
    collectUseBlocks(GV, DI.Blocks);
    bool AllInKernel = !DI.Blocks.empty();
    for (BasicBlock *BB : DI.Blocks)
      if (BB->getParent() != Kernel)
        AllInKernel = false;
    if (AllInKernel)
      Infos.push_back(std::move(DI));
  }
  if (Infos.size() < 2)
    return false;

  auto disjointBefore = [&](const DotInfo &A, const DotInfo &B) {
    for (BasicBlock *BA : A.Blocks)
      if (B.Blocks.count(BA))
        return false;
    for (BasicBlock *BB : B.Blocks)
      for (BasicBlock *BA : A.Blocks)
        if (BA == BB || !DT.dominates(BA, BB))
          return false;
    for (Instruction *Bar : Barriers) {
      BasicBlock *BarBB = Bar->getParent();
      if (A.Blocks.count(BarBB) || B.Blocks.count(BarBB))
        continue;
      bool AfterA = true, BeforeB = true;
      for (BasicBlock *BA : A.Blocks)
        if (!DT.dominates(BA, BarBB)) {
          AfterA = false;
          break;
        }
      if (!AfterA)
        continue;
      for (BasicBlock *BB : B.Blocks)
        if (!DT.dominates(BarBB, BB)) {
          BeforeB = false;
          break;
        }
      if (BeforeB)
        return true;
    }
    return false;
  };

  bool Changed = false;
  for (size_t I = 1; I < Infos.size(); ++I) {
    DotInfo &Cur = Infos[I];
    auto *CurAT = cast<ArrayType>(Cur.GV->getValueType());
    for (size_t J = 0; J < I; ++J) {
      DotInfo &Prev = Infos[J];
      if (!Prev.GV)
        continue;
      auto *PrevAT = cast<ArrayType>(Prev.GV->getValueType());
      if (PrevAT->getElementType() != CurAT->getElementType())
        continue;
      if (!disjointBefore(Prev, Cur) && !disjointBefore(Cur, Prev))
        continue;
      GlobalVariable *Surv = Prev.GV;
      auto *SurvAT = cast<ArrayType>(Surv->getValueType());
      if (CurAT->getNumElements() > SurvAT->getNumElements()) {
        auto *NewAT =
            ArrayType::get(SurvAT->getElementType(), CurAT->getNumElements());
        auto *NewGV = new GlobalVariable(
            M, NewAT, false, Surv->getLinkage(), UndefValue::get(NewAT),
            Surv->getName(), Surv, GlobalVariable::NotThreadLocal,
            Surv->getAddressSpace());
        NewGV->setAlignment(Surv->getAlign());
        Surv->replaceAllUsesWith(NewGV);
        Surv->eraseFromParent();
        Surv = NewGV;
        Prev.GV = NewGV;
      }
      Cur.GV->replaceAllUsesWith(Surv);
      Cur.GV->eraseFromParent();
      Cur.GV = nullptr;
      Prev.Blocks.insert(Cur.Blocks.begin(), Cur.Blocks.end());
      Changed = true;
      break;
    }
  }
  return Changed;
}

static bool tgGlobalCoalesce(Module &M) {
  if (!moduleHasMMA(M))
    return false;

  // Reclaim dead threadgroup globals (e.g. the unused upstream global_smem
  // swizzle scratch) before/independently of the cvt/dot merge below.
  bool Changed = eraseDeadThreadgroupGlobals(M);

  Changed |= mergeDisjointDotBuffers(M);

  SmallVector<GlobalVariable *, 4> CvtGlobals;
  SmallVector<GlobalVariable *, 4> DotAbGlobals;
  for (GlobalVariable &GV : M.globals()) {
    if (GV.getAddressSpace() != metal::AS::Threadgroup)
      continue;
    auto *AT = dyn_cast<ArrayType>(GV.getValueType());
    if (!AT || AT->getNumElements() <= 64)
      continue;
    StringRef Name = GV.getName();
    if (Name.starts_with(kCvtPrefix))
      CvtGlobals.push_back(&GV);
    else if (Name.starts_with(kDotPrefix) && Name.contains(kDotAbInfix))
      DotAbGlobals.push_back(&GV);
  }

  if (CvtGlobals.empty() || DotAbGlobals.empty())
    return Changed;

  const DataLayout &DL = M.getDataLayout();

  for (GlobalVariable *Cvt : CvtGlobals) {
    auto *CvtAT = dyn_cast<ArrayType>(Cvt->getValueType());
    if (!CvtAT)
      continue;

    // Prefer matching element type; fall back to same-size element type.
    GlobalVariable *Target = nullptr;
    size_t TargetIdx = 0;
    for (size_t I = 0; I < DotAbGlobals.size(); ++I) {
      auto *DotAT = dyn_cast<ArrayType>(DotAbGlobals[I]->getValueType());
      if (!DotAT)
        continue;
      if (DotAT->getElementType() == CvtAT->getElementType()) {
        Target = DotAbGlobals[I];
        TargetIdx = I;
        break;
      }
    }

    bool NeedsBitcast = false;
    if (!Target) {
      unsigned CvtElemSize = DL.getTypeSizeInBits(CvtAT->getElementType());
      for (size_t I = 0; I < DotAbGlobals.size(); ++I) {
        auto *DotAT = dyn_cast<ArrayType>(DotAbGlobals[I]->getValueType());
        if (!DotAT)
          continue;
        unsigned DotElemSize = DL.getTypeSizeInBits(DotAT->getElementType());
        if (DotElemSize == CvtElemSize) {
          Target = DotAbGlobals[I];
          TargetIdx = I;
          NeedsBitcast = true;
          break;
        }
      }
    }

    if (!Target)
      continue;

    auto *TargetAT = cast<ArrayType>(Target->getValueType());
    Type *TargetElemTy = TargetAT->getElementType();

    // Resize target if cvt is larger.
    if (CvtAT->getNumElements() > TargetAT->getNumElements()) {
      auto *NewAT = ArrayType::get(TargetElemTy, CvtAT->getNumElements());
      auto *NewGV = new GlobalVariable(
          M, NewAT, false, Target->getLinkage(), UndefValue::get(NewAT),
          Target->getName(), Target, GlobalVariable::NotThreadLocal,
          Target->getAddressSpace());
      Target->replaceAllUsesWith(NewGV);
      Target->eraseFromParent();
      Target = NewGV;
      DotAbGlobals[TargetIdx] = NewGV;
    }

    if (NeedsBitcast) {
      SmallVector<GetElementPtrInst *, 16> GEPsToRewrite;
      for (User *U : Cvt->users())
        if (auto *GEP = dyn_cast<GetElementPtrInst>(U))
          GEPsToRewrite.push_back(GEP);

      for (GetElementPtrInst *GEP : GEPsToRewrite) {
        IRBuilder<> B(GEP);
        SmallVector<Value *, 4> Indices;
        for (Value *Idx : GEP->indices())
          Indices.push_back(Idx);

        Value *NewGEP;
        if (GEP->getSourceElementType() == CvtAT)
          NewGEP = B.CreateGEP(Target->getValueType(), Target, Indices,
                               GEP->getName(), GEP->getNoWrapFlags());
        else
          NewGEP = B.CreateGEP(TargetElemTy, Target, Indices, GEP->getName(),
                               GEP->getNoWrapFlags());

        SmallVector<Instruction *, 8> Users;
        for (User *U : GEP->users())
          Users.push_back(cast<Instruction>(U));

        for (Instruction *U : Users) {
          if (auto *SI = dyn_cast<StoreInst>(U)) {
            if (SI->getPointerOperand() == GEP) {
              IRBuilder<> SB(SI);
              Value *Val = SI->getValueOperand();
              Value *Cast =
                  SB.CreateBitCast(Val, TargetElemTy, Val->getName() + "_bc");
              SB.CreateAlignedStore(Cast, NewGEP, SI->getAlign(),
                                    SI->isVolatile());
              SI->eraseFromParent();
            }
          } else if (auto *LI = dyn_cast<LoadInst>(U)) {
            IRBuilder<> LB(LI);
            auto *NewLoad = LB.CreateAlignedLoad(
                TargetElemTy, NewGEP, LI->getAlign(), LI->getName() + "_fl");
            if (LI->isVolatile())
              NewLoad->setVolatile(true);
            Value *Cast =
                LB.CreateBitCast(NewLoad, LI->getType(), LI->getName());
            LI->replaceAllUsesWith(Cast);
            LI->eraseFromParent();
          } else {
            U->replaceUsesOfWith(GEP, NewGEP);
          }
        }

        if (GEP->use_empty())
          GEP->eraseFromParent();
      }
    } else {
      Cvt->replaceAllUsesWith(Target);
    }

    Cvt->eraseFromParent();
    Changed = true;
  }

  return Changed;
}

PreservedAnalyses AIRTGGlobalCoalescePass::run(Module &M,
                                                 ModuleAnalysisManager &AM) {
  return tgGlobalCoalesce(M) ? PreservedAnalyses::none()
                             : PreservedAnalyses::all();
}

bool AIRTGGlobalCoalesceLegacy::runOnModule(Module &M) {
  return tgGlobalCoalesce(M);
}

char AIRTGGlobalCoalesceLegacy::ID = 0;

INITIALIZE_PASS(AIRTGGlobalCoalesceLegacy, DEBUG_TYPE,
                "AIR Threadgroup Global Coalesce", false, false)

ModulePass *llvm::createAIRTGGlobalCoalesceLegacyPass() {
  return new AIRTGGlobalCoalesceLegacy();
}
