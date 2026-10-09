//===-- AMDGPULDSUtils.cpp - AMDGPU LDS utilities -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared helpers for computing LDS usage and limits for an AMDGPU function.
//
//===----------------------------------------------------------------------===//

#include "AMDGPULDSUtils.h"

#include "AMDGPUSubtarget.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsR600.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Target/TargetMachine.h"
#include <utility>

using namespace llvm;

//===----------------------------------------------------------------------===//
// Work-group / work-item query IR helpers
//===----------------------------------------------------------------------===//

namespace {

static std::pair<Value *, Value *> getLocalSizeYZ(IRBuilderBase &Builder,
                                                  const TargetMachine &TM,
                                                  const AMDGPUSubtarget &ST) {
  Function &F = *Builder.GetInsertBlock()->getParent();
  Module &M = *F.getParent();

  if (TM.getTargetTriple().getOS() != Triple::AMDHSA) {
    CallInst *LocalSizeY = Builder.CreateIntrinsicWithoutFolding(
        Intrinsic::r600_read_local_size_y, {});
    CallInst *LocalSizeZ = Builder.CreateIntrinsicWithoutFolding(
        Intrinsic::r600_read_local_size_z, {});

    ST.makeLIDRangeMetadata(LocalSizeY);
    ST.makeLIDRangeMetadata(LocalSizeZ);
    return {LocalSizeY, LocalSizeZ};
  }

  CallInst *DispatchPtr =
      Builder.CreateIntrinsicWithoutFolding(Intrinsic::amdgcn_dispatch_ptr, {});
  DispatchPtr->addRetAttr(Attribute::NoAlias);
  DispatchPtr->addRetAttr(Attribute::NonNull);
  F.removeFnAttr("amdgpu-no-dispatch-ptr");
  DispatchPtr->addDereferenceableRetAttr(64);

  Type *I32Ty = Type::getInt32Ty(M.getContext());
  Value *GEPXY = Builder.CreateConstInBoundsGEP1_64(I32Ty, DispatchPtr, 1);
  LoadInst *LoadXY = Builder.CreateAlignedLoad(I32Ty, GEPXY, Align(4));
  Value *GEPZU = Builder.CreateConstInBoundsGEP1_64(I32Ty, DispatchPtr, 2);
  LoadInst *LoadZU = Builder.CreateAlignedLoad(I32Ty, GEPZU, Align(4));

  MDNode *MD = MDNode::get(M.getContext(), {});
  LoadXY->setMetadata(LLVMContext::MD_invariant_load, MD);
  LoadZU->setMetadata(LLVMContext::MD_invariant_load, MD);
  ST.makeLIDRangeMetadata(LoadZU);

  Value *Y = Builder.CreateLShr(LoadXY, 16);
  return {Y, LoadZU};
}

static Value *getWorkitemID(IRBuilderBase &Builder, const TargetMachine &TM,
                            const AMDGPUSubtarget &ST, unsigned N) {
  Function *F = Builder.GetInsertBlock()->getParent();
  Intrinsic::ID IntrID = Intrinsic::not_intrinsic;
  StringRef AttrName;

  switch (N) {
  case 0:
    IntrID = TM.getTargetTriple().isAMDGCN()
                 ? static_cast<Intrinsic::ID>(Intrinsic::amdgcn_workitem_id_x)
                 : static_cast<Intrinsic::ID>(Intrinsic::r600_read_tidig_x);
    AttrName = "amdgpu-no-workitem-id-x";
    break;
  case 1:
    IntrID = TM.getTargetTriple().isAMDGCN()
                 ? static_cast<Intrinsic::ID>(Intrinsic::amdgcn_workitem_id_y)
                 : static_cast<Intrinsic::ID>(Intrinsic::r600_read_tidig_y);
    AttrName = "amdgpu-no-workitem-id-y";
    break;
  case 2:
    IntrID = TM.getTargetTriple().isAMDGCN()
                 ? static_cast<Intrinsic::ID>(Intrinsic::amdgcn_workitem_id_z)
                 : static_cast<Intrinsic::ID>(Intrinsic::r600_read_tidig_z);
    AttrName = "amdgpu-no-workitem-id-z";
    break;
  default:
    llvm_unreachable("invalid dimension");
  }

  Function *WorkitemIdFn =
      Intrinsic::getOrInsertDeclaration(F->getParent(), IntrID);
  CallInst *CI = Builder.CreateCall(WorkitemIdFn);
  ST.makeLIDRangeMetadata(CI);
  F->removeFnAttr(AttrName);
  return CI;
}

} // end anonymous namespace

Value *AMDGPU::buildLinearThreadId(IRBuilderBase &Builder,
                                   const TargetMachine &TM) {
  Function &F = *Builder.GetInsertBlock()->getParent();
  const AMDGPUSubtarget &ST = AMDGPUSubtarget::get(TM, F);
  Value *TCntY = nullptr;
  Value *TCntZ = nullptr;
  std::tie(TCntY, TCntZ) = getLocalSizeYZ(Builder, TM, ST);
  Value *TIdX = getWorkitemID(Builder, TM, ST, 0);
  Value *TIdY = getWorkitemID(Builder, TM, ST, 1);
  Value *TIdZ = getWorkitemID(Builder, TM, ST, 2);

  Value *Tmp0 = Builder.CreateMul(TCntY, TCntZ, "", true, true);
  Tmp0 = Builder.CreateMul(Tmp0, TIdX);
  Value *Tmp1 = Builder.CreateMul(TIdY, TCntZ, "", true, true);
  Value *TID = Builder.CreateAdd(Tmp0, Tmp1);
  TID = Builder.CreateAdd(TID, TIdZ);
  return TID;
}

//===----------------------------------------------------------------------===//
// LDS budget computation
//===----------------------------------------------------------------------===//

bool AMDGPU::AMDGPULDSBudget::tryReserve(uint64_t AllocSize, Align Alignment) {
  if (!Promotable || CurrentUsage > Limit)
    return false;

  // The backend may allocate LDS globals in a different order than the IR
  // pass visits them. Reserve the maximum possible leading padding so the
  // budget remains valid for any order.
  uint64_t Padding = CurrentUsage == 0 ? 0 : Alignment.value() - 1;
  if (Padding > Limit - CurrentUsage)
    return false;

  uint64_t PaddedUsage = CurrentUsage + Padding;
  if (AllocSize > Limit - PaddedUsage)
    return false;

  CurrentUsage = PaddedUsage + AllocSize;
  return true;
}

AMDGPU::AMDGPULDSBudget AMDGPU::computeLDSBudget(const Function &F,
                                                 const TargetMachine &TM) {
  AMDGPU::AMDGPULDSBudget Result;

  const AMDGPUSubtarget &ST = AMDGPUSubtarget::get(TM, F);
  const Module *M = F.getParent();
  const DataLayout &DL = M->getDataLayout();

  // If the function has any arguments in the local address space, then it's
  // possible these arguments require the entire local memory space, so
  // we cannot use local memory.
  FunctionType *FTy = F.getFunctionType();
  for (Type *ParamTy : FTy->params()) {
    PointerType *PtrTy = dyn_cast<PointerType>(ParamTy);
    if (PtrTy && PtrTy->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS) {
      Result.DisabledDueToLocalArg = true;
      return Result;
    }
  }

  uint32_t LocalMemLimit = ST.getAddressableLocalMemorySize();
  if (LocalMemLimit == 0)
    return Result;

  SmallVector<const Constant *, 16> Stack;
  SmallPtrSet<const Constant *, 8> VisitedConstants;
  SmallPtrSet<const GlobalVariable *, 8> UsedLDS;

  auto VisitUsers = [&](const Constant *Val) -> bool {
    for (const User *U : Val->users()) {
      if (const Instruction *Use = dyn_cast<Instruction>(U)) {
        if (Use->getParent()->getParent() == &F)
          return true;
      } else {
        const Constant *C = cast<Constant>(U);
        if (VisitedConstants.insert(C).second)
          Stack.push_back(C);
      }
    }
    return false;
  };

  for (const GlobalVariable &GV : M->globals()) {
    if (GV.getAddressSpace() != AMDGPUAS::LOCAL_ADDRESS)
      continue;

    if (VisitUsers(&GV)) {
      UsedLDS.insert(&GV);
      Stack.clear();
      continue;
    }

    while (!Stack.empty()) {
      const Constant *C = Stack.pop_back_val();
      if (VisitUsers(C)) {
        UsedLDS.insert(&GV);
        Stack.clear();
        break;
      }
    }
  }

  SmallVector<std::pair<uint64_t, Align>, 16> AllocatedSizes;
  AllocatedSizes.reserve(UsedLDS.size());

  for (const GlobalVariable *GV : UsedLDS) {
    Align Alignment =
        DL.getValueOrABITypeAlignment(GV->getAlign(), GV->getValueType());
    uint64_t AllocSize = DL.getTypeAllocSize(GV->getValueType());

    // HIP uses an extern unsized array in local address space for dynamically
    // allocated shared memory.
    if (GV->hasExternalLinkage() && AllocSize == 0) {
      Result.DisabledDueToExternDynShared = true;
      return Result;
    }

    AllocatedSizes.emplace_back(AllocSize, Alignment);
  }

  // Sort to try to estimate the worst case alignment padding.
  llvm::sort(AllocatedSizes, llvm::less_second());

  Result.Limit = LocalMemLimit;
  Result.Promotable = true;
  for (const std::pair<uint64_t, Align> &Alloc : AllocatedSizes) {
    uint64_t NewUsage = alignTo(Result.CurrentUsage, Alloc.second);
    if (NewUsage > Result.Limit || Alloc.first > Result.Limit - NewUsage) {
      Result.Promotable = false;
      return Result;
    }
    Result.CurrentUsage = NewUsage + Alloc.first;
  }

  unsigned MaxOccupancy =
      ST.getWavesPerEU(ST.getFlatWorkGroupSizes(F),
                       static_cast<uint32_t>(Result.CurrentUsage), F)
          .second;

  unsigned MaxSizeWithWaveCount =
      ST.getMaxLocalMemSizeWithWaveCount(MaxOccupancy, F);

  if (Result.CurrentUsage > MaxSizeWithWaveCount) {
    Result.Limit = MaxSizeWithWaveCount;
    Result.MaxOccupancy = MaxOccupancy;
    Result.Promotable = false;
    return Result;
  }

  Result.Limit = MaxSizeWithWaveCount;
  Result.MaxOccupancy = MaxOccupancy;
  return Result;
}
