//===-- AMDGPULDSUtils.cpp - AMDGPU LDS utilities ------------------------===//
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

#include "Utils/AMDGPULDSUtils.h"

#include "AMDGPU.h"
#include "GCNSubtarget.h"
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
#include "llvm/IR/Module.h"
#include "llvm/Support/Alignment.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
// Work-group / work-item query IR helpers
//===----------------------------------------------------------------------===//

namespace {

// Read local size Y/Z from the HSA dispatch packet.
static std::pair<Value *, Value *>
getLocalSizeYZFromDispatch(IRBuilderBase &Builder, Module &M,
                           const AMDGPUSubtarget &ST) {
  Function &F = *Builder.GetInsertBlock()->getParent();

  CallInst *DispatchPtr = cast<CallInst>(
      Builder.CreateIntrinsic(Intrinsic::amdgcn_dispatch_ptr, {}));
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

} // end anonymous namespace

Value *AMDGPU::getWorkitemID(IRBuilderBase &Builder, Module &M,
                             const AMDGPUSubtarget &ST, unsigned N) {
  Function *F = Builder.GetInsertBlock()->getParent();
  Intrinsic::ID IntrID = Intrinsic::not_intrinsic;
  StringRef AttrName;

  switch (N) {
  case 0:
    IntrID = Intrinsic::amdgcn_workitem_id_x;
    AttrName = "amdgpu-no-workitem-id-x";
    break;
  case 1:
    IntrID = Intrinsic::amdgcn_workitem_id_y;
    AttrName = "amdgpu-no-workitem-id-y";
    break;
  case 2:
    IntrID = Intrinsic::amdgcn_workitem_id_z;
    AttrName = "amdgpu-no-workitem-id-z";
    break;
  default:
    llvm_unreachable("invalid dimension");
  }

  Function *WorkitemIdFn = Intrinsic::getOrInsertDeclaration(&M, IntrID);
  CallInst *CI = cast<CallInst>(Builder.CreateCall(WorkitemIdFn));
  ST.makeLIDRangeMetadata(CI);
  F->removeFnAttr(AttrName);
  return CI;
}

Value *AMDGPU::buildLinearThreadId(IRBuilderBase &Builder, Module &M,
                                   const AMDGPUSubtarget &ST) {
  Value *TCntY = nullptr;
  Value *TCntZ = nullptr;
  std::tie(TCntY, TCntZ) = getLocalSizeYZFromDispatch(Builder, M, ST);
  Value *TIdX = getWorkitemID(Builder, M, ST, 0);
  Value *TIdY = getWorkitemID(Builder, M, ST, 1);
  Value *TIdZ = getWorkitemID(Builder, M, ST, 2);

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
      Result.limit = 0;
      Result.promotable = false;
      Result.disabledDueToLocalArg = true;
      return Result;
    }
  }

  uint32_t LocalMemLimit = ST.getAddressableLocalMemorySize();
  if (LocalMemLimit == 0) {
    Result.limit = 0;
    Result.promotable = false;
    return Result;
  }

  SmallVector<const Constant *, 16> Stack;
  SmallPtrSet<const Constant *, 8> VisitedConstants;
  SmallPtrSet<const GlobalVariable *, 8> UsedLDS;

  auto visitUsers = [&](const GlobalVariable *GV, const Constant *Val) -> bool {
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

    if (visitUsers(&GV, &GV)) {
      UsedLDS.insert(&GV);
      Stack.clear();
      continue;
    }

    while (!Stack.empty()) {
      const Constant *C = Stack.pop_back_val();
      if (visitUsers(&GV, C)) {
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
      Result.limit = 0;
      Result.promotable = false;
      Result.disabledDueToExternDynShared = true;
      return Result;
    }

    AllocatedSizes.emplace_back(AllocSize, Alignment);
  }

  // Sort to try to estimate the worst case alignment padding.
  llvm::sort(AllocatedSizes, llvm::less_second());

  uint32_t CurrentLocalMemUsage = 0;
  for (auto Alloc : AllocatedSizes) {
    CurrentLocalMemUsage = alignTo(CurrentLocalMemUsage, Alloc.second);
    CurrentLocalMemUsage += Alloc.first;
  }

  unsigned MaxOccupancy =
      ST.getWavesPerEU(ST.getFlatWorkGroupSizes(F), CurrentLocalMemUsage, F)
          .second;

  unsigned MaxSizeWithWaveCount =
      ST.getMaxLocalMemSizeWithWaveCount(MaxOccupancy, F);

  if (CurrentLocalMemUsage > MaxSizeWithWaveCount) {
    Result.currentUsage = CurrentLocalMemUsage;
    Result.limit = MaxSizeWithWaveCount;
    Result.maxOccupancy = MaxOccupancy;
    Result.promotable = false;
    return Result;
  }

  Result.currentUsage = CurrentLocalMemUsage;
  Result.limit = MaxSizeWithWaveCount;
  Result.maxOccupancy = MaxOccupancy;
  Result.promotable = true;
  return Result;
}
