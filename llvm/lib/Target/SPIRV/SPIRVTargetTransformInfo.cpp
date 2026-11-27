//===- SPIRVTargetTransformInfo.cpp - SPIR-V specific TTI -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SPIRVTargetTransformInfo.h"
#include "SPIRVSubtarget.h"
#include "SPIRVUtils.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IntrinsicsSPIRV.h"

using namespace llvm;

// Builtin variables are not always in the Input address space, the backend
// also recognizes them by name or by spirv.Decorations metadata.
static bool isBuiltInVariable(const Value *Ptr) {
  const auto *GV = dyn_cast<GlobalVariable>(getUnderlyingObject(Ptr));
  if (!GV)
    return false;
  SPIRV::BuiltIn::BuiltIn BI;
  if (getSpirvBuiltInIdByName(GV->getName(), BI))
    return true;
  const MDNode *MD = GV->getMetadata("spirv.Decorations");
  return MD && any_of(MD->operands(), [](const MDOperand &Op) {
           const auto *Dec = dyn_cast<MDNode>(Op);
           if (!Dec || Dec->getNumOperands() == 0)
             return false;
           const auto *Id =
               mdconst::dyn_extract<ConstantInt>(Dec->getOperand(0));
           return Id && Id->getZExtValue() == SPIRV::Decoration::BuiltIn;
         });
}

// Without inter-procedural or pointer analysis, conservatively treat values
// that may differ between invocations as divergent.
static bool isSourceOfDivergence(const Value *V) {
  if (const auto *Arg = dyn_cast<Argument>(V))
    return !isEntryPoint(*Arg->getParent());
  if (const auto *LI = dyn_cast<LoadInst>(V)) {
    unsigned AS = LI->getPointerAddressSpace();
    return AS == SPIRVAS::Function || AS == SPIRVAS::Generic ||
           AS == SPIRVAS::Private || AS == SPIRVAS::Input ||
           AS == SPIRVAS::Output || LI->isAtomic() ||
           isBuiltInVariable(LI->getPointerOperand());
  }
  // Calls cover work-item ID builtins and intrinsics.
  return isa<CallBase, AtomicRMWInst, AtomicCmpXchgInst>(V);
}

ValueUniformity llvm::SPIRVTTIImpl::getValueUniformity(const Value *V) const {
  return isSourceOfDivergence(V) ? ValueUniformity::NeverUniform
                                 : ValueUniformity::Default;
}

bool llvm::SPIRVTTIImpl::collectFlatAddressOperands(
    SmallVectorImpl<int> &OpIndexes, Intrinsic::ID IID) const {
  switch (IID) {
  case Intrinsic::spv_generic_cast_to_ptr_explicit:
    OpIndexes.push_back(0);
    return true;
  default:
    return false;
  }
}

Value *llvm::SPIRVTTIImpl::rewriteIntrinsicWithAddressSpace(IntrinsicInst *II,
                                                            Value *OldV,
                                                            Value *NewV) const {
  auto IntrID = II->getIntrinsicID();
  switch (IntrID) {
  case Intrinsic::spv_generic_cast_to_ptr_explicit: {
    unsigned NewAS = NewV->getType()->getPointerAddressSpace();
    unsigned DstAS = II->getType()->getPointerAddressSpace();
    return NewAS == DstAS ? NewV
                          : ConstantPointerNull::get(
                                PointerType::get(NewV->getContext(), DstAS));
  }
  default:
    return nullptr;
  }
}

bool SPIRVTTIImpl::isLegalMaskedGather(Type *DataType, Align Alignment) const {
  return ST->canUseExtension(SPIRV::Extension::SPV_INTEL_masked_gather_scatter);
}

bool SPIRVTTIImpl::isLegalMaskedScatter(Type *DataType, Align Alignment) const {
  return ST->canUseExtension(SPIRV::Extension::SPV_INTEL_masked_gather_scatter);
}
