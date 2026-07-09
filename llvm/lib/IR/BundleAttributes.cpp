//===- llvm/BundleAttributes.cpp - LLVM Bundle Attributes -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/BundleAttributes.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Constants.h"

using namespace llvm;

StringRef llvm::getNameFromBundleAttr(BundleAttr BA) {
  switch (BA) {
#define ATTR(Name, Str)                                                        \
  case BundleAttr::Name:                                                       \
    return #Str;
#include "llvm/IR/BundleAttributes.def"
  case BundleAttr::None:
    return "none";
  }
  llvm_unreachable("unknonwn bundle attribute");
}

BundleAttr llvm::getBundleAttrFromID(uint32_t ID) {
  switch (ID) {
#define ATTR(Name, Str)                                                        \
  case LLVMContext::OB_##Name:                                                 \
    return BundleAttr::Name;
#include "llvm/IR/BundleAttributes.def"
  default:
    return BundleAttr::None;
  }
}

AssumeAlignInfo llvm::getAssumeAlignInfo(OperandBundleUse OBU) {
  assert(OBU.getTagID() == LLVMContext::OB_Align && OBU.Inputs.size() >= 2 &&
         OBU.Inputs.size() <= 3);
  AssumeAlignInfo Ret{OBU.Inputs[0], OBU.Inputs[1], nullptr, std::nullopt,
                      std::nullopt};
  if (auto *Align = dyn_cast<ConstantInt>(OBU.Inputs[1]))
    Ret.AlignmentVal = Align->getZExtValue();
  if (OBU.Inputs.size() == 3) {
    Ret.Offset = &OBU.Inputs[2];
    if (auto *Offset = dyn_cast<ConstantInt>(OBU.Inputs[2]))
      Ret.OffsetVal = Offset->getZExtValue();
  } else {
    Ret.OffsetVal = 0;
  }
  return Ret;
}

AssumeNoUndefInfo llvm::getAssumeNoUndefInfo(OperandBundleUse OBU) {
  assert(OBU.getTagID() == LLVMContext::OB_NoUndef && OBU.Inputs.size() == 1);
  return {OBU.Inputs[0]};
}

AssumeSeparateStorageInfo
llvm::getAssumeSeparateStorageInfo(OperandBundleUse OBU) {
  assert(OBU.getTagID() == LLVMContext::OB_SeparateStorage &&
         OBU.Inputs.size() == 2);
  return {OBU.Inputs[0], OBU.Inputs[1]};
}

AssumeNonNullInfo llvm::getAssumeNonNullInfo(OperandBundleUse OBU) {
  assert(OBU.getTagID() == LLVMContext::OB_NonNull && OBU.Inputs.size() == 1);
  return {OBU.Inputs[0]};
}

AssumeDereferenceableInfo
llvm::getAssumeDereferenceableInfo(OperandBundleUse OBU) {
  assert(OBU.getTagID() == LLVMContext::OB_Dereferenceable &&
         OBU.Inputs.size() == 2);
  AssumeDereferenceableInfo Ret{OBU.Inputs[0], OBU.Inputs[1], std::nullopt};

  if (auto *Size = dyn_cast<ConstantInt>(OBU.Inputs[1]))
    Ret.CountVal = Size->getZExtValue();
  return Ret;
}

bool llvm::assumeBundleImpliesNonNull(const Value *Val, const Function *Context,
                                      OperandBundleUse OBU) {
  switch (getBundleAttrFromOBU(OBU)) {
  case BundleAttr::Align: {
    auto [Ptr, _, _2, Alignment, Offset] = getAssumeAlignInfo(OBU);
    return Ptr == Val && Alignment && Offset && isPowerOf2_64(*Alignment) &&
           *Offset % *Alignment != 0;
  }

  case BundleAttr::Dereferenceable: {
    auto [Ptr, _, Count] = getAssumeDereferenceableInfo(OBU);
    return Ptr == Val && Count && *Count != 0 &&
           !NullPointerIsDefined(Context,
                                 Val->getType()->getPointerAddressSpace());
  }

  case BundleAttr::NonNull:
    return getAssumeNonNullInfo(OBU).Ptr == Val;

  default:
    return false;
  }
}
