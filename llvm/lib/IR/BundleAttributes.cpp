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
}

BundleAttr llvm::getBundleAttrFromString(StringRef Str) {
  return StringSwitch<BundleAttr>(Str)
#define ATTR(Name, Str) .Case(#Str, BundleAttr::Name)
#include "llvm/IR/BundleAttributes.def"
      .Default(BundleAttr::None);
}

AssumeAlignInfo llvm::getAssumeAlignInfo(OperandBundleUse OBU) {
  assert(OBU.getTagName() == "align" && OBU.Inputs.size() >= 2 &&
         OBU.Inputs.size() <= 3);
  AssumeAlignInfo Ret{OBU.Inputs[0], OBU.Inputs[1], std::nullopt, std::nullopt};
  if (auto *Align = dyn_cast<ConstantInt>(OBU.Inputs[1]))
    Ret.AlignmentVal = Align->getZExtValue();
  if (OBU.Inputs.size() == 3) {
    if (auto *Offset = dyn_cast<ConstantInt>(OBU.Inputs[2]))
      Ret.OffsetVal = Offset->getZExtValue();
  } else {
    Ret.OffsetVal = 0;
  }
  return Ret;
}

AssumeSeparateStorageInfo
llvm::getAssumeSeparateStorageInfo(OperandBundleUse OBU) {
  assert(OBU.getTagName() == "separate_storage" && OBU.Inputs.size() == 2);
  return {OBU.Inputs[0], OBU.Inputs[1]};
}

AssumeNonNullInfo llvm::getAssumeNonNullInfo(OperandBundleUse OBU) {
  assert(OBU.getTagName() == "nonnull" && OBU.Inputs.size() == 1);
  return {OBU.Inputs[0]};
}

AssumeDereferenceableInfo
llvm::getAssumeDereferenceableInfo(OperandBundleUse OBU) {
  assert(OBU.getTagName() == "dereferenceable" && OBU.Inputs.size() == 2);
  AssumeDereferenceableInfo Ret{OBU.Inputs[0], OBU.Inputs[1], std::nullopt};

  if (auto *Size = dyn_cast<ConstantInt>(OBU.Inputs[1]))
    Ret.CountVal = Size->getZExtValue();
  return Ret;
}
