//===- llvm/BundleAttributes.h - LLVM Bundle Attributes ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_BUNDLE_ATTRIBUTES_H
#define LLVM_IR_BUNDLE_ATTRIBUTES_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/InstrTypes.h"

enum class BundleAttr {
  None,
#define ATTR(Name, String) Name,
#include "BundleAttributes.def"
};

namespace llvm {

LLVM_ABI StringRef getNameFromBundleAttr(BundleAttr);
LLVM_ABI BundleAttr getBundleAttrFromString(StringRef);

inline BundleAttr getBundleAttrFromOBU(OperandBundleUse OBU) {
  return getBundleAttrFromString(OBU.getTagName());
}

struct AssumeAlignInfo {
  Value *Ptr;
  Value *Alignment;
  Value *Offset;
};

LLVM_ABI AssumeAlignInfo getAssumeAlignInfo(OperandBundleUse);

struct AssumeSeparateStorageInfo {
  const Use *Ptr1;
  const Use *Ptr2;
};

LLVM_ABI
AssumeSeparateStorageInfo getAssumeSeparateStorageInfo(OperandBundleUse);

struct AssumeNonNullInfo {
  Value *Ptr;
};

LLVM_ABI AssumeNonNullInfo getAssumeNonNullInfo(OperandBundleUse);

struct AssumeDereferenceableInfo {
  Value *Ptr;
  Value *Count;
};

LLVM_ABI
AssumeDereferenceableInfo getAssumeDereferenceableInfo(OperandBundleUse);

} // namespace llvm

#endif // LLVM_IR_BUNDLE_ATTRIBUTES_H
