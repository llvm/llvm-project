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
LLVM_ABI BundleAttr getBundleAttrFromID(uint32_t);

inline BundleAttr getBundleAttrFromOBU(OperandBundleUse OBU) {
  return getBundleAttrFromID(OBU.getTagID());
}

struct AssumeAlignInfo {
  const Use &Ptr;
  const Use &Alignment;
  const Use *Offset;
  std::optional<uint64_t> AlignmentVal;
  std::optional<uint64_t> OffsetVal;
};

LLVM_ABI AssumeAlignInfo getAssumeAlignInfo(OperandBundleUse);

struct AssumeDereferenceableInfo {
  const Use &Ptr;
  const Use &Count;
  std::optional<uint64_t> CountVal;
};

LLVM_ABI
AssumeDereferenceableInfo getAssumeDereferenceableInfo(OperandBundleUse);

struct AssumeNonNullInfo {
  const Use &Ptr;
};

LLVM_ABI AssumeNonNullInfo getAssumeNonNullInfo(OperandBundleUse);

struct AssumeNoUndefInfo {
  const Use &Val;
};

LLVM_ABI AssumeNoUndefInfo getAssumeNoUndefInfo(OperandBundleUse);

struct AssumeSeparateStorageInfo {
  const Use &Ptr1;
  const Use &Ptr2;
};

LLVM_ABI
AssumeSeparateStorageInfo getAssumeSeparateStorageInfo(OperandBundleUse);

LLVM_ABI bool assumeBundleImpliesNonNull(const Value *Val,
                                         const Function *Context,
                                         OperandBundleUse OBU);

} // namespace llvm

#endif // LLVM_IR_BUNDLE_ATTRIBUTES_H
