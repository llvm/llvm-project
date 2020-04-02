//===- AssumeBundleQueries.h - utilities to query assume bundles *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contain tools to query into assume bundles. assume bundles can be
// built using utilities from Transform/Utils/AssumeBundleBuilder.h
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_ASSUMEBUNDLEQUERIES_H
#define LLVM_TRANSFORMS_UTILS_ASSUMEBUNDLEQUERIES_H

#include "llvm/IR/Attributes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {
class IntrinsicInst;

/// Index of elements in the operand bundle.
/// If the element exist it is guaranteed to be what is specified in this enum
/// but it may not exist.
enum AssumeBundleArg {
  ABA_WasOn = 0,
  ABA_Argument = 1,
};

/// It is possible to have multiple Value for the argument of an attribute in
/// the same llvm.assume on the same llvm::Value. This is rare but need to be
/// dealt with.
enum class AssumeQuery {
  Highest, ///< Take the highest value available.
  Lowest,  ///< Take the lowest value available.
};

/// Query the operand bundle of an llvm.assume to find a single attribute of
/// the specified kind applied on a specified Value.
///
/// This has a non-constant complexity. It should only be used when a single
/// attribute is going to be queried.
///
/// Return true iff the queried attribute was found.
/// If ArgVal is set. the argument will be stored to ArgVal.
bool hasAttributeInAssume(CallInst &AssumeCI, Value *IsOn, StringRef AttrName,
                          uint64_t *ArgVal = nullptr,
                          AssumeQuery AQR = AssumeQuery::Highest);
inline bool hasAttributeInAssume(CallInst &AssumeCI, Value *IsOn,
                                 Attribute::AttrKind Kind,
                                 uint64_t *ArgVal = nullptr,
                                 AssumeQuery AQR = AssumeQuery::Highest) {
  return hasAttributeInAssume(
      AssumeCI, IsOn, Attribute::getNameFromAttrKind(Kind), ArgVal, AQR);
}

template<> struct DenseMapInfo<Attribute::AttrKind> {
  static Attribute::AttrKind getEmptyKey() {
    return Attribute::EmptyKey;
  }
  static Attribute::AttrKind getTombstoneKey() {
    return Attribute::TombstoneKey;
  }
  static unsigned getHashValue(Attribute::AttrKind AK) {
    return hash_combine(AK);
  }
  static bool isEqual(Attribute::AttrKind LHS, Attribute::AttrKind RHS) {
    return LHS == RHS;
  }
};

/// The map Key contains the Value on for which the attribute is valid and
/// the Attribute that is valid for that value.
/// If the Attribute is not on any value, the Value is nullptr.
using RetainedKnowledgeKey = std::pair<Value *, Attribute::AttrKind>;

struct MinMax {
  unsigned Min;
  unsigned Max;
};

/// A mapping from intrinsics (=`llvm.assume` calls) to a value range
/// (=knowledge) that is encoded in them. How the value range is interpreted
/// depends on the RetainedKnowledgeKey that was used to get this out of the
/// RetainedKnowledgeMap.
using Assume2KnowledgeMap = DenseMap<IntrinsicInst *, MinMax>;

using RetainedKnowledgeMap =
    DenseMap<RetainedKnowledgeKey, Assume2KnowledgeMap>;

/// Insert into the map all the informations contained in the operand bundles of
/// the llvm.assume. This should be used instead of hasAttributeInAssume when
/// many queries are going to be made on the same llvm.assume.
/// String attributes are not inserted in the map.
/// If the IR changes the map will be outdated.
void fillMapFromAssume(CallInst &AssumeCI, RetainedKnowledgeMap &Result);

/// Represent one information held inside an operand bundle of an llvm.assume.
/// AttrKind is the property that hold.
/// WasOn if not null is that Value for which AttrKind holds.
/// ArgValue is optionally an argument.
struct RetainedKnowledge {
  Attribute::AttrKind AttrKind = Attribute::None;
  Value *WasOn = nullptr;
  unsigned ArgValue = 0;
};

/// Retreive the information help by Assume on the operand at index Idx.
/// Assume should be an llvm.assume and Idx should be in the operand bundle.
RetainedKnowledge getKnowledgeFromOperandInAssume(CallInst &Assume,
                                                  unsigned Idx);

/// Retreive the information help by the Use U of an llvm.assume. the use should
/// be in the operand bundle.
inline RetainedKnowledge getKnowledgeFromUseInAssume(const Use *U) {
  return getKnowledgeFromOperandInAssume(*cast<CallInst>(U->getUser()),
                                         U->getOperandNo());
}

/// Return true iff the operand bundles of the provided llvm.assume doesn't
/// contain any valuable information. This is true when:
///  - The operand bundle is empty
///  - The operand bundle only contains information about dropped values or
///    constant folded values.
///
/// the argument to the call of llvm.assume may still be useful even if the
/// function returned true.
bool isAssumeWithEmptyBundle(CallInst &Assume);

} // namespace llvm

#endif
