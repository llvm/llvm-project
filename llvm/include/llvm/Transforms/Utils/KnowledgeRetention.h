//===- KnowledgeRetention.h - utilities to preserve informations *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contain tools to preserve informations. They should be used before
// performing a transformation that may move and delete instructions as those
// transformation may destroy or worsen information that can be derived from the
// IR.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_ASSUMEBUILDER_H
#define LLVM_TRANSFORMS_UTILS_ASSUMEBUILDER_H

#include "llvm/IR/Attributes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/PassManager.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {

/// Build a call to llvm.assume to preserve informations that can be derived
/// from the given instruction.
/// If no information derived from \p I, this call returns null.
/// The returned instruction is not inserted anywhere.
CallInst *BuildAssumeFromInst(const Instruction *I, Module *M);
inline CallInst *BuildAssumeFromInst(Instruction *I) {
  return BuildAssumeFromInst(I, I->getModule());
}

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

using RetainedKnowledgeMap = DenseMap<RetainedKnowledgeKey, MinMax>;

/// Insert into the map all the informations contained in the operand bundles of
/// the llvm.assume. This should be used instead of hasAttributeInAssume when
/// many queries are going to be made on the same llvm.assume.
/// String attributes are not inserted in the map.
/// If the IR changes the map will be outdated.
void fillMapFromAssume(CallInst &AssumeCI, RetainedKnowledgeMap &Result);

//===----------------------------------------------------------------------===//
// Utilities for testing
//===----------------------------------------------------------------------===//

/// This pass will try to build an llvm.assume for every instruction in the
/// function. Its main purpose is testing.
struct AssumeBuilderPass : public PassInfoMixin<AssumeBuilderPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif
