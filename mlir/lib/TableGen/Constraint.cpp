//===- Constraint.cpp - Constraint class ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Constraint wrapper to simplify using TableGen Record for constraints.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Constraint.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;

Constraint::Constraint(const llvm::Record *record)
    : Constraint(record, CK_Uncategorized) {
  // Look through OpVariable's to their constraint.
  if (def->isSubClassOf("OpVariable"))
    def = def->getValueAsDef("constraint");

  if (def->isSubClassOf("TypeConstraint")) {
    kind = CK_Type;
  } else if (def->isSubClassOf("AttrConstraint")) {
    kind = CK_Attr;
  } else if (def->isSubClassOf("RegionConstraint")) {
    kind = CK_Region;
  } else if (def->isSubClassOf("SuccessorConstraint")) {
    kind = CK_Successor;
  } else {
    assert(def->isSubClassOf("Constraint"));
  }
}

Pred Constraint::getPredicate() const {
  auto *val = def->getValue("predicate");

  // If no predicate is specified, then return the null predicate (which
  // corresponds to true).
  if (!val)
    return Pred();

  const auto *pred = dyn_cast<llvm::DefInit>(val->getValue());
  return Pred(pred);
}

std::string Constraint::getConditionTemplate() const {
  return getPredicate().getCondition();
}

StringRef Constraint::getSummary() const {
  if (Optional<StringRef> summary = def->getValueAsOptionalString("summary"))
    return *summary;
  return def->getName();
}

StringRef Constraint::getDescription() const {
  return def->getValueAsOptionalString("description").getValueOr("");
}

StringRef Constraint::getDefName() const {
  if (Optional<StringRef> baseDefName = getBaseDefName())
    return *baseDefName;
  return def->getName();
}

std::string Constraint::getUniqueDefName() const {
  std::string defName = def->getName().str();

  // Non-anonymous classes already have a unique name from the def.
  if (!def->isAnonymous())
    return defName;

  // Otherwise, this is an anonymous class. In these cases we still use the def
  // name, but we also try attach the name of the base def when present to make
  // the name more obvious.
  if (Optional<StringRef> baseDefName = getBaseDefName())
    return (*baseDefName + "(" + defName + ")").str();
  return defName;
}

Optional<StringRef> Constraint::getBaseDefName() const {
  // Functor used to check a base def in the case where the current def is
  // anonymous.
  auto checkBaseDefFn = [&](StringRef baseName) -> Optional<StringRef> {
    if (const auto *defValue = def->getValue(baseName)) {
      if (const auto *defInit = dyn_cast<llvm::DefInit>(defValue->getValue()))
        return Constraint(defInit->getDef(), kind).getDefName();
    }
    return llvm::None;
  };

  switch (kind) {
  case CK_Attr:
    if (def->isAnonymous())
      return checkBaseDefFn("baseAttr");
    return llvm::None;
  case CK_Type:
    if (def->isAnonymous())
      return checkBaseDefFn("baseType");
    return llvm::None;
  default:
    return llvm::None;
  }
}

AppliedConstraint::AppliedConstraint(Constraint &&constraint,
                                     llvm::StringRef self,
                                     std::vector<std::string> &&entities)
    : constraint(constraint), self(std::string(self)),
      entities(std::move(entities)) {}

Constraint DenseMapInfo<Constraint>::getEmptyKey() {
  return Constraint(RecordDenseMapInfo::getEmptyKey(),
                    Constraint::CK_Uncategorized);
}

Constraint DenseMapInfo<Constraint>::getTombstoneKey() {
  return Constraint(RecordDenseMapInfo::getTombstoneKey(),
                    Constraint::CK_Uncategorized);
}

unsigned DenseMapInfo<Constraint>::getHashValue(Constraint constraint) {
  if (constraint == getEmptyKey())
    return RecordDenseMapInfo::getHashValue(RecordDenseMapInfo::getEmptyKey());
  if (constraint == getTombstoneKey()) {
    return RecordDenseMapInfo::getHashValue(
        RecordDenseMapInfo::getTombstoneKey());
  }
  return llvm::hash_combine(constraint.getPredicate(), constraint.getSummary());
}

bool DenseMapInfo<Constraint>::isEqual(Constraint lhs, Constraint rhs) {
  if (lhs == rhs)
    return true;
  if (lhs == getEmptyKey() || lhs == getTombstoneKey())
    return false;
  if (rhs == getEmptyKey() || rhs == getTombstoneKey())
    return false;
  return lhs.getPredicate() == rhs.getPredicate() &&
         lhs.getSummary() == rhs.getSummary();
}
