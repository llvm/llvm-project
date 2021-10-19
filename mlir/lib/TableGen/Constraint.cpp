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
    : def(record), kind(CK_Uncategorized) {
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

Constraint::Constraint(Kind kind, const llvm::Record *record)
    : def(record), kind(kind) {
  // Look through OpVariable's to their constraint.
  if (def->isSubClassOf("OpVariable"))
    def = def->getValueAsDef("constraint");
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

AppliedConstraint::AppliedConstraint(Constraint &&constraint,
                                     llvm::StringRef self,
                                     std::vector<std::string> &&entities)
    : constraint(constraint), self(std::string(self)),
      entities(std::move(entities)) {}
