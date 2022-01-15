//===- GIMatchDagPredicateDependencyEdge - Ensure predicates have inputs --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_GIMATCHDAGPREDICATEEDGE_H
#define LLVM_UTILS_TABLEGEN_GIMATCHDAGPREDICATEEDGE_H

#include "GIMatchDagOperands.h"

namespace llvm {
class GIMatchDag;
class GIMatchDagInstr;
class GIMatchDagPredicate;

/// Represents a dependency that must be met to evaluate a predicate.
///
/// Instances of this class objects are owned by the GIMatchDag and are not
/// shareable between instances of GIMatchDag.
class GIMatchDagPredicateDependencyEdge {
  /// The MI that must be available in order to test the predicate.
  const GIMatchDagInstr *RequiredMI;
  /// The MO that must be available in order to test the predicate. May be
  /// nullptr when only the MI is required.
  const GIMatchDagOperand *RequiredMO;
  /// The Predicate that requires information from RequiredMI/RequiredMO.
  const GIMatchDagPredicate *Predicate;
  /// The Predicate operand that requires information from
  /// RequiredMI/RequiredMO.
  const GIMatchDagOperand *PredicateOp;

public:
  GIMatchDagPredicateDependencyEdge(const GIMatchDagInstr *RequiredMI,
                                    const GIMatchDagOperand *RequiredMO,
                                    const GIMatchDagPredicate *Predicate,
                                    const GIMatchDagOperand *PredicateOp)
      : RequiredMI(RequiredMI), RequiredMO(RequiredMO), Predicate(Predicate),
        PredicateOp(PredicateOp) {}

  const GIMatchDagInstr *getRequiredMI() const { return RequiredMI; }
  const GIMatchDagOperand *getRequiredMO() const { return RequiredMO; }
  const GIMatchDagPredicate *getPredicate() const { return Predicate; }
  const GIMatchDagOperand *getPredicateOp() const { return PredicateOp; }

  void print(raw_ostream &OS) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif // if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
};

raw_ostream &operator<<(raw_ostream &OS,
                        const GIMatchDagPredicateDependencyEdge &N);

} // end namespace llvm
#endif // ifndef LLVM_UTILS_TABLEGEN_GIMATCHDAGPREDICATEEDGE_H
