//===- CmpPredicate.h - CmpInst Predicate with samesign information -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A CmpInst::Predicate with any samesign information (applicable to ICmpInst).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_CMPPREDICATE_H
#define LLVM_IR_CMPPREDICATE_H

#include "llvm/IR/InstrTypes.h"

namespace llvm {
/// An abstraction over a floating-point predicate, and a pack of an integer
/// predicate with samesign information. Some functions in ICmpInst construct
/// and return this type in place of a Predicate.
class CmpPredicate {
  CmpInst::Predicate Pred;
  bool HasSameSign;

public:
  /// Default constructor.
  CmpPredicate() : Pred(CmpInst::BAD_ICMP_PREDICATE), HasSameSign(false) {}

  /// Constructed implictly with a either Predicate and samesign information, or
  /// just a Predicate, dropping samesign information.
  CmpPredicate(CmpInst::Predicate Pred, bool HasSameSign = false)
      : Pred(Pred), HasSameSign(HasSameSign) {
    assert(!HasSameSign || CmpInst::isIntPredicate(Pred));
  }

  /// Implictly converts to the underlying Predicate, dropping samesign
  /// information.
  operator CmpInst::Predicate() const { return Pred; }

  /// Query samesign information, for optimizations.
  bool hasSameSign() const { return HasSameSign; }

  /// Compares two CmpPredicates taking samesign into account and returns the
  /// canonicalized CmpPredicate if they match. An alternative to operator==.
  ///
  /// For example,
  ///   samesign ult + samesign ult -> samesign ult
  ///   samesign ult + ult -> ult
  ///   samesign ult + slt -> slt
  ///   ult + ult -> ult
  ///   ult + slt -> std::nullopt
  static std::optional<CmpPredicate> getMatching(CmpPredicate A,
                                                 CmpPredicate B);

  /// Attempts to return a signed CmpInst::Predicate from the CmpPredicate. If
  /// the CmpPredicate has samesign, return ICmpInst::getSignedPredicate,
  /// dropping samesign information. Otherwise, return the predicate, dropping
  /// samesign information.
  CmpInst::Predicate getPreferredSignedPredicate() const;

  /// An operator== on the underlying Predicate.
  bool operator==(CmpInst::Predicate P) const { return Pred == P; }
  bool operator!=(CmpInst::Predicate P) const { return Pred != P; }

  /// There is no operator== defined on CmpPredicate. Use getMatching instead to
  /// get the canonicalized matching CmpPredicate.
  bool operator==(CmpPredicate) const = delete;
  bool operator!=(CmpPredicate) const = delete;

  /// Do a ICmpInst::getCmpPredicate() or CmpInst::getPredicate(), as
  /// appropriate.
  static CmpPredicate get(const CmpInst *Cmp);

  /// Get the swapped predicate of a CmpPredicate.
  static CmpPredicate getSwapped(CmpPredicate P);

  /// Get the swapped predicate of a CmpInst.
  static CmpPredicate getSwapped(const CmpInst *Cmp);
};
} // namespace llvm

#endif
