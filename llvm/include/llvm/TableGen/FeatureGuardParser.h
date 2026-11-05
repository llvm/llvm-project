//===- FeatureGuardParser.h - Parse feature guards --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TABLEGEN_FEATUREGUARDPARSER_H
#define LLVM_TABLEGEN_FEATUREGUARDPARSER_H

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include <set>
#include <string>
#include <utility>

namespace llvm {

// A clause is a conjunction: all features in the set must be present.
using RequiredFeatureSet = std::set<std::string>;
// A guard is a disjunction: any one clause may satisfy it. An empty guard
// cannot be satisfied. Repeated clauses are discarded.
using FeatureGuardAlternatives = std::set<RequiredFeatureSet>;

// Parse ',' as conjunction and '|' as disjunction. Only spaces are treated as
// whitespace; tabs and line breaks are invalid. The result is in disjunctive
// normal form.
class FeatureGuardParser {
public:
  // The handler must not return.
  using ErrorHandler = function_ref<void(const Twine &Message)>;

private:
  StringRef Remaining;
  ErrorHandler HandleError;

  [[noreturn]] void error(const Twine &Message) const {
    HandleError(Message);
    llvm_unreachable("feature guard error handler returned");
  }

  bool consume(StringRef C) {
    Remaining = Remaining.ltrim(' ');
    return Remaining.consume_front(C);
  }

  FeatureGuardAlternatives parseAtom() {
    if (consume("(")) {
      FeatureGuardAlternatives G = parseOr();
      if (!consume(")"))
        error("unclosed target guard expression");
      return G;
    }
    size_t End = Remaining.find_first_of(",|() \t\n\r");
    StringRef Name = Remaining.take_front(End);
    Remaining = Remaining.drop_front(Name.size());
    if (Name.empty())
      error("expected feature in target guard");
    return {{Name.str()}};
  }

  FeatureGuardAlternatives parseAnd() {
    FeatureGuardAlternatives LHS = parseAtom();
    while (consume(",")) {
      FeatureGuardAlternatives RHS = parseAtom();
      FeatureGuardAlternatives Product;
      for (const RequiredFeatureSet &L : LHS) {
        for (const RequiredFeatureSet &R : RHS) {
          RequiredFeatureSet C = L;
          C.insert(R.begin(), R.end());
          Product.insert(std::move(C));
        }
      }
      LHS = std::move(Product);
    }
    return LHS;
  }

  FeatureGuardAlternatives parseOr() {
    FeatureGuardAlternatives LHS = parseAnd();
    while (consume("|")) {
      FeatureGuardAlternatives RHS = parseAnd();
      LHS.insert(RHS.begin(), RHS.end());
    }
    return LHS;
  }

public:
  FeatureGuardParser(StringRef Text, ErrorHandler HandleError)
      : Remaining(Text), HandleError(HandleError) {}

  FeatureGuardAlternatives parse() {
    Remaining = Remaining.trim(' ');
    if (Remaining.empty())
      return {};
    FeatureGuardAlternatives G = parseOr();
    if (!Remaining.trim(' ').empty())
      error("invalid target guard expression: " + Remaining);
    return G;
  }
};

} // namespace llvm

#endif // LLVM_TABLEGEN_FEATUREGUARDPARSER_H
