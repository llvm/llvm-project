//===- DirectiveNameParser.h  ------------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_OPENMP_DIRECTIVENAMEPARSER_H
#define LLVM_FRONTEND_OPENMP_DIRECTIVENAMEPARSER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/OpenMP/OMP.h"

#include <memory>

namespace llvm::omp {
/// Parser class for OpenMP directive names. It only recognizes names listed
/// in OMP.td, in particular it does not recognize Fortran's end-directives
/// if they are not explicitly listed in OMP.td.
///
/// The class itself may be a singleton, once it's constructed it never
/// changes.
///
/// Usage:
/// {
///   DirectiveNameParser Parser;   // Could be static const.
///
///   DirectiveNameParser::State *S = Parser.initial();
///   for (StringRef Token : Tokens)
///     S = Parser.consume(S, Token); // Passing nullptr is ok.
///
///   if (S == nullptr) {
///     // Error: ended up in a state from which there is no possible path
///     // to a successful parse.
///   } else if (S->Value == OMPD_unknown) {
///     // Parsed a sequence of tokens that are not a complete name, but
///     // parsing more tokens could lead to a successful parse.
///   } else {
///     // Success.
///     ParsedId = S->Value;
///   }
/// }
struct DirectiveNameParser {
  DirectiveNameParser(SourceLanguage L = SourceLanguage::C);

  struct State {
    Directive Value = Directive::OMPD_unknown;

  private:
    using TransitionMapTy = StringMap<State>;
    std::unique_ptr<TransitionMapTy> Transition;

    State *next(StringRef Tok);
    const State *next(StringRef Tok) const;
    bool isValid() const {
      return Value != Directive::OMPD_unknown || !Transition->empty();
    }
    friend struct DirectiveNameParser;
  };

  const State *initial() const { return &InitialState; }
  const State *consume(const State *Current, StringRef Tok) const;

  static SmallVector<StringRef> tokenize(StringRef N);

private:
  void insertName(StringRef Name, Directive D);
  State *insertTransition(State *From, StringRef Tok);

  State InitialState;
};
} // namespace llvm::omp

#endif // LLVM_FRONTEND_OPENMP_DIRECTIVENAMEPARSER_H
