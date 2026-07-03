//===- DirectiveNameParser.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/OpenMP/DirectiveNameParser.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/OpenMP/OMP.h"

#include <cassert>
#include <memory>

namespace llvm::omp {
DirectiveNameParser::DirectiveNameParser(SourceLanguage L) {
  // Take every directive, get its name in every version, break the name up
  // into whitespace-separated tokens, and insert each token.
  for (size_t I : llvm::seq<size_t>(Directive_enumSize)) {
    auto D = static_cast<Directive>(I);
    if (D == Directive::OMPD_unknown || !(getDirectiveLanguages(D) & L))
      continue;
    for (unsigned Ver : getOpenMPVersions())
      insertName(getOpenMPDirectiveName(D, Ver), D);
  }
}

const DirectiveNameParser::State *
DirectiveNameParser::consume(const State *Current, StringRef Tok) const {
  if (!Current)
    return Current;
  assert(Current->isValid() && "Invalid input state");
  if (const State *Next = Current->next(Tok))
    return Next->isValid() ? Next : nullptr;
  return nullptr;
}

SmallVector<StringRef> DirectiveNameParser::tokenize(StringRef Str) {
  SmallVector<StringRef> Tokens;
  SplitString(Str, Tokens);
  return Tokens;
}

void DirectiveNameParser::insertName(StringRef Name, Directive D) {
  State *Where = &InitialState;

  for (StringRef Tok : tokenize(Name))
    Where = insertTransition(Where, Tok);

  Where->Value = D;
}

DirectiveNameParser::State *
DirectiveNameParser::insertTransition(State *From, StringRef Tok) {
  assert(From && "Expecting state");
  if (!From->Transition)
    From->Transition = std::make_unique<State::TransitionMapTy>();
  if (State *Next = From->next(Tok))
    return Next;

  auto [Where, DidIt] = From->Transition->try_emplace(Tok, State());
  assert(DidIt && "Map insertion failed");
  return &Where->second;
}

const DirectiveNameParser::State *
DirectiveNameParser::State::next(StringRef Tok) const {
  if (!Transition)
    return nullptr;
  auto F = Transition->find(Tok);
  return F != Transition->end() ? &F->second : nullptr;
}

DirectiveNameParser::State *DirectiveNameParser::State::next(StringRef Tok) {
  if (!Transition)
    return nullptr;
  auto F = Transition->find(Tok);
  return F != Transition->end() ? &F->second : nullptr;
}
} // namespace llvm::omp
