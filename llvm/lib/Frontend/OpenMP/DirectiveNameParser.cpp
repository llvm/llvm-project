//===- DirectiveNameParser.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/OpenMP/DirectiveNameParser.h"
#include "llvm/ADT/Sequence.h"
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
DirectiveNameParser::apply(const State *Current, StringRef Tok) const {
  if (!Current)
    return Current;
  assert(Current->isValid() && "Invalid input state");
  if (const State *Next = const_cast<State *>(Current)->next(Tok))
    return Next->isValid() ? Next : nullptr;
  return nullptr;
}

SmallVector<StringRef> DirectiveNameParser::tokenize(StringRef Str) {
  SmallVector<StringRef> Tokens;

  auto nextChar = [](StringRef N, size_t I) {
    while (I < N.size() && N[I] == ' ')
      ++I;
    return I;
  };
  auto nextSpace = [](StringRef N, size_t I) {
    size_t S = N.find(' ', I);
    return S != StringRef::npos ? S : N.size();
  };

  size_t From = nextChar(Str, 0);
  size_t To = 0;

  while (From != Str.size()) {
    To = nextSpace(Str, From);
    Tokens.push_back(Str.substr(From, To - From));
    From = nextChar(Str, To);
  }

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
  if (!From->Transition) {
    From->Transition = std::make_unique<State::TransitionMapTy>();
  }
  if (State *Next = From->next(Tok))
    return Next;

  auto [Where, DidIt] = From->Transition->try_emplace(Tok, State());
  assert(DidIt && "Map insertion failed");
  return &Where->second;
}

DirectiveNameParser::State *DirectiveNameParser::State::next(StringRef Tok) {
  if (!Transition)
    return nullptr;
  auto F = Transition->find(Tok);
  return F != Transition->end() ? &F->second : nullptr;
}
} // namespace llvm::omp
