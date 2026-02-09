//===- Option.cpp - Abstract Driver Options -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Option/Option.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace llvm;
using namespace llvm::opt;

Option::Option(const OptTable::Info *info, const OptTable *owner)
  : Info(info), Owner(owner) {
  // Multi-level aliases are not supported. This just simplifies option
  // tracking, it is not an inherent limitation.
  assert((!Info || !getAlias().isValid() || !getAlias().getAlias().isValid()) &&
         "Multi-level aliases are not supported.");

  if (Info && getAliasArgs()) {
    assert(getAlias().isValid() && "Only alias options can have alias args.");
    assert(getKind() == FlagClass && "Only Flag aliases can have alias args.");
    assert(getAlias().getKind() != FlagClass &&
           "Cannot provide alias args to a flag option.");
  }
}

void Option::print(raw_ostream &O, bool AddNewLine) const {
  O << "<";
  switch (getKind()) {
#define P(N) case N: O << #N; break
    P(GroupClass);
    P(InputClass);
    P(UnknownClass);
    P(FlagClass);
    P(JoinedClass);
    P(ValuesClass);
    P(SeparateClass);
    P(CommaJoinedClass);
    P(MultiArgClass);
    P(JoinedOrSeparateClass);
    P(JoinedAndSeparateClass);
    P(RemainingArgsClass);
    P(RemainingArgsJoinedClass);
#undef P
  }

  if (!Info->hasNoPrefix()) {
    O << " Prefixes:[";
    for (size_t I = 0, N = Info->getNumPrefixes(Owner->getPrefixesTable());
         I != N; ++I)
      O << '"'
        << Info->getPrefix(Owner->getStrTable(), Owner->getPrefixesTable(), I)
        << (I == N - 1 ? "\"" : "\", ");
    O << ']';
  }

  O << " Name:\"" << getName() << '"';

  const Option Group = getGroup();
  if (Group.isValid()) {
    O << " Group:";
    Group.print(O, /*AddNewLine=*/false);
  }

  const Option Alias = getAlias();
  if (Alias.isValid()) {
    O << " Alias:";
    Alias.print(O, /*AddNewLine=*/false);
  }

  if (getKind() == MultiArgClass)
    O << " NumArgs:" << getNumArgs();

  O << ">";
  if (AddNewLine)
    O << "\n";
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void Option::dump() const { print(dbgs()); }
#endif

bool Option::matches(OptSpecifier Opt) const {
  // Aliases are never considered in matching, look through them.
  const Option Alias = getAlias();
  if (Alias.isValid())
    return Alias.matches(Opt);

  // Check exact match.
  if (getID() == Opt.getID())
    return true;

  const Option Group = getGroup();
  if (Group.isValid())
    return Group.matches(Opt);
  return false;
}

/// Describes how to treat `nullptr` argument strings produced for
/// newlines and empty lines when advancing through the argument
/// list.
enum class NextArgStringBehavior {
  /// Skip over `nullptr` entries and return the next non-null
  /// argument string, if any.
  SkipEmpty,
  /// Treat a `nullptr` entry as a hard boundary and stop
  /// advancing. This is used for options that should not
  /// consume arguments past such boundaries (for example,
  /// RemainingArgs* options that must not cross @file
  /// segments).
  StopAtNull,
};

/// Advances to and returns the next argument string according to the given
/// behavior.
static const char *advanceToNextArgString(
    const ArgList &Args, unsigned &Index,
    NextArgStringBehavior Behavior = NextArgStringBehavior::SkipEmpty) {
  while (Index <= Args.getNumInputArgStrings()) {
    if (const char *ArgStr = Args.getArgString(Index - 1))
      return ArgStr;
    if (Behavior == NextArgStringBehavior::StopAtNull)
      return nullptr;
    ++Index;
  }
  return nullptr;
}

std::unique_ptr<Arg> Option::acceptInternal(const ArgList &Args,
                                            StringRef Spelling,
                                            unsigned &Index) const {
  const size_t SpellingSize = Spelling.size();
  const size_t ArgStringSize = StringRef(Args.getArgString(Index)).size();
  switch (getKind()) {
  case FlagClass: {
    if (SpellingSize != ArgStringSize)
      return nullptr;
    return std::make_unique<Arg>(*this, Spelling, Index++);
  }
  case JoinedClass: {
    const char *Value = Args.getArgString(Index) + SpellingSize;
    return std::make_unique<Arg>(*this, Spelling, Index++, Value);
  }
  case CommaJoinedClass: {
    // Always matches.
    const char *Str = Args.getArgString(Index) + SpellingSize;
    auto A = std::make_unique<Arg>(*this, Spelling, Index++);

    // Parse out the comma separated values.
    const char *Prev = Str;
    for (;; ++Str) {
      char c = *Str;

      if (!c || c == ',') {
        if (Prev != Str) {
          char *Value = new char[Str - Prev + 1];
          memcpy(Value, Prev, Str - Prev);
          Value[Str - Prev] = '\0';
          A->getValues().push_back(Value);
        }

        if (!c)
          break;

        Prev = Str + 1;
      }
    }
    A->setOwnsValues(true);

    return A;
  }
  case SeparateClass: {
    // Matches iff this is an exact match.
    if (SpellingSize != ArgStringSize)
      return nullptr;

    unsigned StartIndex = Index;
    Index += 2;

    const char *ArgString = advanceToNextArgString(Args, Index);
    if (!ArgString)
      return nullptr;

    return std::make_unique<Arg>(*this, Spelling, StartIndex, ArgString);
  }
  case MultiArgClass: {
    // Matches iff this is an exact match.
    if (SpellingSize != ArgStringSize)
      return nullptr;

    unsigned StartIndex = Index;
    Index += 2;

    SmallVector<const char *, 4> Values;
    for (unsigned i = 0; i < getNumArgs(); ++i) {
      const char *ArgString = advanceToNextArgString(Args, Index);
      if (!ArgString)
        return nullptr;
      Values.push_back(ArgString);
      if (i + 1 < getNumArgs())
        ++Index;
    }

    auto A = std::make_unique<Arg>(*this, Spelling, StartIndex, Values[0]);
    for (unsigned i = 1; i != getNumArgs(); ++i)
      A->getValues().push_back(Values[i]);
    return A;
  }
  case JoinedOrSeparateClass: {
    // If this is not an exact match, it is a joined arg.
    if (SpellingSize != ArgStringSize) {
      const char *Value = Args.getArgString(Index) + SpellingSize;
      return std::make_unique<Arg>(*this, Spelling, Index++, Value);
    }

    // Otherwise it must be separate.
    unsigned StartIndex = Index;
    Index += 2;

    const char *ArgString = advanceToNextArgString(Args, Index);
    if (!ArgString)
      return nullptr;

    return std::make_unique<Arg>(*this, Spelling, StartIndex, ArgString);
  }
  case JoinedAndSeparateClass: {
    // Always matches.
    unsigned StartIndex = Index;
    const char *JoinedValue = Args.getArgString(StartIndex) + SpellingSize;
    Index += 2;

    const char *SeparateValue = advanceToNextArgString(Args, Index);
    if (!SeparateValue)
      return nullptr;

    return std::make_unique<Arg>(*this, Spelling, StartIndex, JoinedValue,
                                 SeparateValue);
  }
  case RemainingArgsClass: {
    // Matches iff this is an exact match.
    if (SpellingSize != ArgStringSize)
      return nullptr;
    auto A = std::make_unique<Arg>(*this, Spelling, Index);
    Index += 2;
    while (const char *ArgString = advanceToNextArgString(
               Args, Index, NextArgStringBehavior::StopAtNull)) {
      A->getValues().push_back(ArgString);
      Index++;
    }
    return A;
  }
  case RemainingArgsJoinedClass: {
    auto A = std::make_unique<Arg>(*this, Spelling, Index);
    if (SpellingSize != ArgStringSize) {
      // An inexact match means there is a joined arg.
      A->getValues().push_back(Args.getArgString(Index) + SpellingSize);
    }
    Index += 2;
    while (const char *ArgString = advanceToNextArgString(
               Args, Index, NextArgStringBehavior::StopAtNull)) {
      A->getValues().push_back(ArgString);
      Index++;
    }
    return A;
  }

  default:
    llvm_unreachable("Invalid option kind!");
  }
}

std::unique_ptr<Arg> Option::accept(const ArgList &Args, StringRef CurArg,
                                    bool GroupedShortOption,
                                    unsigned &Index) const {
  auto A(GroupedShortOption && getKind() == FlagClass
                             ? std::make_unique<Arg>(*this, CurArg, Index)
                             : acceptInternal(Args, CurArg, Index));
  if (!A)
    return nullptr;

  const Option &UnaliasedOption = getUnaliasedOption();
  if (getID() == UnaliasedOption.getID())
    return A;

  // "A" is an alias for a different flag. For most clients it's more convenient
  // if this function returns unaliased Args, so create an unaliased arg for
  // returning.

  // This creates a completely new Arg object for the unaliased Arg because
  // the alias and the unaliased arg can have different Kinds and different
  // Values (due to AliasArgs<>).

  // Get the spelling from the unaliased option.
  StringRef UnaliasedSpelling = Args.MakeArgString(
      Twine(UnaliasedOption.getPrefix()) + Twine(UnaliasedOption.getName()));

  // It's a bit weird that aliased and unaliased arg share one index, but
  // the index is mostly use as a memory optimization in render().
  // Due to this, ArgList::getArgString(A->getIndex()) will return the spelling
  // of the aliased arg always, while A->getSpelling() returns either the
  // unaliased or the aliased arg, depending on which Arg object it's called on.
  auto UnaliasedA =
      std::make_unique<Arg>(UnaliasedOption, UnaliasedSpelling, A->getIndex());
  Arg *RawA = A.get();
  UnaliasedA->setAlias(std::move(A));

  if (getKind() != FlagClass) {
    // Values are usually owned by the ArgList. The exception are
    // CommaJoined flags, where the Arg owns the values. For aliased flags,
    // make the unaliased Arg the owner of the values.
    // FIXME: There aren't many uses of CommaJoined -- try removing
    // CommaJoined in favor of just calling StringRef::split(',') instead.
    UnaliasedA->getValues() = RawA->getValues();
    UnaliasedA->setOwnsValues(RawA->getOwnsValues());
    RawA->setOwnsValues(false);
    return UnaliasedA;
  }

  // FlagClass aliases can have AliasArgs<>; add those to the unaliased arg.
  if (const char *Val = getAliasArgs()) {
    while (*Val != '\0') {
      UnaliasedA->getValues().push_back(Val);

      // Move past the '\0' to the next argument.
      Val += strlen(Val) + 1;
    }
  }
  if (UnaliasedOption.getKind() == JoinedClass && !getAliasArgs())
    // A Flag alias for a Joined option must provide an argument.
    UnaliasedA->getValues().push_back("");
  return UnaliasedA;
}
