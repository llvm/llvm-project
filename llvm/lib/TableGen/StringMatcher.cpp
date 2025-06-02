//===- StringMatcher.cpp - Generate a matcher for input strings -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the StringMatcher class.
//
//===----------------------------------------------------------------------===//

#include "llvm/TableGen/StringMatcher.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <map>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

/// FindFirstNonCommonLetter - Find the first character in the keys of the
/// string pairs that is not shared across the whole set of strings.  All
/// strings are assumed to have the same length.
static unsigned
FindFirstNonCommonLetter(ArrayRef<const StringMatcher::StringPair *> Matches) {
  assert(!Matches.empty());
  for (auto [Idx, Letter] : enumerate(Matches[0]->first)) {
    // Check to see if `Letter` is the same across the set.
    for (const StringMatcher::StringPair *Match : Matches)
      if (Match->first[Idx] != Letter)
        return Idx;
  }

  return Matches[0]->first.size();
}

/// EmitStringMatcherForChar - Given a set of strings that are known to be the
/// same length and whose characters leading up to CharNo are the same, emit
/// code to verify that CharNo and later are the same.
///
/// \return - True if control can leave the emitted code fragment.
bool StringMatcher::EmitStringMatcherForChar(
    ArrayRef<const StringPair *> Matches, unsigned CharNo, unsigned IndentCount,
    bool IgnoreDuplicates) const {
  assert(!Matches.empty() && "Must have at least one string to match!");
  std::string Indent(IndentCount * 2 + 4, ' ');

  // If we have verified that the entire string matches, we're done: output the
  // matching code.
  if (CharNo == Matches[0]->first.size()) {
    if (Matches.size() > 1 && !IgnoreDuplicates)
      report_fatal_error("Had duplicate keys to match on");

    // If the to-execute code has \n's in it, indent each subsequent line.
    StringRef Code = Matches[0]->second;

    std::pair<StringRef, StringRef> Split = Code.split('\n');
    OS << Indent << Split.first << "\t // \"" << Matches[0]->first << "\"\n";

    Code = Split.second;
    while (!Code.empty()) {
      Split = Code.split('\n');
      OS << Indent << Split.first << "\n";
      Code = Split.second;
    }
    return false;
  }

  // Bucket the matches by the character we are comparing.
  std::map<char, std::vector<const StringPair*>> MatchesByLetter;

  for (const StringPair *Match : Matches)
    MatchesByLetter[Match->first[CharNo]].push_back(Match);

  // If we have exactly one bucket to match, see how many characters are common
  // across the whole set and match all of them at once.
  if (MatchesByLetter.size() == 1) {
    unsigned FirstNonCommonLetter = FindFirstNonCommonLetter(Matches);
    unsigned NumChars = FirstNonCommonLetter-CharNo;

    // Emit code to break out if the prefix doesn't match.
    if (NumChars == 1) {
      // Do the comparison with if (Str[1] != 'f')
      // FIXME: Need to escape general characters.
      OS << Indent << "if (" << StrVariableName << "[" << CharNo << "] != '"
      << Matches[0]->first[CharNo] << "')\n";
      OS << Indent << "  break;\n";
    } else {
      // Do the comparison with if memcmp(Str.data()+1, "foo", 3).
      // FIXME: Need to escape general strings.
      OS << Indent << "if (memcmp(" << StrVariableName << ".data()+" << CharNo
         << ", \"" << Matches[0]->first.substr(CharNo, NumChars) << "\", "
         << NumChars << ") != 0)\n";
      OS << Indent << "  break;\n";
    }

    return EmitStringMatcherForChar(Matches, FirstNonCommonLetter, IndentCount,
                                    IgnoreDuplicates);
  }

  // Otherwise, we have multiple possible things, emit a switch on the
  // character.
  OS << Indent << "switch (" << StrVariableName << "[" << CharNo << "]) {\n";
  OS << Indent << "default: break;\n";

  for (const auto &[Letter, Matches] : MatchesByLetter) {
    // TODO: escape hard stuff (like \n) if we ever care about it.
    OS << Indent << "case '" << Letter << "':\t // " << Matches.size()
       << " string";
    if (Matches.size() != 1)
      OS << 's';
    OS << " to match.\n";
    if (EmitStringMatcherForChar(Matches, CharNo + 1, IndentCount + 1,
                                 IgnoreDuplicates))
      OS << Indent << "  break;\n";
  }

  OS << Indent << "}\n";
  return true;
}

/// Emit - Top level entry point.
///
void StringMatcher::Emit(unsigned Indent, bool IgnoreDuplicates) const {
  // If nothing to match, just fall through.
  if (Matches.empty()) return;

  // First level categorization: group strings by length.
  std::map<unsigned, std::vector<const StringPair*>> MatchesByLength;

  for (const StringPair &Match : Matches)
    MatchesByLength[Match.first.size()].push_back(&Match);

  // Output a switch statement on length and categorize the elements within each
  // bin.
  OS.indent(Indent*2+2) << "switch (" << StrVariableName << ".size()) {\n";
  OS.indent(Indent*2+2) << "default: break;\n";

  for (const auto &[Length, Matches] : MatchesByLength) {
    OS.indent(Indent * 2 + 2)
        << "case " << Length << ":\t // " << Matches.size() << " string"
        << (Matches.size() == 1 ? "" : "s") << " to match.\n";
    if (EmitStringMatcherForChar(Matches, 0, Indent, IgnoreDuplicates))
      OS.indent(Indent*2+4) << "break;\n";
  }

  OS.indent(Indent*2+2) << "}\n";
}
