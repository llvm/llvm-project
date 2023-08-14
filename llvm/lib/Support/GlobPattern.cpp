//===-- GlobPattern.cpp - Glob pattern matcher implementation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a glob pattern matcher.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/GlobPattern.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"

using namespace llvm;

// Expands character ranges and returns a bitmap.
// For example, "a-cf-hz" is expanded to "abcfghz".
static Expected<BitVector> expand(StringRef S, StringRef Original) {
  BitVector BV(256, false);

  // Expand X-Y.
  for (;;) {
    if (S.size() < 3)
      break;

    uint8_t Start = S[0];
    uint8_t End = S[2];

    // If it doesn't start with something like X-Y,
    // consume the first character and proceed.
    if (S[1] != '-') {
      BV[Start] = true;
      S = S.substr(1);
      continue;
    }

    // It must be in the form of X-Y.
    // Validate it and then interpret the range.
    if (Start > End)
      return make_error<StringError>("invalid glob pattern: " + Original,
                                     errc::invalid_argument);

    for (int C = Start; C <= End; ++C)
      BV[(uint8_t)C] = true;
    S = S.substr(3);
  }

  for (char C : S)
    BV[(uint8_t)C] = true;
  return BV;
}

Expected<GlobPattern> GlobPattern::create(StringRef S) {
  GlobPattern Pat;

  // Store the prefix that does not contain any metacharacter.
  size_t PrefixSize = S.find_first_of("?*[\\");
  Pat.Prefix = S.substr(0, PrefixSize);
  if (PrefixSize == std::string::npos)
    return Pat;
  S = S.substr(PrefixSize);

  // Parse brackets.
  Pat.Pat = S;
  for (size_t I = 0, E = S.size(); I != E; ++I) {
    if (S[I] == '[') {
      // ']' is allowed as the first character of a character class. '[]' is
      // invalid. So, just skip the first character.
      ++I;
      size_t J = S.find(']', I + 1);
      if (J == StringRef::npos)
        return make_error<StringError>("invalid glob pattern, unmatched '['",
                                       errc::invalid_argument);
      StringRef Chars = S.substr(I, J - I);
      bool Invert = S[I] == '^' || S[I] == '!';
      Expected<BitVector> BV =
          Invert ? expand(Chars.substr(1), S) : expand(Chars, S);
      if (!BV)
        return BV.takeError();
      if (Invert)
        BV->flip();
      Pat.Brackets.push_back(Bracket{S.data() + J + 1, std::move(*BV)});
      I = J;
    } else if (S[I] == '\\') {
      if (++I == E)
        return make_error<StringError>("invalid glob pattern, stray '\\'",
                                       errc::invalid_argument);
    }
  }
  return Pat;
}

bool GlobPattern::match(StringRef S) const {
  return S.consume_front(Prefix) && matchOne(S);
}

// Factor the pattern into segments split by '*'. The segment is matched
// sequentianlly by finding the first occurrence past the end of the previous
// match.
bool GlobPattern::matchOne(StringRef Str) const {
  const char *P = Pat.data(), *SegmentBegin = nullptr, *S = Str.data(),
             *SavedS = S;
  const char *const PEnd = P + Pat.size(), *const End = S + Str.size();
  size_t B = 0, SavedB = 0;
  while (S != End) {
    if (P == PEnd)
      ;
    else if (*P == '*') {
      // The non-* substring on the left of '*' matches the tail of S. Save the
      // positions to be used by backtracking if we see a mismatch later.
      SegmentBegin = ++P;
      SavedS = S;
      SavedB = B;
      continue;
    } else if (*P == '[') {
      if (Brackets[B].Bytes[uint8_t(*S)]) {
        P = Brackets[B++].Next;
        ++S;
        continue;
      }
    } else if (*P == '\\') {
      if (*++P == *S) {
        ++P;
        ++S;
        continue;
      }
    } else if (*P == *S || *P == '?') {
      ++P;
      ++S;
      continue;
    }
    if (!SegmentBegin)
      return false;
    // We have seen a '*'. Backtrack to the saved positions. Shift the S
    // position to probe the next starting position in the segment.
    P = SegmentBegin;
    S = ++SavedS;
    B = SavedB;
  }
  // All bytes in Str have been matched. Return true if the rest part of Pat is
  // empty or contains only '*'.
  return Pat.find_first_not_of('*', P - Pat.data()) == std::string::npos;
}
