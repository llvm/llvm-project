//===- FormatVariadic.cpp - Format string parsing and analysis ----*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "llvm/Support/FormatVariadic.h"
#include <cassert>
#include <optional>

using namespace llvm;

static std::optional<AlignStyle> translateLocChar(char C) {
  switch (C) {
  case '-':
    return AlignStyle::Left;
  case '=':
    return AlignStyle::Center;
  case '+':
    return AlignStyle::Right;
  default:
    return std::nullopt;
  }
  LLVM_BUILTIN_UNREACHABLE;
}

static bool consumeFieldLayout(StringRef &Spec, AlignStyle &Where,
                               size_t &Align, char &Pad) {
  Where = AlignStyle::Right;
  Align = 0;
  Pad = ' ';
  if (Spec.empty())
    return true;

  if (Spec.size() > 1) {
    // A maximum of 2 characters at the beginning can be used for something
    // other than the width.
    // If Spec[1] is a loc char, then Spec[0] is a pad char and Spec[2:...]
    // contains the width.
    // Otherwise, if Spec[0] is a loc char, then Spec[1:...] contains the width.
    // Otherwise, Spec[0:...] contains the width.
    if (auto Loc = translateLocChar(Spec[1])) {
      Pad = Spec[0];
      Where = *Loc;
      Spec = Spec.drop_front(2);
    } else if (auto Loc = translateLocChar(Spec[0])) {
      Where = *Loc;
      Spec = Spec.drop_front(1);
    }
  }

  bool Failed = Spec.consumeInteger(0, Align);
  return !Failed;
}

static std::optional<ReplacementItem> parseReplacementItem(StringRef Spec) {
  StringRef RepString = Spec.trim("{}");

  // If the replacement sequence does not start with a non-negative integer,
  // this is an error.
  char Pad = ' ';
  std::size_t Align = 0;
  AlignStyle Where = AlignStyle::Right;
  StringRef Options;
  size_t Index = 0;
  RepString = RepString.trim();
  if (RepString.consumeInteger(0, Index)) {
    assert(false && "Invalid replacement sequence index!");
    return ReplacementItem{};
  }
  RepString = RepString.trim();
  if (RepString.consume_front(",")) {
    if (!consumeFieldLayout(RepString, Where, Align, Pad))
      assert(false && "Invalid replacement field layout specification!");
  }
  RepString = RepString.trim();
  if (RepString.consume_front(":")) {
    Options = RepString.trim();
    RepString = StringRef();
  }
  RepString = RepString.trim();
  assert(RepString.empty() &&
         "Unexpected characters found in replacement string!");

  return ReplacementItem{Spec, Index, Align, Where, Pad, Options};
}

static std::pair<ReplacementItem, StringRef>
splitLiteralAndReplacement(StringRef Fmt) {
  while (!Fmt.empty()) {
    // Everything up until the first brace is a literal.
    if (Fmt.front() != '{') {
      std::size_t BO = Fmt.find_first_of('{');
      return std::make_pair(ReplacementItem{Fmt.substr(0, BO)}, Fmt.substr(BO));
    }

    StringRef Braces = Fmt.take_while([](char C) { return C == '{'; });
    // If there is more than one brace, then some of them are escaped.  Treat
    // these as replacements.
    if (Braces.size() > 1) {
      size_t NumEscapedBraces = Braces.size() / 2;
      StringRef Middle = Fmt.take_front(NumEscapedBraces);
      StringRef Right = Fmt.drop_front(NumEscapedBraces * 2);
      return std::make_pair(ReplacementItem{Middle}, Right);
    }
    // An unterminated open brace is undefined. Assert to indicate that this is
    // undefined and that we consider it an error. When asserts are disabled,
    // build a replacement item with an error message.
    std::size_t BC = Fmt.find_first_of('}');
    if (BC == StringRef::npos) {
      assert(
          false &&
          "Unterminated brace sequence. Escape with {{ for a literal brace.");
      return std::make_pair(
          ReplacementItem{"Unterminated brace sequence. Escape with {{ for a "
                          "literal brace."},
          StringRef());
    }

    // Even if there is a closing brace, if there is another open brace before
    // this closing brace, treat this portion as literal, and try again with the
    // next one.
    std::size_t BO2 = Fmt.find_first_of('{', 1);
    if (BO2 < BC)
      return std::make_pair(ReplacementItem{Fmt.substr(0, BO2)},
                            Fmt.substr(BO2));

    StringRef Spec = Fmt.slice(1, BC);
    StringRef Right = Fmt.substr(BC + 1);

    auto RI = parseReplacementItem(Spec);
    if (RI)
      return std::make_pair(*RI, Right);

    // If there was an error parsing the replacement item, treat it as an
    // invalid replacement spec, and just continue.
    Fmt = Fmt.drop_front(BC + 1);
  }
  return std::make_pair(ReplacementItem{Fmt}, StringRef());
}

#ifndef NDEBUG
#define ENABLE_VALIDATION 1
#else
#define ENABLE_VALIDATION 0 // Conveniently enable validation in release mode.
#endif

SmallVector<ReplacementItem, 2>
formatv_object_base::parseFormatString(StringRef Fmt, size_t NumArgs,
                                       bool Validate) {
  SmallVector<ReplacementItem, 2> Replacements;

#if ENABLE_VALIDATION
  const StringRef SavedFmtStr = Fmt;
  size_t NumExpectedArgs = 0;
#endif

  while (!Fmt.empty()) {
    ReplacementItem I;
    std::tie(I, Fmt) = splitLiteralAndReplacement(Fmt);
    if (I.Type != ReplacementType::Empty)
      Replacements.push_back(I);
#if ENABLE_VALIDATION
    if (I.Type == ReplacementType::Format)
      NumExpectedArgs = std::max(NumExpectedArgs, I.Index + 1);
#endif
  }

#if ENABLE_VALIDATION
  if (!Validate)
    return Replacements;

  // Perform additional validation. Verify that the number of arguments matches
  // the number of replacement indices and that there are no holes in the
  // replacement indices.

  // When validation fails, return an array of replacement items that
  // will print an error message as the outout of this formatv() (used when
  // validation is enabled in release mode).
  auto getErrorReplacements = [SavedFmtStr](StringLiteral ErrorMsg) {
    return SmallVector<ReplacementItem, 2>{
        ReplacementItem("Invalid formatv() call: "), ReplacementItem(ErrorMsg),
        ReplacementItem(" for format string: "), ReplacementItem(SavedFmtStr)};
  };

  if (NumExpectedArgs != NumArgs) {
    errs() << formatv(
        "Expected {0} Args, but got {1} for format string '{2}'\n",
        NumExpectedArgs, NumArgs, SavedFmtStr);
    assert(0 && "Invalid formatv() call");
    return getErrorReplacements("Unexpected number of arguments");
  }

  // Find the number of unique indices seen. All replacement indices
  // are < NumExpectedArgs.
  SmallVector<bool> Indices(NumExpectedArgs);
  size_t Count = 0;
  for (const ReplacementItem &I : Replacements) {
    if (I.Type != ReplacementType::Format || Indices[I.Index])
      continue;
    Indices[I.Index] = true;
    ++Count;
  }

  if (Count != NumExpectedArgs) {
    errs() << formatv(
        "Replacement field indices cannot have holes for format string '{0}'\n",
        SavedFmtStr);
    assert(0 && "Invalid format string");
    return getErrorReplacements("Replacement indices have holes");
  }
#endif // ENABLE_VALIDATION
  return Replacements;
}

void support::detail::format_adapter::anchor() {}
