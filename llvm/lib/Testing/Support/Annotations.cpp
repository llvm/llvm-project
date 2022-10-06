//===--- Annotations.cpp - Annotated source code for unit tests --*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Testing/Support/Annotations.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

// Crash if the assertion fails, printing the message and testcase.
// More elegant error handling isn't needed for unit tests.
static void require(bool Assertion, const char *Msg, llvm::StringRef Code) {
  if (!Assertion) {
    llvm::errs() << "Annotated testcase: " << Msg << "\n" << Code << "\n";
    llvm_unreachable("Annotated testcase assertion failed!");
  }
}

Annotations::Annotations(llvm::StringRef Text) {
  auto Require = [Text](bool Assertion, const char *Msg) {
    require(Assertion, Msg, Text);
  };
  llvm::Optional<llvm::StringRef> Name;
  llvm::SmallVector<std::pair<llvm::StringRef, size_t>, 8> OpenRanges;

  Code.reserve(Text.size());
  while (!Text.empty()) {
    if (Text.consume_front("^")) {
      Points[Name.value_or("")].push_back(Code.size());
      Name = llvm::None;
      continue;
    }
    if (Text.consume_front("[[")) {
      OpenRanges.emplace_back(Name.value_or(""), Code.size());
      Name = llvm::None;
      continue;
    }
    Require(!Name, "$name should be followed by ^ or [[");
    if (Text.consume_front("]]")) {
      Require(!OpenRanges.empty(), "unmatched ]]");
      Range R;
      R.Begin = OpenRanges.back().second;
      R.End = Code.size();
      Ranges[OpenRanges.back().first].push_back(R);
      OpenRanges.pop_back();
      continue;
    }
    if (Text.consume_front("$")) {
      Name =
          Text.take_while([](char C) { return llvm::isAlnum(C) || C == '_'; });
      Text = Text.drop_front(Name->size());
      continue;
    }
    Code.push_back(Text.front());
    Text = Text.drop_front();
  }
  Require(!Name, "unterminated $name");
  Require(OpenRanges.empty(), "unmatched [[");
}

size_t Annotations::point(llvm::StringRef Name) const {
  auto I = Points.find(Name);
  require(I != Points.end() && I->getValue().size() == 1,
          "expected exactly one point", Code);
  return I->getValue()[0];
}

std::vector<size_t> Annotations::points(llvm::StringRef Name) const {
  auto I = Points.find(Name);
  if (I == Points.end())
    return {};
  return {I->getValue().begin(), I->getValue().end()};
}

const llvm::StringMap<llvm::SmallVector<size_t, 1>> &
Annotations::all_points() const {
  return Points;
}

Annotations::Range Annotations::range(llvm::StringRef Name) const {
  auto I = Ranges.find(Name);
  require(I != Ranges.end() && I->getValue().size() == 1,
          "expected exactly one range", Code);
  return I->getValue()[0];
}

std::vector<Annotations::Range>
Annotations::ranges(llvm::StringRef Name) const {
  auto I = Ranges.find(Name);
  if (I == Ranges.end())
    return {};
  return {I->getValue().begin(), I->getValue().end()};
}

const llvm::StringMap<llvm::SmallVector<Annotations::Range, 1>> &
Annotations::all_ranges() const {
  return Ranges;
}

llvm::raw_ostream &llvm::operator<<(llvm::raw_ostream &O,
                                    const llvm::Annotations::Range &R) {
  return O << llvm::formatv("[{0}, {1})", R.Begin, R.End);
}
