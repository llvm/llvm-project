//===---- MangleAndInternerTest.cpp - Unit tests for MangleAndInterner ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/Support/Error.h"

#include "OrcTestCommon.h"

using namespace llvm;
using namespace llvm::orc;

namespace {

ExecutionSession makeES(StringRef TT) {
  return ExecutionSession(std::make_unique<UnsupportedExecutorProcessControl>(
      nullptr, nullptr, TT.str()));
}

struct ManglingCase {
  StringRef Triple;
  StringRef Input;
  StringRef Expected;
};

} // namespace

TEST(MangleAndInternerTest, FromTripleAcrossFormats) {
  static const ManglingCase Cases[] = {
      // ELF: no prefix.
      {"x86_64-unknown-linux-gnu", "foo", "foo"},
      // MachO: leading underscore.
      {"x86_64-apple-darwin", "foo", "_foo"},
      {"arm64-apple-darwin", "foo", "_foo"},
      // Windows COFF, x86_64: no prefix.
      {"x86_64-pc-windows-msvc", "foo", "foo"},
      // Windows COFF, x86 (32-bit): leading underscore.
      {"i686-pc-windows-msvc", "foo", "_foo"},
      // AIX XCOFF: no prefix.
      {"powerpc64-ibm-aix", "foo", "foo"},
      // z/OS GOFF: no prefix.
      {"s390x-ibm-zos", "foo", "foo"},
      // MIPS O32: no prefix.
      {"mipsel-unknown-linux-gnu", "foo", "foo"},
  };

  for (const auto &C : Cases) {
    SCOPED_TRACE(C.Triple);
    ExecutionSession ES = makeES(C.Triple);
    MangleAndInterner Mangle(ES);
    EXPECT_EQ(*Mangle(C.Input), C.Expected);
    cantFail(ES.endSession());
  }
}

TEST(MangleAndInternerTest, DoNotMangleLeadingBackslash1) {
  ExecutionSession ES = makeES("x86_64-apple-darwin");
  MangleAndInterner Mangle(ES);
  EXPECT_EQ(*Mangle("\1foo"), "foo");
  cantFail(ES.endSession());
}

TEST(MangleAndInternerTest, WindowsQuestionMarkNotMangled) {
  ExecutionSession ES = makeES("x86_64-pc-windows-msvc");
  MangleAndInterner Mangle(ES);
  EXPECT_EQ(*Mangle("?foo@@YAHXZ"), "?foo@@YAHXZ");
  cantFail(ES.endSession());
}

TEST(MangleAndInternerTest, MachOQuestionMarkIsMangled) {
  // MachO has no question-mark suppression: gets the usual '_' prefix.
  ExecutionSession ES = makeES("x86_64-apple-darwin");
  MangleAndInterner Mangle(ES);
  EXPECT_EQ(*Mangle("?foo"), "_?foo");
  cantFail(ES.endSession());
}

TEST(MangleAndInternerTest, ExplicitManglingMode) {
  ExecutionSession ES = makeES("x86_64-unknown-linux-gnu");
  MangleAndInterner Mangle(ES, MangleAndInterner::ManglingMode::MachO);
  EXPECT_EQ(*Mangle("foo"), "_foo");
  cantFail(ES.endSession());
}
