//===-------------- ManglerTest.cpp - Unit tests for Mangler --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Shared/Mangler.h"
#include "llvm/ExecutionEngine/Orc/Shared/SymbolNameSpec.h"

#include "OrcTestCommon.h"

using namespace llvm;
using namespace llvm::orc;

namespace {

struct ManglingCase {
  StringRef Triple;
  StringRef Input;
  StringRef Expected;
};

} // namespace

TEST(ManglerTest, FromTripleAcrossFormats) {
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
    Mangler Mangle(Triple(C.Triple));
    EXPECT_EQ(Mangle.mangledCopy(SymbolNameSpec::ir(C.Input)), C.Expected);
  }
}

TEST(ManglerTest, DoNotMangleLeadingBackslash1) {
  Mangler Mangle(Triple("x86_64-apple-darwin"));
  EXPECT_EQ(Mangle.mangledCopy(SymbolNameSpec::ir("\1foo")), "foo");
}

TEST(ManglerTest, WindowsQuestionMarkNotMangled) {
  Mangler Mangle(Triple("x86_64-pc-windows-msvc"));
  EXPECT_EQ(Mangle.mangledCopy(SymbolNameSpec::ir("?foo@@YAHXZ")),
            "?foo@@YAHXZ");
}

TEST(ManglerTest, MachOQuestionMarkIsMangled) {
  // MachO has no question-mark suppression: gets the usual '_' prefix.
  Mangler Mangle(Triple("x86_64-apple-darwin"));
  EXPECT_EQ(Mangle.mangledCopy(SymbolNameSpec::ir("?foo")), "_?foo");
}

TEST(ManglerTest, ExplicitManglingMode) {
  Mangler Mangle(Mangler::Mode::MachO);
  EXPECT_EQ(Mangle.mangledCopy(SymbolNameSpec::ir("foo")), "_foo");
}

// Use a MachO triple so that mangling ("_" prefix) is observably different from
// interning verbatim.
TEST(ManglerTest, SymbolNameSpecKindDispatch) {
  Mangler Mangle(Triple("x86_64-apple-darwin"));

  // Verbatim and Linker are interned unmodified.
  EXPECT_EQ(Mangle.mangledCopy(SymbolNameSpec::verbatim("foo")), "foo");
  EXPECT_EQ(Mangle.mangledCopy(SymbolNameSpec::linker("foo")), "foo");

  // IR and C are mangled (leading "_" on MachO).
  EXPECT_EQ(Mangle.mangledCopy(SymbolNameSpec::ir("foo")), "_foo");
  EXPECT_EQ(Mangle.mangledCopy(SymbolNameSpec::c("foo")), "_foo");
}
