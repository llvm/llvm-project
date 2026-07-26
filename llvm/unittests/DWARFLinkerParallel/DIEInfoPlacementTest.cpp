//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DWARFLinkerCompileUnit.h"
#include "gtest/gtest.h"
#include <thread>

using namespace llvm;
using namespace llvm::dwarf_linker::parallel;

namespace {

using DIEInfo = CompileUnit::DIEInfo;

// joinPlacement is the least-upper-bound over the lattice
// NotSet < {TypeTable, PlainDwarf} < Both. It only ever raises the placement,
// so no interleaving of marks can demote a DIE.
TEST(DIEInfoPlacement, JoinIsMonotoneLatticeLUB) {
  DIEInfo Info;
  EXPECT_EQ(Info.getPlacement(), CompileUnit::NotSet);

  Info.joinPlacement(CompileUnit::TypeTable);
  EXPECT_EQ(Info.getPlacement(), CompileUnit::TypeTable);

  // The other mark raises TypeTable to Both.
  Info.joinPlacement(CompileUnit::PlainDwarf);
  EXPECT_EQ(Info.getPlacement(), CompileUnit::Both);

  // Re-applying either mark never demotes.
  Info.joinPlacement(CompileUnit::TypeTable);
  Info.joinPlacement(CompileUnit::PlainDwarf);
  EXPECT_EQ(Info.getPlacement(), CompileUnit::Both);
}

// joinPlacement leaves the non-placement flag bits untouched.
TEST(DIEInfoPlacement, JoinPreservesOtherFlags) {
  DIEInfo Info;
  Info.setKeep();
  Info.setODRAvailable();
  Info.joinPlacement(CompileUnit::TypeTable);
  Info.joinPlacement(CompileUnit::PlainDwarf);
  EXPECT_EQ(Info.getPlacement(), CompileUnit::Both);
  EXPECT_TRUE(Info.getKeep());
  EXPECT_TRUE(Info.getODRAvailable());
}

// For a DW_TAG_variable, PlainDwarf is absorbing: a variable cannot occupy the
// type table and plain DWARF at once, so joining a TypeTable mark into a
// PlainDwarf placement keeps PlainDwarf rather than producing Both.
TEST(DIEInfoPlacement, VariableJoinKeepsPlainDwarfAbsorbing) {
  // TypeTable-then-PlainDwarf: the plain mark forces PlainDwarf (handled by the
  // overwrite path), and a subsequent type mark must not resurrect TypeTable.
  {
    DIEInfo Info;
    Info.joinVariablePlacement(CompileUnit::TypeTable);
    EXPECT_EQ(Info.getPlacement(), CompileUnit::TypeTable);
    Info.setPlacement(CompileUnit::PlainDwarf);
    Info.joinVariablePlacement(CompileUnit::TypeTable);
    EXPECT_EQ(Info.getPlacement(), CompileUnit::PlainDwarf);
  }

  // A stale Both is collapsed back to PlainDwarf, never left as Both.
  {
    DIEInfo Info;
    Info.setPlacement(CompileUnit::Both);
    Info.joinVariablePlacement(CompileUnit::TypeTable);
    EXPECT_EQ(Info.getPlacement(), CompileUnit::PlainDwarf);
  }

  // Only a TypeTable mark ever reaches an unset variable: it lands in
  // TypeTable.
  {
    DIEInfo Info;
    Info.joinVariablePlacement(CompileUnit::TypeTable);
    EXPECT_EQ(Info.getPlacement(), CompileUnit::TypeTable);
  }
}

// Concurrently applying a PlainDwarf overwrite and a TypeTable variable-join to
// one DIEInfo must never leave it in Both, regardless of interleaving. A plain
// joinPlacement (OR) would produce Both here. joinVariablePlacement recomputes
// on each attempt so PlainDwarf stays absorbing under the race.
TEST(DIEInfoPlacement, VariableJoinRaceNeverProducesBoth) {
  for (unsigned Trial = 0; Trial < 2000; ++Trial) {
    DIEInfo Info;
    std::thread Plain([&] { Info.setPlacement(CompileUnit::PlainDwarf); });
    std::thread Type(
        [&] { Info.joinVariablePlacement(CompileUnit::TypeTable); });
    Plain.join();
    Type.join();
    // The final placement is one of the single values, never Both.
    CompileUnit::DieOutputPlacement P = Info.getPlacement();
    EXPECT_TRUE(P == CompileUnit::PlainDwarf || P == CompileUnit::TypeTable)
        << "placement was " << static_cast<unsigned>(P) << " on trial "
        << Trial;
  }
}

} // namespace
