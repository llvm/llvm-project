//===- GlobalISelMatchTableTest.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Common/GlobalISel/MatchTable/MatchTable.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::gi;

TEST(GlobalISelMatchTableTest, CompactFailureTargets) {
  MatchTable Table(/*WithCoverage=*/false, /*IsCombinerTable=*/false);

  // Cross the production compaction threshold before adding representative
  // match records. Keeping the padding before the jumps does not affect their
  // relative distances.
  for (unsigned I = 0; I != 8192; ++I)
    Table << MatchTable::IntValue(8, 0);

  unsigned FarLabel = Table.allocateLabelID();
  Table << MatchTable::Opcode("GIM_Try") << MatchTable::JumpTarget(FarLabel);
  for (unsigned I = 0; I != 300; ++I)
    Table << MatchTable::IntValue(1, 0);
  Table << MatchTable::Label(FarLabel);

  unsigned NearLabel = Table.allocateLabelID();
  Table << MatchTable::Opcode("GIM_Try_CheckFeatures")
        << MatchTable::JumpTarget(NearLabel)
        << MatchTable::NamedValue(2, "FeatureBitset")
        << MatchTable::IntValue(1, 0) << MatchTable::Label(NearLabel);

  Table.compactFailureTargets();

  std::string Output;
  raw_string_ostream OS(Output);
  Table.emitDeclaration(OS);

  EXPECT_NE(Output.find("GIM_Try16, /*Label 0*/ GIMT_Encode2(300)"),
            std::string::npos);
  EXPECT_NE(Output.find("GIM_Try_CheckFeatures8, /*Label 1*/ 3"),
            std::string::npos);
}
