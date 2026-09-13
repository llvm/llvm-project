//===- DiagnosticInfoTest.cpp - DiagnosticInfo unit tests -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(DiagnosticInfoTest, DebugMetadataKindsMatchClassof) {
  LLVMContext C;
  Module M("M", C);

  DiagnosticInfoDebugMetadataVersion Version(M, 1);
  const DiagnosticInfo *VersionInfo = &Version;
  EXPECT_EQ(DK_DebugMetadataVersion, VersionInfo->getKind());
  EXPECT_TRUE(isa<DiagnosticInfoDebugMetadataVersion>(VersionInfo));
  EXPECT_FALSE(isa<DiagnosticInfoIgnoringInvalidDebugMetadata>(VersionInfo));

  DiagnosticInfoIgnoringInvalidDebugMetadata Invalid(M);
  const DiagnosticInfo *InvalidInfo = &Invalid;
  EXPECT_EQ(DK_DebugMetadataInvalid, InvalidInfo->getKind());
  EXPECT_TRUE(isa<DiagnosticInfoIgnoringInvalidDebugMetadata>(InvalidInfo));
  EXPECT_FALSE(isa<DiagnosticInfoDebugMetadataVersion>(InvalidInfo));
}

// A plugin diagnostic can override getWarningGroup() so the frontend routes it
// into a user-controllable warning group. The base default is empty, and the
// override is visible through a DiagnosticInfo base reference -- the path the
// clang backend bridge takes.
TEST(DiagnosticInfoTest, WarningGroup) {
  LLVMContext C;
  Module M("M", C);

  DiagnosticInfoDebugMetadataVersion Version(M, 1);
  EXPECT_TRUE(
      static_cast<const DiagnosticInfo &>(Version).getWarningGroup().empty());

  class GroupedDiag : public DiagnosticInfo {
  public:
    GroupedDiag()
        : DiagnosticInfo(getNextAvailablePluginDiagnosticKind(), DS_Warning) {}
    void print(DiagnosticPrinter &) const override {}
    StringRef getWarningGroup() const override { return "example-plugin"; }
  };

  GroupedDiag Grouped;
  const DiagnosticInfo &Base = Grouped;
  EXPECT_EQ(Base.getWarningGroup(), "example-plugin");
}

} // end anonymous namespace
