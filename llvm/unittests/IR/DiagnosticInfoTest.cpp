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

} // end anonymous namespace
