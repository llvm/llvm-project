//===- PassBuilderTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/NullPass.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/SandboxVectorizerPassBuilder.h"
#include "gtest/gtest.h"

using namespace llvm::sandboxir;

// Check that the PassBuilder passes the AuxArg to the RegionPass upon
// construction.
TEST(PassBuilderTest, RegionPassAuxArg) {
  SandboxVectorizerPassBuilder Builder;
  std::string AuxArgStr("aux-arg-test");
  auto RgnPassPtr =
      Builder.createRegionPass("null", /*Args=*/"", /*AuxArg=*/AuxArgStr);
  NullPass *NPass = static_cast<NullPass *>(RgnPassPtr.get());
  EXPECT_EQ(NPass->getAuxArg(), AuxArgStr);
}
