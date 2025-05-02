//===-------- HLSLRootSignatureDumpTest.cpp - RootSignature dump tests ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/HLSL/HLSLRootSignature.h"
#include "gtest/gtest.h"

using namespace llvm::hlsl::rootsig;

namespace {

TEST(HLSLRootSignatureTest, DescriptorTablesDump) {
  // Default clause
  DescriptorTableClause Clause;
  Clause.Type = ClauseType::CBuffer;
  Clause.Reg = { RegisterType::BReg, 0 };

  std::string Out;
  llvm::raw_string_ostream OS(Out);
  Clause.dump(OS);
  OS.flush();

  EXPECT_EQ(Out, "Clause!");
}

} // namespace
