//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace llvm;

// Prints a ret instruction in C, which -print-inst-addrs suffixes with its
// address.
static std::string printRet(LLVMContext &C) {
  std::unique_ptr<ReturnInst> Ret(ReturnInst::Create(C));
  std::string S;
  raw_string_ostream(S) << *Ret;
  return S;
}

// parseCommandLineOptions sets IR's options in the given context only, so two
// contexts in one process print instructions differently.
TEST(LibraryOptionsContextTest, IR) {
  LLVMContext C1, C2;
  const char *Args[] = {"prog", "-print-inst-addrs"};
  ASSERT_TRUE(
      cl::parseCommandLineOptions(C2.getLibraryOptions(), Args, "", &nulls()));
  EXPECT_EQ(printRet(C1), "  ret void");
  EXPECT_NE(printRet(C2).find("ret void ; 0x"), std::string::npos);
}
