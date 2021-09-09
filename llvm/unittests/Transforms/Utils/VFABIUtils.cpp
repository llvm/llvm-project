//===------- VFABIUtils.cpp - VFABI Unittests  ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "gtest/gtest.h"

using namespace llvm;

class VFABIAttrTest : public testing::Test {
protected:
  void SetUp() override {
    M = parseAssemblyString(IR, Err, Ctx);
    // Get the only call instruction in the block, which is the first
    // instruction.
    CI = dyn_cast<CallInst>(&*(instructions(M->getFunction("f")).begin()));
  }
  const char *IR = "define i32 @f(i32 %a) {\n"
                   " %1 = call i32 @g(i32 %a) #0\n"
                   "  ret i32 %1\n"
                   "}\n"
                   "declare i32 @g(i32)\n"
                   "declare <2 x i32> @custom_vg(<2 x i32>)"
                   "declare <4 x i32> @_ZGVnN4v_g(<4 x i32>)"
                   "declare <8 x i32> @_ZGVnN8v_g(<8 x i32>)"
                   "attributes #0 = { "
                   "\"vector-function-abi-variant\"=\""
                   "_ZGVnN2v_g(custom_vg),_ZGVnN4v_g\" }";
  LLVMContext Ctx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M;
  CallInst *CI;
  SmallVector<std::string, 8> Mappings;
};

TEST_F(VFABIAttrTest, Write) {
  Mappings.push_back("_ZGVnN8v_g");
  Mappings.push_back("_ZGVnN2v_g(custom_vg)");
  VFABI::setVectorVariantNames(CI, Mappings);
  const StringRef S =
      CI->getFnAttr("vector-function-abi-variant").getValueAsString();
  EXPECT_EQ(S, "_ZGVnN8v_g,_ZGVnN2v_g(custom_vg)");
}
