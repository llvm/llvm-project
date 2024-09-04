//===- llvm/unittest/CodeGen/SPIRVAPITest.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Test that SPIR-V Backend provides an API call that translates LLVM IR Module
/// into SPIR-V.
//
//===----------------------------------------------------------------------===//

// #include "llvm/IR/LegacyPassManager.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/IR/Module.h"
// #include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Support/SourceMgr.h"
// #include "llvm/Support/TargetSelect.h"
// #include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"
#include <string>
#include <utility>

namespace llvm {

extern "C" bool SPIRVTranslateModule(Module *M, std::string &Buffer,
                                     std::string &ErrMsg,
                                     const std::vector<std::string> &Opts);

class SPIRVAPITest : public testing::Test {
protected:
  /*
    void SetUp() override {
      EXPECT_TRUE(Status && Error.empty() && !Result.empty());
    }
  */

  bool toSpirv(StringRef Assembly, std::string &Result, std::string &ErrMsg,
               const std::vector<std::string> &Opts) {
    SMDiagnostic ParseError;
    M = parseAssemblyString(Assembly, ParseError, Context);
    if (!M) {
      ParseError.print("IR parsing failed: ", errs());
      report_fatal_error("Can't parse input assembly.");
    }
    return SPIRVTranslateModule(M.get(), Result, ErrMsg, Opts);
  }

  LLVMContext Context;
  std::unique_ptr<Module> M;
};

TEST_F(SPIRVAPITest, checkTranslateExtError) {
  StringRef Assembly = R"(
    define dso_local spir_func void @test1() {
    entry:
      %res1 = tail call spir_func i32 @_Z26__spirv_GroupBitwiseAndKHR(i32 2, i32 0, i32 0)
      ret void
    }

    declare dso_local spir_func i32  @_Z26__spirv_GroupBitwiseAndKHR(i32, i32, i32)
  )";
  std::string Result, Error;
  std::vector<std::string> Opts;
  bool Status = toSpirv(Assembly, Result, Error, Opts);
  EXPECT_TRUE(Status && Error.empty() && !Result.empty());
  EXPECT_EQ(identify_magic(Result), file_magic::spirv_object);
}

TEST_F(SPIRVAPITest, checkTranslateOk) {
  StringRef Assemblies[] = {"", R"(
    %struct = type { [1 x i64] }

    define spir_kernel void @foo(ptr noundef byval(%struct) %arg) {
    entry:
      call spir_func void @bar(<2 x i32> noundef <i32 0, i32 1>)
      ret void
    }

    define spir_func void @bar(<2 x i32> noundef) {
    entry:
      ret void
    }
  )"};
  for (StringRef &Assembly : Assemblies) {
    std::string Result, Error;
    std::vector<std::string> Opts;
    bool Status = toSpirv(Assembly, Result, Error, Opts);
    EXPECT_TRUE(Status && Error.empty() && !Result.empty());
    EXPECT_EQ(identify_magic(Result), file_magic::spirv_object);
  }
}

} // end namespace llvm
