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

#include "llvm/AsmParser/Parser.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include <string>
#include <utility>

using ::testing::StartsWith;

namespace llvm {

extern "C" bool
SPIRVTranslateModule(Module *M, std::string &SpirvObj, std::string &ErrMsg,
                     const std::vector<std::string> &AllowExtNames,
                     const std::vector<std::string> &Opts);

class SPIRVAPITest : public testing::Test {
protected:
  bool toSpirv(StringRef Assembly, std::string &Result, std::string &ErrMsg,
               const std::vector<std::string> &AllowExtNames,
               const std::vector<std::string> &Opts) {
    SMDiagnostic ParseError;
    LLVMContext Context;
    std::unique_ptr<Module> M =
        parseAssemblyString(Assembly, ParseError, Context);
    if (!M) {
      ParseError.print("IR parsing failed: ", errs());
      report_fatal_error("Can't parse input assembly.");
    }
    bool Status =
        SPIRVTranslateModule(M.get(), Result, ErrMsg, AllowExtNames, Opts);
    if (!Status)
      errs() << ErrMsg;
    return Status;
  }

  static constexpr StringRef ExtensionAssembly = R"(
    define dso_local spir_func void @test1() {
    entry:
      %res1 = tail call spir_func i32 @_Z26__spirv_GroupBitwiseAndKHR(i32 2, i32 0, i32 0)
      ret void
    }

    declare dso_local spir_func i32  @_Z26__spirv_GroupBitwiseAndKHR(i32, i32, i32)
  )";
  static constexpr StringRef OkAssembly = R"(
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
  )";
};

TEST_F(SPIRVAPITest, checkTranslateOk) {
  StringRef Assemblies[] = {"", OkAssembly};
  // Those command line arguments that overlap with registered by llc/codegen
  // are to be started with the ' ' symbol.
  std::vector<std::string> SetOfOpts[] = {
      {}, {"--spirv-mtriple=spirv32-unknown-unknown"}};
  for (const auto &Opts : SetOfOpts) {
    for (StringRef &Assembly : Assemblies) {
      std::string Result, Error;
      bool Status = toSpirv(Assembly, Result, Error, {}, Opts);
      EXPECT_TRUE(Status && Error.empty() && !Result.empty());
      EXPECT_EQ(identify_magic(Result), file_magic::spirv_object);
    }
  }
}

TEST_F(SPIRVAPITest, checkTranslateError) {
  {
    std::string Result, Error;
    bool Status = toSpirv(OkAssembly, Result, Error, {},
                          {"-mtriple=spirv32-unknown-unknown"});
    EXPECT_FALSE(Status);
    EXPECT_TRUE(Result.empty());
    EXPECT_THAT(
        Error, StartsWith("SPIRVTranslateModule: Unknown command line argument "
                          "'-mtriple=spirv32-unknown-unknown'"));
  }
  {
    std::string Result, Error;
    bool Status = toSpirv(OkAssembly, Result, Error, {}, {"--spirv-O 5"});
    EXPECT_FALSE(Status);
    EXPECT_TRUE(Result.empty());
    EXPECT_EQ(Error, "Invalid optimization level!");
  }
  {
    std::string Result, Error;
    bool Status = toSpirv(OkAssembly, Result, Error, {}, {});
    EXPECT_TRUE(Status && Error.empty() && !Result.empty());
    EXPECT_EQ(identify_magic(Result), file_magic::spirv_object);
  }
}

TEST_F(SPIRVAPITest, checkTranslateSupportExtensionByOpts) {
  std::string Result, Error;
  std::vector<std::string> Opts{
      "--spirv-ext=+SPV_KHR_uniform_group_instructions"};
  bool Status = toSpirv(ExtensionAssembly, Result, Error, {}, Opts);
  EXPECT_TRUE(Status && Error.empty() && !Result.empty());
  EXPECT_EQ(identify_magic(Result), file_magic::spirv_object);
}

TEST_F(SPIRVAPITest, checkTranslateSupportExtensionByArg) {
  std::string Result, Error;
  std::vector<std::string> ExtNames{"SPV_KHR_uniform_group_instructions"};
  bool Status = toSpirv(ExtensionAssembly, Result, Error, ExtNames, {});
  EXPECT_TRUE(Status && Error.empty() && !Result.empty());
  EXPECT_EQ(identify_magic(Result), file_magic::spirv_object);
}

TEST_F(SPIRVAPITest, checkTranslateSupportExtensionByArgList) {
  std::string Result, Error;
  std::vector<std::string> ExtNames{"SPV_KHR_subgroup_rotate",
                                    "SPV_KHR_uniform_group_instructions",
                                    "SPV_KHR_subgroup_rotate"};
  bool Status = toSpirv(ExtensionAssembly, Result, Error, ExtNames, {});
  EXPECT_TRUE(Status && Error.empty() && !Result.empty());
  EXPECT_EQ(identify_magic(Result), file_magic::spirv_object);
}

TEST_F(SPIRVAPITest, checkTranslateAllExtensions) {
  std::string Result, Error;
  std::vector<std::string> Opts{"--spirv-ext=all"};
  bool Status = toSpirv(ExtensionAssembly, Result, Error, {}, Opts);
  EXPECT_TRUE(Status && Error.empty() && !Result.empty());
  EXPECT_EQ(identify_magic(Result), file_magic::spirv_object);
}

TEST_F(SPIRVAPITest, checkTranslateUnknownExtensionByArg) {
  std::string Result, Error;
  std::vector<std::string> ExtNames{"SPV_XYZ_my_unknown_extension"};
  bool Status = toSpirv(ExtensionAssembly, Result, Error, ExtNames, {});
  EXPECT_FALSE(Status);
  EXPECT_TRUE(Result.empty());
  EXPECT_EQ(Error, "Unknown SPIR-V extension: SPV_XYZ_my_unknown_extension");
}

#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST
TEST_F(SPIRVAPITest, checkTranslateExtensionError) {
  std::string Result, Error;
  std::vector<std::string> Opts;
  EXPECT_DEATH_IF_SUPPORTED(
      { toSpirv(ExtensionAssembly, Result, Error, {}, Opts); },
      "LLVM ERROR: __spirv_GroupBitwiseAndKHR: the builtin requires the "
      "following SPIR-V extension: SPV_KHR_uniform_group_instructions");
}

TEST_F(SPIRVAPITest, checkTranslateUnknownExtensionByOpts) {
  std::string Result, Error;
  std::vector<std::string> Opts{"--spirv-ext=+SPV_XYZ_my_unknown_extension"};
  EXPECT_DEATH_IF_SUPPORTED(
      { toSpirv(ExtensionAssembly, Result, Error, {}, Opts); },
      "SPIRVTranslateModule: for the --spirv-ext option: Unknown SPIR-V");
}

TEST_F(SPIRVAPITest, checkTranslateWrongExtensionByOpts) {
  std::string Result, Error;
  std::vector<std::string> Opts{"--spirv-ext=+SPV_KHR_subgroup_rotate"};
  EXPECT_DEATH_IF_SUPPORTED(
      { toSpirv(ExtensionAssembly, Result, Error, {}, Opts); },
      "LLVM ERROR: __spirv_GroupBitwiseAndKHR: the builtin requires the "
      "following SPIR-V extension: SPV_KHR_uniform_group_instructions");
}

TEST_F(SPIRVAPITest, checkTranslateWrongExtensionByArg) {
  std::string Result, Error;
  std::vector<std::string> ExtNames{"SPV_KHR_subgroup_rotate"};
  EXPECT_DEATH_IF_SUPPORTED(
      { toSpirv(ExtensionAssembly, Result, Error, ExtNames, {}); },
      "LLVM ERROR: __spirv_GroupBitwiseAndKHR: the builtin requires the "
      "following SPIR-V extension: SPV_KHR_uniform_group_instructions");
}
#endif

} // end namespace llvm
