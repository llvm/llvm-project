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
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include <string>
#include <utility>

using ::testing::StartsWith;

namespace llvm {

extern "C" LLVM_EXTERNAL_VISIBILITY bool
SPIRVTranslate(Module *M, std::string &SpirvObj, std::string &ErrMsg,
               const std::vector<std::string> &AllowExtNames,
               llvm::CodeGenOptLevel OLevel, Triple TargetTriple);

extern "C" bool
SPIRVTranslateModule(Module *M, std::string &SpirvObj, std::string &ErrMsg,
                     const std::vector<std::string> &AllowExtNames,
                     const std::vector<std::string> &Opts);

class SPIRVAPITest : public testing::Test {
protected:
  bool toSpirv(StringRef Assembly, std::string &Result, std::string &ErrMsg,
               const std::vector<std::string> &AllowExtNames,
               llvm::CodeGenOptLevel OLevel, Triple TargetTriple) {
    SMDiagnostic ParseError;
    LLVMContext Context;
    std::unique_ptr<Module> M =
        parseAssemblyString(Assembly, ParseError, Context);
    if (!M) {
      ParseError.print("IR parsing failed: ", errs());
      report_fatal_error("Can't parse input assembly.");
    }
    bool Status = SPIRVTranslate(M.get(), Result, ErrMsg, AllowExtNames, OLevel,
                                 TargetTriple);
    if (!Status)
      errs() << ErrMsg;
    return Status;
  }
  // TODO: Remove toSpirvLegacy() and related tests after existing clients
  // switch into a newer implementation of SPIRVTranslate().
  bool toSpirvLegacy(StringRef Assembly, std::string &Result,
                     std::string &ErrMsg,
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
  for (StringRef &Assembly : Assemblies) {
    std::string Result, Error;
    bool Status = toSpirv(Assembly, Result, Error, {}, CodeGenOptLevel::Default,
                          Triple("spirv32-unknown-unknown"));
    EXPECT_TRUE(Status && Error.empty() && !Result.empty());
    EXPECT_EQ(identify_magic(Result), file_magic::spirv_object);
  }
}

TEST_F(SPIRVAPITest, checkTranslateSupportExtensionByArg) {
  std::string Result, Error;
  std::vector<std::string> ExtNames{"SPV_KHR_uniform_group_instructions"};
  bool Status =
      toSpirv(ExtensionAssembly, Result, Error, ExtNames,
              CodeGenOptLevel::Aggressive, Triple("spirv64-unknown-unknown"));
  EXPECT_TRUE(Status && Error.empty() && !Result.empty());
  EXPECT_EQ(identify_magic(Result), file_magic::spirv_object);
}

TEST_F(SPIRVAPITest, checkTranslateSupportExtensionByArgList) {
  std::string Result, Error;
  std::vector<std::string> ExtNames{"SPV_KHR_subgroup_rotate",
                                    "SPV_KHR_uniform_group_instructions",
                                    "SPV_KHR_subgroup_rotate"};
  bool Status =
      toSpirv(ExtensionAssembly, Result, Error, ExtNames,
              CodeGenOptLevel::Aggressive, Triple("spirv64-unknown-unknown"));
  EXPECT_TRUE(Status && Error.empty() && !Result.empty());
  EXPECT_EQ(identify_magic(Result), file_magic::spirv_object);
}

TEST_F(SPIRVAPITest, checkTranslateAllExtensions) {
  std::string Result, Error;
  std::vector<std::string> ExtNames{"all"};
  bool Status =
      toSpirv(ExtensionAssembly, Result, Error, ExtNames,
              CodeGenOptLevel::Aggressive, Triple("spirv64-unknown-unknown"));
  EXPECT_TRUE(Status && Error.empty() && !Result.empty());
  EXPECT_EQ(identify_magic(Result), file_magic::spirv_object);
}

TEST_F(SPIRVAPITest, checkTranslateUnknownExtensionByArg) {
  std::string Result, Error;
  std::vector<std::string> ExtNames{"SPV_XYZ_my_unknown_extension"};
  bool Status =
      toSpirv(ExtensionAssembly, Result, Error, ExtNames,
              CodeGenOptLevel::Aggressive, Triple("spirv64-unknown-unknown"));
  EXPECT_FALSE(Status);
  EXPECT_TRUE(Result.empty());
  EXPECT_EQ(Error, "Unknown SPIR-V extension: SPV_XYZ_my_unknown_extension");
}

#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST
TEST_F(SPIRVAPITest, checkTranslateExtensionError) {
  std::string Result, Error;
  EXPECT_DEATH_IF_SUPPORTED(
      {
        toSpirv(ExtensionAssembly, Result, Error, {},
                CodeGenOptLevel::Aggressive, Triple("spirv64-unknown-unknown"));
      },
      "LLVM ERROR: __spirv_GroupBitwiseAndKHR: the builtin requires the "
      "following SPIR-V extension: SPV_KHR_uniform_group_instructions");
}

TEST_F(SPIRVAPITest, checkTranslateWrongExtensionByArg) {
  std::string Result, Error;
  std::vector<std::string> ExtNames{"SPV_KHR_subgroup_rotate"};
  EXPECT_DEATH_IF_SUPPORTED(
      {
        toSpirv(ExtensionAssembly, Result, Error, ExtNames,
                CodeGenOptLevel::Aggressive, Triple("spirv64-unknown-unknown"));
      },
      "LLVM ERROR: __spirv_GroupBitwiseAndKHR: the builtin requires the "
      "following SPIR-V extension: SPV_KHR_uniform_group_instructions");
}
#endif

// Legacy API calls. TODO: Remove after transition into a newer API.
TEST_F(SPIRVAPITest, checkTranslateStringOptsOk) {
  StringRef Assemblies[] = {"", OkAssembly};
  std::vector<std::string> SetOfOpts[] = {{}, {"spirv32-unknown-unknown"}};
  for (const auto &Opts : SetOfOpts) {
    for (StringRef &Assembly : Assemblies) {
      std::string Result, Error;
      bool Status = toSpirvLegacy(Assembly, Result, Error, {}, Opts);
      EXPECT_TRUE(Status && Error.empty() && !Result.empty());
      EXPECT_EQ(identify_magic(Result), file_magic::spirv_object);
    }
  }
}

TEST_F(SPIRVAPITest, checkTranslateStringOptsError) {
  std::string Result, Error;
  bool Status = toSpirvLegacy(OkAssembly, Result, Error, {},
                              {"spirv64v1.6-unknown-unknown", "5"});
  EXPECT_FALSE(Status);
  EXPECT_TRUE(Result.empty());
  EXPECT_EQ(Error, "Invalid optimization level!");
}

TEST_F(SPIRVAPITest, checkTranslateStringOptsErrorOk) {
  {
    std::string Result, Error;
    bool Status = toSpirvLegacy(OkAssembly, Result, Error, {},
                                {"spirv64v1.6-unknown-unknown", "5"});
    EXPECT_FALSE(Status);
    EXPECT_TRUE(Result.empty());
    EXPECT_EQ(Error, "Invalid optimization level!");
  }
  {
    std::string Result, Error;
    bool Status = toSpirvLegacy(OkAssembly, Result, Error, {},
                                {"spirv64v1.6-unknown-unknown", "3"});
    EXPECT_TRUE(Status && Error.empty() && !Result.empty());
    EXPECT_EQ(identify_magic(Result), file_magic::spirv_object);
  }
}

TEST_F(SPIRVAPITest, checkTranslateStringOptsSupportExtensionByArg) {
  std::string Result, Error;
  std::vector<std::string> ExtNames{"SPV_KHR_uniform_group_instructions"};
  bool Status = toSpirvLegacy(ExtensionAssembly, Result, Error, ExtNames, {});
  EXPECT_TRUE(Status && Error.empty() && !Result.empty());
  EXPECT_EQ(identify_magic(Result), file_magic::spirv_object);
}

TEST_F(SPIRVAPITest, checkTranslateStringOptsSupportExtensionByArgList) {
  std::string Result, Error;
  std::vector<std::string> ExtNames{"SPV_KHR_subgroup_rotate",
                                    "SPV_KHR_uniform_group_instructions",
                                    "SPV_KHR_subgroup_rotate"};
  bool Status = toSpirvLegacy(ExtensionAssembly, Result, Error, ExtNames, {});
  EXPECT_TRUE(Status && Error.empty() && !Result.empty());
  EXPECT_EQ(identify_magic(Result), file_magic::spirv_object);
}

TEST_F(SPIRVAPITest, checkTranslateStringOptsAllExtensions) {
  std::string Result, Error;
  std::vector<std::string> ExtNames{"all"};
  bool Status = toSpirvLegacy(ExtensionAssembly, Result, Error, ExtNames, {});
  EXPECT_TRUE(Status && Error.empty() && !Result.empty());
  EXPECT_EQ(identify_magic(Result), file_magic::spirv_object);
}

TEST_F(SPIRVAPITest, checkTranslateStringOptsUnknownExtensionByArg) {
  std::string Result, Error;
  std::vector<std::string> ExtNames{"SPV_XYZ_my_unknown_extension"};
  bool Status = toSpirvLegacy(ExtensionAssembly, Result, Error, ExtNames, {});
  EXPECT_FALSE(Status);
  EXPECT_TRUE(Result.empty());
  EXPECT_EQ(Error, "Unknown SPIR-V extension: SPV_XYZ_my_unknown_extension");
}

#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST
TEST_F(SPIRVAPITest, checkTranslateStringOptsExtensionError) {
  std::string Result, Error;
  EXPECT_DEATH_IF_SUPPORTED(
      { toSpirvLegacy(ExtensionAssembly, Result, Error, {}, {}); },
      "LLVM ERROR: __spirv_GroupBitwiseAndKHR: the builtin requires the "
      "following SPIR-V extension: SPV_KHR_uniform_group_instructions");
}

TEST_F(SPIRVAPITest, checkTranslateStringOptsWrongExtensionByArg) {
  std::string Result, Error;
  std::vector<std::string> ExtNames{"SPV_KHR_subgroup_rotate"};
  EXPECT_DEATH_IF_SUPPORTED(
      { toSpirvLegacy(ExtensionAssembly, Result, Error, ExtNames, {}); },
      "LLVM ERROR: __spirv_GroupBitwiseAndKHR: the builtin requires the "
      "following SPIR-V extension: SPV_KHR_uniform_group_instructions");
}
#endif

} // end namespace llvm
