//===- RemarkTest.cpp - Remark unit tests -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Remarks.h"
#include "mlir/Remark/RemarkStreamer.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMRemarkStreamer.h"
#include "llvm/Remarks/RemarkFormat.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/YAMLParser.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <optional>

using namespace llvm;
using namespace mlir;
using namespace testing;
namespace {

TEST(Remark, TestOutputOptimizationRemark) {
  std::string categoryVectorizer("Vectorizer");
  std::string categoryRegister("Register");
  std::string categoryUnroll("Unroll");
  std::string categoryInliner("Inliner");
  std::string categoryReroller("Reroller");
  std::string myPassname1("myPass1");
  SmallString<64> tmpPathStorage;
  sys::fs::createUniquePath("remarks-%%%%%%.yaml", tmpPathStorage,
                            /*MakeAbsolute=*/true);
  std::string yamlFile =
      std::string(tmpPathStorage.data(), tmpPathStorage.size());
  ASSERT_FALSE(yamlFile.empty());

  {
    MLIRContext context;
    Location loc = UnknownLoc::get(&context);

    context.printOpOnDiagnostic(true);
    context.printStackTraceOnDiagnostic(true);

    // Setup the remark engine
    mlir::remark::RemarkCategories cats{/*passed=*/categoryVectorizer,
                                        /*missed=*/categoryUnroll,
                                        /*analysis=*/categoryRegister,
                                        /*failed=*/categoryInliner};

    LogicalResult isEnabled =
        mlir::remark::enableOptimizationRemarksWithLLVMStreamer(
            context, yamlFile, llvm::remarks::Format::YAML, cats);
    ASSERT_TRUE(succeeded(isEnabled)) << "Failed to enable remark engine";

    // PASS: something succeeded
    remark::passed(loc, remark::RemarkOpts::name("Pass1")
                            .category(categoryVectorizer)
                            .subCategory(myPassname1)
                            .function("bar"))
        << "vectorized loop" << remark::metric("tripCount", 128);

    // ANALYSIS: neutral insight
    remark::analysis(
        loc, remark::RemarkOpts::name("Analysis1").category(categoryRegister))
        << "Kernel uses 168 registers";

    // MISSED: explain why + suggest a fix
    remark::missed(loc, remark::RemarkOpts::name("Miss1")
                            .category(categoryUnroll)
                            .subCategory(myPassname1))
        << remark::reason("not profitable at this size")
        << remark::suggest("increase unroll factor to >=4");

    // FAILURE: action attempted but failed
    remark::failed(loc, remark::RemarkOpts::name("Failed1")
                            .category(categoryInliner)
                            .subCategory(myPassname1))
        << remark::reason("failed due to unsupported pattern");

    // FAILURE: Won't show up
    remark::failed(loc, remark::RemarkOpts::name("Failed2")
                            .category(categoryReroller)
                            .subCategory(myPassname1))
        << remark::reason("failed due to rerolling pattern");
  }

  // Read the file
  auto bufferOrErr = MemoryBuffer::getFile(yamlFile);
  ASSERT_TRUE(static_cast<bool>(bufferOrErr)) << "Failed to open remarks file";
  std::string content = bufferOrErr.get()->getBuffer().str();

  EXPECT_THAT(content, HasSubstr("--- !Passed"));
  EXPECT_THAT(content, HasSubstr("Name:            Pass1"));
  EXPECT_THAT(content, HasSubstr("Pass:            'Vectorizer:myPass1'"));
  EXPECT_THAT(content, HasSubstr("Function:        bar"));
  EXPECT_THAT(content, HasSubstr("Remark:          vectorized loop"));
  EXPECT_THAT(content, HasSubstr("tripCount:       '128'"));

  EXPECT_THAT(content, HasSubstr("--- !Analysis"));
  EXPECT_THAT(content, HasSubstr("Pass:            Register"));
  EXPECT_THAT(content, HasSubstr("Name:            Analysis1"));
  EXPECT_THAT(content, HasSubstr("Function:        '<unknown function>'"));
  EXPECT_THAT(content, HasSubstr("Remark:          Kernel uses 168 registers"));

  EXPECT_THAT(content, HasSubstr("--- !Missed"));
  EXPECT_THAT(content, HasSubstr("Pass:            'Unroll:myPass1'"));
  EXPECT_THAT(content, HasSubstr("Name:            Miss1"));
  EXPECT_THAT(content, HasSubstr("Function:        '<unknown function>'"));
  EXPECT_THAT(content,
              HasSubstr("Reason:          not profitable at this size"));
  EXPECT_THAT(content,
              HasSubstr("Suggestion:      'increase unroll factor to >=4'"));

  EXPECT_THAT(content, HasSubstr("--- !Failure"));
  EXPECT_THAT(content, HasSubstr("Pass:            'Inliner:myPass1'"));
  EXPECT_THAT(content, HasSubstr("Name:            Failed1"));
  EXPECT_THAT(content, HasSubstr("Function:        '<unknown function>'"));
  EXPECT_THAT(content,
              HasSubstr("Reason:          failed due to unsupported pattern"));

  EXPECT_THAT(content, Not(HasSubstr("Failed2")));
  EXPECT_THAT(content, Not(HasSubstr("Reroller")));

  // Also verify document order to avoid false positives.
  size_t iPassed = content.find("--- !Passed");
  size_t iAnalysis = content.find("--- !Analysis");
  size_t iMissed = content.find("--- !Missed");
  size_t iFailure = content.find("--- !Failure");

  ASSERT_NE(iPassed, std::string::npos);
  ASSERT_NE(iAnalysis, std::string::npos);
  ASSERT_NE(iMissed, std::string::npos);
  ASSERT_NE(iFailure, std::string::npos);

  EXPECT_LT(iPassed, iAnalysis);
  EXPECT_LT(iAnalysis, iMissed);
  EXPECT_LT(iMissed, iFailure);
}

TEST(Remark, TestNoOutputOptimizationRemark) {
  const auto *pass1Msg = "My message";

  std::string categoryFailName("myImportantCategory");
  std::string myPassname1("myPass1");
  std::string funcName("myFunc");
  SmallString<64> tmpPathStorage;
  sys::fs::createUniquePath("remarks-%%%%%%.yaml", tmpPathStorage,
                            /*MakeAbsolute=*/true);
  std::string yamlFile =
      std::string(tmpPathStorage.data(), tmpPathStorage.size());
  ASSERT_FALSE(yamlFile.empty());
  std::error_code ec =
      llvm::sys::fs::remove(yamlFile, /*IgnoreNonExisting=*/true);
  if (ec) {
    FAIL() << "Failed to remove file " << yamlFile << ": " << ec.message();
  }
  {
    MLIRContext context;
    Location loc = UnknownLoc::get(&context);
    remark::failed(loc, remark::RemarkOpts::name("myfail")
                            .category(categoryFailName)
                            .subCategory(myPassname1))
        << remark::reason(pass1Msg);
  }
  // No setup, so no output file should be created
  // check!
  bool fileExists = llvm::sys::fs::exists(yamlFile);
  EXPECT_FALSE(fileExists)
      << "Expected no YAML file to be created without setupOptimizationRemarks";
}

TEST(Remark, TestOutputOptimizationRemarkDiagnostic) {
  std::string categoryVectorizer("Vectorizer");
  std::string categoryRegister("Register");
  std::string categoryUnroll("Unroll");
  std::string myPassname1("myPass1");
  std::string fName("foo");

  llvm::SmallVector<std::string> seenMsg;
  {
    MLIRContext context;
    Location loc = UnknownLoc::get(&context);

    context.printOpOnDiagnostic(true);
    context.printStackTraceOnDiagnostic(true);

    // Register a handler that captures the diagnostic.
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diag) {
      seenMsg.push_back(diag.str());
      return success();
    });

    // Setup the remark engine
    mlir::remark::RemarkCategories cats{/*passed=*/categoryVectorizer,
                                        /*missed=*/categoryUnroll,
                                        /*analysis=*/categoryRegister,
                                        /*failed=*/categoryUnroll};

    LogicalResult isEnabled =
        remark::enableOptimizationRemarks(context, nullptr, cats, true);

    ASSERT_TRUE(succeeded(isEnabled)) << "Failed to enable remark engine";

    // PASS: something succeeded
    remark::passed(loc, remark::RemarkOpts::name("pass1")
                            .category(categoryVectorizer)
                            .function(fName)
                            .subCategory(myPassname1))
        << "vectorized loop" << remark::metric("tripCount", 128);

    // ANALYSIS: neutral insight
    remark::analysis(loc, remark::RemarkOpts::name("Analysis1")
                              .category(categoryRegister)
                              .function(fName))
        << "Kernel uses 168 registers";

    // MISSED: explain why + suggest a fix
    int target = 128;
    int tripBad = 4;
    int threshold = 256;

    remark::missed(loc, {"", categoryUnroll, "unroller2", ""})
        << remark::reason("tripCount={0} < threshold={1}", tripBad, threshold);

    remark::missed(loc, {"", categoryUnroll, "", ""})
        << remark::reason("tripCount={0} < threshold={1}", tripBad, threshold)
        << remark::suggest("increase unroll to {0}", target);

    // FAILURE: action attempted but failed
    remark::failed(loc, {"", categoryUnroll, "", ""})
        << remark::reason("failed due to unsupported pattern");
  }
  // clang-format off
  unsigned long expectedSize = 5;
  ASSERT_EQ(seenMsg.size(), expectedSize);
  EXPECT_EQ(seenMsg[0], "[Passed] pass1 | Category:Vectorizer:myPass1 | Function=foo | Remark=\"vectorized loop\", tripCount=128");
  EXPECT_EQ(seenMsg[1], "[Analysis] Analysis1 | Category:Register | Function=foo | Remark=\"Kernel uses 168 registers\"");
  EXPECT_EQ(seenMsg[2], "[Missed]  | Category:Unroll:unroller2 | Reason=\"tripCount=4 < threshold=256\"");
  EXPECT_EQ(seenMsg[3], "[Missed]  | Category:Unroll | Reason=\"tripCount=4 < threshold=256\", Suggestion=\"increase unroll to 128\"");
  EXPECT_EQ(seenMsg[4], "[Failure]  | Category:Unroll | Reason=\"failed due to unsupported pattern\"");
  // clang-format on
}

/// Custom remark streamer that prints remarks to stderr.
class MyCustomStreamer : public remark::detail::MLIRRemarkStreamerBase {
public:
  MyCustomStreamer() = default;

  void streamOptimizationRemark(const remark::detail::Remark &remark) override {
    llvm::errs() << "Custom remark: ";
    remark.print(llvm::errs(), true);
    llvm::errs() << "\n";
  }
};

TEST(Remark, TestCustomOptimizationRemarkDiagnostic) {
  testing::internal::CaptureStderr();
  const auto *pass1Msg = "My message";
  const auto *pass2Msg = "My another message";
  const auto *pass3Msg = "Do not show this message";

  std::string categoryLoopunroll("LoopUnroll");
  std::string categoryInline("Inliner");
  std::string myPassname1("myPass1");
  std::string myPassname2("myPass2");
  std::string funcName("myFunc");

  std::string seenMsg = "";

  {
    MLIRContext context;
    Location loc = UnknownLoc::get(&context);

    // Setup the remark engine
    mlir::remark::RemarkCategories cats{/*passed=*/categoryLoopunroll,
                                        /*missed=*/std::nullopt,
                                        /*analysis=*/std::nullopt,
                                        /*failed=*/categoryLoopunroll};

    LogicalResult isEnabled = remark::enableOptimizationRemarks(
        context, std::make_unique<MyCustomStreamer>(), cats, true);
    ASSERT_TRUE(succeeded(isEnabled)) << "Failed to enable remark engine";

    // Remark 1: pass, category LoopUnroll
    remark::passed(loc, {"", categoryLoopunroll, myPassname1, ""}) << pass1Msg;
    // Remark 2: failure, category LoopUnroll
    remark::failed(loc, {"", categoryLoopunroll, myPassname2, ""})
        << remark::reason(pass2Msg);
    // Remark 3: pass, category Inline (should not be printed)
    remark::passed(loc, {"", categoryInline, myPassname1, ""}) << pass3Msg;
  }

  llvm::errs().flush();
  std::string errOut = ::testing::internal::GetCapturedStderr();

  // Expect exactly two "Custom remark:" lines.
  auto first = errOut.find("Custom remark:");
  EXPECT_NE(first, std::string::npos);
  auto second = errOut.find("Custom remark:", first + 1);
  EXPECT_NE(second, std::string::npos);
  auto third = errOut.find("Custom remark:", second + 1);
  EXPECT_EQ(third, std::string::npos);

  // Containment checks for messages.
  EXPECT_NE(errOut.find(pass1Msg), std::string::npos); // printed
  EXPECT_NE(errOut.find(pass2Msg), std::string::npos); // printed
  EXPECT_EQ(errOut.find(pass3Msg), std::string::npos); // filtered out
}
} // namespace
