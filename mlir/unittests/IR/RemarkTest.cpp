//===- RemarkTest.cpp - Remark unit tests -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
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
  llvm::SmallString<64> tmpPathStorage;
  llvm::sys::fs::createUniquePath("remarks-%%%%%%.yaml", tmpPathStorage,
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
    mlir::remark::RemarkCategories cats{/*all=*/"",
                                        /*passed=*/categoryVectorizer,
                                        /*missed=*/categoryUnroll,
                                        /*analysis=*/categoryRegister,
                                        /*failed=*/categoryInliner};
    std::unique_ptr<remark::RemarkEmittingPolicyAll> policy =
        std::make_unique<remark::RemarkEmittingPolicyAll>();
    LogicalResult isEnabled =
        mlir::remark::enableOptimizationRemarksWithLLVMStreamer(
            context, yamlFile, llvm::remarks::Format::YAML, std::move(policy),
            cats);
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
  auto bufferOrErr = llvm::MemoryBuffer::getFile(yamlFile);
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
  SmallString<64> tmpPathStorage;
  llvm::sys::fs::createUniquePath("remarks-%%%%%%.yaml", tmpPathStorage,
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
    mlir::remark::RemarkCategories cats{/*all=*/"",
                                        /*passed=*/categoryVectorizer,
                                        /*missed=*/categoryUnroll,
                                        /*analysis=*/categoryRegister,
                                        /*failed=*/categoryUnroll};
    std::unique_ptr<remark::RemarkEmittingPolicyAll> policy =
        std::make_unique<remark::RemarkEmittingPolicyAll>();
    LogicalResult isEnabled = remark::enableOptimizationRemarks(
        context, nullptr, std::move(policy), cats, true);

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

    remark::missed(loc, remark::RemarkOpts::name("")
                            .category(categoryUnroll)
                            .subCategory("unroller2"))
        << remark::reason("tripCount={0} < threshold={1}", tripBad, threshold);

    remark::missed(loc, remark::RemarkOpts::name("").category(categoryUnroll))
        << remark::reason("tripCount={0} < threshold={1}", tripBad, threshold)
        << remark::suggest("increase unroll to {0}", target);

    // FAILURE: action attempted but failed
    remark::failed(loc, remark::RemarkOpts::name("").category(categoryUnroll))
        << remark::reason("failed due to unsupported pattern");
  }
  // clang-format off
  // Remarks now include RemarkId=N, so use substring checks.
  unsigned long expectedSize = 5;
  ASSERT_EQ(seenMsg.size(), expectedSize);
  EXPECT_THAT(seenMsg[0], HasSubstr("[Passed]"));
  EXPECT_THAT(seenMsg[0], HasSubstr("pass1 | Category:Vectorizer:myPass1 | Function=foo |"));
  EXPECT_THAT(seenMsg[0], HasSubstr("Remark=\"vectorized loop\""));
  EXPECT_THAT(seenMsg[0], HasSubstr("tripCount=128"));

  EXPECT_THAT(seenMsg[1], HasSubstr("[Analysis]"));
  EXPECT_THAT(seenMsg[1], HasSubstr("Analysis1 | Category:Register | Function=foo |"));
  EXPECT_THAT(seenMsg[1], HasSubstr("Remark=\"Kernel uses 168 registers\""));

  EXPECT_THAT(seenMsg[2], HasSubstr("[Missed]"));
  EXPECT_THAT(seenMsg[2], HasSubstr("Category:Unroll:unroller2"));
  EXPECT_THAT(seenMsg[2], HasSubstr("Reason=\"tripCount=4 < threshold=256\""));

  EXPECT_THAT(seenMsg[3], HasSubstr("[Missed]"));
  EXPECT_THAT(seenMsg[3], HasSubstr("Category:Unroll |"));
  EXPECT_THAT(seenMsg[3], HasSubstr("Suggestion=\"increase unroll to 128\""));

  EXPECT_THAT(seenMsg[4], HasSubstr("[Failure]"));
  EXPECT_THAT(seenMsg[4], HasSubstr("Category:Unroll |"));
  EXPECT_THAT(seenMsg[4], HasSubstr("Reason=\"failed due to unsupported pattern\""));
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

  {
    MLIRContext context;
    Location loc = UnknownLoc::get(&context);

    // Setup the remark engine
    mlir::remark::RemarkCategories cats{/*all=*/"",
                                        /*passed=*/categoryLoopunroll,
                                        /*missed=*/std::nullopt,
                                        /*analysis=*/std::nullopt,
                                        /*failed=*/categoryLoopunroll};

    std::unique_ptr<remark::RemarkEmittingPolicyAll> policy =
        std::make_unique<remark::RemarkEmittingPolicyAll>();
    LogicalResult isEnabled = remark::enableOptimizationRemarks(
        context, std::make_unique<MyCustomStreamer>(), std::move(policy), cats,
        true);
    ASSERT_TRUE(succeeded(isEnabled)) << "Failed to enable remark engine";

    // Remark 1: pass, category LoopUnroll
    remark::passed(loc, remark::RemarkOpts::name("")
                            .category(categoryLoopunroll)
                            .subCategory(myPassname1))
        << pass1Msg;
    // Remark 2: failure, category LoopUnroll
    remark::failed(loc, remark::RemarkOpts::name("")
                            .category(categoryLoopunroll)
                            .subCategory(myPassname2))
        << remark::reason(pass2Msg);
    // Remark 3: pass, category Inline (should not be printed)
    remark::passed(loc, remark::RemarkOpts::name("")
                            .category(categoryInline)
                            .subCategory(myPassname1))
        << pass3Msg;
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

TEST(Remark, TestRemarkFinal) {
  testing::internal::CaptureStderr();
  const auto *pass1Msg = "I failed";
  const auto *pass2Msg = "I failed too";
  const auto *pass3Msg = "I succeeded";
  const auto *pass4Msg = "I succeeded too";

  std::string categoryLoopunroll("LoopUnroll");

  {
    MLIRContext context;
    Location loc = FileLineColLoc::get(&context, "test.cpp", 1, 5);
    Location locOther = FileLineColLoc::get(&context, "test.cpp", 55, 5);

    // Setup the remark engine
    mlir::remark::RemarkCategories cats{/*all=*/"",
                                        /*passed=*/categoryLoopunroll,
                                        /*missed=*/categoryLoopunroll,
                                        /*analysis=*/categoryLoopunroll,
                                        /*failed=*/categoryLoopunroll};

    std::unique_ptr<remark::RemarkEmittingPolicyFinal> policy =
        std::make_unique<remark::RemarkEmittingPolicyFinal>();
    LogicalResult isEnabled = remark::enableOptimizationRemarks(
        context, std::make_unique<MyCustomStreamer>(), std::move(policy), cats,
        true);
    ASSERT_TRUE(succeeded(isEnabled)) << "Failed to enable remark engine";

    // Remark 1: failure
    remark::failed(
        loc, remark::RemarkOpts::name("Unroller").category(categoryLoopunroll))
        << pass1Msg;

    // Remark 2: failure
    remark::missed(
        loc, remark::RemarkOpts::name("Unroller").category(categoryLoopunroll))
        << remark::reason(pass2Msg);

    // Remark 3: pass
    remark::passed(
        loc, remark::RemarkOpts::name("Unroller").category(categoryLoopunroll))
        << pass3Msg;

    // Remark 4: pass
    remark::passed(
        locOther,
        remark::RemarkOpts::name("Unroller").category(categoryLoopunroll))
        << pass4Msg;
  }

  llvm::errs().flush();
  std::string errOut = ::testing::internal::GetCapturedStderr();

  // PolicyFinal deduplicates by (location, name, category, kind).
  // Remarks 1 (failed), 2 (missed), 3 (passed) have different kinds, so all
  // survive. Remark 4 (passed) has a different location, so it also survives.
  EXPECT_NE(errOut.find(pass1Msg), std::string::npos); // shown (failed)
  EXPECT_NE(errOut.find(pass2Msg), std::string::npos); // shown (missed)
  EXPECT_NE(errOut.find(pass3Msg), std::string::npos); // shown (passed)
  EXPECT_NE(errOut.find(pass4Msg),
            std::string::npos); // shown (passed, diff loc)
}

TEST(Remark, TestArgWithAttribute) {
  MLIRContext context;

  llvm::SmallVector<Attribute> elements;
  elements.push_back(IntegerAttr::get(IntegerType::get(&context, 32), 1));
  elements.push_back(IntegerAttr::get(IntegerType::get(&context, 32), 2));
  elements.push_back(IntegerAttr::get(IntegerType::get(&context, 32), 3));
  ArrayAttr arrayAttr = ArrayAttr::get(&context, elements);
  remark::detail::Remark::Arg argWithArray("Values", arrayAttr);

  // Verify the attribute is stored
  EXPECT_TRUE(argWithArray.hasAttribute());
  EXPECT_EQ(argWithArray.getAttribute(), arrayAttr);

  // Ensure it can be retrieved as an ArrayAttr.
  auto retrievedAttr = dyn_cast<ArrayAttr>(argWithArray.getAttribute());
  EXPECT_TRUE(retrievedAttr);
  EXPECT_EQ(retrievedAttr.size(), 3u);
  EXPECT_EQ(cast<IntegerAttr>(retrievedAttr[0]).getInt(), 1);
  EXPECT_EQ(cast<IntegerAttr>(retrievedAttr[1]).getInt(), 2);
  EXPECT_EQ(cast<IntegerAttr>(retrievedAttr[2]).getInt(), 3);

  // Create an Arg without an Attribute (string-based)
  remark::detail::Remark::Arg argWithoutAttr("Key", "Value");

  // Verify no attribute is stored
  EXPECT_FALSE(argWithoutAttr.hasAttribute());
  EXPECT_FALSE(argWithoutAttr.getAttribute()); // Returns null Attribute
  EXPECT_EQ(argWithoutAttr.val, "Value");
}

// Test that Remark correctly owns its string data and doesn't have
// use-after-free issues when the original strings go out of scope.
// This is particularly important for RemarkEmittingPolicyFinal which
// stores remarks and emits them later during finalize().
TEST(Remark, TestRemarkOwnsStringData) {
  testing::internal::CaptureStderr();

  // These are the expected values we'll check for in the output.
  // They must match what we create in the inner scope below.
  const char *expectedCategory = "DynamicCategory";
  const char *expectedName = "DynamicRemarkName";
  const char *expectedFunction = "dynamicFunction";
  const char *expectedMessage = "Dynamic message content";

  {
    MLIRContext context;
    Location loc = FileLineColLoc::get(&context, "test.cpp", 42, 10);

    // Setup with RemarkEmittingPolicyFinal - this stores remarks and emits
    // them only when the engine is destroyed (during finalize).
    // Note: The 'passed' filter must be set for remark::passed() to emit.
    mlir::remark::RemarkCategories cats{
        /*all=*/std::nullopt,
        /*passed=*/expectedCategory, // Enable passed remarks for this category
        /*missed=*/std::nullopt,
        /*analysis=*/std::nullopt,
        /*failed=*/std::nullopt};

    std::unique_ptr<remark::RemarkEmittingPolicyFinal> policy =
        std::make_unique<remark::RemarkEmittingPolicyFinal>();
    LogicalResult isEnabled = remark::enableOptimizationRemarks(
        context, std::make_unique<MyCustomStreamer>(), std::move(policy), cats,
        /*printAsEmitRemarks=*/true);
    ASSERT_TRUE(succeeded(isEnabled)) << "Failed to enable remark engine";

    // Create dynamic strings in an inner scope that will go out of scope
    // BEFORE the RemarkEngine is destroyed and finalize() is called.
    {
      std::string dynamicCategory(expectedCategory);
      std::string dynamicName(expectedName);
      std::string dynamicFunction(expectedFunction);
      std::string dynamicSubCategory("DynamicSubCategory");
      std::string dynamicMessage(expectedMessage);

      // Emit a remark with all dynamic strings
      remark::passed(loc, remark::RemarkOpts::name(dynamicName)
                              .category(dynamicCategory)
                              .subCategory(dynamicSubCategory)
                              .function(dynamicFunction))
          << dynamicMessage;

      // dynamicCategory, dynamicName, dynamicFunction, dynamicSubCategory,
      // and dynamicMessage all go out of scope here!
    }

    // At this point, all the dynamic strings have been destroyed.
    // The Remark stored in RemarkEmittingPolicyFinal must have its own
    // copies of the string data, otherwise we'd have dangling pointers.

    // Context destruction triggers RemarkEngine destruction, which calls
    // finalize() on the policy, which then emits the stored remarks.
    // If Remark doesn't own its strings, this would crash or produce garbage.
  }

  llvm::errs().flush();
  std::string errOut = ::testing::internal::GetCapturedStderr();

  // Verify the output contains our expected strings - this proves the
  // Remark correctly copied and owns the string data.
  EXPECT_NE(errOut.find(expectedCategory), std::string::npos)
      << "Expected category not found in output. Got: " << errOut;
  EXPECT_NE(errOut.find(expectedName), std::string::npos)
      << "Expected name not found in output. Got: " << errOut;
  EXPECT_NE(errOut.find(expectedFunction), std::string::npos)
      << "Expected function not found in output. Got: " << errOut;
  EXPECT_NE(errOut.find(expectedMessage), std::string::npos)
      << "Expected message not found in output. Got: " << errOut;
}

// Test that remarks can be linked together using RemarkId.
TEST(Remark, TestRemarkLinking) {
  testing::internal::CaptureStderr();

  std::string categoryOpt("Optimizer");

  {
    MLIRContext context;
    Location loc = FileLineColLoc::get(&context, "test.cpp", 10, 5);

    // Setup the remark engine
    mlir::remark::RemarkCategories cats{/*all=*/std::nullopt,
                                        /*passed=*/categoryOpt,
                                        /*missed=*/std::nullopt,
                                        /*analysis=*/categoryOpt,
                                        /*failed=*/std::nullopt};

    std::unique_ptr<remark::RemarkEmittingPolicyAll> policy =
        std::make_unique<remark::RemarkEmittingPolicyAll>();
    LogicalResult isEnabled = remark::enableOptimizationRemarks(
        context, std::make_unique<MyCustomStreamer>(), std::move(policy), cats,
        /*printAsEmitRemarks=*/true);
    ASSERT_TRUE(succeeded(isEnabled)) << "Failed to enable remark engine";

    // Emit an analysis remark and capture its ID.
    auto analysisRemark = remark::analysis(
        loc, remark::RemarkOpts::name("LoopAnalysis").category(categoryOpt));
    analysisRemark << "analyzed loop with trip count 128";
    remark::RemarkId analysisId = analysisRemark.getId();

    // Verify we got a valid ID.
    EXPECT_TRUE(static_cast<bool>(analysisId));
    EXPECT_GT(analysisId.getValue(), 0u);

    // Emit a passed remark that links to the analysis via RemarkOpts.
    remark::passed(loc, remark::RemarkOpts::name("LoopOptimized")
                            .category(categoryOpt)
                            .relatedTo(analysisId))
        << "vectorized loop";
  }

  llvm::errs().flush();
  std::string errOut = ::testing::internal::GetCapturedStderr();

  // Verify the analysis remark has an ID.
  EXPECT_THAT(errOut, HasSubstr("RemarkId="));

  // Verify the passed remark links to the analysis remark.
  EXPECT_THAT(errOut, HasSubstr("RelatedTo="));

  // Verify both remarks are present.
  EXPECT_THAT(errOut, HasSubstr("LoopAnalysis"));
  EXPECT_THAT(errOut, HasSubstr("LoopOptimized"));
  EXPECT_THAT(errOut, HasSubstr("analyzed loop"));
  EXPECT_THAT(errOut, HasSubstr("vectorized loop"));
}

} // namespace
