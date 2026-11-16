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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMRemarkStreamer.h"
#include "llvm/Remarks/RemarkFormat.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/YAMLParser.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

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

/// Streamer that records "<remark name>: <Remark arg>" for every remark it
/// receives, independent of how Remark::print formats its output.
class RecordingStreamer : public remark::detail::MLIRRemarkStreamerBase {
public:
  explicit RecordingStreamer(std::vector<std::string> &out) : out(out) {}
  void streamOptimizationRemark(const remark::detail::Remark &remark) override {
    std::string entry = remark.getRemarkName().str();
    for (const remark::detail::Remark::Arg &arg : remark.getArgs())
      if (arg.key == "Remark")
        entry += ": " + arg.val;
    out.push_back(std::move(entry));
  }

private:
  std::vector<std::string> &out;
};

static LogicalResult enableFinalPolicy(MLIRContext &context,
                                       std::vector<std::string> &emitted,
                                       StringRef category) {
  mlir::remark::RemarkCategories cats{/*all=*/std::nullopt,
                                      /*passed=*/category.str(),
                                      /*missed=*/std::nullopt,
                                      /*analysis=*/category.str(),
                                      /*failed=*/std::nullopt};
  return remark::enableOptimizationRemarks(
      context, std::make_unique<RecordingStreamer>(emitted),
      std::make_unique<remark::RemarkEmittingPolicyFinal>(), cats,
      /*printAsEmitRemarks=*/false);
}

// The final policy emits remarks in creation order. A later report of an
// identity replaces the earlier one and takes its own position.
TEST(Remark, TestRemarkFinalOrder) {
  std::vector<std::string> emitted;
  {
    MLIRContext context;
    ASSERT_TRUE(succeeded(enableFinalPolicy(context, emitted, "LoopUnroll")));
    Location locA = FileLineColLoc::get(&context, "test.cpp", 2, 5);
    Location locB = FileLineColLoc::get(&context, "test.cpp", 1, 5);
    auto opts = remark::RemarkOpts::name("Unroller").category("LoopUnroll");

    remark::passed(locA, opts) << "A first";
    remark::passed(locB, opts) << "B";
    // The function name is not part of the identity.
    remark::passed(locA, opts.function("other")) << "A last";
  }
  EXPECT_THAT(emitted, ElementsAre("Unroller: B", "Unroller: A last"));
}

// finalize() drains the stored remarks. mlir-opt calls it explicitly and the
// engine destructor calls it again; each call emits only the remarks reported
// since the previous one, and an identity drained by one call can be reported
// again for the next.
TEST(Remark, TestRemarkFinalDrains) {
  std::vector<std::string> emitted;
  {
    MLIRContext context;
    ASSERT_TRUE(succeeded(enableFinalPolicy(context, emitted, "LoopUnroll")));
    Location loc = FileLineColLoc::get(&context, "test.cpp", 1, 5);
    auto *policy = context.getRemarkEngine()->getRemarkEmittingPolicy();
    auto first = remark::RemarkOpts::name("First").category("LoopUnroll");
    auto second = remark::RemarkOpts::name("Second").category("LoopUnroll");

    remark::passed(loc, first) << "first";
    remark::passed(loc, second) << "second";
    policy->finalize();
    EXPECT_THAT(emitted, ElementsAre("First: first", "Second: second"));

    // Nothing pending: a repeated call emits nothing.
    policy->finalize();
    EXPECT_EQ(emitted.size(), 2u);

    // A drained identity can be reported again; it waits for the next call,
    // which here is the engine destructor's.
    remark::passed(loc, first) << "first again";
    EXPECT_EQ(emitted.size(), 2u);
  }
  EXPECT_THAT(emitted, ElementsAre("First: first", "Second: second",
                                   "First: first again"));
}

// A RelatedTo link only resolves between remarks drained by the same
// finalize() call. A parent reported after its child was drained is emitted
// without the child being repeated.
TEST(Remark, TestRemarkFinalLinkAcrossDrains) {
  std::vector<std::string> emitted;
  {
    MLIRContext context;
    ASSERT_TRUE(succeeded(enableFinalPolicy(context, emitted, "LoopUnroll")));
    Location loc = FileLineColLoc::get(&context, "test.cpp", 1, 5);
    auto *policy = context.getRemarkEngine()->getRemarkEmittingPolicy();

    remark::RemarkId analysisId;
    {
      // The remark is reported when the InFlightRemark goes out of scope.
      auto analysis = remark::analysis(
          loc, remark::RemarkOpts::name("Analysis").category("LoopUnroll"));
      analysis << "trip count 128";
      analysisId = analysis.getId();
    }
    policy->finalize();
    EXPECT_THAT(emitted, ElementsAre("Analysis: trip count 128"));

    remark::passed(loc, remark::RemarkOpts::name("Unroller")
                            .category("LoopUnroll")
                            .relatedTo(analysisId))
        << "unrolled";
  }
  EXPECT_THAT(emitted,
              ElementsAre("Analysis: trip count 128", "Unroller: unrolled"));
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

// Test that the engine exposes which remark kinds and categories are enabled,
// so that external remark producers can query the filters before building
// remarks.
TEST(Remark, TestEngineFilterQueries) {
  using remark::RemarkKind;

  // No filter is set: nothing is enabled.
  {
    MLIRContext context;
    mlir::remark::RemarkCategories cats{/*all=*/std::nullopt,
                                        /*passed=*/std::nullopt,
                                        /*missed=*/std::nullopt,
                                        /*analysis=*/std::nullopt,
                                        /*failed=*/std::nullopt};
    ASSERT_TRUE(succeeded(remark::enableOptimizationRemarks(
        context, nullptr, std::make_unique<remark::RemarkEmittingPolicyAll>(),
        cats)));
    remark::detail::RemarkEngine *engine = context.getRemarkEngine();
    ASSERT_TRUE(engine);
    EXPECT_FALSE(engine->isAnyRemarkEnabled());
    EXPECT_FALSE(engine->isAnyRemarkEnabled("Vectorizer"));
    EXPECT_FALSE(engine->isPassedOptRemarkEnabled("Vectorizer"));
  }

  // Empty filter strings are equivalent to unset filters.
  {
    MLIRContext context;
    mlir::remark::RemarkCategories cats{/*all=*/"", /*passed=*/"",
                                        /*missed=*/"", /*analysis=*/"",
                                        /*failed=*/""};
    ASSERT_TRUE(succeeded(remark::enableOptimizationRemarks(
        context, nullptr, std::make_unique<remark::RemarkEmittingPolicyAll>(),
        cats)));
    remark::detail::RemarkEngine *engine = context.getRemarkEngine();
    ASSERT_TRUE(engine);
    EXPECT_FALSE(engine->isAnyRemarkEnabled());
  }

  // Per-kind filters are matched against the whole category name.
  {
    MLIRContext context;
    mlir::remark::RemarkCategories cats{/*all=*/std::nullopt,
                                        /*passed=*/"Vectorizer",
                                        /*missed=*/"Unroll|Inliner",
                                        /*analysis=*/std::nullopt,
                                        /*failed=*/std::nullopt};
    ASSERT_TRUE(succeeded(remark::enableOptimizationRemarks(
        context, nullptr, std::make_unique<remark::RemarkEmittingPolicyAll>(),
        cats)));
    remark::detail::RemarkEngine *engine = context.getRemarkEngine();
    ASSERT_TRUE(engine);

    EXPECT_TRUE(engine->isAnyRemarkEnabled());

    EXPECT_TRUE(engine->isPassedOptRemarkEnabled("Vectorizer"));
    EXPECT_FALSE(engine->isPassedOptRemarkEnabled("Vectorizer2"));
    EXPECT_FALSE(engine->isPassedOptRemarkEnabled("Unroll"));

    EXPECT_TRUE(engine->isMissedOptRemarkEnabled("Unroll"));
    EXPECT_TRUE(engine->isMissedOptRemarkEnabled("Inliner"));
    EXPECT_FALSE(engine->isMissedOptRemarkEnabled("Vectorizer"));

    EXPECT_FALSE(engine->isAnalysisOptRemarkEnabled("Vectorizer"));
    EXPECT_FALSE(engine->isFailedOptRemarkEnabled("Vectorizer"));

    EXPECT_TRUE(engine->isAnyRemarkEnabled("Vectorizer"));
    EXPECT_TRUE(engine->isAnyRemarkEnabled("Unroll"));
    EXPECT_FALSE(engine->isAnyRemarkEnabled("Register"));

    EXPECT_TRUE(
        engine->isRemarkEnabled(RemarkKind::RemarkPassed, "Vectorizer"));
    EXPECT_TRUE(engine->isRemarkEnabled(RemarkKind::RemarkMissed, "Unroll"));
    EXPECT_FALSE(
        engine->isRemarkEnabled(RemarkKind::RemarkAnalysis, "Vectorizer"));
    EXPECT_FALSE(
        engine->isRemarkEnabled(RemarkKind::RemarkFailure, "Vectorizer"));
    EXPECT_FALSE(
        engine->isRemarkEnabled(RemarkKind::RemarkUnknown, "Vectorizer"));
  }

  // The `all` filter enables every kind for the matching categories.
  {
    MLIRContext context;
    mlir::remark::RemarkCategories cats{/*all=*/"loop-.*",
                                        /*passed=*/std::nullopt,
                                        /*missed=*/"",
                                        /*analysis=*/"",
                                        /*failed=*/""};
    ASSERT_TRUE(succeeded(remark::enableOptimizationRemarks(
        context, nullptr, std::make_unique<remark::RemarkEmittingPolicyAll>(),
        cats)));
    remark::detail::RemarkEngine *engine = context.getRemarkEngine();
    ASSERT_TRUE(engine);

    EXPECT_TRUE(engine->isAnyRemarkEnabled());
    // `passed` is unset, so `all` does not enable it.
    EXPECT_FALSE(engine->isPassedOptRemarkEnabled("loop-unroll"));
    EXPECT_TRUE(engine->isMissedOptRemarkEnabled("loop-unroll"));
    EXPECT_TRUE(engine->isAnalysisOptRemarkEnabled("loop-vectorize"));
    EXPECT_TRUE(engine->isFailedOptRemarkEnabled("loop-unroll"));
    EXPECT_FALSE(engine->isMissedOptRemarkEnabled("inline"));
  }
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

class RemarkHistoryTest : public TestWithParam<bool> {
protected:
  struct Record {
    remark::RemarkId id;
    SmallVector<remark::RemarkId> related;
    std::vector<std::string> serializedLinks;
  };

  class RecordingStreamer final
      : public remark::detail::MLIRRemarkStreamerBase {
  public:
    explicit RecordingStreamer(std::vector<Record> &records)
        : records(records) {}

    void
    streamOptimizationRemark(const remark::detail::Remark &remark) override {
      Record record{remark.getId(), {}, {}};
      llvm::append_range(record.related, remark.getRelatedRemarkIds());
      for (const auto &arg : remark.generateRemark().Args)
        if (arg.Key == "RelatedTo")
          record.serializedLinks.push_back(arg.Val.str());
      std::scoped_lock<std::mutex> lock(mutex);
      records.push_back(std::move(record));
    }

  private:
    std::mutex mutex;
    std::vector<Record> &records;
  };

  void SetUp() override {
    // Also match an empty category so that RemarkOpts{} is accepted.
    remark::RemarkCategories categories;
    categories.passed = "(Opt.*)?";
    categories.missed = "(Opt.*)?";
    categories.analysis = "(Opt.*)?";
    categories.failed = "(Opt.*)?";
    std::unique_ptr<remark::detail::RemarkEmittingPolicyBase> policy;
    if (isFinal())
      policy = std::make_unique<remark::RemarkEmittingPolicyFinal>();
    else
      policy = std::make_unique<remark::RemarkEmittingPolicyAll>();
    auto streamer = std::make_unique<RecordingStreamer>(records);
    ASSERT_TRUE(succeeded(remark::enableOptimizationRemarks(
        context, std::move(streamer), std::move(policy), categories)));
    ASSERT_TRUE(context.getRemarkEngine());
    ASSERT_TRUE(engine().isPassedOptRemarkEnabled("Opt"));
    ASSERT_TRUE(engine().isAnalysisOptRemarkEnabled("Opt"));
    ASSERT_TRUE(engine().isPassedOptRemarkEnabled(""));
    ASSERT_TRUE(engine().isAnalysisOptRemarkEnabled(""));
  }

  bool isFinal() const { return GetParam(); }

  remark::detail::RemarkEngine &engine() { return *context.getRemarkEngine(); }

  remark::RemarkId emit(Location loc, remark::RemarkOpts opts) {
    auto r = remark::passed(loc, opts);
    return r.getId();
  }

  remark::RemarkId emitAnalysis(Location loc, remark::RemarkOpts opts) {
    auto r = remark::analysis(loc, opts);
    return r.getId();
  }

  remark::RemarkId emitMissed(Location loc, remark::RemarkOpts opts) {
    auto r = remark::missed(loc, opts);
    return r.getId();
  }

  remark::RemarkId emitFailed(Location loc, remark::RemarkOpts opts) {
    auto r = remark::failed(loc, opts);
    return r.getId();
  }

  void flush() { engine().getRemarkEmittingPolicy()->finalize(); }

  void expectLinks(remark::RemarkId id, ArrayRef<remark::RemarkId> expected) {
    std::vector<std::string> serialized;
    for (remark::RemarkId related : expected)
      serialized.push_back(std::to_string(related.getValue()));
    EXPECT_THAT(records, Contains(AllOf(Field(&Record::id, id),
                                        Field(&Record::related,
                                              ElementsAreArray(expected)),
                                        Field(&Record::serializedLinks,
                                              ElementsAreArray(serialized)))));
  }

  std::vector<Record> records;
  // Keep records alive while the context drains its remark engine.
  MLIRContext context;
};

TEST_P(RemarkHistoryTest, LifecycleAndCreationOrder) {
  Location loc = UnknownLoc::get(&context);
  auto opts = remark::RemarkOpts::name("Candidate").category("Opt");
  auto any = remark::RemarkQuery{};
  emit(loc, opts);
  flush();
  EXPECT_THAT(engine().findRemarks(any), IsEmpty());

  engine().enableHistory();
  remark::RemarkId firstCreated, secondCreated;
  {
    auto pending = remark::passed(loc, opts);
    firstCreated = pending.getId();
    EXPECT_THAT(engine().findRemarks(any), IsEmpty());
    secondCreated =
        emit(loc, remark::RemarkOpts::name("Second").category("Opt"));
    EXPECT_THAT(engine().findRemarks(any), ElementsAre(secondCreated));
  }
  EXPECT_LT(firstCreated.getValue(), secondCreated.getValue());
  // Query ordering follows creation IDs, even when reports arrive out of order.
  EXPECT_THAT(engine().findRemarks(any),
              ElementsAre(firstCreated, secondCreated));
  flush();
  engine().enableHistory();
  EXPECT_THAT(engine().findRemarks(any),
              ElementsAre(firstCreated, secondCreated));
  engine().clearHistory();
  EXPECT_THAT(engine().findRemarks(any), IsEmpty());
  auto afterClear = emit(loc, opts);
  flush();
  EXPECT_THAT(engine().findRemarks(any), ElementsAre(afterClear));
  EXPECT_GT(afterClear.getValue(), secondCreated.getValue());

  engine().enableHistory(false);
  EXPECT_THAT(engine().findRemarks(any), IsEmpty());
  emit(loc, opts);
  flush();
  engine().clearHistory();
  emit(loc, opts);
  flush();
  EXPECT_THAT(engine().findRemarks(any), IsEmpty());
  engine().enableHistory();
  EXPECT_THAT(engine().findRemarks(any), IsEmpty());
  auto afterEnable = emit(loc, opts);
  flush();
  EXPECT_THAT(engine().findRemarks(any), ElementsAre(afterEnable));
  EXPECT_GT(afterEnable.getValue(), afterClear.getValue());

  EXPECT_FALSE(static_cast<bool>(
      emit(loc, remark::RemarkOpts::name("Filtered").category("Unselected"))));
  EXPECT_THAT(engine().findRemarks(any), ElementsAre(afterEnable));
}

TEST_P(RemarkHistoryTest, ExactFieldsAndConjunction) {
  engine().enableHistory();
  Location loc = FileLineColLoc::get(&context, "input.mlir", 7, 2);
  Location otherLoc = NameLoc::get(StringAttr::get(&context, "wrapped"), loc);
  auto opts = remark::RemarkOpts::name("Candidate")
                  .category("Opt")
                  .subCategory("Vector")
                  .function("kernel");
  auto target = emit(loc, opts);
  auto otherSub = emit(loc, opts.subCategory("Unroll"));
  auto analysis = emitAnalysis(loc, opts);
  auto other = emit(otherLoc, remark::RemarkOpts::name("Other")
                                  .category("OptOther")
                                  .subCategory("Other")
                                  .function("helper"));
  auto empty = emit(UnknownLoc::get(&context), remark::RemarkOpts{});
  auto any = remark::RemarkQuery{};
  flush();

  EXPECT_THAT(engine().findRemarks(any.name("Candidate")),
              ElementsAre(target, otherSub, analysis));
  EXPECT_THAT(engine().findRemarks(any.category("Opt")),
              ElementsAre(target, otherSub, analysis));
  EXPECT_THAT(engine().findRemarks(any.subCategory("Vector")),
              ElementsAre(target, analysis));
  EXPECT_THAT(engine().findRemarks(any.function("kernel")),
              ElementsAre(target, otherSub, analysis));
  EXPECT_THAT(engine().findRemarks(any.kind(remark::RemarkKind::RemarkPassed)),
              ElementsAre(target, otherSub, other, empty));
  EXPECT_THAT(engine().findRemarks(any.at(loc)),
              ElementsAre(target, otherSub, analysis));
  EXPECT_THAT(engine().findRemarks(any.at(otherLoc)), ElementsAre(other));

  auto exact = any.name("Candidate")
                   .category("Opt")
                   .subCategory("Vector")
                   .function("kernel")
                   .kind(remark::RemarkKind::RemarkPassed)
                   .at(loc);
  EXPECT_THAT(engine().findRemarks(exact), ElementsAre(target));
  EXPECT_THAT(engine().findRemarks(exact.function("helper")), IsEmpty());
  EXPECT_THAT(engine().findRemarks(any.category("Opt.*")), IsEmpty());
  EXPECT_THAT(engine().findRemarks(any.category("Opt:Vector")), IsEmpty());
  EXPECT_THAT(engine().findRemarks(any.name("")), ElementsAre(empty));
  EXPECT_THAT(engine().findRemarks(any.category("")), ElementsAre(empty));
  EXPECT_THAT(engine().findRemarks(any.subCategory("")), ElementsAre(empty));
  EXPECT_THAT(engine().findRemarks(any.function("")), ElementsAre(empty));
  EXPECT_THAT(engine().findRemarks(any.name("<unknown remark name>")),
              IsEmpty());
  EXPECT_THAT(engine().findRemarks(any.function("<unknown function>")),
              IsEmpty());

  // The existing options overload treats empty fields as wildcards.
  EXPECT_THAT(engine().findRemarks(remark::RemarkOpts{}),
              ElementsAre(target, otherSub, analysis, other, empty));
  EXPECT_THAT(engine().findRemarks(opts, remark::RemarkKind::RemarkAnalysis),
              ElementsAre(analysis));
  EXPECT_THAT(engine().findRemarks(opts.relatedTo(other)),
              ElementsAre(target, analysis));
  EXPECT_THAT(engine().findRemarks(remark::RemarkOpts{},
                                   remark::RemarkKind::RemarkAnalysis),
              ElementsAre(analysis));
  EXPECT_THAT(engine().findRemarks(any.kind(remark::RemarkKind::RemarkUnknown)),
              IsEmpty());
}

TEST_P(RemarkHistoryTest, OwnsQueryAndHistoryStrings) {
  engine().enableHistory();
  Location loc = UnknownLoc::get(&context);
  remark::RemarkQuery query;
  remark::RemarkId target;
  {
    std::string name = "Candidate", category = "Opt", sub = "Vector";
    std::string function = "kernel";
    query = remark::RemarkQuery::named(name)
                .category(category)
                .subCategory(sub)
                .function(function);
    target = emit(loc, remark::RemarkOpts::name(name)
                           .category(category)
                           .subCategory(sub)
                           .function(function));
    name.assign("Changed");
    category.assign("Changed");
    sub.assign("Changed");
    function.assign("Changed");
  }
  flush();
  EXPECT_THAT(engine().findRemarks(query), ElementsAre(target));
  const auto original = query;
  auto changed = original.name("Other");
  EXPECT_THAT(engine().findRemarks(changed), IsEmpty());
  EXPECT_THAT(engine().findRemarks(original), ElementsAre(target));

  // Saved queries remain usable after the interned pool resets; IDs are not
  // reset.
  for (bool disable : {false, true}) {
    if (disable) {
      engine().enableHistory(false);
      engine().enableHistory();
    } else {
      engine().clearHistory();
    }
    EXPECT_THAT(engine().findRemarks(query), IsEmpty());
    auto next = emit(loc, remark::RemarkOpts::name("Candidate")
                              .category("Opt")
                              .subCategory("Vector")
                              .function("kernel"));
    flush();
    EXPECT_THAT(engine().findRemarks(original), ElementsAre(next));
    EXPECT_GT(next.getValue(), target.getValue());
  }
}

TEST_P(RemarkHistoryTest, QueryLinksSnapshotAndSerialization) {
  engine().enableHistory();
  Location loc = UnknownLoc::get(&context);
  auto query = remark::RemarkQuery{}.kind(remark::RemarkKind::RemarkAnalysis);
  auto first =
      emitAnalysis(loc, remark::RemarkOpts::name("First").category("Opt"));
  auto second =
      emitAnalysis(loc, remark::RemarkOpts::name("Second").category("Opt"));
  flush();
  remark::RemarkId linked, later, pendingId, noMatches, copiedWrapper;
  {
    auto pending = remark::analysis(
        loc, remark::RemarkOpts::name("Pending").category("Opt"));
    pendingId = pending.getId();
    {
      auto r = remark::passed(
          loc, remark::RemarkOpts::name("Linked").category("Opt"));
      linked = r.getId();
      r << remark::relatedTo(query);
      // This report occurs before Linked is reported, but after its query.
      later =
          emitAnalysis(loc, remark::RemarkOpts::name("Later").category("Opt"));
    }
  }
  flush();
  {
    const auto wrapper = remark::relatedTo(query);
    auto r = remark::passed(
        loc, remark::RemarkOpts::name("CopiedWrapper").category("Opt"));
    copiedWrapper = r.getId();
    r << wrapper;
  }
  {
    auto r = remark::passed(
        loc, remark::RemarkOpts::name("NoMatches").category("Opt"));
    noMatches = r.getId();
    r << remark::relatedTo(query.name("Absent"));
  }
  flush();

  expectLinks(linked, {first, second});
  expectLinks(copiedWrapper, {first, second, pendingId, later});
  expectLinks(noMatches, {});
}

TEST_P(RemarkHistoryTest, QueryLinksDoNotEnableHistory) {
  Location loc = UnknownLoc::get(&context);
  auto query = remark::RemarkQuery{};
  emit(loc, remark::RemarkOpts::name("First").category("Opt"));
  remark::RemarkId linked;
  {
    auto r =
        remark::passed(loc, remark::RemarkOpts::name("Linked").category("Opt"));
    linked = r.getId();
    auto wrapper = remark::relatedTo(query);
    r << wrapper;
  }
  // Both filtered and ownerless remarks accept the lazy wrapper harmlessly.
  remark::passed(loc, remark::RemarkOpts::name("Filtered").category("Other"))
      << remark::relatedTo(query);
  remark::detail::InFlightRemark{} << remark::relatedTo(query);
  EXPECT_THAT(engine().findRemarks(query), IsEmpty());
  flush();
  engine().enableHistory();
  EXPECT_THAT(engine().findRemarks(query), IsEmpty());
  EXPECT_THAT(records, Contains(AllOf(Field(&Record::id, linked),
                                      Field(&Record::related, IsEmpty()))));
}

TEST_P(RemarkHistoryTest, HistoryControlsPreserveFinalPendingRemarks) {
  Location loc = UnknownLoc::get(&context);
  auto query = remark::RemarkQuery{};
  auto beforeEnable =
      emit(loc, remark::RemarkOpts::name("BeforeEnable").category("Opt"));
  EXPECT_THAT(engine().findRemarks(query), IsEmpty());
  engine().enableHistory();
  if (isFinal())
    EXPECT_THAT(engine().findRemarks(query), ElementsAre(beforeEnable));
  else
    EXPECT_THAT(engine().findRemarks(query), IsEmpty());
  flush();

  auto beforeClear =
      emit(loc, remark::RemarkOpts::name("BeforeClear").category("Opt"));
  engine().clearHistory();
  if (isFinal())
    EXPECT_THAT(engine().findRemarks(query), ElementsAre(beforeClear));
  else
    EXPECT_THAT(engine().findRemarks(query), IsEmpty());
  flush();

  auto beforeDisable =
      emit(loc, remark::RemarkOpts::name("BeforeDisable").category("Opt"));
  engine().enableHistory(false);
  EXPECT_THAT(engine().findRemarks(query), IsEmpty());
  engine().enableHistory();
  if (isFinal())
    EXPECT_THAT(engine().findRemarks(query), ElementsAre(beforeDisable));
  else
    EXPECT_THAT(engine().findRemarks(query), IsEmpty());
  engine().enableHistory(false);
  flush();
  engine().enableHistory();
  EXPECT_THAT(engine().findRemarks(query), IsEmpty());
}

TEST_P(RemarkHistoryTest, SameKeyQueriesLinkPendingAndEmittedTargets) {
  engine().enableHistory();
  Location loc = UnknownLoc::get(&context);
  auto opts = remark::RemarkOpts::name("Candidate").category("Opt");
  auto query = remark::RemarkQuery::named("Candidate").category("Opt");
  auto first = emit(loc, opts);
  auto second = emit(loc, opts);
  if (isFinal())
    EXPECT_THAT(engine().findRemarks(query), ElementsAre(second));
  else
    EXPECT_THAT(engine().findRemarks(query), ElementsAre(first, second));

  remark::RemarkId beforeFlush;
  {
    auto r = remark::passed(
        loc, remark::RemarkOpts::name("BeforeFlush").category("Opt"));
    beforeFlush = r.getId();
    r << remark::relatedTo(query);
  }
  flush();
  if (isFinal()) {
    EXPECT_THAT(engine().findRemarks(query), ElementsAre(second));
    EXPECT_THAT(records, Not(Contains(Field(&Record::id, first))));
    expectLinks(beforeFlush, {second});
  } else {
    EXPECT_THAT(engine().findRemarks(query), ElementsAre(first, second));
    expectLinks(beforeFlush, {first, second});
  }

  // The same identity can link to an earlier batch without suppressing it.
  remark::RemarkId sameKey;
  {
    auto r = remark::passed(loc, opts);
    sameKey = r.getId();
    r << remark::relatedTo(query);
  }
  flush();
  if (isFinal())
    expectLinks(sameKey, {second});
  else
    expectLinks(sameKey, {first, second});

  // A pending FINAL replacement must not become a dangling query target.
  auto replaced = emit(loc, opts);
  remark::RemarkId consumer;
  {
    auto r = remark::passed(
        loc, remark::RemarkOpts::name("Consumer").category("Opt"));
    consumer = r.getId();
    r << remark::relatedTo(query);
    emit(loc, opts);
  }
  flush();
  if (isFinal()) {
    expectLinks(consumer, {second, sameKey});
    EXPECT_THAT(records, Not(Contains(Field(&Record::id, replaced))));
  } else {
    expectLinks(consumer, {first, second, sameKey, replaced});
  }
  for (const Record &record : records)
    for (remark::RemarkId related : record.related)
      EXPECT_THAT(records, Contains(Field(&Record::id, related)));
}

// Querying before replacement captures r1, but FINAL must remove that link
// when r2 supersedes r1. It must not redirect the link to r2.
TEST_P(RemarkHistoryTest, QueryBeforeReplacementPrunesDiscardedTarget) {
  engine().enableHistory();
  Location loc = UnknownLoc::get(&context);
  auto opts = remark::RemarkOpts::name("Candidate").category("Opt");
  auto query = remark::RemarkQuery::named("Candidate").at(loc);
  auto r1 = emitMissed(loc, opts);
  EXPECT_THAT(engine().findRemarks(query), ElementsAre(r1));
  remark::RemarkId r3;
  {
    auto r = remark::missed(
        loc, remark::RemarkOpts::name("Consumer").category("Opt"));
    r3 = r.getId();
    r << remark::relatedTo(query);
  }
  auto r2 = emitMissed(loc, opts);
  flush();
  if (isFinal()) {
    EXPECT_THAT(records,
                ElementsAre(Field(&Record::id, r3), Field(&Record::id, r2)));
    expectLinks(r3, {});
    EXPECT_THAT(engine().findRemarks(query), ElementsAre(r2));
  } else {
    expectLinks(r3, {r1});
    EXPECT_THAT(engine().findRemarks(query), ElementsAre(r1, r2));
  }
}

TEST_P(RemarkHistoryTest, QueryAfterReplacementLinksSurvivingTarget) {
  engine().enableHistory();
  Location loc = UnknownLoc::get(&context);
  auto opts = remark::RemarkOpts::name("Candidate").category("Opt");
  auto r1 = emitMissed(loc, opts);
  auto r2 = emitMissed(loc, opts);
  auto query = remark::RemarkQuery::named("Candidate").at(loc);
  if (isFinal())
    EXPECT_THAT(engine().findRemarks(query), ElementsAre(r2));
  else
    EXPECT_THAT(engine().findRemarks(query), ElementsAre(r1, r2));
  remark::RemarkId r3;
  {
    auto r = remark::missed(
        loc, remark::RemarkOpts::name("Consumer").category("Opt"));
    r3 = r.getId();
    r << remark::relatedTo(query);
  }
  flush();
  if (isFinal()) {
    EXPECT_THAT(records,
                ElementsAre(Field(&Record::id, r3), Field(&Record::id, r2)));
    expectLinks(r3, {r2});
  } else {
    expectLinks(r3, {r1, r2});
  }
}

TEST_P(RemarkHistoryTest, QueryLinksBeforeFinalization) {
  engine().enableHistory();
  Location loc = UnknownLoc::get(&context);
  auto target =
      emitMissed(loc, remark::RemarkOpts::name("Candidate").category("Opt"));
  remark::RemarkId consumer;
  {
    auto r = remark::missed(
        loc, remark::RemarkOpts::name("Consumer").category("Opt"));
    consumer = r.getId();
    r << remark::relatedTo(remark::RemarkQuery::named("Candidate"));
  }
  if (isFinal())
    EXPECT_THAT(records, IsEmpty());
  else
    EXPECT_EQ(records.size(), 2u);
  flush();
  expectLinks(consumer, {target});
  EXPECT_EQ(records.size(), 2u);
}

// Final only prunes references to targets superseded in the consumer's batch.
// Keeping a consumer in flight across finalization moves its report into the
// next batch, after the policy has discarded the previous batch's bookkeeping.
TEST_P(RemarkHistoryTest, InFlightConsumerOutlivesDiscardedTargetBatch) {
  engine().enableHistory();
  Location loc = UnknownLoc::get(&context);
  auto opts = remark::RemarkOpts::name("Candidate").category("Opt");
  auto query = remark::RemarkQuery::named("Candidate").at(loc);
  auto target = emitMissed(loc, opts);
  remark::RemarkId consumer, replacement;
  {
    auto r = remark::missed(
        loc, remark::RemarkOpts::name("Consumer").category("Opt"));
    consumer = r.getId();
    r << remark::relatedTo(query);
    replacement = emitMissed(loc, opts);
    flush();
    EXPECT_THAT(records, Not(Contains(Field(&Record::id, consumer))));
    if (isFinal()) {
      EXPECT_THAT(records, ElementsAre(Field(&Record::id, replacement)));
      EXPECT_THAT(engine().findRemarks(query), ElementsAre(replacement));
    } else {
      EXPECT_THAT(engine().findRemarks(query),
                  ElementsAre(target, replacement));
    }
  }
  flush();
  // Final no longer knows that the previous batch superseded this target.
  // The captured link remains unchanged, including in serialized arguments.
  expectLinks(consumer, {target});
  if (isFinal())
    EXPECT_THAT(records, ElementsAre(Field(&Record::id, replacement),
                                     Field(&Record::id, consumer)));
  else
    EXPECT_EQ(records.size(), 3u);
}

TEST_P(RemarkHistoryTest, EarlierBatchLinksSurviveHistoryReset) {
  Location loc = UnknownLoc::get(&context);
  for (bool disable : {false, true}) {
    SCOPED_TRACE(disable);
    // Earlier-batch links remain intact even when history was disabled.
    engine().enableHistory(!disable);
    auto target = emit(loc, remark::RemarkOpts::name("Target").category("Opt"));
    flush();
    if (disable)
      engine().enableHistory();
    else
      engine().clearHistory();
    EXPECT_THAT(engine().findRemarks(remark::RemarkQuery{}), IsEmpty());
    engine().enableHistory(false);
    auto consumer = emit(
        loc,
        remark::RemarkOpts::name("Consumer").category("Opt").relatedTo(target));
    flush();
    expectLinks(consumer, {target});
    EXPECT_EQ(llvm::count_if(records,
                             [&](const Record &r) { return r.id == target; }),
              1);
  }
}

TEST_P(RemarkHistoryTest, SharedTargetsEmitOnceAndQueriesUseCreationOrder) {
  engine().enableHistory();
  Location loc = UnknownLoc::get(&context);
  auto child =
      emitAnalysis(loc, remark::RemarkOpts::name("Child").category("Opt"));
  auto firstParent = emit(
      loc,
      remark::RemarkOpts::name("FirstParent").category("Opt").relatedTo(child));
  auto secondParent = emitAnalysis(loc, remark::RemarkOpts::name("SecondParent")
                                            .category("Opt")
                                            .relatedTo(child));
  flush();
  auto query = remark::RemarkQuery{};
  EXPECT_THAT(engine().findRemarks(query),
              ElementsAre(child, firstParent, secondParent));
  EXPECT_EQ(
      llvm::count_if(records, [&](const Record &r) { return r.id == child; }),
      1);
  EXPECT_EQ(records.size(), 3u);
  expectLinks(firstParent, {child});
  expectLinks(secondParent, {child});

  // Grouping can emit a parent first; queries still return creation order.
  auto analyses = query.kind(remark::RemarkKind::RemarkAnalysis);
  EXPECT_THAT(engine().findRemarks(analyses), ElementsAre(child, secondParent));
  remark::RemarkId linked;
  {
    auto r =
        remark::passed(loc, remark::RemarkOpts::name("Linked").category("Opt"));
    linked = r.getId();
    r << remark::relatedTo(analyses);
  }
  flush();
  expectLinks(linked, {child, secondParent});
}

// The consumer shares the Final identity of the pending target it queries, so
// its own report replaces that target and the link is pruned; the link is not
// redirected to the consumer.
TEST_P(RemarkHistoryTest, ConsumerReplacingItsQueryTargetLosesTheLink) {
  engine().enableHistory();
  Location loc = UnknownLoc::get(&context);
  auto opts = remark::RemarkOpts::name("Candidate").category("Opt");
  auto target = emitMissed(loc, opts);
  remark::RemarkId consumer;
  {
    auto r = remark::missed(loc, opts);
    consumer = r.getId();
    r << remark::relatedTo(remark::RemarkQuery::named("Candidate").at(loc));
  }
  flush();
  if (isFinal()) {
    EXPECT_THAT(records, ElementsAre(Field(&Record::id, consumer)));
    expectLinks(consumer, {});
    EXPECT_THAT(engine().findRemarks(remark::RemarkQuery::named("Candidate")),
                ElementsAre(consumer));
  } else {
    EXPECT_EQ(records.size(), 2u);
    expectLinks(consumer, {target});
  }
}

// function() narrows a query, but the function is not part of the Final
// identity: a same-identity report from another function replaces the matched
// target and the link is pruned.
TEST_P(RemarkHistoryTest, FunctionScopedQueryTargetReplacedByOtherFunction) {
  engine().enableHistory();
  Location loc = UnknownLoc::get(&context);
  auto opts = remark::RemarkOpts::name("Candidate").category("Opt");
  auto kernelTarget = emit(loc, opts.function("kernel"));
  remark::RemarkId consumer;
  {
    auto r = remark::passed(
        loc, remark::RemarkOpts::name("Consumer").category("Opt"));
    consumer = r.getId();
    r << remark::relatedTo(
        remark::RemarkQuery::named("Candidate").function("kernel"));
  }
  auto helperTarget = emit(loc, opts.function("helper"));
  flush();
  if (isFinal()) {
    EXPECT_THAT(records, Not(Contains(Field(&Record::id, kernelTarget))));
    EXPECT_THAT(records, Contains(Field(&Record::id, helperTarget)));
    expectLinks(consumer, {});
  } else {
    expectLinks(consumer, {kernelTarget});
  }
}

// An explicit link and query matches for the same target produce one link.
TEST_P(RemarkHistoryTest, ExplicitAndQueryLinksToSameTargetAreLinkedOnce) {
  engine().enableHistory();
  Location loc = UnknownLoc::get(&context);
  auto target =
      emitAnalysis(loc, remark::RemarkOpts::name("Target").category("Opt"));
  auto query = remark::RemarkQuery::named("Target");
  remark::RemarkId consumer;
  {
    auto r = remark::passed(
        loc,
        remark::RemarkOpts::name("Consumer").category("Opt").relatedTo(target));
    consumer = r.getId();
    r << remark::relatedTo(query) << remark::relatedTo(query);
  }
  flush();
  expectLinks(consumer, {target});
}

// A Failure and a Passed remark at the same site have different Final
// identities: both are searchable, both can be linked, and both are emitted.
TEST_P(RemarkHistoryTest, FailedAndPassedAtOneSiteAreDistinctIdentities) {
  engine().enableHistory();
  Location loc = UnknownLoc::get(&context);
  auto opts = remark::RemarkOpts::name("Candidate").category("Opt");
  auto failed = emitFailed(loc, opts);
  auto passed = emit(loc, opts);
  auto query = remark::RemarkQuery::named("Candidate").at(loc);
  EXPECT_THAT(engine().findRemarks(query), ElementsAre(failed, passed));
  EXPECT_THAT(
      engine().findRemarks(query.kind(remark::RemarkKind::RemarkFailure)),
      ElementsAre(failed));
  remark::RemarkId consumer;
  {
    auto r = remark::passed(
        loc, remark::RemarkOpts::name("Consumer").category("Opt"));
    consumer = r.getId();
    r << remark::relatedTo(query);
  }
  flush();
  expectLinks(consumer, {failed, passed});
  EXPECT_EQ(records.size(), 3u);
}

#if LLVM_ENABLE_THREADS
// Concurrent reports, query-based links, and queries with history enabled.
TEST_P(RemarkHistoryTest, ConcurrentReportsAndQueries) {
  engine().enableHistory();
  Location loc = UnknownLoc::get(&context);
  constexpr unsigned numThreads = 4, perThread = 50;
  std::vector<std::thread> threads;
  for (unsigned t = 0; t < numThreads; ++t)
    threads.emplace_back([&, t] {
      std::string function = "f" + std::to_string(t);
      for (unsigned i = 0; i < perThread; ++i) {
        // Distinct names: Final identity ignores the function, so same-named
        // remarks at one location would replace each other across threads.
        std::string name = function + "_" + std::to_string(i);
        remark::passed(
            loc,
            remark::RemarkOpts::name(name).category("Opt").function(function))
            << remark::relatedTo(remark::RemarkQuery{}.function(function));
        (void)engine().findRemarks(remark::RemarkQuery{});
      }
    });
  for (std::thread &thread : threads)
    thread.join();
  flush();
  ASSERT_EQ(records.size(), size_t{numThreads * perThread});
  llvm::DenseSet<uint64_t> ids;
  for (const Record &record : records)
    ids.insert(record.id.getValue());
  for (const Record &record : records)
    for (remark::RemarkId related : record.related)
      EXPECT_TRUE(ids.contains(related.getValue()));
}
#endif

INSTANTIATE_TEST_SUITE_P(EmittingPolicies, RemarkHistoryTest, Bool(),
                         [](const TestParamInfo<bool> &info) {
                           return info.param ? "Final" : "All";
                         });

TEST(Remark, RawMetadataAccessorsPreserveEmptyFields) {
  MLIRContext context;
  Location loc = UnknownLoc::get(&context);
  remark::detail::OptRemarkPass empty(loc, remark::RemarkOpts{});
  EXPECT_TRUE(empty.getRawRemarkName().empty());
  EXPECT_TRUE(empty.getRawFunction().empty());
  EXPECT_TRUE(empty.getSubCategoryName().empty());
  EXPECT_EQ(empty.getRemarkName(), "<unknown remark name>");
  EXPECT_EQ(empty.getFunction(), "<unknown function>");

  remark::detail::OptRemarkPass populated(loc,
                                          remark::RemarkOpts::name("Candidate")
                                              .category("Opt")
                                              .subCategory("Vector")
                                              .function("kernel"));
  EXPECT_EQ(populated.getRawRemarkName(), "Candidate");
  EXPECT_EQ(populated.getRawFunction(), "kernel");
  EXPECT_EQ(populated.getCategoryName(), "Opt");
  EXPECT_EQ(populated.getSubCategoryName(), "Vector");
  EXPECT_EQ(populated.getCombinedCategoryName(), "Opt:Vector");
}

TEST(Remark, TestNestedLocation) {
  MLIRContext context;
  Location fileLoc = FileLineColLoc::get(&context, "test.mlir", 42, 7);
  Location otherLoc = FileLineColLoc::get(&context, "other.mlir", 9, 3);
  Location nameLoc = NameLoc::get(StringAttr::get(&context, "name"), fileLoc);
  Location fusedLoc = FusedLoc::get(&context, {nameLoc, otherLoc});
  Location callSiteLoc = CallSiteLoc::get(nameLoc, otherLoc);

  std::pair<const char *, Location> cases[] = {{"NameLoc", nameLoc},
                                               {"FusedLoc", fusedLoc},
                                               {"CallSiteLoc", callSiteLoc}};
  for (auto [kind, loc] : cases) {
    SCOPED_TRACE(kind);
    remark::detail::OptRemarkPass remark(loc, remark::RemarkOpts::name("R"));

    llvm::remarks::Remark llvmRemark = remark.generateRemark();
    ASSERT_TRUE(llvmRemark.Loc.has_value());
    EXPECT_EQ(llvmRemark.Loc->SourceFilePath, "test.mlir");
    EXPECT_EQ(llvmRemark.Loc->SourceLine, 42u);
    EXPECT_EQ(llvmRemark.Loc->SourceColumn, 7u);

    std::string printed;
    llvm::raw_string_ostream os(printed);
    remark.print(os, /*printLocation=*/true);
    EXPECT_THAT(printed, HasSubstr(" @\"test.mlir\":42:7"));
  }

  // Without a file location inside, the remark has no source location.
  remark::detail::OptRemarkPass remark(
      NameLoc::get(StringAttr::get(&context, "name")),
      remark::RemarkOpts::name("R"));
  EXPECT_EQ(remark.generateRemark().Loc->SourceFilePath, "<unknown file>");
  std::string printed;
  llvm::raw_string_ostream os(printed);
  remark.print(os, /*printLocation=*/true);
  EXPECT_THAT(printed, Not(HasSubstr(" @")));
}

TEST(Remark, TestLLVMStreamerErrors) {
  llvm::SmallString<64> dir;
  ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory("remarks", dir));
  llvm::scope_exit removeDir([&] { llvm::sys::fs::remove_directories(dir); });
  llvm::SmallString<64> missingPath(dir);
  llvm::sys::path::append(missingPath, "missing", "out.yaml");
  llvm::SmallString<64> filePath(dir);
  llvm::sys::path::append(filePath, "out.yaml");
  std::string noSuchFile =
      std::make_error_code(std::errc::no_such_file_or_directory).message();

  // A file that cannot be opened.
  auto missing = remark::detail::LLVMRemarkStreamer::createToFile(
      missingPath, llvm::remarks::Format::YAML);
  ASSERT_FALSE(static_cast<bool>(missing));
  EXPECT_EQ(llvm::toString(missing.takeError()), "cannot open output file '" +
                                                     missingPath.str().str() +
                                                     "': " + noSuchFile);

  // A serializer that cannot be created: the file is not left behind.
  auto badFormat = remark::detail::LLVMRemarkStreamer::createToFile(
      filePath, llvm::remarks::Format::Unknown);
  ASSERT_FALSE(static_cast<bool>(badFormat));
  EXPECT_THAT(llvm::toString(badFormat.takeError()),
              StartsWith("'" + filePath.str().str() + "': "));
  EXPECT_FALSE(llvm::sys::fs::exists(filePath));
}
} // namespace
