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
#include "gtest/gtest.h"
#include <optional>

using namespace llvm;
using namespace mlir;
using namespace mlir::detail;

namespace {

TEST(Remark, TestOutputOptimizationRemark) {
  const auto *pass1Msg = "My message";
  const auto *pass2Msg = "My another message";
  const auto *pass3Msg = "Do not show this message";

  std::string categoryLoopunroll("LoopUnroll");
  std::string categoryInline("Inliner");
  std::string myPassname1("myPass1");
  std::string myPassname2("myPass2");
  std::string funcName("myFunc");
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
    mlir::MLIRContext::RemarkCategories cats{/*passed=*/categoryLoopunroll,
                                             /*missed=*/std::nullopt,
                                             /*analysis=*/std::nullopt,
                                             /*failed=*/categoryLoopunroll};

    LogicalResult isEnabled = mlir::remark::enableOptimizationRemarksToFile(
        context, yamlFile, llvm::remarks::Format::YAML, cats);
    ASSERT_TRUE(succeeded(isEnabled)) << "Failed to enable remark engine";

    // Remark 1: pass, category LoopUnroll
    reportOptimizationPass(loc, categoryLoopunroll, myPassname1) << pass1Msg;
    // Remark 2: failure, category LoopUnroll
    reportOptimizationFail(loc, categoryLoopunroll, myPassname2) << pass2Msg;
    // Remark 3: pass, category Inline (should not be printed)
    reportOptimizationPass(loc, categoryInline, myPassname1) << pass3Msg;
  }

  // Read the file
  auto bufferOrErr = MemoryBuffer::getFile(yamlFile);
  ASSERT_TRUE(static_cast<bool>(bufferOrErr)) << "Failed to open remarks file";
  std::string content = bufferOrErr.get()->getBuffer().str();

  // Remark 1: pass, should be printed
  EXPECT_NE(content.find("--- !Passed"), std::string::npos);
  EXPECT_NE(content.find("Pass:            " + categoryLoopunroll),
            std::string::npos);
  EXPECT_NE(content.find("Name:            " + myPassname1), std::string::npos);
  EXPECT_NE(content.find("String:          " + std::string(pass1Msg)),
            std::string::npos);

  // Remark 2: failure, should be printed
  EXPECT_NE(content.find("--- !Failure"), std::string::npos);
  EXPECT_NE(content.find("Name:            " + myPassname2), std::string::npos);
  EXPECT_NE(content.find("String:          " + std::string(pass2Msg)),
            std::string::npos);

  // Remark 3: pass, category Inline (should not be printed)
  EXPECT_EQ(content.find("String:          " + std::string(pass3Msg)),
            std::string::npos);
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
    reportOptimizationFail(loc, categoryFailName, myPassname1) << pass1Msg;
  }
  // No setup, so no output file should be created
  // check!
  bool fileExists = llvm::sys::fs::exists(yamlFile);
  EXPECT_FALSE(fileExists)
      << "Expected no YAML file to be created without setupOptimizationRemarks";
}

TEST(Remark, TestOutputOptimizationRemarkDiagnostic) {
  const auto *pass1Msg = "My message";

  std::string categoryLoopunroll("LoopUnroll");
  std::string myPassname1("myPass1");
  std::string funcName("myFunc");

  std::string seenMsg = "";
  std::string expectedMsg = "Passed\nLoopUnroll:myPass1\n func=<unknown "
                            "function>\n {\nString=My message\n\n}";
  {
    MLIRContext context;
    Location loc = UnknownLoc::get(&context);

    context.printOpOnDiagnostic(true);
    context.printStackTraceOnDiagnostic(true);

    // Register a handler that captures the diagnostic.
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diag) {
      seenMsg = diag.str();
      return success();
    });

    // Setup the remark engine
    mlir::MLIRContext::RemarkCategories cats{/*passed=*/categoryLoopunroll,
                                             /*missed=*/std::nullopt,
                                             /*analysis=*/std::nullopt,
                                             /*failed=*/categoryLoopunroll};

    LogicalResult isEnabled =
        context.enableOptimizationRemarks(nullptr, cats, true);
    ASSERT_TRUE(succeeded(isEnabled)) << "Failed to enable remark engine";

    // Remark 1: pass, category LoopUnroll
    reportOptimizationPass(loc, categoryLoopunroll, myPassname1) << pass1Msg;
  }
  EXPECT_EQ(seenMsg, expectedMsg);
}

/// Custom remark streamer that prints remarks to stderr.
class MyCustomStreamer : public MLIRRemarkStreamerBase {
public:
  MyCustomStreamer() = default;

  void streamOptimizationRemark(const Remark &remark) override {
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
    mlir::MLIRContext::RemarkCategories cats{/*passed=*/categoryLoopunroll,
                                             /*missed=*/std::nullopt,
                                             /*analysis=*/std::nullopt,
                                             /*failed=*/categoryLoopunroll};

    LogicalResult isEnabled = context.enableOptimizationRemarks(
        std::make_unique<MyCustomStreamer>(), cats, true);
    ASSERT_TRUE(succeeded(isEnabled)) << "Failed to enable remark engine";

    // Remark 1: pass, category LoopUnroll
    reportOptimizationPass(loc, categoryLoopunroll, myPassname1) << pass1Msg;
    // Remark 2: failure, category LoopUnroll
    reportOptimizationFail(loc, categoryLoopunroll, myPassname2) << pass2Msg;
    // Remark 3: pass, category Inline (should not be printed)
    reportOptimizationPass(loc, categoryInline, myPassname1) << pass3Msg;
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
