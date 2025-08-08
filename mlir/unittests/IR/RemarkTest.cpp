//===- RemarkTest.cpp - Remark unit tests -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Remarks.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
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

  auto *context = new MLIRContext();

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
  Location loc = UnknownLoc::get(context);

  // Setup the remark engine
  context->setupOptimizationRemarks(yamlFile, "yaml", true, categoryLoopunroll,
                                    std::nullopt, std::nullopt,
                                    categoryLoopunroll);

  // Remark 1: pass, category LoopUnroll
  reportOptimizationPass(loc, categoryLoopunroll, myPassname1) << pass1Msg;
  // Remark 2: failure, category LoopUnroll
  reportOptimizationFail(loc, categoryLoopunroll, myPassname2) << pass2Msg;
  // Remark 3: pass, category Inline (should not be printed)
  reportOptimizationPass(loc, categoryInline, myPassname1) << pass3Msg;

  delete context;

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
  auto *context = new MLIRContext();

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

  Location loc = UnknownLoc::get(context);
  reportOptimizationFail(loc, categoryFailName, myPassname1) << pass1Msg;

  delete context;

  // No setup, so no output file should be created
  // check!
  bool fileExists = llvm::sys::fs::exists(yamlFile);
  EXPECT_FALSE(fileExists)
      << "Expected no YAML file to be created without setupOptimizationRemarks";
}

} // namespace
