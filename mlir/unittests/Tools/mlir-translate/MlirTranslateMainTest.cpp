//===- MlirTranslateMainTest.cpp - mlir-translate tests -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace mlir;

namespace {
TEST(MlirTranslateMainTest, RunsTranslationsWithoutCommandLineParsing) {
  llvm::cl::opt<std::string> existingOutputOption("o");
  (void)existingOutputOption;

  auto append = [](StringRef suffix) {
    return Translation(
        [suffix](const std::shared_ptr<llvm::SourceMgr> &sourceMgr,
                 llvm::raw_ostream &output, MLIRContext *) {
          const llvm::MemoryBuffer *input =
              sourceMgr->getMemoryBuffer(sourceMgr->getMainFileID());
          output << input->getBuffer() << suffix;
          return success();
        },
        "append text", std::nullopt);
  };

  Translation first = append(" first");
  Translation second = append(" second");
  const Translation *translations[] = {&first, &second};
  std::string output;
  llvm::raw_string_ostream outputStream(output);

  EXPECT_TRUE(succeeded(mlirTranslateMain(
      llvm::MemoryBuffer::getMemBuffer("input"), outputStream, translations)));
  EXPECT_EQ(output, "input first second");
}
} // namespace
