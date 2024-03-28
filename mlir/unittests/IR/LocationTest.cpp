//===- LocationTest.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Location.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace mlir;

TEST(LocationTest, FileRange) {
  MLIRContext context;
  std::string text = R"(
    Secti<on o>f text where the error is reported from.
  )";
  std::string fileName = "file.mlir";
  std::unique_ptr<llvm::MemoryBuffer> buffer =
      llvm::MemoryBuffer::getMemBuffer(text, fileName);
  std::string str;
  llvm::raw_string_ostream os(str);
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context, os);

  auto fileRange = FileRangeLoc::get(&context, fileName, /*line=*/2,
                                     /*column=*/10, /*byteSize=*/6);
  sourceMgrHandler.emitDiagnostic(fileRange, "Update this",
                                  DiagnosticSeverity::Warning, true);

  llvm::MemoryBufferRef resBuffer(os.str(), "result");
  llvm::line_iterator it(resBuffer);
  size_t ltIndex = -1, carrotIndex = -2;
  size_t gtIndex = -3, tildeIndex = -4;
  for (; !it.is_at_eof(); ++it) {
    if (size_t id = it->find_first_of('<'); id != StringRef::npos) {
      ltIndex = id;
      gtIndex = it->find_last_of('>');
    }
    if (size_t id = it->find_first_of('^'); id != StringRef::npos) {
      carrotIndex = id;
      tildeIndex = it->find_last_of('~');
    }
  }
  EXPECT_EQ(ltIndex, carrotIndex);
  EXPECT_EQ(gtIndex, tildeIndex);
}
