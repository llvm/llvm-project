//===- llvm/unittest/Support/FileUtilitiesTest.cpp - unit tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "gtest/gtest.h"
#include <fstream>

using namespace llvm;
using namespace llvm::sys;

#define ASSERT_NO_ERROR(x)                                                     \
  if (std::error_code ASSERT_NO_ERROR_ec = x) {                                \
    SmallString<128> MessageStorage;                                           \
    raw_svector_ostream Message(MessageStorage);                               \
    Message << #x ": did not return errc::success.\n"                          \
            << "error number: " << ASSERT_NO_ERROR_ec.value() << "\n"          \
            << "error message: " << ASSERT_NO_ERROR_ec.message() << "\n";      \
    GTEST_FATAL_FAILURE_(MessageStorage.c_str());                              \
  } else {                                                                     \
  }

namespace {
TEST(writeFileAtomicallyTest, Test) {
  // Create unique temporary directory for these tests
  SmallString<128> RootTestDirectory;
  ASSERT_NO_ERROR(
    fs::createUniqueDirectory("writeFileAtomicallyTest", RootTestDirectory));

  SmallString<128> FinalTestfilePath(RootTestDirectory);
  sys::path::append(FinalTestfilePath, "foo.txt");
  const std::string TempUniqTestFileModel = FinalTestfilePath.str().str() + "-%%%%%%%%";
  const std::string TestfileContent = "fooFOOfoo";

  llvm::Error Err = llvm::writeFileAtomically(TempUniqTestFileModel, FinalTestfilePath, TestfileContent);
  ASSERT_FALSE(static_cast<bool>(Err));

  std::ifstream FinalFileStream(FinalTestfilePath.str());
  std::string FinalFileContent;
  FinalFileStream >> FinalFileContent;
  ASSERT_EQ(FinalFileContent, TestfileContent);
}
} // anonymous namespace
