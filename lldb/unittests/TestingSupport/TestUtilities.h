//===- TestUtilities.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UNITTESTS_TESTINGSUPPORT_TESTUTILITIES_H
#define LLDB_UNITTESTS_TESTINGSUPPORT_TESTUTILITIES_H

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileUtilities.h"
#include <string>

#define ASSERT_NO_ERROR(x)                                                     \
  if (std::error_code ASSERT_NO_ERROR_ec = x) {                                \
    llvm::SmallString<128> MessageStorage;                                     \
    llvm::raw_svector_ostream Message(MessageStorage);                         \
    Message << #x ": did not return errc::success.\n"                          \
            << "error number: " << ASSERT_NO_ERROR_ec.value() << "\n"          \
            << "error message: " << ASSERT_NO_ERROR_ec.message() << "\n";      \
    GTEST_FATAL_FAILURE_(MessageStorage.c_str());                              \
  } else {                                                                     \
  }

namespace lldb_private {
std::string GetInputFilePath(const llvm::Twine &name);

class TestFile {
public:
  static llvm::Expected<TestFile> fromYaml(llvm::StringRef Yaml);
  static llvm::Expected<TestFile> fromYamlFile(const llvm::Twine &Name);

  TestFile(TestFile &&RHS) : Name(std::move(RHS.Name)) {
    RHS.Name = llvm::None;
  }

  ~TestFile();

  llvm::StringRef name() { return *Name; }

private:
  TestFile(llvm::StringRef Name, llvm::FileRemover &&Remover)
      : Name(std::string(Name)) {
    Remover.releaseFile();
  }
  void operator=(const TestFile &) = delete;

  llvm::Optional<std::string> Name;
};
}

#endif
