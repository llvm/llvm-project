//===-- TestBase.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SERVER_TESTS_TESTBASE_H
#define LLDB_SERVER_TESTS_TESTBASE_H

#include "TestClient.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "llvm/Support/Path.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

namespace llgs_tests {

class TestBase: public ::testing::Test {
public:
  static void SetUpTestCase() {
    lldb_private::FileSystem::Initialize();
    lldb_private::HostInfo::Initialize();
  }

  static void TearDownTestCase() {
    lldb_private::HostInfo::Terminate();
    lldb_private::FileSystem::Terminate();
  }

  static std::string getInferiorPath(llvm::StringRef Name) {
    llvm::SmallString<64> Path(LLDB_TEST_INFERIOR_PATH);
    llvm::sys::path::append(Path, Name + LLDB_TEST_INFERIOR_SUFFIX);
    return Path.str();
  }

  static std::string getLogFileName();
};

class StandardStartupTest: public TestBase {
public:
  void SetUp() override {
    auto ClientOr = TestClient::launch(getLogFileName());
    ASSERT_THAT_EXPECTED(ClientOr, llvm::Succeeded());
    Client = std::move(*ClientOr);
  }

protected:
  std::unique_ptr<TestClient> Client;
};

} // namespace llgs_tests

#endif // LLDB_SERVER_TESTS_TESTBASE_H
