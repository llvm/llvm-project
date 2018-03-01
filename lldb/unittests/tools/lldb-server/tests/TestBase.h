//===-- TestBase.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SERVER_TESTS_TESTBASE_H
#define LLDB_SERVER_TESTS_TESTBASE_H

#include "TestClient.h"
#include "lldb/Host/HostInfo.h"
#include "llvm/Support/Path.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

namespace llgs_tests {

class TestBase: public ::testing::Test {
public:
  static void SetUpTestCase() { lldb_private::HostInfo::Initialize(); }

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
