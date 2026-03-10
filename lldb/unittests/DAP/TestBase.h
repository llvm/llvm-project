//===-- TestBase.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "DAPLog.h"
#include "Handler/RequestHandler.h"
#include "Handler/ResponseHandler.h"
#include "TestingSupport/Host/JSONTransportTestUtilities.h"
#include "TestingSupport/SubsystemRAII.h"
#include "Transport.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/MainLoop.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "gtest/gtest.h"
#include <memory>
#include <optional>

namespace lldb_dap_tests {

using TestDAPTransport = TestTransport<lldb_dap::ProtocolDescriptor>;

/// A base class for tests that need transport configured for communicating DAP
/// messages.
class TransportBase : public testing::Test {
protected:
  lldb_private::SubsystemRAII<lldb_private::FileSystem, lldb_private::HostInfo>
      subsystems;
  lldb_private::MainLoop loop;

  std::unique_ptr<lldb_dap::Log> log;
  lldb_dap::Log::Mutex log_mutex;

  std::unique_ptr<TestDAPTransport> to_client;
  MockMessageHandler<lldb_dap::ProtocolDescriptor> client;

  std::unique_ptr<TestDAPTransport> to_server;
  std::unique_ptr<lldb_dap::DAP> dap;

  void SetUp() override;

  void Run();
};

/// A base class for tests that interact with a `lldb_dap::DAP` instance.
class DAPTestBase : public TransportBase {
protected:
  std::optional<llvm::sys::fs::TempFile> core;
  std::optional<llvm::sys::fs::TempFile> binary;
  lldb::SBProcess process;

  static void SetUpTestSuite();
  static void TearDownTestSuite();
  void SetUp() override;
  void TearDown() override;

  void ConfigureDebugger();
  void LoadCore(llvm::StringRef binary_path, llvm::StringRef core_path);
};

} // namespace lldb_dap_tests
