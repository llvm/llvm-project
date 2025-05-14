//===-- TestBase.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "Protocol/ProtocolBase.h"
#include "Transport.h"
#include "lldb/Host/Pipe.h"
#include "gtest/gtest.h"

namespace lldb_dap_tests {

/// A base class for tests that need a pair of pipes for communication.
class PipeBase : public testing::Test {
protected:
  lldb_private::Pipe input;
  lldb_private::Pipe output;

  void SetUp() override;
};

/// A base class for tests that need transport configured for communicating DAP
/// messages.
class TransportBase : public PipeBase {
protected:
  std::unique_ptr<lldb_dap::Transport> to_dap;
  std::unique_ptr<lldb_dap::Transport> from_dap;

  void SetUp() override;
};

/// A base class for tests that interact with a `lldb_dap::DAP` instance.
class DAPTestBase : public TransportBase {
protected:
  std::unique_ptr<lldb_dap::DAP> dap;

  void SetUp() override;

  /// Closes the DAP output pipe and returns the remaining protocol messages in
  /// the buffer.
  std::vector<lldb_dap::protocol::Message> DrainOutput();
};

} // namespace lldb_dap_tests
