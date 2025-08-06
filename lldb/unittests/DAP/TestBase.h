//===-- TestBase.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "Protocol/ProtocolBase.h"
#include "TestingSupport/Host/PipeTestUtilities.h"
#include "Transport.h"
#include "llvm/ADT/StringRef.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace lldb_dap_tests {

/// A base class for tests that need transport configured for communicating DAP
/// messages.
class TransportBase : public PipePairTest {
protected:
  std::unique_ptr<lldb_dap::Transport> to_dap;
  std::unique_ptr<lldb_dap::Transport> from_dap;

  void SetUp() override;
};

/// Matches an "output" event.
inline auto OutputMatcher(const llvm::StringRef output,
                          const llvm::StringRef category = "console") {
  return testing::VariantWith<lldb_dap::protocol::Event>(testing::FieldsAre(
      /*event=*/"output", /*body=*/testing::Optional<llvm::json::Value>(
          llvm::json::Object{{"category", category}, {"output", output}})));
}

/// A base class for tests that interact with a `lldb_dap::DAP` instance.
class DAPTestBase : public TransportBase {
protected:
  std::unique_ptr<lldb_dap::DAP> dap;
  std::optional<llvm::sys::fs::TempFile> core;
  std::optional<llvm::sys::fs::TempFile> binary;

  static constexpr llvm::StringLiteral k_linux_binary = "linux-x86_64.out.yaml";
  static constexpr llvm::StringLiteral k_linux_core = "linux-x86_64.core.yaml";

  static void SetUpTestSuite();
  static void TeatUpTestSuite();
  void SetUp() override;
  void TearDown() override;

  bool GetDebuggerSupportsTarget(llvm::StringRef platform);
  void CreateDebugger();
  void LoadCore();

  /// Closes the DAP output pipe and returns the remaining protocol messages in
  /// the buffer.
  std::vector<lldb_dap::protocol::Message> DrainOutput();
};

} // namespace lldb_dap_tests
