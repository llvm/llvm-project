//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UNITTESTS_DAP_TESTFIXTURES_H
#define LLDB_UNITTESTS_DAP_TESTFIXTURES_H

#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBTarget.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include <optional>

namespace lldb_dap_tests {

class DAPTestBase;

struct TestFixtures {
  TestFixtures() = default;
  ~TestFixtures();
  TestFixtures(const TestFixtures &) = delete;
  TestFixtures &operator=(const TestFixtures &) = delete;

  static constexpr llvm::StringLiteral k_linux_binary = "linux-x86_64.out.yaml";
  static constexpr llvm::StringLiteral k_linux_core = "linux-x86_64.core.yaml";

  bool IsPlatformSupported(llvm::StringRef platform);

  lldb::SBDebugger debugger;
  lldb::SBTarget target;
  lldb::SBProcess process;

  void LoadDebugger();
  void LoadTarget(llvm::StringRef path = k_linux_binary);
  void LoadProcess(llvm::StringRef path = k_linux_core);

private:
  friend DAPTestBase;

  std::optional<llvm::sys::fs::TempFile> m_binary;
  std::optional<llvm::sys::fs::TempFile> m_core;
};

} // namespace lldb_dap_tests

#endif
