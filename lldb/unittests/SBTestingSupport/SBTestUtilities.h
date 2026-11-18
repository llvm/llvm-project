//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UNITTESTS_SBTESTINGSUPPORT_SBTESTUTILITIES_H
#define LLDB_UNITTESTS_SBTESTINGSUPPORT_SBTESTUTILITIES_H

#include "TestingSupport/SubsystemRAII.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBTarget.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"

#include <utility>

namespace lldb_private {

/// Returns true if the host permits opening a listening socket.
bool HostSupportsListeningSockets();

/// Check if the debugger supports the given platform.
bool DebuggerSupportsLLVMTarget(llvm::StringRef target);

std::pair<lldb::SBTarget, lldb::SBProcess> LoadCore(lldb::SBDebugger &debugger,
                                                    llvm::StringRef binary_path,
                                                    llvm::StringRef core_path);

/// Owns a debugger that reads no init files. LLDB is initialized per test.
class SBDebuggerFixture : public testing::Test {
protected:
  void SetUp() override {
    m_debugger = lldb::SBDebugger::Create(/*source_init_files=*/false);
  }

  void TearDown() override { lldb::SBDebugger::Destroy(m_debugger); }

  /// Returns a target that has an architecture and no modules.
  lldb::SBTarget CreateEmptyTarget() {
    lldb::SBError error;
    return m_debugger.CreateTarget("", "arm64-apple-macosx", "remote-macosx",
                                   /*add_dependent_modules=*/false, error);
  }

  SubsystemRAII<lldb::SBDebugger> m_subsystems;
  lldb::SBDebugger m_debugger;
};

} // namespace lldb_private

#endif
