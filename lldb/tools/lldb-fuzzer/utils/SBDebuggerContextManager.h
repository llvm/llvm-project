//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBDebugger.h"

using namespace lldb;

namespace lldb_fuzzer {

class SBDebuggerContextManager {
  inline static int instance_count;

public:
  SBDebuggerContextManager() { ++instance_count; }

  ~SBDebuggerContextManager() {
    --instance_count;
    if (instance_count == 0)
      SBDebugger::Terminate();
  }
};

} // namespace lldb_fuzzer
