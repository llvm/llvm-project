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
public:
  SBDebuggerContextManager() { SBDebugger::Initialize(); }

  ~SBDebuggerContextManager() { SBDebugger::Terminate(); }
};

} // namespace lldb_fuzzer
