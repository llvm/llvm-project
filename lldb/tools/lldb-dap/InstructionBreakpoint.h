//===-- InstructionBreakpoint.h --------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_INSTRUCTIONBREAKPOINT_H
#define LLDB_TOOLS_LLDB_DAP_INSTRUCTIONBREAKPOINT_H

#include "Breakpoint.h"
#include "DAPForward.h"
#include "lldb/lldb-types.h"
#include <cstdint>

namespace lldb_dap {

// Instruction Breakpoint
struct InstructionBreakpoint : public Breakpoint {

  lldb::addr_t instructionAddressReference;
  int32_t offset;

  InstructionBreakpoint(DAP &d, const llvm::json::Object &obj);

  // Set instruction breakpoint in LLDB as a new breakpoint
  void SetBreakpoint();
};

} // namespace lldb_dap

#endif
