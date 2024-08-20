//===-- InstructionBreakpoint.cpp ------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InstructionBreakpoint.h"
#include "DAP.h"

namespace lldb_dap {

// Instruction Breakpoint
InstructionBreakpoint::InstructionBreakpoint(const llvm::json::Object &obj)
    : Breakpoint(obj), instructionReference(LLDB_INVALID_ADDRESS), id(0),
      offset(GetSigned(obj, "offset", 0)) {
  GetString(obj, "instructionReference").getAsInteger(0, instructionReference);
  instructionReference += offset;
}

void InstructionBreakpoint::SetInstructionBreakpoint() {
  bp = g_dap.target.BreakpointCreateByAddress(instructionReference);
  id = bp.GetID();
}
} // namespace lldb_dap
