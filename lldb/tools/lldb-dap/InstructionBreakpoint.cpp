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
#include "JSONUtils.h"
#include "lldb/API/SBBreakpoint.h"
#include "lldb/API/SBTarget.h"
#include "llvm/ADT/StringRef.h"

namespace lldb_dap {

// Instruction Breakpoint
InstructionBreakpoint::InstructionBreakpoint(DAP &d,
                                             const llvm::json::Object &obj)
    : Breakpoint(d, obj), instructionAddressReference(LLDB_INVALID_ADDRESS),
      offset(GetSigned(obj, "offset", 0)) {
  GetString(obj, "instructionReference")
      .getAsInteger(0, instructionAddressReference);
  instructionAddressReference += offset;
}

void InstructionBreakpoint::SetBreakpoint() {
  bp = dap.target.BreakpointCreateByAddress(instructionAddressReference);
  Breakpoint::SetBreakpoint();
}

} // namespace lldb_dap
