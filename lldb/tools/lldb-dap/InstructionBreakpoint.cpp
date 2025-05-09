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

InstructionBreakpoint::InstructionBreakpoint(DAP &d,
                                             const llvm::json::Object &obj)
    : Breakpoint(d, obj), m_instruction_address_reference(LLDB_INVALID_ADDRESS),
      m_offset(GetInteger<int64_t>(obj, "offset").value_or(0)) {
  GetString(obj, "instructionReference")
      .value_or("")
      .getAsInteger(0, m_instruction_address_reference);
  m_instruction_address_reference += m_offset;
}

void InstructionBreakpoint::SetBreakpoint() {
  m_bp =
      m_dap.target.BreakpointCreateByAddress(m_instruction_address_reference);
  Breakpoint::SetBreakpoint();
}

} // namespace lldb_dap
