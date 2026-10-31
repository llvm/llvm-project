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
#include "lldb/API/SBBreakpoint.h"
#include "lldb/API/SBTarget.h"
#include "llvm/ADT/StringRef.h"

namespace lldb_dap {

InstructionBreakpoint::InstructionBreakpoint(
    DAP &d, const protocol::InstructionBreakpoint &breakpoint)
    : Breakpoint(d, breakpoint.condition, breakpoint.hitCondition),
      m_instruction_address_reference(LLDB_INVALID_ADDRESS) {
  llvm::StringRef instruction_reference(breakpoint.instructionReference);
  instruction_reference.getAsInteger(0, m_instruction_address_reference);
  m_instruction_address_reference += breakpoint.offset.value_or(0);
}

void InstructionBreakpoint::SetBreakpoint() {
  lldb::SBMutex lock = m_dap.GetAPIMutex();
  std::lock_guard<lldb::SBMutex> guard(lock);

  m_bp =
      m_dap.target.BreakpointCreateByAddress(m_instruction_address_reference);
  Breakpoint::SetBreakpoint();
}

} // namespace lldb_dap
