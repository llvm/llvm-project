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
#include "Protocol/ProtocolTypes.h"
#include "lldb/lldb-types.h"
#include <cstdint>

namespace lldb_dap {

/// Instruction Breakpoint
class InstructionBreakpoint : public Breakpoint {
public:
  InstructionBreakpoint(DAP &d,
                        const protocol::InstructionBreakpoint &breakpoint);

  /// Set instruction breakpoint in LLDB as a new breakpoint.
  void SetBreakpoint();

  lldb::addr_t GetInstructionAddressReference() const {
    return m_instruction_address_reference;
  }

protected:
  lldb::addr_t m_instruction_address_reference;
  int32_t m_offset;
};

} // namespace lldb_dap

#endif
