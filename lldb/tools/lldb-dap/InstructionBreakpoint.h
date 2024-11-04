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
#include "llvm/ADT/StringRef.h"

namespace lldb_dap {

// Instruction Breakpoint
struct InstructionBreakpoint : public Breakpoint {

  lldb::addr_t instructionAddressReference;
  int32_t id;
  int32_t offset;

  InstructionBreakpoint()
      : Breakpoint(), instructionAddressReference(LLDB_INVALID_ADDRESS), id(0),
        offset(0) {}
  InstructionBreakpoint(const llvm::json::Object &obj);

  // Set instruction breakpoint in LLDB as a new breakpoint
  void SetInstructionBreakpoint();
};

} // namespace lldb_dap

#endif
