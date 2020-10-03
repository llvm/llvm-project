//===-- ThreadIntelPT.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ThreadIntelPT.h"

#include <memory>

#include "Plugins/Process/Utility/RegisterContextHistory.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::trace_intel_pt;

void ThreadIntelPT::RefreshStateAfterStop() {}

RegisterContextSP ThreadIntelPT::GetRegisterContext() {
  if (!m_reg_context_sp)
    m_reg_context_sp = CreateRegisterContextForFrame(nullptr);

  return m_reg_context_sp;
}

RegisterContextSP
ThreadIntelPT::CreateRegisterContextForFrame(StackFrame *frame) {
  // Eventually this will calculate the register context based on the current
  // trace position.
  return std::make_shared<RegisterContextHistory>(
      *this, 0, GetProcess()->GetAddressByteSize(), LLDB_INVALID_ADDRESS);
}

bool ThreadIntelPT::CalculateStopInfo() { return false; }
