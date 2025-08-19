//===-- HistoryUnwind.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-private.h"

#include "Plugins/Process/Utility/HistoryUnwind.h"
#include "Plugins/Process/Utility/RegisterContextHistory.h"

#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

#include <memory>

using namespace lldb;
using namespace lldb_private;

// Constructor

HistoryUnwind::HistoryUnwind(Thread &thread, std::vector<lldb::addr_t> pcs,
                             HistoryPCType pc_type)
    : Unwind(thread), m_pcs(pcs), m_pc_type(pc_type) {}

// Destructor

HistoryUnwind::~HistoryUnwind() = default;

void HistoryUnwind::DoClear() {
  std::lock_guard<std::recursive_mutex> guard(m_unwind_mutex);
  m_pcs.clear();
}

lldb::RegisterContextSP
HistoryUnwind::DoCreateRegisterContextForFrame(StackFrame *frame) {
  RegisterContextSP rctx;
  if (frame) {
    addr_t pc = frame->GetFrameCodeAddress().GetLoadAddress(
        &frame->GetThread()->GetProcess()->GetTarget());
    if (pc != LLDB_INVALID_ADDRESS) {
      rctx = std::make_shared<RegisterContextHistory>(
          *frame->GetThread().get(), frame->GetConcreteFrameIndex(),
          frame->GetThread()->GetProcess()->GetAddressByteSize(), pc);
    }
  }
  return rctx;
}

static bool BehavesLikeZerothFrame(HistoryPCType pc_type, uint32_t frame_idx) {
  switch (pc_type) {
  case HistoryPCType::Returns:
    return (frame_idx == 0);
  case HistoryPCType::ReturnsNoZerothFrame:
    return false;
  case HistoryPCType::Calls:
    return true;
  }
}

bool HistoryUnwind::DoGetFrameInfoAtIndex(uint32_t frame_idx, lldb::addr_t &cfa,
                                          lldb::addr_t &pc,
                                          bool &behaves_like_zeroth_frame) {
  // FIXME do not throw away the lock after we acquire it..
  std::unique_lock<std::recursive_mutex> guard(m_unwind_mutex);
  guard.unlock();
  if (frame_idx < m_pcs.size()) {
    cfa = frame_idx;
    pc = m_pcs[frame_idx];
    behaves_like_zeroth_frame = BehavesLikeZerothFrame(m_pc_type, frame_idx);
    return true;
  }
  return false;
}

uint32_t HistoryUnwind::DoGetFrameCount() { return m_pcs.size(); }
