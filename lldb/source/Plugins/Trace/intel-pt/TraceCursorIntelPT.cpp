//===-- TraceCursorIntelPT.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TraceCursorIntelPT.h"
#include "DecodedThread.h"
#include "TraceIntelPT.h"

#include <cstdlib>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::trace_intel_pt;
using namespace llvm;

TraceCursorIntelPT::TraceCursorIntelPT(ThreadSP thread_sp,
                                       DecodedThreadSP decoded_thread_sp)
    : TraceCursor(thread_sp), m_decoded_thread_sp(decoded_thread_sp) {
  assert(m_decoded_thread_sp->GetInstructionsCount() > 0 &&
         "a trace should have at least one instruction or error");
  m_pos = m_decoded_thread_sp->GetInstructionsCount() - 1;
  m_tsc_range =
      m_decoded_thread_sp->CalculateTscRange(m_pos, /*hint_range*/ None);
}

size_t TraceCursorIntelPT::GetInternalInstructionSize() {
  return m_decoded_thread_sp->GetInstructionsCount();
}

bool TraceCursorIntelPT::Next() {
  auto canMoveOne = [&]() {
    if (IsForwards())
      return m_pos + 1 < GetInternalInstructionSize();
    return m_pos > 0;
  };

  size_t initial_pos = m_pos;

  while (canMoveOne()) {
    m_pos += IsForwards() ? 1 : -1;

    if (m_tsc_range && !m_tsc_range->InRange(m_pos))
      m_tsc_range = IsForwards() ? m_tsc_range->Next() : m_tsc_range->Prev();

    if (!m_ignore_errors && IsError())
      return true;
    if (GetInstructionControlFlowType() & m_granularity)
      return true;
  }

  // Didn't find any matching instructions
  m_pos = initial_pos;
  return false;
}

uint64_t TraceCursorIntelPT::Seek(int64_t offset, SeekType origin) {
  int64_t last_index = GetInternalInstructionSize() - 1;

  auto fitPosToBounds = [&](int64_t raw_pos) -> int64_t {
    return std::min(std::max((int64_t)0, raw_pos), last_index);
  };

  auto FindDistanceAndSetPos = [&]() -> int64_t {
    switch (origin) {
    case TraceCursor::SeekType::Beginning:
      m_pos = fitPosToBounds(offset);
      return m_pos;
    case TraceCursor::SeekType::End:
      m_pos = fitPosToBounds(offset + last_index);
      return last_index - m_pos;
    case TraceCursor::SeekType::Current:
      int64_t new_pos = fitPosToBounds(offset + m_pos);
      int64_t dist = m_pos - new_pos;
      m_pos = new_pos;
      return std::abs(dist);
    }
  };

  int64_t dist = FindDistanceAndSetPos();
  m_tsc_range = m_decoded_thread_sp->CalculateTscRange(m_pos, m_tsc_range);
  return dist;
}

bool TraceCursorIntelPT::IsError() {
  return m_decoded_thread_sp->IsInstructionAnError(m_pos);
}

const char *TraceCursorIntelPT::GetError() {
  return m_decoded_thread_sp->GetErrorByInstructionIndex(m_pos);
}

lldb::addr_t TraceCursorIntelPT::GetLoadAddress() {
  return m_decoded_thread_sp->GetInstructionLoadAddress(m_pos);
}

Optional<uint64_t>
TraceCursorIntelPT::GetCounter(lldb::TraceCounter counter_type) {
  switch (counter_type) {
  case lldb::eTraceCounterTSC:
    if (m_tsc_range)
      return m_tsc_range->GetTsc();
    else
      return llvm::None;
  }
}

lldb::TraceEvents TraceCursorIntelPT::GetEvents() {
  return m_decoded_thread_sp->GetEvents(m_pos);
}

TraceInstructionControlFlowType
TraceCursorIntelPT::GetInstructionControlFlowType() {
  return m_decoded_thread_sp->GetInstructionControlFlowType(m_pos);
}

bool TraceCursorIntelPT::GoToId(user_id_t id) {
  if (m_decoded_thread_sp->GetInstructionsCount() <= id)
    return false;
  m_pos = id;
  m_tsc_range = m_decoded_thread_sp->CalculateTscRange(m_pos, m_tsc_range);

  return true;
}

user_id_t TraceCursorIntelPT::GetId() const { return m_pos; }
