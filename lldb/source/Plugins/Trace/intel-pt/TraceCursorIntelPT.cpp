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
  Seek(0, SeekType::End);
}

void TraceCursorIntelPT::CalculateTscRange() {
  // If we failed, then we look for the exact range
  if (!m_tsc_range || !m_tsc_range->InRange(m_pos))
    m_tsc_range = m_decoded_thread_sp->CalculateTscRange(
        m_pos, /*hit_range=*/m_tsc_range);
}

void TraceCursorIntelPT::Next() {
  m_pos += IsForwards() ? 1 : -1;

  // We try to go to a neighbor tsc range that might contain the current pos
  if (m_tsc_range && !m_tsc_range->InRange(m_pos))
    m_tsc_range = IsForwards() ? m_tsc_range->Next() : m_tsc_range->Prev();

  // If we failed, this call will fix it
  CalculateTscRange();
}

bool TraceCursorIntelPT::Seek(int64_t offset, SeekType origin) {
  switch (origin) {
  case TraceCursor::SeekType::Beginning:
    m_pos = offset;
    break;
  case TraceCursor::SeekType::End:
    m_pos = m_decoded_thread_sp->GetItemsCount() - 1 + offset;
    break;
  case TraceCursor::SeekType::Current:
    m_pos += offset;
  }
  CalculateTscRange();

  return HasValue();
}

bool TraceCursorIntelPT::HasValue() const {
  return m_pos >= 0 && m_pos < m_decoded_thread_sp->GetItemsCount();
}

lldb::TraceItemKind TraceCursorIntelPT::GetItemKind() const {
  return m_decoded_thread_sp->GetItemKindByIndex(m_pos);
}

const char *TraceCursorIntelPT::GetError() const {
  return m_decoded_thread_sp->GetErrorByIndex(m_pos);
}

lldb::addr_t TraceCursorIntelPT::GetLoadAddress() const {
  return m_decoded_thread_sp->GetInstructionLoadAddress(m_pos);
}

Optional<uint64_t>
TraceCursorIntelPT::GetCounter(lldb::TraceCounter counter_type) const {
  switch (counter_type) {
  case lldb::eTraceCounterTSC:
    if (m_tsc_range)
      return m_tsc_range->GetTsc();
    else
      return llvm::None;
  }
}

lldb::TraceEvent TraceCursorIntelPT::GetEventType() const {
  return m_decoded_thread_sp->GetEventByIndex(m_pos);
}

bool TraceCursorIntelPT::GoToId(user_id_t id) {
  if (!HasId(id))
    return false;
  m_pos = id;
  m_tsc_range = m_decoded_thread_sp->CalculateTscRange(m_pos, m_tsc_range);

  return true;
}

bool TraceCursorIntelPT::HasId(lldb::user_id_t id) const {
  return static_cast<int64_t>(id) < m_decoded_thread_sp->GetItemsCount();
}

user_id_t TraceCursorIntelPT::GetId() const { return m_pos; }
