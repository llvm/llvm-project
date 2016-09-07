//===-- ThreadSpec.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadSpec.h"

using namespace lldb;
using namespace lldb_private;

ThreadSpec::ThreadSpec()
    : m_index(UINT32_MAX), m_tid(LLDB_INVALID_THREAD_ID), m_name(),
      m_queue_name() {}

ThreadSpec::ThreadSpec(const ThreadSpec &rhs)
    : m_index(rhs.m_index), m_tid(rhs.m_tid), m_name(rhs.m_name),
      m_queue_name(rhs.m_queue_name) {}

const ThreadSpec &ThreadSpec::operator=(const ThreadSpec &rhs) {
  m_index = rhs.m_index;
  m_tid = rhs.m_tid;
  m_name = rhs.m_name;
  m_queue_name = rhs.m_queue_name;
  return *this;
}

const char *ThreadSpec::GetName() const {
  return m_name.empty() ? nullptr : m_name.c_str();
}

const char *ThreadSpec::GetQueueName() const {
  return m_queue_name.empty() ? nullptr : m_queue_name.c_str();
}

bool ThreadSpec::TIDMatches(Thread &thread) const {
  if (m_tid == LLDB_INVALID_THREAD_ID)
    return true;

  lldb::tid_t thread_id = thread.GetID();
  return TIDMatches(thread_id);
}

bool ThreadSpec::IndexMatches(Thread &thread) const {
  if (m_index == UINT32_MAX)
    return true;
  uint32_t index = thread.GetIndexID();
  return IndexMatches(index);
}

bool ThreadSpec::NameMatches(Thread &thread) const {
  if (m_name.empty())
    return true;

  const char *name = thread.GetName();
  return NameMatches(name);
}

bool ThreadSpec::QueueNameMatches(Thread &thread) const {
  if (m_queue_name.empty())
    return true;

  const char *queue_name = thread.GetQueueName();
  return QueueNameMatches(queue_name);
}

bool ThreadSpec::ThreadPassesBasicTests(Thread &thread) const {
  if (!HasSpecification())
    return true;

  if (!TIDMatches(thread))
    return false;

  if (!IndexMatches(thread))
    return false;

  if (!NameMatches(thread))
    return false;

  if (!QueueNameMatches(thread))
    return false;

  return true;
}

bool ThreadSpec::HasSpecification() const {
  return (m_index != UINT32_MAX || m_tid != LLDB_INVALID_THREAD_ID ||
          !m_name.empty() || !m_queue_name.empty());
}

void ThreadSpec::GetDescription(Stream *s, lldb::DescriptionLevel level) const {
  if (!HasSpecification()) {
    if (level == eDescriptionLevelBrief) {
      s->PutCString("thread spec: no ");
    }
  } else {
    if (level == eDescriptionLevelBrief) {
      s->PutCString("thread spec: yes ");
    } else {
      if (GetTID() != LLDB_INVALID_THREAD_ID)
        s->Printf("tid: 0x%" PRIx64 " ", GetTID());

      if (GetIndex() != UINT32_MAX)
        s->Printf("index: %d ", GetIndex());

      const char *name = GetName();
      if (name)
        s->Printf("thread name: \"%s\" ", name);

      const char *queue_name = GetQueueName();
      if (queue_name)
        s->Printf("queue name: \"%s\" ", queue_name);
    }
  }
}
