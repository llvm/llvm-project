//===-- QueueList.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/QueueList.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Queue.h"

using namespace lldb;
using namespace lldb_private;

QueueList::QueueList(Process *process)
    : m_process(process), m_stop_id(0), m_queues(), m_mutex() {}

QueueList::~QueueList() { Clear(); }

uint32_t QueueList::GetSize() {
  Mutex::Locker locker(m_mutex);
  return m_queues.size();
}

lldb::QueueSP QueueList::GetQueueAtIndex(uint32_t idx) {
  Mutex::Locker locker(m_mutex);
  if (idx < m_queues.size()) {
    return m_queues[idx];
  } else {
    return QueueSP();
  }
}

void QueueList::Clear() {
  Mutex::Locker locker(m_mutex);
  m_queues.clear();
}

void QueueList::AddQueue(QueueSP queue_sp) {
  Mutex::Locker locker(m_mutex);
  if (queue_sp.get()) {
    m_queues.push_back(queue_sp);
  }
}

lldb::QueueSP QueueList::FindQueueByID(lldb::queue_id_t qid) {
  QueueSP ret;
  for (QueueSP queue_sp : Queues()) {
    if (queue_sp->GetID() == qid) {
      ret = queue_sp;
      break;
    }
  }
  return ret;
}

lldb::QueueSP QueueList::FindQueueByIndexID(uint32_t index_id) {
  QueueSP ret;
  for (QueueSP queue_sp : Queues()) {
    if (queue_sp->GetIndexID() == index_id) {
      ret = queue_sp;
      break;
    }
  }
  return ret;
}

lldb_private::Mutex &QueueList::GetMutex() { return m_mutex; }
