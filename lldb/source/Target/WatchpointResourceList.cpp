//===-- WatchpointResourceList.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/WatchpointResourceList.h"
#include "lldb/Target/WatchpointResource.h"

using namespace lldb;
using namespace lldb_private;

WatchpointResourceList::WatchpointResourceList() : m_resources(), m_mutex() {}

WatchpointResourceList::~WatchpointResourceList() { Clear(); }

uint32_t WatchpointResourceList::GetSize() {
  std::lock_guard<std::mutex> guard(m_mutex);
  return m_resources.size();
}

lldb::WatchpointResourceSP
WatchpointResourceList::GetResourceAtIndex(uint32_t idx) {
  std::lock_guard<std::mutex> guard(m_mutex);
  if (idx < m_resources.size()) {
    return m_resources[idx];
  } else {
    return {};
  }
}

void WatchpointResourceList::RemoveWatchpointResource(
    WatchpointResourceSP wp_resource_sp) {
  assert(wp_resource_sp->GetNumDependantWatchpoints() == 0);

  std::lock_guard<std::mutex> guard(m_mutex);
  collection::iterator pos = m_resources.begin();
  while (pos != m_resources.end()) {
    if (*pos == wp_resource_sp) {
      m_resources.erase(pos);
      return;
    }
    ++pos;
  }
}

void WatchpointResourceList::Clear() {
  std::lock_guard<std::mutex> guard(m_mutex);
  m_resources.clear();
}

void WatchpointResourceList::AddResource(WatchpointResourceSP resource_sp) {
  std::lock_guard<std::mutex> guard(m_mutex);
  if (resource_sp) {
    m_resources.push_back(resource_sp);
  }
}

std::mutex &WatchpointResourceList::GetMutex() { return m_mutex; }
