//===-- WatchpointResource.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/WatchpointResource.h"

using namespace lldb;
using namespace lldb_private;

WatchpointResource::WatchpointResource(lldb::addr_t addr, size_t size,
                                       bool read, bool write)
    : m_addr(addr), m_size(size), m_watch_read(read), m_watch_write(write) {}

WatchpointResource::~WatchpointResource() { m_watchpoints.clear(); }

void WatchpointResource::GetResourceMemoryRange(lldb::addr_t &addr,
                                                size_t &size) const {
  addr = m_addr;
  size = m_size;
}

void WatchpointResource::GetResourceType(bool &read, bool &write) const {
  read = m_watch_read;
  write = m_watch_write;
}

void WatchpointResource::RegisterWatchpoint(lldb::WatchpointSP &wp_sp) {
  m_watchpoints.insert(wp_sp);
}

void WatchpointResource::DeregisterWatchpoint(lldb::WatchpointSP &wp_sp) {
  m_watchpoints.erase(wp_sp);
}

size_t WatchpointResource::GetNumDependantWatchpoints() {
  return m_watchpoints.size();
}

bool WatchpointResource::DependantWatchpointsContains(
    WatchpointSP wp_sp_to_match) {
  for (const WatchpointSP &wp_sp : m_watchpoints)
    if (wp_sp == wp_sp_to_match)
      return true;
  return false;
}
