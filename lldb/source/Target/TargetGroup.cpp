//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/TargetGroup.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/TargetGroupList.h"

#include <algorithm>

using namespace lldb;
using namespace lldb_private;

bool TargetGroup::AddTarget(const TargetSP &target_sp, TargetRole role) {
  if (!target_sp || role == TargetRole::None)
    return false;

  std::lock_guard<std::recursive_mutex> guard(*m_mutex);
  if (!m_active || &target_sp->GetDebugger() != m_debugger)
    return false;

  for (auto it = m_entries.begin(); it != m_entries.end();) {
    TargetSP member_sp = it->target_wp.lock();
    if (!member_sp) {
      it = m_entries.erase(it);
      continue;
    }
    if (member_sp == target_sp) {
      it->role = role;
      target_sp->AddTargetGroup(shared_from_this());
      return true;
    }
    ++it;
  }

  m_entries.push_back({target_sp, role});
  target_sp->AddTargetGroup(shared_from_this());
  return true;
}

bool TargetGroup::RemoveTarget(const TargetSP &target_sp) {
  if (!target_sp)
    return false;

  std::lock_guard<std::recursive_mutex> guard(*m_mutex);
  if (!m_active)
    return false;

  bool removed = false;
  auto it = std::remove_if(m_entries.begin(), m_entries.end(),
                           [&](const Entry &entry) {
                             TargetSP member_sp = entry.target_wp.lock();
                             if (!member_sp)
                               return true;
                             if (member_sp == target_sp) {
                               removed = true;
                               return true;
                             }
                             return false;
                           });
  m_entries.erase(it, m_entries.end());
  if (removed)
    target_sp->RemoveTargetGroup(this);
  return removed;
}

TargetRole TargetGroup::GetRole(const TargetSP &target_sp) const {
  if (!target_sp)
    return TargetRole::None;

  std::lock_guard<std::recursive_mutex> guard(*m_mutex);
  if (!m_active)
    return TargetRole::None;
  for (const Entry &entry : m_entries) {
    if (entry.target_wp.lock() == target_sp)
      return entry.role;
  }
  return TargetRole::None;
}

std::vector<TargetSP> TargetGroup::GetTargets(TargetRole role) const {
  std::vector<TargetSP> targets;
  std::lock_guard<std::recursive_mutex> guard(*m_mutex);
  if (!m_active)
    return targets;
  for (const Entry &entry : m_entries) {
    if (TargetSP target_sp = entry.target_wp.lock();
        target_sp && (role == TargetRole::None || entry.role == role))
      targets.push_back(std::move(target_sp));
  }
  return targets;
}

void TargetGroup::Deactivate() {
  std::lock_guard<std::recursive_mutex> guard(*m_mutex);
  if (!m_active)
    return;

  m_active = false;
  for (const Entry &entry : m_entries) {
    if (TargetSP target_sp = entry.target_wp.lock())
      target_sp->RemoveTargetGroup(this);
  }
  m_entries.clear();
}
