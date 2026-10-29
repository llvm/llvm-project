//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/TargetGroupList.h"
#include "lldb/Core/Debugger.h"
#include "llvm/ADT/STLExtras.h"

using namespace lldb;
using namespace lldb_private;

TargetGroupList::~TargetGroupList() { Clear(); }

TargetGroupSP TargetGroupList::CreateTargetGroup(llvm::StringRef description) {
  return CreateTargetGroup<TargetGroup>(description);
}

bool TargetGroupList::DeleteTargetGroup(const TargetGroupSP &group_sp) {
  if (!group_sp)
    return false;

  std::lock_guard<std::recursive_mutex> guard(*m_mutex);
  auto it = llvm::find(m_groups, group_sp);
  if (it == m_groups.end())
    return false;
  m_groups.erase(it);
  return true;
}

std::vector<TargetGroupSP> TargetGroupList::GetGroups() const {
  std::lock_guard<std::recursive_mutex> guard(*m_mutex);
  return m_groups;
}

void TargetGroupList::Add(const TargetGroupSP &group_sp) {
  if (!group_sp)
    return;

  std::lock_guard<std::recursive_mutex> guard(*m_mutex);
  if (!llvm::is_contained(m_groups, group_sp))
    m_groups.push_back(group_sp);
}

void TargetGroupList::Remove(const TargetGroup *group) {
  std::lock_guard<std::recursive_mutex> guard(*m_mutex);
  llvm::erase_if(m_groups, [group](const TargetGroupSP &group_sp) {
    return group_sp.get() == group;
  });
}

void TargetGroupList::RemoveTarget(const TargetSP &target_sp) {
  if (!target_sp)
    return;

  std::lock_guard<std::recursive_mutex> guard(*m_mutex);
  std::vector<TargetGroupSP> groups = m_groups;
  for (const TargetGroupSP &group_sp : groups) {
    group_sp->RemoveTarget(target_sp);
    if (group_sp->GetTargets().empty())
      m_debugger.GetTargetGroupList().DeleteTargetGroup(group_sp);
  }
  m_groups.clear();
}

void TargetGroupList::Clear() {
  std::lock_guard<std::recursive_mutex> guard(*m_mutex);
  m_groups.clear();
}
