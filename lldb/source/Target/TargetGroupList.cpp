//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/TargetGroupList.h"
#include "llvm/ADT/STLExtras.h"

using namespace lldb;
using namespace lldb_private;

TargetGroupList::~TargetGroupList() { Clear(); }

TargetGroupSP TargetGroupList::CreateTargetGroup() {
  std::lock_guard<std::recursive_mutex> guard(*m_mutex);
  TargetGroupSP group_sp(new TargetGroup(m_debugger, m_mutex));
  m_groups.push_back(group_sp);
  return group_sp;
}

bool TargetGroupList::DeleteTargetGroup(const TargetGroupSP &group_sp) {
  if (!group_sp)
    return false;

  std::lock_guard<std::recursive_mutex> guard(*m_mutex);
  auto it = llvm::find(m_groups, group_sp);
  if (it == m_groups.end())
    return false;
  group_sp->Deactivate();
  m_groups.erase(it);
  return true;
}

size_t TargetGroupList::GetNumTargetGroups() const {
  std::lock_guard<std::recursive_mutex> guard(*m_mutex);
  return m_groups.size();
}

TargetGroupSP TargetGroupList::GetTargetGroupAtIndex(size_t index) const {
  std::lock_guard<std::recursive_mutex> guard(*m_mutex);
  if (index >= m_groups.size())
    return {};
  return m_groups[index];
}

void TargetGroupList::Clear() {
  std::lock_guard<std::recursive_mutex> guard(*m_mutex);
  for (const TargetGroupSP &group_sp : m_groups)
    group_sp->Deactivate();
  m_groups.clear();
}
