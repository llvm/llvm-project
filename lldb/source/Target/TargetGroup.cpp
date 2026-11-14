//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/TargetGroup.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/TargetGroupList.h"

#include <algorithm>
#include <cassert>

using namespace lldb;
using namespace lldb_private;

char TargetGroup::ID;
char TargetGroupCPUAndAccelerators::ID;

TargetGroup::TargetGroup(TargetGroupList &group_list,
                         llvm::StringRef description)
    : m_debugger_wp(group_list.m_debugger.shared_from_this()),
      m_description(description), m_mutex(group_list.m_mutex) {}

TargetGroup::~TargetGroup() {
  assert(m_entries.empty() &&
         "target group destroyed while it still contains targets");
}

llvm::Error TargetGroup::ValidateTargetAddition(const TargetSP &, Role) const {
  return llvm::Error::success();
}

TargetGroupCPUAndAccelerators::TargetGroupCPUAndAccelerators(
    TargetGroupList &group_list, llvm::StringRef description)
    : Base(group_list, description) {}

llvm::Error
TargetGroupCPUAndAccelerators::ValidateTargetAddition(const TargetSP &,
                                                      Role role) const {
  switch (role) {
  case Role::CPU:
    if (!GetTargets().empty())
      return llvm::createStringError(
          "the CPU target must be the first target in the group");
    break;
  case Role::Accelerator:
    if (GetTargets(Role::CPU).size() != 1)
      return llvm::createStringError(
          "the group must contain one CPU target before adding accelerators");
    break;
  }
  return llvm::Error::success();
}

TargetSP TargetGroupCPUAndAccelerators::GetCPUTarget() const {
  std::vector<TargetSP> cpu_targets = GetTargets(Role::CPU);
  assert(cpu_targets.size() <= 1 &&
         "CPU and accelerator group contains multiple CPU targets");
  return cpu_targets.empty() ? nullptr : cpu_targets.front();
}

llvm::Error TargetGroup::AddTarget(const TargetSP &target_sp, Role role) {
  if (!target_sp)
    return llvm::createStringError(
        "cannot add a null target to a target group");
  if (!target_sp->IsValid())
    return llvm::createStringError(
        "cannot add an invalid target to a target group");

  DebuggerSP debugger_sp = m_debugger_wp.lock();
  if (!debugger_sp)
    return llvm::createStringError("target group's debugger no longer exists");
  if (&target_sp->GetDebugger() != debugger_sp.get())
    return llvm::createStringError("target belongs to a different debugger");

  std::lock_guard<std::recursive_mutex> guard(*m_mutex);
  for (const Entry &entry : m_entries) {
    TargetSP member_sp = entry.target_wp.lock();
    assert(member_sp && "target did not remove itself from its target group");
    if (!member_sp)
      continue;
    if (member_sp != target_sp)
      continue;
    if (entry.role == role)
      return llvm::Error::success();
    return llvm::createStringError(
        "target is already in the group with a different role");
  }

  if (llvm::Error error = ValidateTargetAddition(target_sp, role))
    return error;

  m_entries.push_back({target_sp, role});
  target_sp->AddTargetGroup(shared_from_this());
  return llvm::Error::success();
}

bool TargetGroup::RemoveTarget(const TargetSP &target_sp) {
  if (!target_sp)
    return false;

  std::lock_guard<std::recursive_mutex> guard(*m_mutex);
  bool removed = false;
  auto it = std::remove_if(
      m_entries.begin(), m_entries.end(), [&](const Entry &entry) {
        TargetSP member_sp = entry.target_wp.lock();
        assert(member_sp && "target did not remove itself from its "
                            "target group");
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

std::optional<TargetGroup::Role>
TargetGroup::GetRole(const TargetSP &target_sp) const {
  if (!target_sp)
    return std::nullopt;

  std::lock_guard<std::recursive_mutex> guard(*m_mutex);
  for (const Entry &entry : m_entries) {
    TargetSP member_sp = entry.target_wp.lock();
    assert(member_sp && "target did not remove itself from its target group");
    if (member_sp == target_sp)
      return entry.role;
  }
  return std::nullopt;
}

std::vector<TargetSP> TargetGroup::GetTargets() const {
  std::vector<TargetSP> targets;
  std::lock_guard<std::recursive_mutex> guard(*m_mutex);
  for (const Entry &entry : m_entries) {
    TargetSP target_sp = entry.target_wp.lock();
    assert(target_sp && "target did not remove itself from its target group");
    if (target_sp)
      targets.push_back(std::move(target_sp));
  }
  return targets;
}

std::vector<TargetSP> TargetGroup::GetTargets(Role role) const {
  std::vector<TargetSP> targets;
  std::lock_guard<std::recursive_mutex> guard(*m_mutex);
  for (const Entry &entry : m_entries) {
    if (entry.role != role)
      continue;
    TargetSP target_sp = entry.target_wp.lock();
    assert(target_sp && "target did not remove itself from its target group");
    if (target_sp)
      targets.push_back(std::move(target_sp));
  }
  return targets;
}
