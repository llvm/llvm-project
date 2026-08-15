//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/TargetAPIMutex.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/Policy.h"

using namespace lldb_private;

void TargetAPIMutex::lock() {
  if (m_target_sp) {
    Policy policy = PolicyStack::Get().Current();
    std::recursive_mutex &real_mutex = policy.view == Policy::View::Private
                                           ? m_target_sp->m_private_mutex
                                           : m_target_sp->m_mutex;
    m_mutex = std::shared_ptr<std::recursive_mutex>(m_target_sp, &real_mutex);
  }
  if (m_mutex)
    m_mutex->lock();
}

bool TargetAPIMutex::try_lock() {
  if (m_target_sp) {
    Policy policy = PolicyStack::Get().Current();
    std::recursive_mutex &real_mutex = policy.view == Policy::View::Private
                                           ? m_target_sp->m_private_mutex
                                           : m_target_sp->m_mutex;
    m_mutex = std::shared_ptr<std::recursive_mutex>(m_target_sp, &real_mutex);
  }
  return m_mutex ? m_mutex->try_lock() : true;
}
