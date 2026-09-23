//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/TargetAPIMutex.h"
#include "lldb/Target/Target.h"

using namespace lldb_private;

void TargetAPIMutex::Resolve() {
  if (!m_target_sp)
    return;

  std::recursive_mutex *real_mutex = m_target_sp->GetAPIMutexForCurrentPolicy();
  m_mutex = real_mutex
                ? std::shared_ptr<std::recursive_mutex>(m_target_sp, real_mutex)
                : nullptr;
}

void TargetAPIMutex::lock() {
  Resolve();
  if (m_mutex)
    m_mutex->lock();
}

bool TargetAPIMutex::try_lock() {
  Resolve();
  return m_mutex ? m_mutex->try_lock() : true;
}
