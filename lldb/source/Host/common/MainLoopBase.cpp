//===-- MainLoopBase.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/MainLoopBase.h"

using namespace lldb;
using namespace lldb_private;

void MainLoopBase::ProcessPendingCallbacks() {
  for (const Callback &callback : m_pending_callbacks)
    callback(*this);
  m_pending_callbacks.clear();
}
