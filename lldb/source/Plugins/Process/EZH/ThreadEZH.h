//===-- ThreadEZH.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadEZH_h_
#define liblldb_ThreadEZH_h_

#include "../gdb-remote/ThreadGDBRemote.h"
#include "lldb/lldb-private.h"

class ThreadEZH : public lldb_private::process_gdb_remote::ThreadGDBRemote {
public:
  ThreadEZH(lldb_private::Process &process, lldb::tid_t tid);

  ~ThreadEZH() override = default;

  lldb::RegisterContextSP GetRegisterContext() override;

  lldb::RegisterContextSP
  CreateRegisterContextForFrame(lldb_private::StackFrame *frame) override;

  bool CalculateStopInfo() override;

  void RefreshStateAfterStop() override;
};

#endif // liblldb_ThreadEZH_h_
