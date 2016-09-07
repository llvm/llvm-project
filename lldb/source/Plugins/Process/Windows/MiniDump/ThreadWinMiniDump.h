//===-- ThreadWinMiniDump.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadWinMiniDump_h_
#define liblldb_ThreadWinMiniDump_h_

#include <string>

#include "lldb/Core/DataExtractor.h"
#include "lldb/Target/Thread.h"

class ThreadWinMiniDump : public lldb_private::Thread {
public:
  ThreadWinMiniDump(lldb_private::Process &process, lldb::tid_t tid);

  virtual ~ThreadWinMiniDump();

  void RefreshStateAfterStop() override;

  lldb::RegisterContextSP GetRegisterContext() override;

  lldb::RegisterContextSP
  CreateRegisterContextForFrame(lldb_private::StackFrame *frame) override;

  void ClearStackFrames() override;

  void SetContext(const void *context);

protected:
  lldb::RegisterContextSP m_reg_context_sp;
  class Data;
  std::unique_ptr<Data> m_data; // for WinAPI-specific data

  bool CalculateStopInfo() override;
};

#endif
