//===-- TargetThreadWindowsLive.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Plugins_Process_Windows_TargetThreadWindowsLive_H_
#define liblldb_Plugins_Process_Windows_TargetThreadWindowsLive_H_

#include "lldb/Host/HostThread.h"
#include "lldb/Target/Thread.h"
#include "lldb/lldb-forward.h"

#include "Plugins/Process/Windows/Common/TargetThreadWindows.h"

namespace lldb_private {
class ProcessWindows;
class HostThread;
class StackFrame;

class TargetThreadWindowsLive : public lldb_private::TargetThreadWindows {
public:
  TargetThreadWindowsLive(ProcessWindows &process, const HostThread &thread);
  virtual ~TargetThreadWindowsLive();

  // lldb_private::Thread overrides
  void RefreshStateAfterStop() override;
  void WillResume(lldb::StateType resume_state) override;
  void DidStop() override;
  lldb::RegisterContextSP GetRegisterContext() override;
  lldb::RegisterContextSP
  CreateRegisterContextForFrame(StackFrame *frame) override;
  bool CalculateStopInfo() override;
  Unwind *GetUnwinder() override;

  bool DoResume();

  HostThread GetHostThread() const { return m_host_thread; }

private:
  lldb::RegisterContextSP CreateRegisterContextForFrameIndex(uint32_t idx);

  HostThread m_host_thread;
};
}

#endif
