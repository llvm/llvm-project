//===-- TargetThreadWindows.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Plugins_Process_Windows_TargetThreadWindows_H_
#define liblldb_Plugins_Process_Windows_TargetThreadWindows_H_

//#include "ForwardDecl.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Target/Thread.h"
#include "lldb/lldb-forward.h"

namespace lldb_private {
class ProcessWindows;
class HostThread;
class StackFrame;

class TargetThreadWindows : public lldb_private::Thread {
public:
  TargetThreadWindows(ProcessWindows &process, const HostThread &thread);
  virtual ~TargetThreadWindows();

  // lldb_private::Thread overrides
  void RefreshStateAfterStop() override;
  void WillResume(lldb::StateType resume_state) override;
  void DidStop() override;
  bool CalculateStopInfo() override;
  Unwind *GetUnwinder() override;

  bool DoResume();

  HostThread GetHostThread() const { return m_host_thread; }

private:
  HostThread m_host_thread;
};
}

#endif
