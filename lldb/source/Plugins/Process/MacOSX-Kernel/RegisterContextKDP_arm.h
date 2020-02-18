//===-- RegisterContextKDP_arm.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_MACOSX_KERNEL_REGISTERCONTEXTKDP_ARM_H
#define LLDB_SOURCE_PLUGINS_PROCESS_MACOSX_KERNEL_REGISTERCONTEXTKDP_ARM_H

#include "Plugins/Process/Utility/RegisterContextDarwin_arm.h"

class ThreadKDP;

class RegisterContextKDP_arm : public RegisterContextDarwin_arm {
public:
  RegisterContextKDP_arm(ThreadKDP &thread, uint32_t concrete_frame_idx);

  virtual ~RegisterContextKDP_arm();

protected:
  virtual int DoReadGPR(lldb::tid_t tid, int flavor, GPR &gpr);

  int DoReadFPU(lldb::tid_t tid, int flavor, FPU &fpu);

  int DoReadEXC(lldb::tid_t tid, int flavor, EXC &exc);

  int DoReadDBG(lldb::tid_t tid, int flavor, DBG &dbg);

  int DoWriteGPR(lldb::tid_t tid, int flavor, const GPR &gpr);

  int DoWriteFPU(lldb::tid_t tid, int flavor, const FPU &fpu);

  int DoWriteEXC(lldb::tid_t tid, int flavor, const EXC &exc);

  int DoWriteDBG(lldb::tid_t tid, int flavor, const DBG &dbg);

  ThreadKDP &m_kdp_thread;
};

#endif // LLDB_SOURCE_PLUGINS_PROCESS_MACOSX_KERNEL_REGISTERCONTEXTKDP_ARM_H
