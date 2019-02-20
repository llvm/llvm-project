//===-- RegisterContextKDP_arm64.h --------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextKDP_arm64_h_
#define liblldb_RegisterContextKDP_arm64_h_


#include "Plugins/Process/Utility/RegisterContextDarwin_arm64.h"

class ThreadKDP;

class RegisterContextKDP_arm64 : public RegisterContextDarwin_arm64 {
public:
  RegisterContextKDP_arm64(ThreadKDP &thread, uint32_t concrete_frame_idx);

  virtual ~RegisterContextKDP_arm64();

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

#endif // liblldb_RegisterContextKDP_arm64_h_
