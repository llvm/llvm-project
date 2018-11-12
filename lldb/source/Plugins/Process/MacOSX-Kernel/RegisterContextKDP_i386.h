//===-- RegisterContextKDP_i386.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextKDP_i386_h_
#define liblldb_RegisterContextKDP_i386_h_

#include "Plugins/Process/Utility/RegisterContextDarwin_i386.h"

class ThreadKDP;

class RegisterContextKDP_i386 : public RegisterContextDarwin_i386 {
public:
  RegisterContextKDP_i386(ThreadKDP &thread, uint32_t concrete_frame_idx);

  virtual ~RegisterContextKDP_i386();

protected:
  virtual int DoReadGPR(lldb::tid_t tid, int flavor, GPR &gpr);

  int DoReadFPU(lldb::tid_t tid, int flavor, FPU &fpu);

  int DoReadEXC(lldb::tid_t tid, int flavor, EXC &exc);

  int DoWriteGPR(lldb::tid_t tid, int flavor, const GPR &gpr);

  int DoWriteFPU(lldb::tid_t tid, int flavor, const FPU &fpu);

  int DoWriteEXC(lldb::tid_t tid, int flavor, const EXC &exc);

  ThreadKDP &m_kdp_thread;
};

#endif // liblldb_RegisterContextKDP_i386_h_
