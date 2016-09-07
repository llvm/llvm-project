//===-- RegisterContextWindowsLive_x86.h ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextWindowsLive_x86_H_
#define liblldb_RegisterContextWindowsLive_x86_H_

#include "../../Common/x86/RegisterContextWindows_x86.h"
#include "lldb/lldb-forward.h"

namespace lldb_private {

class Thread;

class RegisterContextWindowsLive_x86 : public RegisterContextWindows_x86 {
public:
  //------------------------------------------------------------------
  // Constructors and Destructors
  //------------------------------------------------------------------
  RegisterContextWindowsLive_x86(Thread &thread, uint32_t concrete_frame_idx);

  virtual ~RegisterContextWindowsLive_x86();

  bool WriteRegister(const RegisterInfo *reg_info,
                     const RegisterValue &reg_value) override;
};
}

#endif // #ifndef liblldb_RegisterContextWindowsLive_x86_H_
