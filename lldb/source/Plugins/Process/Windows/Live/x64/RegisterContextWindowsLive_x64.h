//===-- RegisterContextWindowsLive_x64.h ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextWindowsLive_x64_H_
#define liblldb_RegisterContextWindowsLive_x64_H_

#include "../../Common/x64/RegisterContextWindows_x64.h"
#include "lldb/lldb-forward.h"

namespace lldb_private {

class Thread;

class RegisterContextWindowsLive_x64 : public RegisterContextWindows_x64 {
public:
  //------------------------------------------------------------------
  // Constructors and Destructors
  //------------------------------------------------------------------
  RegisterContextWindowsLive_x64(Thread &thread, uint32_t concrete_frame_idx);

  virtual ~RegisterContextWindowsLive_x64();

  //------------------------------------------------------------------
  // Subclasses must override these functions
  //------------------------------------------------------------------
  bool ReadRegister(const RegisterInfo *reg_info,
                    RegisterValue &reg_value) override;

  bool WriteRegister(const RegisterInfo *reg_info,
                     const RegisterValue &reg_value) override;
};
}

#endif // #ifndef liblldb_RegisterContextWindowsLive_x64_H_
