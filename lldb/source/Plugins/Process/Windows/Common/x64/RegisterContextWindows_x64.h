//===-- RegisterContextWindows_x64.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextWindows_x64_H_
#define liblldb_RegisterContextWindows_x64_H_

#include "RegisterContextWindows.h"
#include "lldb/lldb-forward.h"

namespace lldb_private {

class Thread;

class RegisterContextWindows_x64 : public RegisterContextWindows {
public:
  //------------------------------------------------------------------
  // Constructors and Destructors
  //------------------------------------------------------------------
  RegisterContextWindows_x64(Thread &thread, uint32_t concrete_frame_idx);

  virtual ~RegisterContextWindows_x64();

  //------------------------------------------------------------------
  // Subclasses must override these functions
  //------------------------------------------------------------------
  size_t GetRegisterCount() override;

  const RegisterInfo *GetRegisterInfoAtIndex(size_t reg) override;

  size_t GetRegisterSetCount() override;

  const RegisterSet *GetRegisterSet(size_t reg_set) override;

  bool ReadRegister(const RegisterInfo *reg_info,
                    RegisterValue &reg_value) override;

  bool WriteRegister(const RegisterInfo *reg_info,
                     const RegisterValue &reg_value) override;
};
}

#endif // #ifndef liblldb_RegisterContextWindows_x64_H_
