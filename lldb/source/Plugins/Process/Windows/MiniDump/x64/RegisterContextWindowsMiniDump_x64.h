//===-- RegisterContextWindowsMiniDump_x64.h --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextWindowsMiniDump_x64_H_
#define liblldb_RegisterContextWindowsMiniDump_x64_H_

#include "Plugins/Process/Windows/Common/x64/RegisterContextWindows_x64.h"
#include "lldb/lldb-forward.h"

namespace lldb_private {

class Thread;

class RegisterContextWindowsMiniDump_x64 : public RegisterContextWindows_x64 {
public:
  RegisterContextWindowsMiniDump_x64(Thread &thread,
                                     uint32_t concrete_frame_idx,
                                     const CONTEXT *context);

  virtual ~RegisterContextWindowsMiniDump_x64();

  bool WriteRegister(const RegisterInfo *reg_info,
                     const RegisterValue &reg_value) override;

protected:
  bool CacheAllRegisterValues() override;
};
}

#endif // #ifndef liblldb_RegisterContextWindowsMiniDump_x64_H_
