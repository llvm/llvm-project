//===-- RegisterContextWindowsMiniDump_x86.h ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextWindowsMiniDump_x86_H_
#define liblldb_RegisterContextWindowsMiniDump_x86_H_

#include "lldb/lldb-forward.h"
#include "Plugins/Process/Windows/Common/x86/RegisterContextWindows_x86.h"

namespace lldb_private
{

class Thread;

class RegisterContextWindowsMiniDump_x86 : public RegisterContextWindows_x86
{
  public:
    RegisterContextWindowsMiniDump_x86(Thread &thread, uint32_t concrete_frame_idx, const CONTEXT *context);

    virtual ~RegisterContextWindowsMiniDump_x86();

    bool WriteRegister(const RegisterInfo *reg_info, const RegisterValue &reg_value) override;

  protected:
    bool CacheAllRegisterValues() override;
};

}

#endif // #ifndef liblldb_RegisterContextWindowsMiniDump_x86_H_
