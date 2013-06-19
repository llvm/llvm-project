//===-- RegisterContextFreeBSD_x86_64.h ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextFreeBSD_x86_64_H_
#define liblldb_RegisterContextFreeBSD_x86_64_H_

#include "Plugins/Process/POSIX/RegisterContext_x86_64.h"

class RegisterContextFreeBSD_x86_64:
    public RegisterContext_x86_64
{
public:
    RegisterContextFreeBSD_x86_64(lldb_private::Thread &thread, uint32_t concrete_frame_idx);

    size_t
    GetGPRSize();

protected:
    virtual const lldb_private::RegisterInfo *
    GetRegisterInfo();

    virtual void
    UpdateRegisterInfo();
};

#endif
