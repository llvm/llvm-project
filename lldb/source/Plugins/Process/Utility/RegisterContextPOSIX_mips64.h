//===-- RegisterContextPOSIX_mips64.h ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextPOSIX_mips64_H_
#define liblldb_RegisterContextPOSIX_mips64_H_

#include "lldb/Core/Log.h"
#include "RegisterContextPOSIX.h"
#include "RegisterContext_mips.h"
#include "lldb-mips-freebsd-register-enums.h"

using namespace lldb_private;

class ProcessMonitor;

class RegisterContextPOSIX_mips64
  : public lldb_private::RegisterContext
{
public:
    RegisterContextPOSIX_mips64 (lldb_private::Thread &thread,
                            uint32_t concrete_frame_idx,
                            lldb_private::RegisterInfoInterface *register_info);

    ~RegisterContextPOSIX_mips64();

    void
    Invalidate();

    void
    InvalidateAllRegisters();

    size_t
    GetRegisterCount();

    virtual size_t
    GetGPRSize();

    virtual unsigned
    GetRegisterSize(unsigned reg);

    virtual unsigned
    GetRegisterOffset(unsigned reg);

    const lldb_private::RegisterInfo *
    GetRegisterInfoAtIndex(size_t reg);

    size_t
    GetRegisterSetCount();

    const lldb_private::RegisterSet *
    GetRegisterSet(size_t set);

    const char *
    GetRegisterName(unsigned reg);

    uint32_t
    ConvertRegisterKindToRegisterNumber(lldb::RegisterKind kind, uint32_t num);

protected:
    uint64_t m_gpr_mips64[k_num_gpr_registers_mips64];         // general purpose registers.
    std::unique_ptr<lldb_private::RegisterInfoInterface> m_register_info_ap; // Register Info Interface (FreeBSD or Linux)

    // Determines if an extended register set is supported on the processor running the inferior process.
    virtual bool
    IsRegisterSetAvailable(size_t set_index);

    virtual const lldb_private::RegisterInfo *
    GetRegisterInfo();

    bool
    IsGPR(unsigned reg);

    bool
    IsFPR(unsigned reg);

    lldb::ByteOrder GetByteOrder();

    virtual bool ReadGPR() = 0;
    virtual bool ReadFPR() = 0;
    virtual bool WriteGPR() = 0;
    virtual bool WriteFPR() = 0;
};

#endif // #ifndef liblldb_RegisterContextPOSIX_mips64_H_
