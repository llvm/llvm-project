//===-- RegisterContext_x86_64.h ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContext_x86_64_H_
#define liblldb_RegisterContext_x86_64_H_

#include "lldb/Core/Log.h"
#include "RegisterContextPOSIX.h"

#ifdef __FreeBSD__
#include "RegisterContextFreeBSD_x86_64.h"
#endif

#ifdef __linux__
#include "RegisterContextLinux_x86_64.h"
#endif

class ProcessMonitor;

class RegisterContext_x86_64
  : public RegisterContextPOSIX
{
public:
    RegisterContext_x86_64 (lldb_private::Thread &thread,
                                 uint32_t concrete_frame_idx);

    ~RegisterContext_x86_64();

    void
    Invalidate();

    void
    InvalidateAllRegisters();

    size_t
    GetRegisterCount();

    const lldb_private::RegisterInfo *
    GetRegisterInfoAtIndex(uint32_t reg);

    size_t
    GetRegisterSetCount();

    const lldb_private::RegisterSet *
    GetRegisterSet(uint32_t set);

    static unsigned
    GetRegisterIndexFromOffset(unsigned offset);

    static const char *
    GetRegisterName(unsigned reg);

    virtual bool
    ReadRegister(const lldb_private::RegisterInfo *reg_info,
                 lldb_private::RegisterValue &value);

    bool
    ReadAllRegisterValues(lldb::DataBufferSP &data_sp);

    virtual bool
    WriteRegister(const lldb_private::RegisterInfo *reg_info,
                  const lldb_private::RegisterValue &value);

    bool
    WriteAllRegisterValues(const lldb::DataBufferSP &data_sp);

    uint32_t
    ConvertRegisterKindToRegisterNumber(uint32_t kind, uint32_t num);

    bool
    HardwareSingleStep(bool enable);

    bool
    UpdateAfterBreakpoint();

    struct MMSReg
    {
        uint8_t bytes[10];
        uint8_t pad[6];
    };

    struct XMMReg
    {
        uint8_t bytes[16];
    };

    struct FPU
    {
        uint16_t fcw;
        uint16_t fsw;
        uint16_t ftw;
        uint16_t fop;
        uint64_t ip;
        uint64_t dp;
        uint32_t mxcsr;
        uint32_t mxcsrmask;
        MMSReg   stmm[8];
        XMMReg   xmm[16];
        uint32_t padding[24];
    };

    struct UserArea
    {
        GPR      regs;          // General purpose registers.
        int32_t  fpvalid;       // True if FPU is being used.
        int32_t  pad0;
        FPU      i387;          // FPU registers.
        uint64_t tsize;         // Text segment size.
        uint64_t dsize;         // Data segment size.
        uint64_t ssize;         // Stack segment size.
        uint64_t start_code;    // VM address of text.
        uint64_t start_stack;   // VM address of stack bottom (top in rsp).
        int64_t  signal;        // Signal causing core dump.
        int32_t  reserved;      // Unused.
        int32_t  pad1;
        uint64_t ar0;           // Location of GPR's.
        FPU*     fpstate;       // Location of FPR's.
        uint64_t magic;         // Identifier for core dumps.
        char     u_comm[32];    // Command causing core dump.
        uint64_t u_debugreg[8]; // Debug registers (DR0 - DR7).
        uint64_t error_code;    // CPU error code.
        uint64_t fault_address; // Control register CR3.
    };

private:
    UserArea user;

    ProcessMonitor &GetMonitor();

    bool ReadGPR();
    bool ReadFPR();

    bool WriteGPR();
    bool WriteFPR();
};

#endif // #ifndef liblldb_RegisterContext_x86_64_H_
