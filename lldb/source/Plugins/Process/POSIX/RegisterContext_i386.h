//===-- RegisterContext_i386.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContext_i386_h_
#define liblldb_RegisterContext_i386_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Log.h"
#include "RegisterContextPOSIX.h"

class RegisterContext_i386 : public RegisterContextPOSIX
{
public:
    RegisterContext_i386(lldb_private::Thread &thread,
                              uint32_t concreate_frame_idx);

    ~RegisterContext_i386();

    void
    Invalidate();

    void
    InvalidateAllRegisters();

    size_t
    GetRegisterCount();

    const lldb_private::RegisterInfo *
    GetRegisterInfoAtIndex(size_t reg);

    size_t
    GetRegisterSetCount();

    const lldb_private::RegisterSet *
    GetRegisterSet(size_t set);

    unsigned
    GetRegisterIndexFromOffset(unsigned offset);

    const char *
    GetRegisterName(unsigned reg);

    bool
    ReadRegisterValue(uint32_t reg, lldb_private::Scalar &value);

    bool
    ReadRegisterBytes(uint32_t reg, lldb_private::DataExtractor &data);

    virtual bool
    ReadRegister(const lldb_private::RegisterInfo *reg_info,
                 lldb_private::RegisterValue &value);

    bool
    ReadAllRegisterValues(lldb::DataBufferSP &data_sp);

    bool
    WriteRegisterValue(uint32_t reg, const lldb_private::Scalar &value);

    bool
    WriteRegisterBytes(uint32_t reg, lldb_private::DataExtractor &data,
                       uint32_t data_offset = 0);

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

    struct GPR
    {
        uint32_t ebx;
        uint32_t ecx;
        uint32_t edx;
        uint32_t esi;
        uint32_t edi;
        uint32_t ebp;
        uint32_t eax;
        uint32_t ds;
        uint32_t es;
        uint32_t fs;
        uint32_t gs;
        uint32_t orig_ax;
        uint32_t eip;
        uint32_t cs;
        uint32_t eflags;
        uint32_t esp;
        uint32_t ss;
    };

    struct MMSReg
    {
        uint8_t bytes[8];
    };

    struct XMMReg
    {
        uint8_t bytes[16];
    };

    struct FPU
    {
        uint16_t    fcw;
        uint16_t    fsw;
        uint16_t    ftw;
        uint16_t    fop;
        uint32_t    ip;
        uint32_t    cs;
        uint32_t    foo;
        uint32_t    fos;
        uint32_t    mxcsr;
        uint32_t    reserved;
        MMSReg      stmm[8];
        XMMReg      xmm[8];
        uint32_t    pad[56];
    };

    // A user area like this no longer exists on FreeBSD
    // making this a Linux artifact. Nonetheless, it is safe
    // leaving it here while the code is being cleaned up and generalized.

    struct UserArea
    {
        GPR      regs;          // General purpose registers.
        int32_t  fpvalid;       // True if FPU is being used.
        FPU      i387;          // FPU registers.
        uint32_t tsize;         // Text segment size.
        uint32_t dsize;         // Data segment size.
        uint32_t ssize;         // Stack segment size.
        uint32_t start_code;    // VM address of text.
        uint32_t start_stack;   // VM address of stack bottom (top in rsp).
        int32_t  signal;        // Signal causing core dump.
        int32_t  reserved;      // Unused.
        uint32_t ar0;           // Location of GPR's.
        FPU*     fpstate;       // Location of FPR's.
        uint32_t magic;         // Identifier for core dumps.
        char     u_comm[32];    // Command causing core dump.
        uint32_t u_debugreg[8]; // Debug registers (DR0 - DR7).
    };
private:
    UserArea user;

    ProcessMonitor &GetMonitor();

    void LogGPR(const char *title);

    bool ReadGPR();
    bool ReadFPR();
};

#endif // #ifndef liblldb_RegisterContext_i386_h_
