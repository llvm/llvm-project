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

class ProcessMonitor;

// Internal codes for all x86_64 registers.
enum
{
    k_first_gpr,
    gpr_rax = k_first_gpr,
    gpr_rbx,
    gpr_rcx,
    gpr_rdx,
    gpr_rdi,
    gpr_rsi,
    gpr_rbp,
    gpr_rsp,
    gpr_r8,
    gpr_r9,
    gpr_r10,
    gpr_r11,
    gpr_r12,
    gpr_r13,
    gpr_r14,
    gpr_r15,
    gpr_rip,
    gpr_rflags,
    gpr_cs,
    gpr_fs,
    gpr_gs,
    gpr_ss,
    gpr_ds,
    gpr_es,
    k_first_i386,
    gpr_eax = k_first_i386,
    gpr_ebx,
    gpr_ecx,
    gpr_edx,
    gpr_edi,
    gpr_esi,
    gpr_ebp,
    gpr_esp,
    gpr_eip,
    gpr_eflags, // eRegisterKindLLDB == 33
    k_last_i386 = gpr_eflags,
    k_last_gpr = gpr_eflags,

    k_first_fpr,
    fpu_fcw = k_first_fpr,
    fpu_fsw,
    fpu_ftw,
    fpu_fop,
    fpu_ip,
    fpu_cs,
    fpu_dp,
    fpu_ds,
    fpu_mxcsr,
    fpu_mxcsrmask,
    fpu_stmm0,
    fpu_stmm1,
    fpu_stmm2,
    fpu_stmm3,
    fpu_stmm4,
    fpu_stmm5,
    fpu_stmm6,
    fpu_stmm7,
    fpu_xmm0,
    fpu_xmm1,
    fpu_xmm2,
    fpu_xmm3,
    fpu_xmm4,
    fpu_xmm5,
    fpu_xmm6,
    fpu_xmm7,
    fpu_xmm8,
    fpu_xmm9,
    fpu_xmm10,
    fpu_xmm11,
    fpu_xmm12,
    fpu_xmm13,
    fpu_xmm14,
    fpu_xmm15,
    k_last_fpr = fpu_xmm15,
    k_first_avx,
    fpu_ymm0 = k_first_avx,
    fpu_ymm1,
    fpu_ymm2,
    fpu_ymm3,
    fpu_ymm4,
    fpu_ymm5,
    fpu_ymm6,
    fpu_ymm7,
    fpu_ymm8,
    fpu_ymm9,
    fpu_ymm10,
    fpu_ymm11,
    fpu_ymm12,
    fpu_ymm13,
    fpu_ymm14,
    fpu_ymm15,
    k_last_avx = fpu_ymm15,

    dr0,
    dr1,
    dr2,
    dr3,
    dr4,
    dr5,
    dr6,
    dr7,

    k_num_registers,
    k_num_gpr_registers = k_last_gpr - k_first_gpr + 1,
    k_num_fpr_registers = k_last_fpr - k_first_fpr + 1,
    k_num_avx_registers = k_last_avx - k_first_avx + 1
};

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

    virtual size_t
    GetGPRSize() = 0;

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

    unsigned
    GetRegisterIndexFromOffset(unsigned offset);

    const char *
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

    uint32_t
    NumSupportedHardwareWatchpoints();

    uint32_t
    SetHardwareWatchpoint(lldb::addr_t, size_t size, bool read, bool write);

    bool
    SetHardwareWatchpointWithIndex(lldb::addr_t, size_t size, bool read,
                                   bool write, uint32_t hw_index);

    bool
    ClearHardwareWatchpoint(uint32_t hw_index);

    bool
    HardwareSingleStep(bool enable);

    bool
    UpdateAfterBreakpoint();

    bool
    IsWatchpointVacant(uint32_t hw_index);

    bool
    IsWatchpointHit (uint32_t hw_index);

    lldb::addr_t
    GetWatchpointAddress (uint32_t hw_index);

    bool
    ClearWatchpointHits();

    //---------------------------------------------------------------------------
    // Generic floating-point registers
    //---------------------------------------------------------------------------

    struct MMSReg
    {
        uint8_t bytes[10];
        uint8_t pad[6];
    };

    struct XMMReg
    {
        uint8_t bytes[16]; // 128-bits for each XMM register
    };

    struct FXSAVE
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

    //---------------------------------------------------------------------------
    // Extended floating-point registers
    //---------------------------------------------------------------------------
    struct YMMHReg
    {
        uint8_t  bytes[16];     // 16 * 8 bits for the high bytes of each YMM register
    };

    struct YMMReg
    {
        uint8_t  bytes[32];     // 16 * 16 bits for each YMM register
    };

    struct YMM
    {
        YMMReg   ymm[16];       // assembled from ymmh and xmm registers
    };

    struct XSAVE_HDR
    {
        uint64_t  xstate_bv;    // OS enabled xstate mask to determine the extended states supported by the processor
        uint64_t  reserved1[2];
        uint64_t  reserved2[5];
    } __attribute__((packed));

    // x86 extensions to FXSAVE (i.e. for AVX processors) 
    struct XSAVE 
    {
        FXSAVE    i387;         // floating point registers typical in i387_fxsave_struct
        XSAVE_HDR header;       // The xsave_hdr_struct can be used to determine if the following extensions are usable
        YMMHReg   ymmh[16];     // High 16 bytes of each of 16 YMM registers (the low bytes are in FXSAVE.xmm for compatibility with SSE)
        // Slot any extensions to the register file here
    } __attribute__((packed, aligned (64)));

    struct IOVEC
    {
        void    *iov_base;      // pointer to XSAVE
        size_t   iov_len;       // sizeof(XSAVE)
    };

    //---------------------------------------------------------------------------
    // Note: prefer kernel definitions over user-land
    //---------------------------------------------------------------------------
    enum FPRType
    {
        eNotValid = 0,
        eFSAVE,  // TODO
        eFXSAVE,
        eSOFT,   // TODO
        eXSAVE
    };

    // Floating-point registers
    struct FPR
    {
        // Thread state for the floating-point unit of the processor read by ptrace.
        union XSTATE {
            FXSAVE   fxsave;    // Generic floating-point registers.
            XSAVE    xsave;     // x86 extended processor state.
        } xstate;
    };

protected:
    // Determines if an extended register set is supported on the processor running the inferior process.
    virtual bool
    IsRegisterSetAvailable(size_t set_index);

    virtual const lldb_private::RegisterInfo *
    GetRegisterInfo();

    virtual bool
    ReadRegister(const unsigned reg, lldb_private::RegisterValue &value);

    virtual bool
    WriteRegister(const unsigned reg, const lldb_private::RegisterValue &value);

private:
    uint64_t m_gpr[k_num_gpr_registers]; // general purpose registers.
    FPRType  m_fpr_type;                 // determines the type of data stored by union FPR, if any.
    FPR      m_fpr;                      // floating-point registers including extended register sets.
    IOVEC    m_iovec;                    // wrapper for xsave.
    YMM      m_ymm_set;                  // copy of ymmh and xmm register halves.

    ProcessMonitor &GetMonitor();
    lldb::ByteOrder GetByteOrder();

    bool CopyXSTATEtoYMM(uint32_t reg, lldb::ByteOrder byte_order);
    bool CopyYMMtoXSTATE(uint32_t reg, lldb::ByteOrder byte_order);
    bool IsFPR(unsigned reg, FPRType fpr_type);

    bool ReadGPR();
    bool ReadFPR();

    bool WriteGPR();
    bool WriteFPR();
};

#endif // #ifndef liblldb_RegisterContext_x86_64_H_
