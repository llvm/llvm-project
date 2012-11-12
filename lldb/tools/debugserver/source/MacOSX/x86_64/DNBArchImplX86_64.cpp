//===-- DNBArchImplX86_64.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 6/25/07.
//
//===----------------------------------------------------------------------===//

#if defined (__i386__) || defined (__x86_64__)

#include <sys/cdefs.h>
#include <sys/types.h>
#include <sys/sysctl.h>

#include "MacOSX/x86_64/DNBArchImplX86_64.h"
#include "DNBLog.h"
#include "MachThread.h"
#include "MachProcess.h"
#include <mach/mach.h>
#include <stdlib.h>

#if defined (LLDB_DEBUGSERVER_RELEASE) || defined (LLDB_DEBUGSERVER_DEBUG)
enum debugState {
    debugStateUnknown,
    debugStateOff,
    debugStateOn
};

static debugState sFPUDebugState = debugStateUnknown;
static debugState sAVXForceState = debugStateUnknown;

static bool DebugFPURegs ()
{
    if (sFPUDebugState == debugStateUnknown)
    {
        if (getenv("DNB_DEBUG_FPU_REGS"))
            sFPUDebugState = debugStateOn;
        else
            sFPUDebugState = debugStateOff;
    }
    
    return (sFPUDebugState == debugStateOn);
}

static bool ForceAVXRegs ()
{
    if (sFPUDebugState == debugStateUnknown)
    {
        if (getenv("DNB_DEBUG_X86_FORCE_AVX_REGS"))
            sAVXForceState = debugStateOn;
        else
            sAVXForceState = debugStateOff;
    }
    
    return (sAVXForceState == debugStateOn);
}

#define DEBUG_FPU_REGS (DebugFPURegs())
#define FORCE_AVX_REGS (ForceAVXRegs())
#else
#define DEBUG_FPU_REGS (0)
#define FORCE_AVX_REGS (0)
#endif


extern "C" bool
CPUHasAVX()
{
    enum AVXPresence
    {
        eAVXUnknown     = -1,
        eAVXNotPresent  =  0,
        eAVXPresent     =  1
    };

    static AVXPresence g_has_avx = eAVXUnknown;
    if (g_has_avx == eAVXUnknown)
    {
        g_has_avx = eAVXNotPresent;

        // Only xnu-2020 or later has AVX support, any versions before
        // this have a busted thread_get_state RPC where it would truncate
        // the thread state buffer (<rdar://problem/10122874>). So we need to
        // verify the kernel version number manually or disable AVX support.
        int mib[2];
        char buffer[1024];
        size_t length = sizeof(buffer);
        uint64_t xnu_version = 0;
        mib[0] = CTL_KERN;
        mib[1] = KERN_VERSION;
        int err = ::sysctl(mib, 2, &buffer, &length, NULL, 0);
        if (err == 0)
        {
            const char *xnu = strstr (buffer, "xnu-");
            if (xnu)
            {
                const char *xnu_version_cstr = xnu + 4;
                xnu_version = strtoull (xnu_version_cstr, NULL, 0);
                if (xnu_version >= 2020 && xnu_version != ULLONG_MAX)
                {
                    if (::HasAVX())
                    {
                        g_has_avx = eAVXPresent;
                    }
                }
            }
        }
        DNBLogThreadedIf (LOG_THREAD, "CPUHasAVX(): g_has_avx = %i (err = %i, errno = %i, xnu_version = %llu)", g_has_avx, err, errno, xnu_version);
    }
    
    return (g_has_avx == eAVXPresent);
}

uint64_t
DNBArchImplX86_64::GetPC(uint64_t failValue)
{
    // Get program counter
    if (GetGPRState(false) == KERN_SUCCESS)
        return m_state.context.gpr.__rip;
    return failValue;
}

kern_return_t
DNBArchImplX86_64::SetPC(uint64_t value)
{
    // Get program counter
    kern_return_t err = GetGPRState(false);
    if (err == KERN_SUCCESS)
    {
        m_state.context.gpr.__rip = value;
        err = SetGPRState();
    }
    return err == KERN_SUCCESS;
}

uint64_t
DNBArchImplX86_64::GetSP(uint64_t failValue)
{
    // Get stack pointer
    if (GetGPRState(false) == KERN_SUCCESS)
        return m_state.context.gpr.__rsp;
    return failValue;
}

// Uncomment the value below to verify the values in the debugger.
//#define DEBUG_GPR_VALUES 1    // DO NOT CHECK IN WITH THIS DEFINE ENABLED

kern_return_t
DNBArchImplX86_64::GetGPRState(bool force)
{
    if (force || m_state.GetError(e_regSetGPR, Read))
    {
#if DEBUG_GPR_VALUES
        m_state.context.gpr.__rax = ('a' << 8) + 'x';
        m_state.context.gpr.__rbx = ('b' << 8) + 'x';
        m_state.context.gpr.__rcx = ('c' << 8) + 'x';
        m_state.context.gpr.__rdx = ('d' << 8) + 'x';
        m_state.context.gpr.__rdi = ('d' << 8) + 'i';
        m_state.context.gpr.__rsi = ('s' << 8) + 'i';
        m_state.context.gpr.__rbp = ('b' << 8) + 'p';
        m_state.context.gpr.__rsp = ('s' << 8) + 'p';
        m_state.context.gpr.__r8  = ('r' << 8) + '8';
        m_state.context.gpr.__r9  = ('r' << 8) + '9';
        m_state.context.gpr.__r10 = ('r' << 8) + 'a';
        m_state.context.gpr.__r11 = ('r' << 8) + 'b';
        m_state.context.gpr.__r12 = ('r' << 8) + 'c';
        m_state.context.gpr.__r13 = ('r' << 8) + 'd';
        m_state.context.gpr.__r14 = ('r' << 8) + 'e';
        m_state.context.gpr.__r15 = ('r' << 8) + 'f';
        m_state.context.gpr.__rip = ('i' << 8) + 'p';
        m_state.context.gpr.__rflags = ('f' << 8) + 'l';
        m_state.context.gpr.__cs = ('c' << 8) + 's';
        m_state.context.gpr.__fs = ('f' << 8) + 's';
        m_state.context.gpr.__gs = ('g' << 8) + 's';
        m_state.SetError(e_regSetGPR, Read, 0);
#else
        mach_msg_type_number_t count = e_regSetWordSizeGPR;
        m_state.SetError(e_regSetGPR, Read, ::thread_get_state(m_thread->ThreadID(), __x86_64_THREAD_STATE, (thread_state_t)&m_state.context.gpr, &count));
        DNBLogThreadedIf (LOG_THREAD, "::thread_get_state (0x%4.4x, %u, &gpr, %u) => 0x%8.8x"
                          "\n\trax = %16.16llx rbx = %16.16llx rcx = %16.16llx rdx = %16.16llx"
                          "\n\trdi = %16.16llx rsi = %16.16llx rbp = %16.16llx rsp = %16.16llx"
                          "\n\t r8 = %16.16llx  r9 = %16.16llx r10 = %16.16llx r11 = %16.16llx"
                          "\n\tr12 = %16.16llx r13 = %16.16llx r14 = %16.16llx r15 = %16.16llx"
                          "\n\trip = %16.16llx"
                          "\n\tflg = %16.16llx  cs = %16.16llx  fs = %16.16llx  gs = %16.16llx",
                          m_thread->ThreadID(), x86_THREAD_STATE64, x86_THREAD_STATE64_COUNT,
                          m_state.GetError(e_regSetGPR, Read),
                          m_state.context.gpr.__rax,m_state.context.gpr.__rbx,m_state.context.gpr.__rcx,
                          m_state.context.gpr.__rdx,m_state.context.gpr.__rdi,m_state.context.gpr.__rsi,
                          m_state.context.gpr.__rbp,m_state.context.gpr.__rsp,m_state.context.gpr.__r8,
                          m_state.context.gpr.__r9, m_state.context.gpr.__r10,m_state.context.gpr.__r11,
                          m_state.context.gpr.__r12,m_state.context.gpr.__r13,m_state.context.gpr.__r14,
                          m_state.context.gpr.__r15,m_state.context.gpr.__rip,m_state.context.gpr.__rflags,
                          m_state.context.gpr.__cs,m_state.context.gpr.__fs, m_state.context.gpr.__gs);
        
        //      DNBLogThreadedIf (LOG_THREAD, "thread_get_state(0x%4.4x, %u, &gpr, %u) => 0x%8.8x"
        //                        "\n\trax = %16.16llx"
        //                        "\n\trbx = %16.16llx"
        //                        "\n\trcx = %16.16llx"
        //                        "\n\trdx = %16.16llx"
        //                        "\n\trdi = %16.16llx"
        //                        "\n\trsi = %16.16llx"
        //                        "\n\trbp = %16.16llx"
        //                        "\n\trsp = %16.16llx"
        //                        "\n\t r8 = %16.16llx"
        //                        "\n\t r9 = %16.16llx"
        //                        "\n\tr10 = %16.16llx"
        //                        "\n\tr11 = %16.16llx"
        //                        "\n\tr12 = %16.16llx"
        //                        "\n\tr13 = %16.16llx"
        //                        "\n\tr14 = %16.16llx"
        //                        "\n\tr15 = %16.16llx"
        //                        "\n\trip = %16.16llx"
        //                        "\n\tflg = %16.16llx"
        //                        "\n\t cs = %16.16llx"
        //                        "\n\t fs = %16.16llx"
        //                        "\n\t gs = %16.16llx",
        //                        m_thread->ThreadID(),
        //                        x86_THREAD_STATE64,
        //                        x86_THREAD_STATE64_COUNT,
        //                        m_state.GetError(e_regSetGPR, Read),
        //                        m_state.context.gpr.__rax,
        //                        m_state.context.gpr.__rbx,
        //                        m_state.context.gpr.__rcx,
        //                        m_state.context.gpr.__rdx,
        //                        m_state.context.gpr.__rdi,
        //                        m_state.context.gpr.__rsi,
        //                        m_state.context.gpr.__rbp,
        //                        m_state.context.gpr.__rsp,
        //                        m_state.context.gpr.__r8,
        //                        m_state.context.gpr.__r9,
        //                        m_state.context.gpr.__r10,
        //                        m_state.context.gpr.__r11,
        //                        m_state.context.gpr.__r12,
        //                        m_state.context.gpr.__r13,
        //                        m_state.context.gpr.__r14,
        //                        m_state.context.gpr.__r15,
        //                        m_state.context.gpr.__rip,
        //                        m_state.context.gpr.__rflags,
        //                        m_state.context.gpr.__cs,
        //                        m_state.context.gpr.__fs,
        //                        m_state.context.gpr.__gs);
#endif
    }
    return m_state.GetError(e_regSetGPR, Read);
}

// Uncomment the value below to verify the values in the debugger.
//#define DEBUG_FPU_REGS 1    // DO NOT CHECK IN WITH THIS DEFINE ENABLED

kern_return_t
DNBArchImplX86_64::GetFPUState(bool force)
{
    if (force || m_state.GetError(e_regSetFPU, Read))
    {
        if (DEBUG_FPU_REGS) {
            if (CPUHasAVX() || FORCE_AVX_REGS)
            {
                m_state.context.fpu.avx.__fpu_reserved[0] = -1;
                m_state.context.fpu.avx.__fpu_reserved[1] = -1;
                *(uint16_t *)&(m_state.context.fpu.avx.__fpu_fcw) = 0x1234;
                *(uint16_t *)&(m_state.context.fpu.avx.__fpu_fsw) = 0x5678;
                m_state.context.fpu.avx.__fpu_ftw = 1;
                m_state.context.fpu.avx.__fpu_rsrv1 = UINT8_MAX;
                m_state.context.fpu.avx.__fpu_fop = 2;
                m_state.context.fpu.avx.__fpu_ip = 3;
                m_state.context.fpu.avx.__fpu_cs = 4;
                m_state.context.fpu.avx.__fpu_rsrv2 = 5;
                m_state.context.fpu.avx.__fpu_dp = 6;
                m_state.context.fpu.avx.__fpu_ds = 7;
                m_state.context.fpu.avx.__fpu_rsrv3 = UINT16_MAX;
                m_state.context.fpu.avx.__fpu_mxcsr = 8;
                m_state.context.fpu.avx.__fpu_mxcsrmask = 9;
                int i;
                for (i=0; i<16; ++i)
                {
                    if (i<10)
                    {
                        m_state.context.fpu.avx.__fpu_stmm0.__mmst_reg[i] = 'a';
                        m_state.context.fpu.avx.__fpu_stmm1.__mmst_reg[i] = 'b';
                        m_state.context.fpu.avx.__fpu_stmm2.__mmst_reg[i] = 'c';
                        m_state.context.fpu.avx.__fpu_stmm3.__mmst_reg[i] = 'd';
                        m_state.context.fpu.avx.__fpu_stmm4.__mmst_reg[i] = 'e';
                        m_state.context.fpu.avx.__fpu_stmm5.__mmst_reg[i] = 'f';
                        m_state.context.fpu.avx.__fpu_stmm6.__mmst_reg[i] = 'g';
                        m_state.context.fpu.avx.__fpu_stmm7.__mmst_reg[i] = 'h';
                    }
                    else
                    {
                        m_state.context.fpu.avx.__fpu_stmm0.__mmst_reg[i] = INT8_MIN;
                        m_state.context.fpu.avx.__fpu_stmm1.__mmst_reg[i] = INT8_MIN;
                        m_state.context.fpu.avx.__fpu_stmm2.__mmst_reg[i] = INT8_MIN;
                        m_state.context.fpu.avx.__fpu_stmm3.__mmst_reg[i] = INT8_MIN;
                        m_state.context.fpu.avx.__fpu_stmm4.__mmst_reg[i] = INT8_MIN;
                        m_state.context.fpu.avx.__fpu_stmm5.__mmst_reg[i] = INT8_MIN;
                        m_state.context.fpu.avx.__fpu_stmm6.__mmst_reg[i] = INT8_MIN;
                        m_state.context.fpu.avx.__fpu_stmm7.__mmst_reg[i] = INT8_MIN;
                    }
                    
                    m_state.context.fpu.avx.__fpu_xmm0.__xmm_reg[i] = '0';
                    m_state.context.fpu.avx.__fpu_xmm1.__xmm_reg[i] = '1';
                    m_state.context.fpu.avx.__fpu_xmm2.__xmm_reg[i] = '2';
                    m_state.context.fpu.avx.__fpu_xmm3.__xmm_reg[i] = '3';
                    m_state.context.fpu.avx.__fpu_xmm4.__xmm_reg[i] = '4';
                    m_state.context.fpu.avx.__fpu_xmm5.__xmm_reg[i] = '5';
                    m_state.context.fpu.avx.__fpu_xmm6.__xmm_reg[i] = '6';
                    m_state.context.fpu.avx.__fpu_xmm7.__xmm_reg[i] = '7';
                    m_state.context.fpu.avx.__fpu_xmm8.__xmm_reg[i] = '8';
                    m_state.context.fpu.avx.__fpu_xmm9.__xmm_reg[i] = '9';
                    m_state.context.fpu.avx.__fpu_xmm10.__xmm_reg[i] = 'A';
                    m_state.context.fpu.avx.__fpu_xmm11.__xmm_reg[i] = 'B';
                    m_state.context.fpu.avx.__fpu_xmm12.__xmm_reg[i] = 'C';
                    m_state.context.fpu.avx.__fpu_xmm13.__xmm_reg[i] = 'D';
                    m_state.context.fpu.avx.__fpu_xmm14.__xmm_reg[i] = 'E';
                    m_state.context.fpu.avx.__fpu_xmm15.__xmm_reg[i] = 'F';
                    
                    m_state.context.fpu.avx.__fpu_ymmh0.__xmm_reg[i] = '0';
                    m_state.context.fpu.avx.__fpu_ymmh1.__xmm_reg[i] = '1';
                    m_state.context.fpu.avx.__fpu_ymmh2.__xmm_reg[i] = '2';
                    m_state.context.fpu.avx.__fpu_ymmh3.__xmm_reg[i] = '3';
                    m_state.context.fpu.avx.__fpu_ymmh4.__xmm_reg[i] = '4';
                    m_state.context.fpu.avx.__fpu_ymmh5.__xmm_reg[i] = '5';
                    m_state.context.fpu.avx.__fpu_ymmh6.__xmm_reg[i] = '6';
                    m_state.context.fpu.avx.__fpu_ymmh7.__xmm_reg[i] = '7';
                    m_state.context.fpu.avx.__fpu_ymmh8.__xmm_reg[i] = '8';
                    m_state.context.fpu.avx.__fpu_ymmh9.__xmm_reg[i] = '9';
                    m_state.context.fpu.avx.__fpu_ymmh10.__xmm_reg[i] = 'A';
                    m_state.context.fpu.avx.__fpu_ymmh11.__xmm_reg[i] = 'B';
                    m_state.context.fpu.avx.__fpu_ymmh12.__xmm_reg[i] = 'C';
                    m_state.context.fpu.avx.__fpu_ymmh13.__xmm_reg[i] = 'D';
                    m_state.context.fpu.avx.__fpu_ymmh14.__xmm_reg[i] = 'E';
                    m_state.context.fpu.avx.__fpu_ymmh15.__xmm_reg[i] = 'F';
                }
                for (i=0; i<sizeof(m_state.context.fpu.avx.__fpu_rsrv4); ++i)
                    m_state.context.fpu.avx.__fpu_rsrv4[i] = INT8_MIN;
                m_state.context.fpu.avx.__fpu_reserved1 = -1;
                for (i=0; i<sizeof(m_state.context.fpu.avx.__avx_reserved1); ++i)
                    m_state.context.fpu.avx.__avx_reserved1[i] = INT8_MIN;
                m_state.SetError(e_regSetFPU, Read, 0);
            }
            else
            {
                m_state.context.fpu.no_avx.__fpu_reserved[0] = -1;
                m_state.context.fpu.no_avx.__fpu_reserved[1] = -1;
                *(uint16_t *)&(m_state.context.fpu.no_avx.__fpu_fcw) = 0x1234;
                *(uint16_t *)&(m_state.context.fpu.no_avx.__fpu_fsw) = 0x5678;
                m_state.context.fpu.no_avx.__fpu_ftw = 1;
                m_state.context.fpu.no_avx.__fpu_rsrv1 = UINT8_MAX;
                m_state.context.fpu.no_avx.__fpu_fop = 2;
                m_state.context.fpu.no_avx.__fpu_ip = 3;
                m_state.context.fpu.no_avx.__fpu_cs = 4;
                m_state.context.fpu.no_avx.__fpu_rsrv2 = 5;
                m_state.context.fpu.no_avx.__fpu_dp = 6;
                m_state.context.fpu.no_avx.__fpu_ds = 7;
                m_state.context.fpu.no_avx.__fpu_rsrv3 = UINT16_MAX;
                m_state.context.fpu.no_avx.__fpu_mxcsr = 8;
                m_state.context.fpu.no_avx.__fpu_mxcsrmask = 9;
                int i;
                for (i=0; i<16; ++i)
                {
                    if (i<10)
                    {
                        m_state.context.fpu.no_avx.__fpu_stmm0.__mmst_reg[i] = 'a';
                        m_state.context.fpu.no_avx.__fpu_stmm1.__mmst_reg[i] = 'b';
                        m_state.context.fpu.no_avx.__fpu_stmm2.__mmst_reg[i] = 'c';
                        m_state.context.fpu.no_avx.__fpu_stmm3.__mmst_reg[i] = 'd';
                        m_state.context.fpu.no_avx.__fpu_stmm4.__mmst_reg[i] = 'e';
                        m_state.context.fpu.no_avx.__fpu_stmm5.__mmst_reg[i] = 'f';
                        m_state.context.fpu.no_avx.__fpu_stmm6.__mmst_reg[i] = 'g';
                        m_state.context.fpu.no_avx.__fpu_stmm7.__mmst_reg[i] = 'h';
                    }
                    else
                    {
                        m_state.context.fpu.no_avx.__fpu_stmm0.__mmst_reg[i] = INT8_MIN;
                        m_state.context.fpu.no_avx.__fpu_stmm1.__mmst_reg[i] = INT8_MIN;
                        m_state.context.fpu.no_avx.__fpu_stmm2.__mmst_reg[i] = INT8_MIN;
                        m_state.context.fpu.no_avx.__fpu_stmm3.__mmst_reg[i] = INT8_MIN;
                        m_state.context.fpu.no_avx.__fpu_stmm4.__mmst_reg[i] = INT8_MIN;
                        m_state.context.fpu.no_avx.__fpu_stmm5.__mmst_reg[i] = INT8_MIN;
                        m_state.context.fpu.no_avx.__fpu_stmm6.__mmst_reg[i] = INT8_MIN;
                        m_state.context.fpu.no_avx.__fpu_stmm7.__mmst_reg[i] = INT8_MIN;
                    }
                    
                    m_state.context.fpu.no_avx.__fpu_xmm0.__xmm_reg[i] = '0';
                    m_state.context.fpu.no_avx.__fpu_xmm1.__xmm_reg[i] = '1';
                    m_state.context.fpu.no_avx.__fpu_xmm2.__xmm_reg[i] = '2';
                    m_state.context.fpu.no_avx.__fpu_xmm3.__xmm_reg[i] = '3';
                    m_state.context.fpu.no_avx.__fpu_xmm4.__xmm_reg[i] = '4';
                    m_state.context.fpu.no_avx.__fpu_xmm5.__xmm_reg[i] = '5';
                    m_state.context.fpu.no_avx.__fpu_xmm6.__xmm_reg[i] = '6';
                    m_state.context.fpu.no_avx.__fpu_xmm7.__xmm_reg[i] = '7';
                    m_state.context.fpu.no_avx.__fpu_xmm8.__xmm_reg[i] = '8';
                    m_state.context.fpu.no_avx.__fpu_xmm9.__xmm_reg[i] = '9';
                    m_state.context.fpu.no_avx.__fpu_xmm10.__xmm_reg[i] = 'A';
                    m_state.context.fpu.no_avx.__fpu_xmm11.__xmm_reg[i] = 'B';
                    m_state.context.fpu.no_avx.__fpu_xmm12.__xmm_reg[i] = 'C';
                    m_state.context.fpu.no_avx.__fpu_xmm13.__xmm_reg[i] = 'D';
                    m_state.context.fpu.no_avx.__fpu_xmm14.__xmm_reg[i] = 'E';
                    m_state.context.fpu.no_avx.__fpu_xmm15.__xmm_reg[i] = 'F';
                }
                for (i=0; i<sizeof(m_state.context.fpu.no_avx.__fpu_rsrv4); ++i)
                    m_state.context.fpu.no_avx.__fpu_rsrv4[i] = INT8_MIN;
                m_state.context.fpu.no_avx.__fpu_reserved1 = -1;
                m_state.SetError(e_regSetFPU, Read, 0);
            }
        }
        else
        {
            if (CPUHasAVX() || FORCE_AVX_REGS)
            {
                mach_msg_type_number_t count = e_regSetWordSizeAVX;
                m_state.SetError(e_regSetFPU, Read, ::thread_get_state(m_thread->ThreadID(), __x86_64_AVX_STATE, (thread_state_t)&m_state.context.fpu.avx, &count));
                DNBLogThreadedIf (LOG_THREAD, "::thread_get_state (0x%4.4x, %u, &avx, %u (%u passed in) carp) => 0x%8.8x",
                                  m_thread->ThreadID(), __x86_64_AVX_STATE, (uint32_t)count, 
                                  e_regSetWordSizeAVX, m_state.GetError(e_regSetFPU, Read));
            }
            else
            {
                mach_msg_type_number_t count = e_regSetWordSizeFPU;
                m_state.SetError(e_regSetFPU, Read, ::thread_get_state(m_thread->ThreadID(), __x86_64_FLOAT_STATE, (thread_state_t)&m_state.context.fpu.no_avx, &count));
                DNBLogThreadedIf (LOG_THREAD, "::thread_get_state (0x%4.4x, %u, &fpu, %u (%u passed in) => 0x%8.8x",
                                  m_thread->ThreadID(), __x86_64_FLOAT_STATE, (uint32_t)count, 
                                  e_regSetWordSizeFPU, m_state.GetError(e_regSetFPU, Read));
            }
        }        
    }
    return m_state.GetError(e_regSetFPU, Read);
}

kern_return_t
DNBArchImplX86_64::GetEXCState(bool force)
{
    if (force || m_state.GetError(e_regSetEXC, Read))
    {
        mach_msg_type_number_t count = e_regSetWordSizeEXC;
        m_state.SetError(e_regSetEXC, Read, ::thread_get_state(m_thread->ThreadID(), __x86_64_EXCEPTION_STATE, (thread_state_t)&m_state.context.exc, &count));
    }
    return m_state.GetError(e_regSetEXC, Read);
}

kern_return_t
DNBArchImplX86_64::SetGPRState()
{
    kern_return_t kret = ::thread_abort_safely(m_thread->ThreadID());
    DNBLogThreadedIf (LOG_THREAD, "thread = 0x%4.4x calling thread_abort_safely (tid) => %u (SetGPRState() for stop_count = %u)", m_thread->ThreadID(), kret, m_thread->Process()->StopCount());    

    m_state.SetError(e_regSetGPR, Write, ::thread_set_state(m_thread->ThreadID(), __x86_64_THREAD_STATE, (thread_state_t)&m_state.context.gpr, e_regSetWordSizeGPR));
    DNBLogThreadedIf (LOG_THREAD, "::thread_set_state (0x%4.4x, %u, &gpr, %u) => 0x%8.8x"
                      "\n\trax = %16.16llx rbx = %16.16llx rcx = %16.16llx rdx = %16.16llx"
                      "\n\trdi = %16.16llx rsi = %16.16llx rbp = %16.16llx rsp = %16.16llx"
                      "\n\t r8 = %16.16llx  r9 = %16.16llx r10 = %16.16llx r11 = %16.16llx"
                      "\n\tr12 = %16.16llx r13 = %16.16llx r14 = %16.16llx r15 = %16.16llx"
                      "\n\trip = %16.16llx"
                      "\n\tflg = %16.16llx  cs = %16.16llx  fs = %16.16llx  gs = %16.16llx",
                      m_thread->ThreadID(), __x86_64_THREAD_STATE, e_regSetWordSizeGPR,
                      m_state.GetError(e_regSetGPR, Write),
                      m_state.context.gpr.__rax,m_state.context.gpr.__rbx,m_state.context.gpr.__rcx,
                      m_state.context.gpr.__rdx,m_state.context.gpr.__rdi,m_state.context.gpr.__rsi,
                      m_state.context.gpr.__rbp,m_state.context.gpr.__rsp,m_state.context.gpr.__r8,
                      m_state.context.gpr.__r9, m_state.context.gpr.__r10,m_state.context.gpr.__r11,
                      m_state.context.gpr.__r12,m_state.context.gpr.__r13,m_state.context.gpr.__r14,
                      m_state.context.gpr.__r15,m_state.context.gpr.__rip,m_state.context.gpr.__rflags,
                      m_state.context.gpr.__cs, m_state.context.gpr.__fs, m_state.context.gpr.__gs);
    return m_state.GetError(e_regSetGPR, Write);
}

kern_return_t
DNBArchImplX86_64::SetFPUState()
{
    if (DEBUG_FPU_REGS)
    {
        m_state.SetError(e_regSetFPU, Write, 0);
        return m_state.GetError(e_regSetFPU, Write);   
    }
    else
    {
        if (CPUHasAVX() || FORCE_AVX_REGS)
        {
            m_state.SetError(e_regSetFPU, Write, ::thread_set_state(m_thread->ThreadID(), __x86_64_AVX_STATE, (thread_state_t)&m_state.context.fpu.avx, e_regSetWordSizeAVX));
            return m_state.GetError(e_regSetFPU, Write);
        }
        else
        {
            m_state.SetError(e_regSetFPU, Write, ::thread_set_state(m_thread->ThreadID(), __x86_64_FLOAT_STATE, (thread_state_t)&m_state.context.fpu.no_avx, e_regSetWordSizeFPU));
            return m_state.GetError(e_regSetFPU, Write);
        }
    }
}

kern_return_t
DNBArchImplX86_64::SetEXCState()
{
    m_state.SetError(e_regSetEXC, Write, ::thread_set_state(m_thread->ThreadID(), __x86_64_EXCEPTION_STATE, (thread_state_t)&m_state.context.exc, e_regSetWordSizeEXC));
    return m_state.GetError(e_regSetEXC, Write);
}

kern_return_t
DNBArchImplX86_64::GetDBGState(bool force)
{
    if (force || m_state.GetError(e_regSetDBG, Read))
    {
        mach_msg_type_number_t count = e_regSetWordSizeDBG;
        m_state.SetError(e_regSetDBG, Read, ::thread_get_state(m_thread->ThreadID(), __x86_64_DEBUG_STATE, (thread_state_t)&m_state.context.dbg, &count));
    }
    return m_state.GetError(e_regSetDBG, Read);
}

kern_return_t
DNBArchImplX86_64::SetDBGState()
{
    m_state.SetError(e_regSetDBG, Write, ::thread_set_state(m_thread->ThreadID(), __x86_64_DEBUG_STATE, (thread_state_t)&m_state.context.dbg, e_regSetWordSizeDBG));
    return m_state.GetError(e_regSetDBG, Write);
}

void
DNBArchImplX86_64::ThreadWillResume()
{
    // Do we need to step this thread? If so, let the mach thread tell us so.
    if (m_thread->IsStepping())
    {
        // This is the primary thread, let the arch do anything it needs
        EnableHardwareSingleStep(true);
    }

    // Reset the debug status register, if necessary, before we resume.
    kern_return_t kret = GetDBGState(false);
    DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchImplX86_64::ThreadWillResume() GetDBGState() => 0x%8.8x.", kret);
    if (kret != KERN_SUCCESS)
        return;

    DBG &debug_state = m_state.context.dbg;
    bool need_reset = false;
    uint32_t i, num = NumSupportedHardwareWatchpoints();
    for (i = 0; i < num; ++i)
        if (IsWatchpointHit(debug_state, i))
            need_reset = true;

    if (need_reset)
    {
        ClearWatchpointHits(debug_state);
        kret = SetDBGState();
        DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchImplX86_64::ThreadWillResume() SetDBGState() => 0x%8.8x.", kret);
    }
}

bool
DNBArchImplX86_64::ThreadDidStop()
{
    bool success = true;
    
    m_state.InvalidateAllRegisterStates();
    
    // Are we stepping a single instruction?
    if (GetGPRState(true) == KERN_SUCCESS)
    {
        // We are single stepping, was this the primary thread?
        if (m_thread->IsStepping())
        {
            // This was the primary thread, we need to clear the trace
            // bit if so.
            success = EnableHardwareSingleStep(false) == KERN_SUCCESS;
        }
        else
        {
            // The MachThread will automatically restore the suspend count
            // in ThreadDidStop(), so we don't need to do anything here if
            // we weren't the primary thread the last time
        }
    }
    return success;
}

bool
DNBArchImplX86_64::NotifyException(MachException::Data& exc)
{
    switch (exc.exc_type)
    {
        case EXC_BAD_ACCESS:
            break;
        case EXC_BAD_INSTRUCTION:
            break;
        case EXC_ARITHMETIC:
            break;
        case EXC_EMULATION:
            break;
        case EXC_SOFTWARE:
            break;
        case EXC_BREAKPOINT:
            if (exc.exc_data.size() >= 2 && exc.exc_data[0] == 2)
            {
                // exc_code = EXC_I386_BPT
                //
                nub_addr_t pc = GetPC(INVALID_NUB_ADDRESS);
                if (pc != INVALID_NUB_ADDRESS && pc > 0)
                {
                    pc -= 1;
                    // Check for a breakpoint at one byte prior to the current PC value
                    // since the PC will be just past the trap.
                    
                    nub_break_t breakID = m_thread->Process()->Breakpoints().FindIDByAddress(pc);
                    if (NUB_BREAK_ID_IS_VALID(breakID))
                    {
                        // Backup the PC for i386 since the trap was taken and the PC
                        // is at the address following the single byte trap instruction.
                        if (m_state.context.gpr.__rip > 0)
                        {
                            m_state.context.gpr.__rip = pc;
                            // Write the new PC back out
                            SetGPRState ();
                        }
                    }
                    return true;
                }
            }
            else if (exc.exc_data.size() >= 2 && exc.exc_data[0] == 1)
            {
                // exc_code = EXC_I386_SGL
                //
                // Check whether this corresponds to a watchpoint hit event.
                // If yes, set the exc_sub_code to the data break address.
                nub_addr_t addr = 0;
                uint32_t hw_index = GetHardwareWatchpointHit(addr);
                if (hw_index != INVALID_NUB_HW_INDEX)
                {
                    exc.exc_data[1] = addr;
                    // Piggyback the hw_index in the exc.data.
                    exc.exc_data.push_back(hw_index);
                }

                return true;
            }
            break;
        case EXC_SYSCALL:
            break;
        case EXC_MACH_SYSCALL:
            break;
        case EXC_RPC_ALERT:
            break;
    }
    return false;
}

uint32_t
DNBArchImplX86_64::NumSupportedHardwareWatchpoints()
{
    // Available debug address registers: dr0, dr1, dr2, dr3.
    return 4;
}

static uint32_t
size_and_rw_bits(nub_size_t size, bool read, bool write)
{
    uint32_t rw;
    if (read) {
        rw = 0x3; // READ or READ/WRITE
    } else if (write) {
        rw = 0x1; // WRITE
    } else {
        assert(0 && "read and write cannot both be false");
    }

    switch (size) {
    case 1:
        return rw;
    case 2:
        return (0x1 << 2) | rw;
    case 4:
        return (0x3 << 2) | rw;
    case 8:
        return (0x2 << 2) | rw;
    default:
        assert(0 && "invalid size, must be one of 1, 2, 4, or 8");
    }    
}
void
DNBArchImplX86_64::SetWatchpoint(DBG &debug_state, uint32_t hw_index, nub_addr_t addr, nub_size_t size, bool read, bool write)
{
    // Set both dr7 (debug control register) and dri (debug address register).
    
    // dr7{7-0} encodes the local/gloabl enable bits:
    //  global enable --. .-- local enable
    //                  | |
    //                  v v
    //      dr0 -> bits{1-0}
    //      dr1 -> bits{3-2}
    //      dr2 -> bits{5-4}
    //      dr3 -> bits{7-6}
    //
    // dr7{31-16} encodes the rw/len bits:
    //  b_x+3, b_x+2, b_x+1, b_x
    //      where bits{x+1, x} => rw
    //            0b00: execute, 0b01: write, 0b11: read-or-write, 0b10: io read-or-write (unused)
    //      and bits{x+3, x+2} => len
    //            0b00: 1-byte, 0b01: 2-byte, 0b11: 4-byte, 0b10: 8-byte
    //
    //      dr0 -> bits{19-16}
    //      dr1 -> bits{23-20}
    //      dr2 -> bits{27-24}
    //      dr3 -> bits{31-28}
    debug_state.__dr7 |= (1 << (2*hw_index) |
                          size_and_rw_bits(size, read, write) << (16+4*hw_index));
    switch (hw_index) {
    case 0:
        debug_state.__dr0 = addr; break;
    case 1:
        debug_state.__dr1 = addr; break;
    case 2:
        debug_state.__dr2 = addr; break;
    case 3:
        debug_state.__dr3 = addr; break;
    default:
        assert(0 && "invalid hardware register index, must be one of 0, 1, 2, or 3");
    }
    return;
}

void
DNBArchImplX86_64::ClearWatchpoint(DBG &debug_state, uint32_t hw_index)
{
    debug_state.__dr7 &= ~(3 << (2*hw_index));
    switch (hw_index) {
    case 0:
        debug_state.__dr0 = 0; break;
    case 1:
        debug_state.__dr1 = 0; break;
    case 2:
        debug_state.__dr2 = 0; break;
    case 3:
        debug_state.__dr3 = 0; break;
    default:
        assert(0 && "invalid hardware register index, must be one of 0, 1, 2, or 3");
    }
    return;
}

bool
DNBArchImplX86_64::IsWatchpointVacant(const DBG &debug_state, uint32_t hw_index)
{
    // Check dr7 (debug control register) for local/global enable bits:
    //  global enable --. .-- local enable
    //                  | |
    //                  v v
    //      dr0 -> bits{1-0}
    //      dr1 -> bits{3-2}
    //      dr2 -> bits{5-4}
    //      dr3 -> bits{7-6}
    return (debug_state.__dr7 & (3 << (2*hw_index))) == 0;
}

// Resets local copy of debug status register to wait for the next debug excpetion.
void
DNBArchImplX86_64::ClearWatchpointHits(DBG &debug_state)
{
    // See also IsWatchpointHit().
    debug_state.__dr6 = 0;
    return;
}

bool
DNBArchImplX86_64::IsWatchpointHit(const DBG &debug_state, uint32_t hw_index)
{
    // Check dr6 (debug status register) whether a watchpoint hits:
    //          is watchpoint hit?
    //                  |
    //                  v
    //      dr0 -> bits{0}
    //      dr1 -> bits{1}
    //      dr2 -> bits{2}
    //      dr3 -> bits{3}
    return (debug_state.__dr6 & (1 << hw_index));
}

nub_addr_t
DNBArchImplX86_64::GetWatchAddress(const DBG &debug_state, uint32_t hw_index)
{
    switch (hw_index) {
    case 0:
        return debug_state.__dr0;
    case 1:
        return debug_state.__dr1;
    case 2:
        return debug_state.__dr2;
    case 3:
        return debug_state.__dr3;
    default:
        assert(0 && "invalid hardware register index, must be one of 0, 1, 2, or 3");
    }
}

bool
DNBArchImplX86_64::StartTransForHWP()
{
    if (m_2pc_trans_state != Trans_Done || m_2pc_trans_state != Trans_Rolled_Back)
        DNBLogError ("%s inconsistent state detected, expected %d or %d, got: %d", __FUNCTION__, Trans_Done, Trans_Rolled_Back, m_2pc_trans_state);
    m_2pc_dbg_checkpoint = m_state.context.dbg;
    m_2pc_trans_state = Trans_Pending;
    return true;
}
bool
DNBArchImplX86_64::RollbackTransForHWP()
{
    m_state.context.dbg = m_2pc_dbg_checkpoint;
    if (m_2pc_trans_state != Trans_Pending)
        DNBLogError ("%s inconsistent state detected, expected %d, got: %d", __FUNCTION__, Trans_Pending, m_2pc_trans_state);
    m_2pc_trans_state = Trans_Rolled_Back;
    kern_return_t kret = SetDBGState();
    DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchImplX86_64::RollbackTransForHWP() SetDBGState() => 0x%8.8x.", kret);

    if (kret == KERN_SUCCESS)
        return true;
    else
        return false;
}
bool
DNBArchImplX86_64::FinishTransForHWP()
{
    m_2pc_trans_state = Trans_Done;
    return true;
}
DNBArchImplX86_64::DBG
DNBArchImplX86_64::GetDBGCheckpoint()
{
    return m_2pc_dbg_checkpoint;
}

uint32_t
DNBArchImplX86_64::EnableHardwareWatchpoint (nub_addr_t addr, nub_size_t size, bool read, bool write)
{
    DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchImplX86_64::EnableHardwareWatchpoint(addr = 0x%llx, size = %llu, read = %u, write = %u)", (uint64_t)addr, (uint64_t)size, read, write);

    const uint32_t num_hw_watchpoints = NumSupportedHardwareWatchpoints();

    // Can only watch 1, 2, 4, or 8 bytes.
    if (!(size == 1 || size == 2 || size == 4 || size == 8))
        return INVALID_NUB_HW_INDEX;

    // We must watch for either read or write
    if (read == false && write == false)
        return INVALID_NUB_HW_INDEX;

    // Read the debug state
    kern_return_t kret = GetDBGState(false);

    if (kret == KERN_SUCCESS)
    {
        // Check to make sure we have the needed hardware support
        uint32_t i = 0;

        DBG &debug_state = m_state.context.dbg;
        for (i = 0; i < num_hw_watchpoints; ++i)
        {
            if (IsWatchpointVacant(debug_state, i))
                break;
        }

        // See if we found an available hw breakpoint slot above
        if (i < num_hw_watchpoints)
        {
            StartTransForHWP();

            // Modify our local copy of the debug state, first.
            SetWatchpoint(debug_state, i, addr, size, read, write);
            // Now set the watch point in the inferior.
            kret = SetDBGState();
            DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchImplX86_64::EnableHardwareWatchpoint() SetDBGState() => 0x%8.8x.", kret);

            if (kret == KERN_SUCCESS)
                return i;
            else // Revert to the previous debug state voluntarily.  The transaction coordinator knows that we have failed.
                m_state.context.dbg = GetDBGCheckpoint();
        }
        else
        {
            DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchImplX86_64::EnableHardwareWatchpoint(): All hardware resources (%u) are in use.", num_hw_watchpoints);
        }
    }
    return INVALID_NUB_HW_INDEX;
}

bool
DNBArchImplX86_64::DisableHardwareWatchpoint (uint32_t hw_index)
{
    kern_return_t kret = GetDBGState(false);

    const uint32_t num_hw_points = NumSupportedHardwareWatchpoints();
    if (kret == KERN_SUCCESS)
    {
        DBG &debug_state = m_state.context.dbg;
        if (hw_index < num_hw_points && !IsWatchpointVacant(debug_state, hw_index))
        {
            StartTransForHWP();

            // Modify our local copy of the debug state, first.
            ClearWatchpoint(debug_state, hw_index);
            // Now disable the watch point in the inferior.
            kret = SetDBGState();
            DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchImplX86_64::DisableHardwareWatchpoint( %u )",
                             hw_index);

            if (kret == KERN_SUCCESS)
                return true;
            else // Revert to the previous debug state voluntarily.  The transaction coordinator knows that we have failed.
                m_state.context.dbg = GetDBGCheckpoint();
        }
    }
    return false;
}

DNBArchImplX86_64::DBG DNBArchImplX86_64::Global_Debug_State = {0,0,0,0,0,0,0,0};
bool DNBArchImplX86_64::Valid_Global_Debug_State = false;

// Use this callback from MachThread, which in turn was called from MachThreadList, to update
// the global view of the hardware watchpoint state, so that when new thread comes along, they
// get to inherit the existing hardware watchpoint state.
void
DNBArchImplX86_64::HardwareWatchpointStateChanged ()
{
    Global_Debug_State = m_state.context.dbg;
    Valid_Global_Debug_State = true;
}

// Iterate through the debug status register; return the index of the first hit.
uint32_t
DNBArchImplX86_64::GetHardwareWatchpointHit(nub_addr_t &addr)
{
    // Read the debug state
    kern_return_t kret = GetDBGState(true);
    DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchImplX86_64::GetHardwareWatchpointHit() GetDBGState() => 0x%8.8x.", kret);
    if (kret == KERN_SUCCESS)
    {
        DBG &debug_state = m_state.context.dbg;
        uint32_t i, num = NumSupportedHardwareWatchpoints();
        for (i = 0; i < num; ++i)
        {
            if (IsWatchpointHit(debug_state, i))
            {
                addr = GetWatchAddress(debug_state, i);
                DNBLogThreadedIf(LOG_WATCHPOINTS,
                                 "DNBArchImplX86_64::GetHardwareWatchpointHit() found => %u (addr = 0x%llx).",
                                 i, 
                                 (uint64_t)addr);
                return i;
            }
        }
    }
    return INVALID_NUB_HW_INDEX;
}

// Set the single step bit in the processor status register.
kern_return_t
DNBArchImplX86_64::EnableHardwareSingleStep (bool enable)
{
    if (GetGPRState(false) == KERN_SUCCESS)
    {
        const uint32_t trace_bit = 0x100u;
        if (enable)
            m_state.context.gpr.__rflags |= trace_bit;
        else
            m_state.context.gpr.__rflags &= ~trace_bit;
        return SetGPRState();
    }
    return m_state.GetError(e_regSetGPR, Read);
}


//----------------------------------------------------------------------
// Register information defintions
//----------------------------------------------------------------------

enum
{
    gpr_rax = 0,
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
    k_num_gpr_regs
};

enum {
    fpu_fcw,
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
    fpu_ymm0,
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
    k_num_fpu_regs,
    
    // Aliases
    fpu_fctrl = fpu_fcw,
    fpu_fstat = fpu_fsw,
    fpu_ftag  = fpu_ftw,
    fpu_fiseg = fpu_cs,
    fpu_fioff = fpu_ip,
    fpu_foseg = fpu_ds,
    fpu_fooff = fpu_dp
};

enum {
    exc_trapno,
    exc_err,
    exc_faultvaddr,
    k_num_exc_regs,
};


enum gcc_dwarf_regnums
{
    gcc_dwarf_rax = 0,
    gcc_dwarf_rdx = 1,
    gcc_dwarf_rcx = 2,
    gcc_dwarf_rbx = 3,
    gcc_dwarf_rsi = 4,
    gcc_dwarf_rdi = 5,
    gcc_dwarf_rbp = 6,
    gcc_dwarf_rsp = 7,
    gcc_dwarf_r8,
    gcc_dwarf_r9,
    gcc_dwarf_r10,
    gcc_dwarf_r11,
    gcc_dwarf_r12,
    gcc_dwarf_r13,
    gcc_dwarf_r14,
    gcc_dwarf_r15,
    gcc_dwarf_rip,
    gcc_dwarf_xmm0,
    gcc_dwarf_xmm1,
    gcc_dwarf_xmm2,
    gcc_dwarf_xmm3,
    gcc_dwarf_xmm4,
    gcc_dwarf_xmm5,
    gcc_dwarf_xmm6,
    gcc_dwarf_xmm7,
    gcc_dwarf_xmm8,
    gcc_dwarf_xmm9,
    gcc_dwarf_xmm10,
    gcc_dwarf_xmm11,
    gcc_dwarf_xmm12,
    gcc_dwarf_xmm13,
    gcc_dwarf_xmm14,
    gcc_dwarf_xmm15,
    gcc_dwarf_stmm0,
    gcc_dwarf_stmm1,
    gcc_dwarf_stmm2,
    gcc_dwarf_stmm3,
    gcc_dwarf_stmm4,
    gcc_dwarf_stmm5,
    gcc_dwarf_stmm6,
    gcc_dwarf_stmm7,
    gcc_dwarf_ymm0 = gcc_dwarf_xmm0,
    gcc_dwarf_ymm1 = gcc_dwarf_xmm1,
    gcc_dwarf_ymm2 = gcc_dwarf_xmm2,
    gcc_dwarf_ymm3 = gcc_dwarf_xmm3,
    gcc_dwarf_ymm4 = gcc_dwarf_xmm4,
    gcc_dwarf_ymm5 = gcc_dwarf_xmm5,
    gcc_dwarf_ymm6 = gcc_dwarf_xmm6,
    gcc_dwarf_ymm7 = gcc_dwarf_xmm7,
    gcc_dwarf_ymm8 = gcc_dwarf_xmm8,
    gcc_dwarf_ymm9 = gcc_dwarf_xmm9,
    gcc_dwarf_ymm10 = gcc_dwarf_xmm10,
    gcc_dwarf_ymm11 = gcc_dwarf_xmm11,
    gcc_dwarf_ymm12 = gcc_dwarf_xmm12,
    gcc_dwarf_ymm13 = gcc_dwarf_xmm13,
    gcc_dwarf_ymm14 = gcc_dwarf_xmm14,
    gcc_dwarf_ymm15 = gcc_dwarf_xmm15
};

enum gdb_regnums
{
    gdb_rax     =   0,
    gdb_rbx     =   1,
    gdb_rcx     =   2,
    gdb_rdx     =   3,
    gdb_rsi     =   4,
    gdb_rdi     =   5,
    gdb_rbp     =   6,
    gdb_rsp     =   7,
    gdb_r8      =   8,
    gdb_r9      =   9,
    gdb_r10     =  10,
    gdb_r11     =  11,
    gdb_r12     =  12,
    gdb_r13     =  13,
    gdb_r14     =  14,
    gdb_r15     =  15,
    gdb_rip     =  16,
    gdb_rflags  =  17,
    gdb_cs      =  18,
    gdb_ss      =  19,
    gdb_ds      =  20,
    gdb_es      =  21,
    gdb_fs      =  22,
    gdb_gs      =  23,
    gdb_stmm0   =  24,
    gdb_stmm1   =  25,
    gdb_stmm2   =  26,
    gdb_stmm3   =  27,
    gdb_stmm4   =  28,
    gdb_stmm5   =  29,
    gdb_stmm6   =  30,
    gdb_stmm7   =  31,
    gdb_fctrl   =  32,  gdb_fcw = gdb_fctrl,
    gdb_fstat   =  33,  gdb_fsw = gdb_fstat,
    gdb_ftag    =  34,  gdb_ftw = gdb_ftag,
    gdb_fiseg   =  35,  gdb_fpu_cs  = gdb_fiseg,
    gdb_fioff   =  36,  gdb_ip  = gdb_fioff,
    gdb_foseg   =  37,  gdb_fpu_ds  = gdb_foseg,
    gdb_fooff   =  38,  gdb_dp  = gdb_fooff,
    gdb_fop     =  39,
    gdb_xmm0    =  40,
    gdb_xmm1    =  41,
    gdb_xmm2    =  42,
    gdb_xmm3    =  43,
    gdb_xmm4    =  44,
    gdb_xmm5    =  45,
    gdb_xmm6    =  46,
    gdb_xmm7    =  47,
    gdb_xmm8    =  48,
    gdb_xmm9    =  49,
    gdb_xmm10   =  50,
    gdb_xmm11   =  51,
    gdb_xmm12   =  52,
    gdb_xmm13   =  53,
    gdb_xmm14   =  54,
    gdb_xmm15   =  55,
    gdb_mxcsr   =  56,
    gdb_ymm0    =  gdb_xmm0,
    gdb_ymm1    =  gdb_xmm1,
    gdb_ymm2    =  gdb_xmm2,
    gdb_ymm3    =  gdb_xmm3,
    gdb_ymm4    =  gdb_xmm4,
    gdb_ymm5    =  gdb_xmm5,
    gdb_ymm6    =  gdb_xmm6,
    gdb_ymm7    =  gdb_xmm7,
    gdb_ymm8    =  gdb_xmm8,
    gdb_ymm9    =  gdb_xmm9,
    gdb_ymm10   =  gdb_xmm10,
    gdb_ymm11   =  gdb_xmm11,
    gdb_ymm12   =  gdb_xmm12,
    gdb_ymm13   =  gdb_xmm13,
    gdb_ymm14   =  gdb_xmm14,
    gdb_ymm15   =  gdb_xmm15
};

#define GPR_OFFSET(reg) (offsetof (DNBArchImplX86_64::GPR, __##reg))
#define FPU_OFFSET(reg) (offsetof (DNBArchImplX86_64::FPU, __fpu_##reg) + offsetof (DNBArchImplX86_64::Context, fpu.no_avx))
#define AVX_OFFSET(reg) (offsetof (DNBArchImplX86_64::AVX, __fpu_##reg) + offsetof (DNBArchImplX86_64::Context, fpu.avx))
#define EXC_OFFSET(reg) (offsetof (DNBArchImplX86_64::EXC, __##reg)     + offsetof (DNBArchImplX86_64::Context, exc))

// This does not accurately identify the location of ymm0...7 in 
// Context.fpu.avx.  That is because there is a bunch of padding
// in Context.fpu.avx that we don't need.  Offset macros lay out
// the register state that Debugserver transmits to the debugger
// -- not to interpret the thread_get_state info.
#define AVX_OFFSET_YMM(n)   (AVX_OFFSET(xmm7) + FPU_SIZE_XMM(xmm7) + (32 * n))

#define GPR_SIZE(reg)       (sizeof(((DNBArchImplX86_64::GPR *)NULL)->__##reg))
#define FPU_SIZE_UINT(reg)  (sizeof(((DNBArchImplX86_64::FPU *)NULL)->__fpu_##reg))
#define FPU_SIZE_MMST(reg)  (sizeof(((DNBArchImplX86_64::FPU *)NULL)->__fpu_##reg.__mmst_reg))
#define FPU_SIZE_XMM(reg)   (sizeof(((DNBArchImplX86_64::FPU *)NULL)->__fpu_##reg.__xmm_reg))
#define FPU_SIZE_YMM(reg)   (32)
#define EXC_SIZE(reg)       (sizeof(((DNBArchImplX86_64::EXC *)NULL)->__##reg))

// These macros will auto define the register name, alt name, register size,
// register offset, encoding, format and native register. This ensures that
// the register state structures are defined correctly and have the correct
// sizes and offsets.
#define DEFINE_GPR(reg) { e_regSetGPR, gpr_##reg, #reg, NULL, Uint, Hex, GPR_SIZE(reg), GPR_OFFSET(reg), gcc_dwarf_##reg, gcc_dwarf_##reg, INVALID_NUB_REGNUM, gdb_##reg }
#define DEFINE_GPR_ALT(reg, alt, gen) { e_regSetGPR, gpr_##reg, #reg, alt, Uint, Hex, GPR_SIZE(reg), GPR_OFFSET(reg), gcc_dwarf_##reg, gcc_dwarf_##reg, gen, gdb_##reg }
#define DEFINE_GPR_ALT2(reg, alt) { e_regSetGPR, gpr_##reg, #reg, alt, Uint, Hex, GPR_SIZE(reg), GPR_OFFSET(reg), INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, gdb_##reg }
#define DEFINE_GPR_ALT3(reg, alt, gen) { e_regSetGPR, gpr_##reg, #reg, alt, Uint, Hex, GPR_SIZE(reg), GPR_OFFSET(reg), INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, gen, gdb_##reg }

// General purpose registers for 64 bit
const DNBRegisterInfo
DNBArchImplX86_64::g_gpr_registers[] =
{
    DEFINE_GPR      (rax),
    DEFINE_GPR      (rbx),
    DEFINE_GPR_ALT  (rcx , "arg4", GENERIC_REGNUM_ARG4),
    DEFINE_GPR_ALT  (rdx , "arg3", GENERIC_REGNUM_ARG3),
    DEFINE_GPR_ALT  (rdi , "arg1", GENERIC_REGNUM_ARG1),
    DEFINE_GPR_ALT  (rsi , "arg2", GENERIC_REGNUM_ARG2),
    DEFINE_GPR_ALT  (rbp , "fp"  , GENERIC_REGNUM_FP),
    DEFINE_GPR_ALT  (rsp , "sp"  , GENERIC_REGNUM_SP),
    DEFINE_GPR_ALT  (r8  , "arg5", GENERIC_REGNUM_ARG5),
    DEFINE_GPR_ALT  (r9  , "arg6", GENERIC_REGNUM_ARG6),
    DEFINE_GPR      (r10),
    DEFINE_GPR      (r11),
    DEFINE_GPR      (r12),
    DEFINE_GPR      (r13),
    DEFINE_GPR      (r14),
    DEFINE_GPR      (r15),
    DEFINE_GPR_ALT  (rip , "pc", GENERIC_REGNUM_PC),
    DEFINE_GPR_ALT3 (rflags, "flags", GENERIC_REGNUM_FLAGS),
    DEFINE_GPR_ALT2 (cs,        NULL),
    DEFINE_GPR_ALT2 (fs,        NULL),
    DEFINE_GPR_ALT2 (gs,        NULL),
};

// Floating point registers 64 bit
const DNBRegisterInfo
DNBArchImplX86_64::g_fpu_registers_no_avx[] =
{
    { e_regSetFPU, fpu_fcw      , "fctrl"       , NULL, Uint, Hex, FPU_SIZE_UINT(fcw)       , FPU_OFFSET(fcw)       , -1U, -1U, -1U, -1U },
    { e_regSetFPU, fpu_fsw      , "fstat"       , NULL, Uint, Hex, FPU_SIZE_UINT(fsw)       , FPU_OFFSET(fsw)       , -1U, -1U, -1U, -1U },
    { e_regSetFPU, fpu_ftw      , "ftag"        , NULL, Uint, Hex, FPU_SIZE_UINT(ftw)       , FPU_OFFSET(ftw)       , -1U, -1U, -1U, -1U },
    { e_regSetFPU, fpu_fop      , "fop"         , NULL, Uint, Hex, FPU_SIZE_UINT(fop)       , FPU_OFFSET(fop)       , -1U, -1U, -1U, -1U },
    { e_regSetFPU, fpu_ip       , "fioff"       , NULL, Uint, Hex, FPU_SIZE_UINT(ip)        , FPU_OFFSET(ip)        , -1U, -1U, -1U, -1U },
    { e_regSetFPU, fpu_cs       , "fiseg"       , NULL, Uint, Hex, FPU_SIZE_UINT(cs)        , FPU_OFFSET(cs)        , -1U, -1U, -1U, -1U },
    { e_regSetFPU, fpu_dp       , "fooff"       , NULL, Uint, Hex, FPU_SIZE_UINT(dp)        , FPU_OFFSET(dp)        , -1U, -1U, -1U, -1U },
    { e_regSetFPU, fpu_ds       , "foseg"       , NULL, Uint, Hex, FPU_SIZE_UINT(ds)        , FPU_OFFSET(ds)        , -1U, -1U, -1U, -1U },
    { e_regSetFPU, fpu_mxcsr    , "mxcsr"       , NULL, Uint, Hex, FPU_SIZE_UINT(mxcsr)     , FPU_OFFSET(mxcsr)     , -1U, -1U, -1U, -1U },
    { e_regSetFPU, fpu_mxcsrmask, "mxcsrmask"   , NULL, Uint, Hex, FPU_SIZE_UINT(mxcsrmask) , FPU_OFFSET(mxcsrmask) , -1U, -1U, -1U, -1U },
    
    { e_regSetFPU, fpu_stmm0, "stmm0", NULL, Vector, VectorOfUInt8, FPU_SIZE_MMST(stmm0), FPU_OFFSET(stmm0), gcc_dwarf_stmm0, gcc_dwarf_stmm0, -1U, gdb_stmm0 },
    { e_regSetFPU, fpu_stmm1, "stmm1", NULL, Vector, VectorOfUInt8, FPU_SIZE_MMST(stmm1), FPU_OFFSET(stmm1), gcc_dwarf_stmm1, gcc_dwarf_stmm1, -1U, gdb_stmm1 },
    { e_regSetFPU, fpu_stmm2, "stmm2", NULL, Vector, VectorOfUInt8, FPU_SIZE_MMST(stmm2), FPU_OFFSET(stmm2), gcc_dwarf_stmm2, gcc_dwarf_stmm2, -1U, gdb_stmm2 },
    { e_regSetFPU, fpu_stmm3, "stmm3", NULL, Vector, VectorOfUInt8, FPU_SIZE_MMST(stmm3), FPU_OFFSET(stmm3), gcc_dwarf_stmm3, gcc_dwarf_stmm3, -1U, gdb_stmm3 },
    { e_regSetFPU, fpu_stmm4, "stmm4", NULL, Vector, VectorOfUInt8, FPU_SIZE_MMST(stmm4), FPU_OFFSET(stmm4), gcc_dwarf_stmm4, gcc_dwarf_stmm4, -1U, gdb_stmm4 },
    { e_regSetFPU, fpu_stmm5, "stmm5", NULL, Vector, VectorOfUInt8, FPU_SIZE_MMST(stmm5), FPU_OFFSET(stmm5), gcc_dwarf_stmm5, gcc_dwarf_stmm5, -1U, gdb_stmm5 },
    { e_regSetFPU, fpu_stmm6, "stmm6", NULL, Vector, VectorOfUInt8, FPU_SIZE_MMST(stmm6), FPU_OFFSET(stmm6), gcc_dwarf_stmm6, gcc_dwarf_stmm6, -1U, gdb_stmm6 },
    { e_regSetFPU, fpu_stmm7, "stmm7", NULL, Vector, VectorOfUInt8, FPU_SIZE_MMST(stmm7), FPU_OFFSET(stmm7), gcc_dwarf_stmm7, gcc_dwarf_stmm7, -1U, gdb_stmm7 },
    
    { e_regSetFPU, fpu_xmm0 , "xmm0"    , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm0)   , FPU_OFFSET(xmm0) , gcc_dwarf_xmm0 , gcc_dwarf_xmm0 , -1U, gdb_xmm0 },
    { e_regSetFPU, fpu_xmm1 , "xmm1"    , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm1)   , FPU_OFFSET(xmm1) , gcc_dwarf_xmm1 , gcc_dwarf_xmm1 , -1U, gdb_xmm1 },
    { e_regSetFPU, fpu_xmm2 , "xmm2"    , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm2)   , FPU_OFFSET(xmm2) , gcc_dwarf_xmm2 , gcc_dwarf_xmm2 , -1U, gdb_xmm2 },
    { e_regSetFPU, fpu_xmm3 , "xmm3"    , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm3)   , FPU_OFFSET(xmm3) , gcc_dwarf_xmm3 , gcc_dwarf_xmm3 , -1U, gdb_xmm3 },
    { e_regSetFPU, fpu_xmm4 , "xmm4"    , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm4)   , FPU_OFFSET(xmm4) , gcc_dwarf_xmm4 , gcc_dwarf_xmm4 , -1U, gdb_xmm4 },
    { e_regSetFPU, fpu_xmm5 , "xmm5"    , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm5)   , FPU_OFFSET(xmm5) , gcc_dwarf_xmm5 , gcc_dwarf_xmm5 , -1U, gdb_xmm5 },
    { e_regSetFPU, fpu_xmm6 , "xmm6"    , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm6)   , FPU_OFFSET(xmm6) , gcc_dwarf_xmm6 , gcc_dwarf_xmm6 , -1U, gdb_xmm6 },
    { e_regSetFPU, fpu_xmm7 , "xmm7"    , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm7)   , FPU_OFFSET(xmm7) , gcc_dwarf_xmm7 , gcc_dwarf_xmm7 , -1U, gdb_xmm7 },
    { e_regSetFPU, fpu_xmm8 , "xmm8"    , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm8)   , FPU_OFFSET(xmm8) , gcc_dwarf_xmm8 , gcc_dwarf_xmm8 , -1U, gdb_xmm8  },
    { e_regSetFPU, fpu_xmm9 , "xmm9"    , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm9)   , FPU_OFFSET(xmm9) , gcc_dwarf_xmm9 , gcc_dwarf_xmm9 , -1U, gdb_xmm9  },
    { e_regSetFPU, fpu_xmm10, "xmm10"   , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm10)  , FPU_OFFSET(xmm10), gcc_dwarf_xmm10, gcc_dwarf_xmm10, -1U, gdb_xmm10 },
    { e_regSetFPU, fpu_xmm11, "xmm11"   , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm11)  , FPU_OFFSET(xmm11), gcc_dwarf_xmm11, gcc_dwarf_xmm11, -1U, gdb_xmm11 },
    { e_regSetFPU, fpu_xmm12, "xmm12"   , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm12)  , FPU_OFFSET(xmm12), gcc_dwarf_xmm12, gcc_dwarf_xmm12, -1U, gdb_xmm12 },
    { e_regSetFPU, fpu_xmm13, "xmm13"   , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm13)  , FPU_OFFSET(xmm13), gcc_dwarf_xmm13, gcc_dwarf_xmm13, -1U, gdb_xmm13 },
    { e_regSetFPU, fpu_xmm14, "xmm14"   , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm14)  , FPU_OFFSET(xmm14), gcc_dwarf_xmm14, gcc_dwarf_xmm14, -1U, gdb_xmm14 },
    { e_regSetFPU, fpu_xmm15, "xmm15"   , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm15)  , FPU_OFFSET(xmm15), gcc_dwarf_xmm15, gcc_dwarf_xmm15, -1U, gdb_xmm15 },
};

const DNBRegisterInfo
DNBArchImplX86_64::g_fpu_registers_avx[] =
{
    { e_regSetFPU, fpu_fcw      , "fctrl"       , NULL, Uint, Hex, FPU_SIZE_UINT(fcw)       , AVX_OFFSET(fcw)       , -1U, -1U, -1U, -1U },
    { e_regSetFPU, fpu_fsw      , "fstat"       , NULL, Uint, Hex, FPU_SIZE_UINT(fsw)       , AVX_OFFSET(fsw)       , -1U, -1U, -1U, -1U },
    { e_regSetFPU, fpu_ftw      , "ftag"        , NULL, Uint, Hex, FPU_SIZE_UINT(ftw)       , AVX_OFFSET(ftw)       , -1U, -1U, -1U, -1U },
    { e_regSetFPU, fpu_fop      , "fop"         , NULL, Uint, Hex, FPU_SIZE_UINT(fop)       , AVX_OFFSET(fop)       , -1U, -1U, -1U, -1U },
    { e_regSetFPU, fpu_ip       , "fioff"       , NULL, Uint, Hex, FPU_SIZE_UINT(ip)        , AVX_OFFSET(ip)        , -1U, -1U, -1U, -1U },
    { e_regSetFPU, fpu_cs       , "fiseg"       , NULL, Uint, Hex, FPU_SIZE_UINT(cs)        , AVX_OFFSET(cs)        , -1U, -1U, -1U, -1U },
    { e_regSetFPU, fpu_dp       , "fooff"       , NULL, Uint, Hex, FPU_SIZE_UINT(dp)        , AVX_OFFSET(dp)        , -1U, -1U, -1U, -1U },
    { e_regSetFPU, fpu_ds       , "foseg"       , NULL, Uint, Hex, FPU_SIZE_UINT(ds)        , AVX_OFFSET(ds)        , -1U, -1U, -1U, -1U },
    { e_regSetFPU, fpu_mxcsr    , "mxcsr"       , NULL, Uint, Hex, FPU_SIZE_UINT(mxcsr)     , AVX_OFFSET(mxcsr)     , -1U, -1U, -1U, -1U },
    { e_regSetFPU, fpu_mxcsrmask, "mxcsrmask"   , NULL, Uint, Hex, FPU_SIZE_UINT(mxcsrmask) , AVX_OFFSET(mxcsrmask) , -1U, -1U, -1U, -1U },
    
    { e_regSetFPU, fpu_stmm0, "stmm0", NULL, Vector, VectorOfUInt8, FPU_SIZE_MMST(stmm0), AVX_OFFSET(stmm0), gcc_dwarf_stmm0, gcc_dwarf_stmm0, -1U, gdb_stmm0 },
    { e_regSetFPU, fpu_stmm1, "stmm1", NULL, Vector, VectorOfUInt8, FPU_SIZE_MMST(stmm1), AVX_OFFSET(stmm1), gcc_dwarf_stmm1, gcc_dwarf_stmm1, -1U, gdb_stmm1 },
    { e_regSetFPU, fpu_stmm2, "stmm2", NULL, Vector, VectorOfUInt8, FPU_SIZE_MMST(stmm2), AVX_OFFSET(stmm2), gcc_dwarf_stmm2, gcc_dwarf_stmm2, -1U, gdb_stmm2 },
    { e_regSetFPU, fpu_stmm3, "stmm3", NULL, Vector, VectorOfUInt8, FPU_SIZE_MMST(stmm3), AVX_OFFSET(stmm3), gcc_dwarf_stmm3, gcc_dwarf_stmm3, -1U, gdb_stmm3 },
    { e_regSetFPU, fpu_stmm4, "stmm4", NULL, Vector, VectorOfUInt8, FPU_SIZE_MMST(stmm4), AVX_OFFSET(stmm4), gcc_dwarf_stmm4, gcc_dwarf_stmm4, -1U, gdb_stmm4 },
    { e_regSetFPU, fpu_stmm5, "stmm5", NULL, Vector, VectorOfUInt8, FPU_SIZE_MMST(stmm5), AVX_OFFSET(stmm5), gcc_dwarf_stmm5, gcc_dwarf_stmm5, -1U, gdb_stmm5 },
    { e_regSetFPU, fpu_stmm6, "stmm6", NULL, Vector, VectorOfUInt8, FPU_SIZE_MMST(stmm6), AVX_OFFSET(stmm6), gcc_dwarf_stmm6, gcc_dwarf_stmm6, -1U, gdb_stmm6 },
    { e_regSetFPU, fpu_stmm7, "stmm7", NULL, Vector, VectorOfUInt8, FPU_SIZE_MMST(stmm7), AVX_OFFSET(stmm7), gcc_dwarf_stmm7, gcc_dwarf_stmm7, -1U, gdb_stmm7 },
    
    { e_regSetFPU, fpu_xmm0 , "xmm0"    , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm0)   , AVX_OFFSET(xmm0) , gcc_dwarf_xmm0 , gcc_dwarf_xmm0 , -1U, gdb_xmm0 },
    { e_regSetFPU, fpu_xmm1 , "xmm1"    , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm1)   , AVX_OFFSET(xmm1) , gcc_dwarf_xmm1 , gcc_dwarf_xmm1 , -1U, gdb_xmm1 },
    { e_regSetFPU, fpu_xmm2 , "xmm2"    , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm2)   , AVX_OFFSET(xmm2) , gcc_dwarf_xmm2 , gcc_dwarf_xmm2 , -1U, gdb_xmm2 },
    { e_regSetFPU, fpu_xmm3 , "xmm3"    , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm3)   , AVX_OFFSET(xmm3) , gcc_dwarf_xmm3 , gcc_dwarf_xmm3 , -1U, gdb_xmm3 },
    { e_regSetFPU, fpu_xmm4 , "xmm4"    , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm4)   , AVX_OFFSET(xmm4) , gcc_dwarf_xmm4 , gcc_dwarf_xmm4 , -1U, gdb_xmm4 },
    { e_regSetFPU, fpu_xmm5 , "xmm5"    , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm5)   , AVX_OFFSET(xmm5) , gcc_dwarf_xmm5 , gcc_dwarf_xmm5 , -1U, gdb_xmm5 },
    { e_regSetFPU, fpu_xmm6 , "xmm6"    , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm6)   , AVX_OFFSET(xmm6) , gcc_dwarf_xmm6 , gcc_dwarf_xmm6 , -1U, gdb_xmm6 },
    { e_regSetFPU, fpu_xmm7 , "xmm7"    , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm7)   , AVX_OFFSET(xmm7) , gcc_dwarf_xmm7 , gcc_dwarf_xmm7 , -1U, gdb_xmm7 },
    { e_regSetFPU, fpu_xmm8 , "xmm8"    , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm8)   , AVX_OFFSET(xmm8) , gcc_dwarf_xmm8 , gcc_dwarf_xmm8 , -1U, gdb_xmm8  },
    { e_regSetFPU, fpu_xmm9 , "xmm9"    , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm9)   , AVX_OFFSET(xmm9) , gcc_dwarf_xmm9 , gcc_dwarf_xmm9 , -1U, gdb_xmm9  },
    { e_regSetFPU, fpu_xmm10, "xmm10"   , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm10)  , AVX_OFFSET(xmm10), gcc_dwarf_xmm10, gcc_dwarf_xmm10, -1U, gdb_xmm10 },
    { e_regSetFPU, fpu_xmm11, "xmm11"   , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm11)  , AVX_OFFSET(xmm11), gcc_dwarf_xmm11, gcc_dwarf_xmm11, -1U, gdb_xmm11 },
    { e_regSetFPU, fpu_xmm12, "xmm12"   , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm12)  , AVX_OFFSET(xmm12), gcc_dwarf_xmm12, gcc_dwarf_xmm12, -1U, gdb_xmm12 },
    { e_regSetFPU, fpu_xmm13, "xmm13"   , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm13)  , AVX_OFFSET(xmm13), gcc_dwarf_xmm13, gcc_dwarf_xmm13, -1U, gdb_xmm13 },
    { e_regSetFPU, fpu_xmm14, "xmm14"   , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm14)  , AVX_OFFSET(xmm14), gcc_dwarf_xmm14, gcc_dwarf_xmm14, -1U, gdb_xmm14 },
    { e_regSetFPU, fpu_xmm15, "xmm15"   , NULL, Vector, VectorOfUInt8, FPU_SIZE_XMM(xmm15)  , AVX_OFFSET(xmm15), gcc_dwarf_xmm15, gcc_dwarf_xmm15, -1U, gdb_xmm15 },

    { e_regSetFPU, fpu_ymm0 , "ymm0"    , NULL, Vector, VectorOfUInt8, FPU_SIZE_YMM(ymm0)   , AVX_OFFSET_YMM(0) , gcc_dwarf_ymm0 , gcc_dwarf_ymm0 , -1U, gdb_ymm0 },
    { e_regSetFPU, fpu_ymm1 , "ymm1"    , NULL, Vector, VectorOfUInt8, FPU_SIZE_YMM(ymm1)   , AVX_OFFSET_YMM(1) , gcc_dwarf_ymm1 , gcc_dwarf_ymm1 , -1U, gdb_ymm1 },
    { e_regSetFPU, fpu_ymm2 , "ymm2"    , NULL, Vector, VectorOfUInt8, FPU_SIZE_YMM(ymm2)   , AVX_OFFSET_YMM(2) , gcc_dwarf_ymm2 , gcc_dwarf_ymm2 , -1U, gdb_ymm2 },
    { e_regSetFPU, fpu_ymm3 , "ymm3"    , NULL, Vector, VectorOfUInt8, FPU_SIZE_YMM(ymm3)   , AVX_OFFSET_YMM(3) , gcc_dwarf_ymm3 , gcc_dwarf_ymm3 , -1U, gdb_ymm3 },
    { e_regSetFPU, fpu_ymm4 , "ymm4"    , NULL, Vector, VectorOfUInt8, FPU_SIZE_YMM(ymm4)   , AVX_OFFSET_YMM(4) , gcc_dwarf_ymm4 , gcc_dwarf_ymm4 , -1U, gdb_ymm4 },
    { e_regSetFPU, fpu_ymm5 , "ymm5"    , NULL, Vector, VectorOfUInt8, FPU_SIZE_YMM(ymm5)   , AVX_OFFSET_YMM(5) , gcc_dwarf_ymm5 , gcc_dwarf_ymm5 , -1U, gdb_ymm5 },
    { e_regSetFPU, fpu_ymm6 , "ymm6"    , NULL, Vector, VectorOfUInt8, FPU_SIZE_YMM(ymm6)   , AVX_OFFSET_YMM(6) , gcc_dwarf_ymm6 , gcc_dwarf_ymm6 , -1U, gdb_ymm6 },
    { e_regSetFPU, fpu_ymm7 , "ymm7"    , NULL, Vector, VectorOfUInt8, FPU_SIZE_YMM(ymm7)   , AVX_OFFSET_YMM(7) , gcc_dwarf_ymm7 , gcc_dwarf_ymm7 , -1U, gdb_ymm7 },
    { e_regSetFPU, fpu_ymm8 , "ymm8"    , NULL, Vector, VectorOfUInt8, FPU_SIZE_YMM(ymm8)   , AVX_OFFSET_YMM(8) , gcc_dwarf_ymm8 , gcc_dwarf_ymm8 , -1U, gdb_ymm8  },
    { e_regSetFPU, fpu_ymm9 , "ymm9"    , NULL, Vector, VectorOfUInt8, FPU_SIZE_YMM(ymm9)   , AVX_OFFSET_YMM(9) , gcc_dwarf_ymm9 , gcc_dwarf_ymm9 , -1U, gdb_ymm9  },
    { e_regSetFPU, fpu_ymm10, "ymm10"   , NULL, Vector, VectorOfUInt8, FPU_SIZE_YMM(ymm10)  , AVX_OFFSET_YMM(10), gcc_dwarf_ymm10, gcc_dwarf_ymm10, -1U, gdb_ymm10 },
    { e_regSetFPU, fpu_ymm11, "ymm11"   , NULL, Vector, VectorOfUInt8, FPU_SIZE_YMM(ymm11)  , AVX_OFFSET_YMM(11), gcc_dwarf_ymm11, gcc_dwarf_ymm11, -1U, gdb_ymm11 },
    { e_regSetFPU, fpu_ymm12, "ymm12"   , NULL, Vector, VectorOfUInt8, FPU_SIZE_YMM(ymm12)  , AVX_OFFSET_YMM(12), gcc_dwarf_ymm12, gcc_dwarf_ymm12, -1U, gdb_ymm12 },
    { e_regSetFPU, fpu_ymm13, "ymm13"   , NULL, Vector, VectorOfUInt8, FPU_SIZE_YMM(ymm13)  , AVX_OFFSET_YMM(13), gcc_dwarf_ymm13, gcc_dwarf_ymm13, -1U, gdb_ymm13 },
    { e_regSetFPU, fpu_ymm14, "ymm14"   , NULL, Vector, VectorOfUInt8, FPU_SIZE_YMM(ymm14)  , AVX_OFFSET_YMM(14), gcc_dwarf_ymm14, gcc_dwarf_ymm14, -1U, gdb_ymm14 },
    { e_regSetFPU, fpu_ymm15, "ymm15"   , NULL, Vector, VectorOfUInt8, FPU_SIZE_YMM(ymm15)  , AVX_OFFSET_YMM(15), gcc_dwarf_ymm15, gcc_dwarf_ymm15, -1U, gdb_ymm15 }
};

// Exception registers

const DNBRegisterInfo
DNBArchImplX86_64::g_exc_registers[] =
{
    { e_regSetEXC, exc_trapno,      "trapno"    , NULL, Uint, Hex, EXC_SIZE (trapno)    , EXC_OFFSET (trapno)       , -1U, -1U, -1U, -1U },
    { e_regSetEXC, exc_err,         "err"       , NULL, Uint, Hex, EXC_SIZE (err)       , EXC_OFFSET (err)          , -1U, -1U, -1U, -1U },
    { e_regSetEXC, exc_faultvaddr,  "faultvaddr", NULL, Uint, Hex, EXC_SIZE (faultvaddr), EXC_OFFSET (faultvaddr)   , -1U, -1U, -1U, -1U }
};

// Number of registers in each register set
const size_t DNBArchImplX86_64::k_num_gpr_registers = sizeof(g_gpr_registers)/sizeof(DNBRegisterInfo);
const size_t DNBArchImplX86_64::k_num_fpu_registers_no_avx = sizeof(g_fpu_registers_no_avx)/sizeof(DNBRegisterInfo);
const size_t DNBArchImplX86_64::k_num_fpu_registers_avx = sizeof(g_fpu_registers_avx)/sizeof(DNBRegisterInfo);
const size_t DNBArchImplX86_64::k_num_exc_registers = sizeof(g_exc_registers)/sizeof(DNBRegisterInfo);
const size_t DNBArchImplX86_64::k_num_all_registers_no_avx = k_num_gpr_registers + k_num_fpu_registers_no_avx + k_num_exc_registers;
const size_t DNBArchImplX86_64::k_num_all_registers_avx = k_num_gpr_registers + k_num_fpu_registers_avx + k_num_exc_registers;

//----------------------------------------------------------------------
// Register set definitions. The first definitions at register set index
// of zero is for all registers, followed by other registers sets. The
// register information for the all register set need not be filled in.
//----------------------------------------------------------------------
const DNBRegisterSetInfo
DNBArchImplX86_64::g_reg_sets_no_avx[] =
{
    { "x86_64 Registers",           NULL,               k_num_all_registers_no_avx },
    { "General Purpose Registers",  g_gpr_registers,    k_num_gpr_registers },
    { "Floating Point Registers",   g_fpu_registers_no_avx, k_num_fpu_registers_no_avx },
    { "Exception State Registers",  g_exc_registers,    k_num_exc_registers }
};

const DNBRegisterSetInfo
DNBArchImplX86_64::g_reg_sets_avx[] =
{
    { "x86_64 Registers",           NULL,               k_num_all_registers_avx },
    { "General Purpose Registers",  g_gpr_registers,    k_num_gpr_registers },
    { "Floating Point Registers",   g_fpu_registers_avx, k_num_fpu_registers_avx },
    { "Exception State Registers",  g_exc_registers,    k_num_exc_registers }
};

// Total number of register sets for this architecture
const size_t DNBArchImplX86_64::k_num_register_sets = sizeof(g_reg_sets_avx)/sizeof(DNBRegisterSetInfo);


DNBArchProtocol *
DNBArchImplX86_64::Create (MachThread *thread)
{
    DNBArchImplX86_64 *obj = new DNBArchImplX86_64 (thread);

    // When new thread comes along, it tries to inherit from the global debug state, if it is valid.
    if (Valid_Global_Debug_State)
    {
        obj->m_state.context.dbg = Global_Debug_State;
        kern_return_t kret = obj->SetDBGState();
        DNBLogThreadedIf(LOG_WATCHPOINTS,
                         "DNBArchImplX86_64::Create() Inherit and SetDBGState() => 0x%8.8x.", kret);
    }
    return obj;
}

const uint8_t * const
DNBArchImplX86_64::SoftwareBreakpointOpcode (nub_size_t byte_size)
{
    static const uint8_t g_breakpoint_opcode[] = { 0xCC };
    if (byte_size == 1)
        return g_breakpoint_opcode;
    return NULL;
}

const DNBRegisterSetInfo *
DNBArchImplX86_64::GetRegisterSetInfo(nub_size_t *num_reg_sets)
{
    *num_reg_sets = k_num_register_sets;
    
    if (CPUHasAVX() || FORCE_AVX_REGS)
        return g_reg_sets_avx;
    else
        return g_reg_sets_no_avx;
}

void
DNBArchImplX86_64::Initialize()
{
    DNBArchPluginInfo arch_plugin_info = 
    {
        CPU_TYPE_X86_64, 
        DNBArchImplX86_64::Create, 
        DNBArchImplX86_64::GetRegisterSetInfo,
        DNBArchImplX86_64::SoftwareBreakpointOpcode
    };
    
    // Register this arch plug-in with the main protocol class
    DNBArchProtocol::RegisterArchPlugin (arch_plugin_info);
}

bool
DNBArchImplX86_64::GetRegisterValue(int set, int reg, DNBRegisterValue *value)
{
    if (set == REGISTER_SET_GENERIC)
    {
        switch (reg)
        {
            case GENERIC_REGNUM_PC:     // Program Counter
                set = e_regSetGPR;
                reg = gpr_rip;
                break;
                
            case GENERIC_REGNUM_SP:     // Stack Pointer
                set = e_regSetGPR;
                reg = gpr_rsp;
                break;
                
            case GENERIC_REGNUM_FP:     // Frame Pointer
                set = e_regSetGPR;
                reg = gpr_rbp;
                break;
                
            case GENERIC_REGNUM_FLAGS:  // Processor flags register
                set = e_regSetGPR;
                reg = gpr_rflags;
                break;
                
            case GENERIC_REGNUM_RA:     // Return Address
            default:
                return false;
        }
    }
    
    if (GetRegisterState(set, false) != KERN_SUCCESS)
        return false;
    
    const DNBRegisterInfo *regInfo = m_thread->GetRegisterInfo(set, reg);
    if (regInfo)
    {
        value->info = *regInfo;
        switch (set)
        {
            case e_regSetGPR:
                if (reg < k_num_gpr_registers)
                {
                    value->value.uint64 = ((uint64_t*)(&m_state.context.gpr))[reg];
                    return true;
                }
                break;
                
            case e_regSetFPU:
                if (CPUHasAVX() || FORCE_AVX_REGS)
                {
                    switch (reg)
                    {
                    case fpu_fcw:       value->value.uint16 = *((uint16_t *)(&m_state.context.fpu.avx.__fpu_fcw));    return true;
                    case fpu_fsw:       value->value.uint16 = *((uint16_t *)(&m_state.context.fpu.avx.__fpu_fsw));    return true;
                    case fpu_ftw:       value->value.uint8  = m_state.context.fpu.avx.__fpu_ftw;                      return true;
                    case fpu_fop:       value->value.uint16 = m_state.context.fpu.avx.__fpu_fop;                      return true;
                    case fpu_ip:        value->value.uint32 = m_state.context.fpu.avx.__fpu_ip;                       return true;
                    case fpu_cs:        value->value.uint16 = m_state.context.fpu.avx.__fpu_cs;                       return true;
                    case fpu_dp:        value->value.uint32 = m_state.context.fpu.avx.__fpu_dp;                       return true;
                    case fpu_ds:        value->value.uint16 = m_state.context.fpu.avx.__fpu_ds;                       return true;
                    case fpu_mxcsr:     value->value.uint32 = m_state.context.fpu.avx.__fpu_mxcsr;                    return true;
                    case fpu_mxcsrmask: value->value.uint32 = m_state.context.fpu.avx.__fpu_mxcsrmask;                return true;
                        
                    case fpu_stmm0:
                    case fpu_stmm1:
                    case fpu_stmm2:
                    case fpu_stmm3:
                    case fpu_stmm4:
                    case fpu_stmm5:
                    case fpu_stmm6:
                    case fpu_stmm7:
                        memcpy(&value->value.uint8, &m_state.context.fpu.avx.__fpu_stmm0 + (reg - fpu_stmm0), 10);
                        return true;
                        
                    case fpu_xmm0:
                    case fpu_xmm1:
                    case fpu_xmm2:
                    case fpu_xmm3:
                    case fpu_xmm4:
                    case fpu_xmm5:
                    case fpu_xmm6:
                    case fpu_xmm7:
                    case fpu_xmm8:
                    case fpu_xmm9:
                    case fpu_xmm10:
                    case fpu_xmm11:
                    case fpu_xmm12:
                    case fpu_xmm13:
                    case fpu_xmm14:
                    case fpu_xmm15:
                        memcpy(&value->value.uint8, &m_state.context.fpu.avx.__fpu_xmm0 + (reg - fpu_xmm0), 16);
                        return true;
                            
                    case fpu_ymm0:
                    case fpu_ymm1:
                    case fpu_ymm2:
                    case fpu_ymm3:
                    case fpu_ymm4:
                    case fpu_ymm5:
                    case fpu_ymm6:
                    case fpu_ymm7:
                    case fpu_ymm8:
                    case fpu_ymm9:
                    case fpu_ymm10:
                    case fpu_ymm11:
                    case fpu_ymm12:
                    case fpu_ymm13:
                    case fpu_ymm14:
                    case fpu_ymm15:
                        memcpy(&value->value.uint8, &m_state.context.fpu.avx.__fpu_xmm0 + (reg - fpu_ymm0), 16);
                        memcpy((&value->value.uint8) + 16, &m_state.context.fpu.avx.__fpu_ymmh0 + (reg - fpu_ymm0), 16);
                        return true;
                    }
                }
                else
                {
                    switch (reg)
                    {
                        case fpu_fcw:       value->value.uint16 = *((uint16_t *)(&m_state.context.fpu.no_avx.__fpu_fcw));    return true;
                        case fpu_fsw:       value->value.uint16 = *((uint16_t *)(&m_state.context.fpu.no_avx.__fpu_fsw));    return true;
                        case fpu_ftw:       value->value.uint8  = m_state.context.fpu.no_avx.__fpu_ftw;                      return true;
                        case fpu_fop:       value->value.uint16 = m_state.context.fpu.no_avx.__fpu_fop;                      return true;
                        case fpu_ip:        value->value.uint32 = m_state.context.fpu.no_avx.__fpu_ip;                       return true;
                        case fpu_cs:        value->value.uint16 = m_state.context.fpu.no_avx.__fpu_cs;                       return true;
                        case fpu_dp:        value->value.uint32 = m_state.context.fpu.no_avx.__fpu_dp;                       return true;
                        case fpu_ds:        value->value.uint16 = m_state.context.fpu.no_avx.__fpu_ds;                       return true;
                        case fpu_mxcsr:     value->value.uint32 = m_state.context.fpu.no_avx.__fpu_mxcsr;                    return true;
                        case fpu_mxcsrmask: value->value.uint32 = m_state.context.fpu.no_avx.__fpu_mxcsrmask;                return true;
                            
                        case fpu_stmm0:
                        case fpu_stmm1:
                        case fpu_stmm2:
                        case fpu_stmm3:
                        case fpu_stmm4:
                        case fpu_stmm5:
                        case fpu_stmm6:
                        case fpu_stmm7:
                            memcpy(&value->value.uint8, &m_state.context.fpu.no_avx.__fpu_stmm0 + (reg - fpu_stmm0), 10);
                            return true;
                            
                        case fpu_xmm0:
                        case fpu_xmm1:
                        case fpu_xmm2:
                        case fpu_xmm3:
                        case fpu_xmm4:
                        case fpu_xmm5:
                        case fpu_xmm6:
                        case fpu_xmm7:
                        case fpu_xmm8:
                        case fpu_xmm9:
                        case fpu_xmm10:
                        case fpu_xmm11:
                        case fpu_xmm12:
                        case fpu_xmm13:
                        case fpu_xmm14:
                        case fpu_xmm15:
                            memcpy(&value->value.uint8, &m_state.context.fpu.no_avx.__fpu_xmm0 + (reg - fpu_xmm0), 16);
                            return true;
                    }
                }
                break;
                
            case e_regSetEXC:
                switch (reg)
                {
                case exc_trapno:    value->value.uint32 = m_state.context.exc.__trapno; return true;
                case exc_err:       value->value.uint32 = m_state.context.exc.__err; return true;
                case exc_faultvaddr:value->value.uint64 = m_state.context.exc.__faultvaddr; return true;
                }
                break;
        }
    }
    return false;
}


bool
DNBArchImplX86_64::SetRegisterValue(int set, int reg, const DNBRegisterValue *value)
{
    if (set == REGISTER_SET_GENERIC)
    {
        switch (reg)
        {
            case GENERIC_REGNUM_PC:     // Program Counter
                set = e_regSetGPR;
                reg = gpr_rip;
                break;
                
            case GENERIC_REGNUM_SP:     // Stack Pointer
                set = e_regSetGPR;
                reg = gpr_rsp;
                break;
                
            case GENERIC_REGNUM_FP:     // Frame Pointer
                set = e_regSetGPR;
                reg = gpr_rbp;
                break;
                
            case GENERIC_REGNUM_FLAGS:  // Processor flags register
                set = e_regSetGPR;
                reg = gpr_rflags;
                break;
                
            case GENERIC_REGNUM_RA:     // Return Address
            default:
                return false;
        }
    }
    
    if (GetRegisterState(set, false) != KERN_SUCCESS)
        return false;
    
    bool success = false;
    const DNBRegisterInfo *regInfo = m_thread->GetRegisterInfo(set, reg);
    if (regInfo)
    {
        switch (set)
        {
            case e_regSetGPR:
                if (reg < k_num_gpr_registers)
                {
                    ((uint64_t*)(&m_state.context.gpr))[reg] = value->value.uint64;
                    success = true;
                }
                break;
                
            case e_regSetFPU:
                if (CPUHasAVX() || FORCE_AVX_REGS)
                {
                    switch (reg)
                    {
                    case fpu_fcw:       *((uint16_t *)(&m_state.context.fpu.avx.__fpu_fcw)) = value->value.uint16;    success = true; break;
                    case fpu_fsw:       *((uint16_t *)(&m_state.context.fpu.avx.__fpu_fsw)) = value->value.uint16;    success = true; break;
                    case fpu_ftw:       m_state.context.fpu.avx.__fpu_ftw = value->value.uint8;                       success = true; break;
                    case fpu_fop:       m_state.context.fpu.avx.__fpu_fop = value->value.uint16;                      success = true; break;
                    case fpu_ip:        m_state.context.fpu.avx.__fpu_ip = value->value.uint32;                       success = true; break;
                    case fpu_cs:        m_state.context.fpu.avx.__fpu_cs = value->value.uint16;                       success = true; break;
                    case fpu_dp:        m_state.context.fpu.avx.__fpu_dp = value->value.uint32;                       success = true; break;
                    case fpu_ds:        m_state.context.fpu.avx.__fpu_ds = value->value.uint16;                       success = true; break;
                    case fpu_mxcsr:     m_state.context.fpu.avx.__fpu_mxcsr = value->value.uint32;                    success = true; break;
                    case fpu_mxcsrmask: m_state.context.fpu.avx.__fpu_mxcsrmask = value->value.uint32;                success = true; break;
                        
                    case fpu_stmm0:
                    case fpu_stmm1:
                    case fpu_stmm2:
                    case fpu_stmm3:
                    case fpu_stmm4:
                    case fpu_stmm5:
                    case fpu_stmm6:
                    case fpu_stmm7:
                        memcpy (&m_state.context.fpu.avx.__fpu_stmm0 + (reg - fpu_stmm0), &value->value.uint8, 10);
                        success = true;
                        break;
                        
                    case fpu_xmm0:
                    case fpu_xmm1:
                    case fpu_xmm2:
                    case fpu_xmm3:
                    case fpu_xmm4:
                    case fpu_xmm5:
                    case fpu_xmm6:
                    case fpu_xmm7:
                    case fpu_xmm8:
                    case fpu_xmm9:
                    case fpu_xmm10:
                    case fpu_xmm11:
                    case fpu_xmm12:
                    case fpu_xmm13:
                    case fpu_xmm14:
                    case fpu_xmm15:
                        memcpy (&m_state.context.fpu.avx.__fpu_xmm0 + (reg - fpu_xmm0), &value->value.uint8, 16);
                        success = true;
                        break;
                    
                    case fpu_ymm0:
                    case fpu_ymm1:
                    case fpu_ymm2:
                    case fpu_ymm3:
                    case fpu_ymm4:
                    case fpu_ymm5:
                    case fpu_ymm6:
                    case fpu_ymm7:
                    case fpu_ymm8:
                    case fpu_ymm9:
                    case fpu_ymm10:
                    case fpu_ymm11:
                    case fpu_ymm12:
                    case fpu_ymm13:
                    case fpu_ymm14:
                    case fpu_ymm15:
                        memcpy(&m_state.context.fpu.avx.__fpu_xmm0 + (reg - fpu_ymm0), &value->value.uint8, 16);
                        memcpy(&m_state.context.fpu.avx.__fpu_ymmh0 + (reg - fpu_ymm0), (&value->value.uint8) + 16, 16);
                        return true;
                    }
                }
                else
                {
                    switch (reg)
                    {
                    case fpu_fcw:       *((uint16_t *)(&m_state.context.fpu.no_avx.__fpu_fcw)) = value->value.uint16;    success = true; break;
                    case fpu_fsw:       *((uint16_t *)(&m_state.context.fpu.no_avx.__fpu_fsw)) = value->value.uint16;    success = true; break;
                    case fpu_ftw:       m_state.context.fpu.no_avx.__fpu_ftw = value->value.uint8;                       success = true; break;
                    case fpu_fop:       m_state.context.fpu.no_avx.__fpu_fop = value->value.uint16;                      success = true; break;
                    case fpu_ip:        m_state.context.fpu.no_avx.__fpu_ip = value->value.uint32;                       success = true; break;
                    case fpu_cs:        m_state.context.fpu.no_avx.__fpu_cs = value->value.uint16;                       success = true; break;
                    case fpu_dp:        m_state.context.fpu.no_avx.__fpu_dp = value->value.uint32;                       success = true; break;
                    case fpu_ds:        m_state.context.fpu.no_avx.__fpu_ds = value->value.uint16;                       success = true; break;
                    case fpu_mxcsr:     m_state.context.fpu.no_avx.__fpu_mxcsr = value->value.uint32;                    success = true; break;
                    case fpu_mxcsrmask: m_state.context.fpu.no_avx.__fpu_mxcsrmask = value->value.uint32;                success = true; break;
                        
                    case fpu_stmm0:
                    case fpu_stmm1:
                    case fpu_stmm2:
                    case fpu_stmm3:
                    case fpu_stmm4:
                    case fpu_stmm5:
                    case fpu_stmm6:
                    case fpu_stmm7:
                        memcpy (&m_state.context.fpu.no_avx.__fpu_stmm0 + (reg - fpu_stmm0), &value->value.uint8, 10);
                        success = true;
                        break;
                        
                    case fpu_xmm0:
                    case fpu_xmm1:
                    case fpu_xmm2:
                    case fpu_xmm3:
                    case fpu_xmm4:
                    case fpu_xmm5:
                    case fpu_xmm6:
                    case fpu_xmm7:
                    case fpu_xmm8:
                    case fpu_xmm9:
                    case fpu_xmm10:
                    case fpu_xmm11:
                    case fpu_xmm12:
                    case fpu_xmm13:
                    case fpu_xmm14:
                    case fpu_xmm15:
                        memcpy (&m_state.context.fpu.no_avx.__fpu_xmm0 + (reg - fpu_xmm0), &value->value.uint8, 16);
                        success = true;
                        break;
                    }
                }
                break;
                
            case e_regSetEXC:
                switch (reg)
            {
                case exc_trapno:    m_state.context.exc.__trapno = value->value.uint32;     success = true; break;
                case exc_err:       m_state.context.exc.__err = value->value.uint32;        success = true; break;
                case exc_faultvaddr:m_state.context.exc.__faultvaddr = value->value.uint64; success = true; break;
            }
                break;
        }
    }
    
    if (success)
        return SetRegisterState(set) == KERN_SUCCESS;
    return false;
}


nub_size_t
DNBArchImplX86_64::GetRegisterContext (void *buf, nub_size_t buf_len)
{
    nub_size_t size = sizeof (m_state.context);
    
    if (buf && buf_len)
    {
        if (size > buf_len)
            size = buf_len;

        bool force = false;
        kern_return_t kret;
        if ((kret = GetGPRState(force)) != KERN_SUCCESS)
        {
            DNBLogThreadedIf (LOG_THREAD, "DNBArchImplX86_64::GetRegisterContext (buf = %p, len = %llu) error: GPR regs failed to read: %u ", buf, (uint64_t)buf_len, kret);
            size = 0;
        }
        else 
        if ((kret = GetFPUState(force)) != KERN_SUCCESS)
        {
            DNBLogThreadedIf (LOG_THREAD, "DNBArchImplX86_64::GetRegisterContext (buf = %p, len = %llu) error: %s regs failed to read: %u", buf, (uint64_t)buf_len, CPUHasAVX() ? "AVX" : "FPU", kret);
            size = 0;
        }
        else 
        if ((kret = GetEXCState(force)) != KERN_SUCCESS)
        {
            DNBLogThreadedIf (LOG_THREAD, "DNBArchImplX86_64::GetRegisterContext (buf = %p, len = %llu) error: EXC regs failed to read: %u", buf, (uint64_t)buf_len, kret);
            size = 0;
        }
        else
        {
            // Success
            ::memcpy (buf, &m_state.context, size);
        }
    }
    DNBLogThreadedIf (LOG_THREAD, "DNBArchImplX86_64::GetRegisterContext (buf = %p, len = %llu) => %llu", buf, (uint64_t)buf_len, (uint64_t)size);
    // Return the size of the register context even if NULL was passed in
    return size;
}

nub_size_t
DNBArchImplX86_64::SetRegisterContext (const void *buf, nub_size_t buf_len)
{
    nub_size_t size = sizeof (m_state.context);
    if (buf == NULL || buf_len == 0)
        size = 0;
    
    if (size)
    {
        if (size > buf_len)
            size = buf_len;

        ::memcpy (&m_state.context, buf, size);
        kern_return_t kret;
        if ((kret = SetGPRState()) != KERN_SUCCESS)
            DNBLogThreadedIf (LOG_THREAD, "DNBArchImplX86_64::SetRegisterContext (buf = %p, len = %llu) error: GPR regs failed to write: %u", buf, (uint64_t)buf_len, kret);
        if ((kret = SetFPUState()) != KERN_SUCCESS)
            DNBLogThreadedIf (LOG_THREAD, "DNBArchImplX86_64::SetRegisterContext (buf = %p, len = %llu) error: %s regs failed to write: %u", buf, (uint64_t)buf_len, CPUHasAVX() ? "AVX" : "FPU", kret);
        if ((kret = SetEXCState()) != KERN_SUCCESS)
            DNBLogThreadedIf (LOG_THREAD, "DNBArchImplX86_64::SetRegisterContext (buf = %p, len = %llu) error: EXP regs failed to write: %u", buf, (uint64_t)buf_len, kret);
    }
    DNBLogThreadedIf (LOG_THREAD, "DNBArchImplX86_64::SetRegisterContext (buf = %p, len = %llu) => %llu", buf, (uint64_t)buf_len, (uint64_t)size);
    return size;
}


kern_return_t
DNBArchImplX86_64::GetRegisterState(int set, bool force)
{
    switch (set)
    {
        case e_regSetALL:    return GetGPRState(force) | GetFPUState(force) | GetEXCState(force);
        case e_regSetGPR:    return GetGPRState(force);
        case e_regSetFPU:    return GetFPUState(force);
        case e_regSetEXC:    return GetEXCState(force);
        default: break;
    }
    return KERN_INVALID_ARGUMENT;
}

kern_return_t
DNBArchImplX86_64::SetRegisterState(int set)
{
    // Make sure we have a valid context to set.
    if (RegisterSetStateIsValid(set))
    {
        switch (set)
        {
            case e_regSetALL:    return SetGPRState() | SetFPUState() | SetEXCState();
            case e_regSetGPR:    return SetGPRState();
            case e_regSetFPU:    return SetFPUState();
            case e_regSetEXC:    return SetEXCState();
            default: break;
        }
    }
    return KERN_INVALID_ARGUMENT;
}

bool
DNBArchImplX86_64::RegisterSetStateIsValid (int set) const
{
    return m_state.RegsAreValid(set);
}



#endif    // #if defined (__i386__) || defined (__x86_64__)
