//===-- DNBArchImplI386.cpp -------------------------------------*- C++ -*-===//
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

#if defined(__i386__) || defined(__x86_64__)

#include <sys/cdefs.h>

#include "DNBLog.h"
#include "MacOSX/i386/DNBArchImplI386.h"
#include "MachProcess.h"
#include "MachThread.h"

extern "C" bool CPUHasAVX(); // Defined over in DNBArchImplX86_64.cpp

#if defined(LLDB_DEBUGSERVER_RELEASE) || defined(LLDB_DEBUGSERVER_DEBUG)
enum debugState { debugStateUnknown, debugStateOff, debugStateOn };

static debugState sFPUDebugState = debugStateUnknown;
static debugState sAVXForceState = debugStateUnknown;

static bool DebugFPURegs() {
  if (sFPUDebugState == debugStateUnknown) {
    if (getenv("DNB_DEBUG_FPU_REGS"))
      sFPUDebugState = debugStateOn;
    else
      sFPUDebugState = debugStateOff;
  }

  return (sFPUDebugState == debugStateOn);
}

static bool ForceAVXRegs() {
  if (sFPUDebugState == debugStateUnknown) {
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

enum {
  gpr_eax = 0,
  gpr_ebx = 1,
  gpr_ecx = 2,
  gpr_edx = 3,
  gpr_edi = 4,
  gpr_esi = 5,
  gpr_ebp = 6,
  gpr_esp = 7,
  gpr_ss = 8,
  gpr_eflags = 9,
  gpr_eip = 10,
  gpr_cs = 11,
  gpr_ds = 12,
  gpr_es = 13,
  gpr_fs = 14,
  gpr_gs = 15,
  gpr_ax,
  gpr_bx,
  gpr_cx,
  gpr_dx,
  gpr_di,
  gpr_si,
  gpr_bp,
  gpr_sp,
  gpr_ah,
  gpr_bh,
  gpr_ch,
  gpr_dh,
  gpr_al,
  gpr_bl,
  gpr_cl,
  gpr_dl,
  gpr_dil,
  gpr_sil,
  gpr_bpl,
  gpr_spl,
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
  fpu_ymm0,
  fpu_ymm1,
  fpu_ymm2,
  fpu_ymm3,
  fpu_ymm4,
  fpu_ymm5,
  fpu_ymm6,
  fpu_ymm7,
  k_num_fpu_regs,

  // Aliases
  fpu_fctrl = fpu_fcw,
  fpu_fstat = fpu_fsw,
  fpu_ftag = fpu_ftw,
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

enum {
  ehframe_eax = 0,
  ehframe_ecx,
  ehframe_edx,
  ehframe_ebx,

  // On i386 Darwin the eh_frame register numbers for ebp and esp are reversed
  // from DWARF.
  // It's due to an ancient compiler bug in the output of the eh_frame.
  // Specifically, on i386 darwin eh_frame, 4 is ebp, 5 is esp.
  // On i386 darwin debug_frame (and debug_info), 4 is esp, 5 is ebp.
  ehframe_ebp,
  ehframe_esp,
  ehframe_esi,
  ehframe_edi,
  ehframe_eip,
  ehframe_eflags
};

enum {
  dwarf_eax = 0,
  dwarf_ecx,
  dwarf_edx,
  dwarf_ebx,
  dwarf_esp,
  dwarf_ebp,
  dwarf_esi,
  dwarf_edi,
  dwarf_eip,
  dwarf_eflags,
  dwarf_stmm0 = 11,
  dwarf_stmm1,
  dwarf_stmm2,
  dwarf_stmm3,
  dwarf_stmm4,
  dwarf_stmm5,
  dwarf_stmm6,
  dwarf_stmm7,
  dwarf_xmm0 = 21,
  dwarf_xmm1,
  dwarf_xmm2,
  dwarf_xmm3,
  dwarf_xmm4,
  dwarf_xmm5,
  dwarf_xmm6,
  dwarf_xmm7,
  dwarf_ymm0 = dwarf_xmm0,
  dwarf_ymm1 = dwarf_xmm1,
  dwarf_ymm2 = dwarf_xmm2,
  dwarf_ymm3 = dwarf_xmm3,
  dwarf_ymm4 = dwarf_xmm4,
  dwarf_ymm5 = dwarf_xmm5,
  dwarf_ymm6 = dwarf_xmm6,
  dwarf_ymm7 = dwarf_xmm7,
};

enum {
  debugserver_eax = 0,
  debugserver_ecx = 1,
  debugserver_edx = 2,
  debugserver_ebx = 3,
  debugserver_esp = 4,
  debugserver_ebp = 5,
  debugserver_esi = 6,
  debugserver_edi = 7,
  debugserver_eip = 8,
  debugserver_eflags = 9,
  debugserver_cs = 10,
  debugserver_ss = 11,
  debugserver_ds = 12,
  debugserver_es = 13,
  debugserver_fs = 14,
  debugserver_gs = 15,
  debugserver_stmm0 = 16,
  debugserver_stmm1 = 17,
  debugserver_stmm2 = 18,
  debugserver_stmm3 = 19,
  debugserver_stmm4 = 20,
  debugserver_stmm5 = 21,
  debugserver_stmm6 = 22,
  debugserver_stmm7 = 23,
  debugserver_fctrl = 24,
  debugserver_fcw = debugserver_fctrl,
  debugserver_fstat = 25,
  debugserver_fsw = debugserver_fstat,
  debugserver_ftag = 26,
  debugserver_ftw = debugserver_ftag,
  debugserver_fiseg = 27,
  debugserver_fpu_cs = debugserver_fiseg,
  debugserver_fioff = 28,
  debugserver_ip = debugserver_fioff,
  debugserver_foseg = 29,
  debugserver_fpu_ds = debugserver_foseg,
  debugserver_fooff = 30,
  debugserver_dp = debugserver_fooff,
  debugserver_fop = 31,
  debugserver_xmm0 = 32,
  debugserver_xmm1 = 33,
  debugserver_xmm2 = 34,
  debugserver_xmm3 = 35,
  debugserver_xmm4 = 36,
  debugserver_xmm5 = 37,
  debugserver_xmm6 = 38,
  debugserver_xmm7 = 39,
  debugserver_mxcsr = 40,
  debugserver_mm0 = 41,
  debugserver_mm1 = 42,
  debugserver_mm2 = 43,
  debugserver_mm3 = 44,
  debugserver_mm4 = 45,
  debugserver_mm5 = 46,
  debugserver_mm6 = 47,
  debugserver_mm7 = 48,
  debugserver_ymm0 = debugserver_xmm0,
  debugserver_ymm1 = debugserver_xmm1,
  debugserver_ymm2 = debugserver_xmm2,
  debugserver_ymm3 = debugserver_xmm3,
  debugserver_ymm4 = debugserver_xmm4,
  debugserver_ymm5 = debugserver_xmm5,
  debugserver_ymm6 = debugserver_xmm6,
  debugserver_ymm7 = debugserver_xmm7
};

uint64_t DNBArchImplI386::GetPC(uint64_t failValue) {
  // Get program counter
  if (GetGPRState(false) == KERN_SUCCESS)
    return m_state.context.gpr.__eip;
  return failValue;
}

kern_return_t DNBArchImplI386::SetPC(uint64_t value) {
  // Get program counter
  kern_return_t err = GetGPRState(false);
  if (err == KERN_SUCCESS) {
    m_state.context.gpr.__eip = static_cast<uint32_t>(value);
    err = SetGPRState();
  }
  return err == KERN_SUCCESS;
}

uint64_t DNBArchImplI386::GetSP(uint64_t failValue) {
  // Get stack pointer
  if (GetGPRState(false) == KERN_SUCCESS)
    return m_state.context.gpr.__esp;
  return failValue;
}

// Uncomment the value below to verify the values in the debugger.
//#define DEBUG_GPR_VALUES 1    // DO NOT CHECK IN WITH THIS DEFINE ENABLED
//#define SET_GPR(reg) m_state.context.gpr.__##reg = gpr_##reg

kern_return_t DNBArchImplI386::GetGPRState(bool force) {
  if (force || m_state.GetError(e_regSetGPR, Read)) {
#if DEBUG_GPR_VALUES
    SET_GPR(eax);
    SET_GPR(ebx);
    SET_GPR(ecx);
    SET_GPR(edx);
    SET_GPR(edi);
    SET_GPR(esi);
    SET_GPR(ebp);
    SET_GPR(esp);
    SET_GPR(ss);
    SET_GPR(eflags);
    SET_GPR(eip);
    SET_GPR(cs);
    SET_GPR(ds);
    SET_GPR(es);
    SET_GPR(fs);
    SET_GPR(gs);
    m_state.SetError(e_regSetGPR, Read, 0);
#else
    mach_msg_type_number_t count = e_regSetWordSizeGPR;
    m_state.SetError(
        e_regSetGPR, Read,
        ::thread_get_state(m_thread->MachPortNumber(), __i386_THREAD_STATE,
                           (thread_state_t)&m_state.context.gpr, &count));
#endif
  }
  return m_state.GetError(e_regSetGPR, Read);
}

// Uncomment the value below to verify the values in the debugger.
//#define DEBUG_FPU_VALUES 1    // DO NOT CHECK IN WITH THIS DEFINE ENABLED

kern_return_t DNBArchImplI386::GetFPUState(bool force) {
  if (force || m_state.GetError(e_regSetFPU, Read)) {
    if (DEBUG_FPU_REGS) {

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
      for (int i = 0; i < 16; ++i) {
        if (i < 10) {
          m_state.context.fpu.no_avx.__fpu_stmm0.__mmst_reg[i] = 'a';
          m_state.context.fpu.no_avx.__fpu_stmm1.__mmst_reg[i] = 'b';
          m_state.context.fpu.no_avx.__fpu_stmm2.__mmst_reg[i] = 'c';
          m_state.context.fpu.no_avx.__fpu_stmm3.__mmst_reg[i] = 'd';
          m_state.context.fpu.no_avx.__fpu_stmm4.__mmst_reg[i] = 'e';
          m_state.context.fpu.no_avx.__fpu_stmm5.__mmst_reg[i] = 'f';
          m_state.context.fpu.no_avx.__fpu_stmm6.__mmst_reg[i] = 'g';
          m_state.context.fpu.no_avx.__fpu_stmm7.__mmst_reg[i] = 'h';
        } else {
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
      }
      for (int i = 0; i < sizeof(m_state.context.fpu.no_avx.__fpu_rsrv4); ++i)
        m_state.context.fpu.no_avx.__fpu_rsrv4[i] = INT8_MIN;
      m_state.context.fpu.no_avx.__fpu_reserved1 = -1;

      if (CPUHasAVX() || FORCE_AVX_REGS) {
        for (int i = 0; i < sizeof(m_state.context.fpu.avx.__avx_reserved1); ++i)
          m_state.context.fpu.avx.__avx_reserved1[i] = INT8_MIN;

        for (int i = 0; i < 16; ++i) {
          m_state.context.fpu.avx.__fpu_ymmh0.__xmm_reg[i] = '0';
          m_state.context.fpu.avx.__fpu_ymmh1.__xmm_reg[i] = '1';
          m_state.context.fpu.avx.__fpu_ymmh2.__xmm_reg[i] = '2';
          m_state.context.fpu.avx.__fpu_ymmh3.__xmm_reg[i] = '3';
          m_state.context.fpu.avx.__fpu_ymmh4.__xmm_reg[i] = '4';
          m_state.context.fpu.avx.__fpu_ymmh5.__xmm_reg[i] = '5';
          m_state.context.fpu.avx.__fpu_ymmh6.__xmm_reg[i] = '6';
          m_state.context.fpu.avx.__fpu_ymmh7.__xmm_reg[i] = '7';
        }
      }
      m_state.SetError(e_regSetFPU, Read, 0);
    } else {
      mach_msg_type_number_t count = e_regSetWordSizeFPU;
      int flavor = __i386_FLOAT_STATE;

      if (CPUHasAVX() || FORCE_AVX_REGS) {
        count = e_regSetWordSizeAVX;
        flavor = __i386_AVX_STATE;
      }
      m_state.SetError(e_regSetFPU, Read,
                       ::thread_get_state(m_thread->MachPortNumber(), flavor,
                                          (thread_state_t)&m_state.context.fpu,
                                          &count));
      DNBLogThreadedIf(LOG_THREAD,
                       "::thread_get_state (0x%4.4x, %u, &fpu, %u => 0x%8.8x",
                       m_thread->MachPortNumber(), flavor, (uint32_t)count,
                       m_state.GetError(e_regSetFPU, Read));
    }
  }
  return m_state.GetError(e_regSetFPU, Read);
}

kern_return_t DNBArchImplI386::GetEXCState(bool force) {
  if (force || m_state.GetError(e_regSetEXC, Read)) {
    mach_msg_type_number_t count = e_regSetWordSizeEXC;
    m_state.SetError(
        e_regSetEXC, Read,
        ::thread_get_state(m_thread->MachPortNumber(), __i386_EXCEPTION_STATE,
                           (thread_state_t)&m_state.context.exc, &count));
  }
  return m_state.GetError(e_regSetEXC, Read);
}

kern_return_t DNBArchImplI386::SetGPRState() {
  kern_return_t kret = ::thread_abort_safely(m_thread->MachPortNumber());
  DNBLogThreadedIf(
      LOG_THREAD, "thread = 0x%4.4x calling thread_abort_safely (tid) => %u "
                  "(SetGPRState() for stop_count = %u)",
      m_thread->MachPortNumber(), kret, m_thread->Process()->StopCount());

  m_state.SetError(e_regSetGPR, Write,
                   ::thread_set_state(m_thread->MachPortNumber(),
                                      __i386_THREAD_STATE,
                                      (thread_state_t)&m_state.context.gpr,
                                      e_regSetWordSizeGPR));
  return m_state.GetError(e_regSetGPR, Write);
}

kern_return_t DNBArchImplI386::SetFPUState() {
  if (DEBUG_FPU_REGS) {
    m_state.SetError(e_regSetFPU, Write, 0);
    return m_state.GetError(e_regSetFPU, Write);
  } else {
    if (CPUHasAVX() || FORCE_AVX_REGS)
      m_state.SetError(
          e_regSetFPU, Write,
          ::thread_set_state(m_thread->MachPortNumber(), __i386_AVX_STATE,
                             (thread_state_t)&m_state.context.fpu.avx,
                             e_regSetWordSizeAVX));
    else
      m_state.SetError(
          e_regSetFPU, Write,
          ::thread_set_state(m_thread->MachPortNumber(), __i386_FLOAT_STATE,
                             (thread_state_t)&m_state.context.fpu.no_avx,
                             e_regSetWordSizeFPU));
    return m_state.GetError(e_regSetFPU, Write);
  }
}

kern_return_t DNBArchImplI386::SetEXCState() {
  m_state.SetError(e_regSetEXC, Write,
                   ::thread_set_state(m_thread->MachPortNumber(),
                                      __i386_EXCEPTION_STATE,
                                      (thread_state_t)&m_state.context.exc,
                                      e_regSetWordSizeEXC));
  return m_state.GetError(e_regSetEXC, Write);
}

kern_return_t DNBArchImplI386::GetDBGState(bool force) {
  if (force || m_state.GetError(e_regSetDBG, Read)) {
    mach_msg_type_number_t count = e_regSetWordSizeDBG;
    m_state.SetError(
        e_regSetDBG, Read,
        ::thread_get_state(m_thread->MachPortNumber(), __i386_DEBUG_STATE,
                           (thread_state_t)&m_state.context.dbg, &count));
  }
  return m_state.GetError(e_regSetDBG, Read);
}

kern_return_t DNBArchImplI386::SetDBGState(bool also_set_on_task) {
  m_state.SetError(e_regSetDBG, Write,
                   ::thread_set_state(m_thread->MachPortNumber(),
                                      __i386_DEBUG_STATE,
                                      (thread_state_t)&m_state.context.dbg,
                                      e_regSetWordSizeDBG));
  if (also_set_on_task) {
    kern_return_t kret = ::task_set_state(
        m_thread->Process()->Task().TaskPort(), __i386_DEBUG_STATE,
        (thread_state_t)&m_state.context.dbg, e_regSetWordSizeDBG);
    if (kret != KERN_SUCCESS)
      DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchImplI386::SetDBGState failed "
                                        "to set debug control register state: "
                                        "0x%8.8x.",
                       kret);
  }
  return m_state.GetError(e_regSetDBG, Write);
}

void DNBArchImplI386::ThreadWillResume() {
  // Do we need to step this thread? If so, let the mach thread tell us so.
  if (m_thread->IsStepping()) {
    // This is the primary thread, let the arch do anything it needs
    EnableHardwareSingleStep(true);
  }

  // Reset the debug status register, if necessary, before we resume.
  kern_return_t kret = GetDBGState(false);
  DNBLogThreadedIf(
      LOG_WATCHPOINTS,
      "DNBArchImplI386::ThreadWillResume() GetDBGState() => 0x%8.8x.", kret);
  if (kret != KERN_SUCCESS)
    return;

  DBG &debug_state = m_state.context.dbg;
  bool need_reset = false;
  uint32_t i, num = NumSupportedHardwareWatchpoints();
  for (i = 0; i < num; ++i)
    if (IsWatchpointHit(debug_state, i))
      need_reset = true;

  if (need_reset) {
    ClearWatchpointHits(debug_state);
    kret = SetDBGState(false);
    DNBLogThreadedIf(
        LOG_WATCHPOINTS,
        "DNBArchImplI386::ThreadWillResume() SetDBGState() => 0x%8.8x.", kret);
  }
}

bool DNBArchImplI386::ThreadDidStop() {
  bool success = true;

  m_state.InvalidateAllRegisterStates();

  // Are we stepping a single instruction?
  if (GetGPRState(true) == KERN_SUCCESS) {
    // We are single stepping, was this the primary thread?
    if (m_thread->IsStepping()) {
      // This was the primary thread, we need to clear the trace
      // bit if so.
      success = EnableHardwareSingleStep(false) == KERN_SUCCESS;
    } else {
      // The MachThread will automatically restore the suspend count
      // in ThreadDidStop(), so we don't need to do anything here if
      // we weren't the primary thread the last time
    }
  }
  return success;
}

bool DNBArchImplI386::NotifyException(MachException::Data &exc) {
  switch (exc.exc_type) {
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
    if (exc.exc_data.size() >= 2 && exc.exc_data[0] == 2) {
      // exc_code = EXC_I386_BPT
      //
      nub_addr_t pc = GetPC(INVALID_NUB_ADDRESS);
      if (pc != INVALID_NUB_ADDRESS && pc > 0) {
        pc -= 1;
        // Check for a breakpoint at one byte prior to the current PC value
        // since the PC will be just past the trap.

        DNBBreakpoint *bp =
            m_thread->Process()->Breakpoints().FindByAddress(pc);
        if (bp) {
          // Backup the PC for i386 since the trap was taken and the PC
          // is at the address following the single byte trap instruction.
          if (m_state.context.gpr.__eip > 0) {
            m_state.context.gpr.__eip = static_cast<uint32_t>(pc);
            // Write the new PC back out
            SetGPRState();
          }
        }
        return true;
      }
    } else if (exc.exc_data.size() >= 2 && exc.exc_data[0] == 1) {
      // exc_code = EXC_I386_SGL
      //
      // Check whether this corresponds to a watchpoint hit event.
      // If yes, set the exc_sub_code to the data break address.
      nub_addr_t addr = 0;
      uint32_t hw_index = GetHardwareWatchpointHit(addr);
      if (hw_index != INVALID_NUB_HW_INDEX) {
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

uint32_t DNBArchImplI386::NumSupportedHardwareWatchpoints() {
  // Available debug address registers: dr0, dr1, dr2, dr3.
  return 4;
}

static uint32_t size_and_rw_bits(nub_size_t size, bool read, bool write) {
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
  }
  assert(0 && "invalid size, must be one of 1, 2, 4, or 8");
  return 0;
}

void DNBArchImplI386::SetWatchpoint(DBG &debug_state, uint32_t hw_index,
                                    nub_addr_t addr, nub_size_t size, bool read,
                                    bool write) {
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
  //            0b00: execute, 0b01: write, 0b11: read-or-write, 0b10: io
  //            read-or-write (unused)
  //      and bits{x+3, x+2} => len
  //            0b00: 1-byte, 0b01: 2-byte, 0b11: 4-byte, 0b10: 8-byte
  //
  //      dr0 -> bits{19-16}
  //      dr1 -> bits{23-20}
  //      dr2 -> bits{27-24}
  //      dr3 -> bits{31-28}
  debug_state.__dr7 |=
      (1 << (2 * hw_index) |
       size_and_rw_bits(size, read, write) << (16 + 4 * hw_index));
  uint32_t addr_32 = addr & 0xffffffff;
  switch (hw_index) {
  case 0:
    debug_state.__dr0 = addr_32;
    break;
  case 1:
    debug_state.__dr1 = addr_32;
    break;
  case 2:
    debug_state.__dr2 = addr_32;
    break;
  case 3:
    debug_state.__dr3 = addr_32;
    break;
  default:
    assert(0 &&
           "invalid hardware register index, must be one of 0, 1, 2, or 3");
  }
  return;
}

void DNBArchImplI386::ClearWatchpoint(DBG &debug_state, uint32_t hw_index) {
  debug_state.__dr7 &= ~(3 << (2 * hw_index));
  switch (hw_index) {
  case 0:
    debug_state.__dr0 = 0;
    break;
  case 1:
    debug_state.__dr1 = 0;
    break;
  case 2:
    debug_state.__dr2 = 0;
    break;
  case 3:
    debug_state.__dr3 = 0;
    break;
  default:
    assert(0 &&
           "invalid hardware register index, must be one of 0, 1, 2, or 3");
  }
  return;
}

bool DNBArchImplI386::IsWatchpointVacant(const DBG &debug_state,
                                         uint32_t hw_index) {
  // Check dr7 (debug control register) for local/global enable bits:
  //  global enable --. .-- local enable
  //                  | |
  //                  v v
  //      dr0 -> bits{1-0}
  //      dr1 -> bits{3-2}
  //      dr2 -> bits{5-4}
  //      dr3 -> bits{7-6}
  return (debug_state.__dr7 & (3 << (2 * hw_index))) == 0;
}

// Resets local copy of debug status register to wait for the next debug
// exception.
void DNBArchImplI386::ClearWatchpointHits(DBG &debug_state) {
  // See also IsWatchpointHit().
  debug_state.__dr6 = 0;
  return;
}

bool DNBArchImplI386::IsWatchpointHit(const DBG &debug_state,
                                      uint32_t hw_index) {
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

nub_addr_t DNBArchImplI386::GetWatchAddress(const DBG &debug_state,
                                            uint32_t hw_index) {
  switch (hw_index) {
  case 0:
    return debug_state.__dr0;
  case 1:
    return debug_state.__dr1;
  case 2:
    return debug_state.__dr2;
  case 3:
    return debug_state.__dr3;
  }
  assert(0 && "invalid hardware register index, must be one of 0, 1, 2, or 3");
  return 0;
}

bool DNBArchImplI386::StartTransForHWP() {
  if (m_2pc_trans_state != Trans_Done && m_2pc_trans_state != Trans_Rolled_Back)
    DNBLogError("%s inconsistent state detected, expected %d or %d, got: %d",
                __FUNCTION__, Trans_Done, Trans_Rolled_Back, m_2pc_trans_state);
  m_2pc_dbg_checkpoint = m_state.context.dbg;
  m_2pc_trans_state = Trans_Pending;
  return true;
}
bool DNBArchImplI386::RollbackTransForHWP() {
  m_state.context.dbg = m_2pc_dbg_checkpoint;
  if (m_2pc_trans_state != Trans_Pending)
    DNBLogError("%s inconsistent state detected, expected %d, got: %d",
                __FUNCTION__, Trans_Pending, m_2pc_trans_state);
  m_2pc_trans_state = Trans_Rolled_Back;
  kern_return_t kret = SetDBGState(false);
  DNBLogThreadedIf(
      LOG_WATCHPOINTS,
      "DNBArchImplI386::RollbackTransForHWP() SetDBGState() => 0x%8.8x.", kret);

  if (kret == KERN_SUCCESS)
    return true;
  else
    return false;
}
bool DNBArchImplI386::FinishTransForHWP() {
  m_2pc_trans_state = Trans_Done;
  return true;
}
DNBArchImplI386::DBG DNBArchImplI386::GetDBGCheckpoint() {
  return m_2pc_dbg_checkpoint;
}

uint32_t DNBArchImplI386::EnableHardwareWatchpoint(nub_addr_t addr,
                                                   nub_size_t size, bool read,
                                                   bool write,
                                                   bool also_set_on_task) {
  DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchImplI386::EnableHardwareWatchpoint("
                                    "addr = 0x%llx, size = %llu, read = %u, "
                                    "write = %u)",
                   (uint64_t)addr, (uint64_t)size, read, write);

  const uint32_t num_hw_watchpoints = NumSupportedHardwareWatchpoints();

  // Can only watch 1, 2, 4, or 8 bytes.
  if (!(size == 1 || size == 2 || size == 4 || size == 8))
    return INVALID_NUB_HW_INDEX;

  // We must watch for either read or write
  if (read == false && write == false)
    return INVALID_NUB_HW_INDEX;

  // Read the debug state
  kern_return_t kret = GetDBGState(false);

  if (kret == KERN_SUCCESS) {
    // Check to make sure we have the needed hardware support
    uint32_t i = 0;

    DBG &debug_state = m_state.context.dbg;
    for (i = 0; i < num_hw_watchpoints; ++i) {
      if (IsWatchpointVacant(debug_state, i))
        break;
    }

    // See if we found an available hw breakpoint slot above
    if (i < num_hw_watchpoints) {
      StartTransForHWP();

      // Modify our local copy of the debug state, first.
      SetWatchpoint(debug_state, i, addr, size, read, write);
      // Now set the watch point in the inferior.
      kret = SetDBGState(also_set_on_task);
      DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchImplI386::"
                                        "EnableHardwareWatchpoint() "
                                        "SetDBGState() => 0x%8.8x.",
                       kret);

      if (kret == KERN_SUCCESS)
        return i;
      else // Revert to the previous debug state voluntarily.  The transaction
           // coordinator knows that we have failed.
        m_state.context.dbg = GetDBGCheckpoint();
    } else {
      DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchImplI386::"
                                        "EnableHardwareWatchpoint(): All "
                                        "hardware resources (%u) are in use.",
                       num_hw_watchpoints);
    }
  }
  return INVALID_NUB_HW_INDEX;
}

bool DNBArchImplI386::DisableHardwareWatchpoint(uint32_t hw_index,
                                                bool also_set_on_task) {
  kern_return_t kret = GetDBGState(false);

  const uint32_t num_hw_points = NumSupportedHardwareWatchpoints();
  if (kret == KERN_SUCCESS) {
    DBG &debug_state = m_state.context.dbg;
    if (hw_index < num_hw_points &&
        !IsWatchpointVacant(debug_state, hw_index)) {
      StartTransForHWP();

      // Modify our local copy of the debug state, first.
      ClearWatchpoint(debug_state, hw_index);
      // Now disable the watch point in the inferior.
      kret = SetDBGState(also_set_on_task);
      DNBLogThreadedIf(LOG_WATCHPOINTS,
                       "DNBArchImplI386::DisableHardwareWatchpoint( %u )",
                       hw_index);

      if (kret == KERN_SUCCESS)
        return true;
      else // Revert to the previous debug state voluntarily.  The transaction
           // coordinator knows that we have failed.
        m_state.context.dbg = GetDBGCheckpoint();
    }
  }
  return false;
}

// Iterate through the debug status register; return the index of the first hit.
uint32_t DNBArchImplI386::GetHardwareWatchpointHit(nub_addr_t &addr) {
  // Read the debug state
  kern_return_t kret = GetDBGState(true);
  DNBLogThreadedIf(
      LOG_WATCHPOINTS,
      "DNBArchImplI386::GetHardwareWatchpointHit() GetDBGState() => 0x%8.8x.",
      kret);
  if (kret == KERN_SUCCESS) {
    DBG &debug_state = m_state.context.dbg;
    uint32_t i, num = NumSupportedHardwareWatchpoints();
    for (i = 0; i < num; ++i) {
      if (IsWatchpointHit(debug_state, i)) {
        addr = GetWatchAddress(debug_state, i);
        DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchImplI386::"
                                          "GetHardwareWatchpointHit() found => "
                                          "%u (addr = 0x%llx).",
                         i, (uint64_t)addr);
        return i;
      }
    }
  }
  return INVALID_NUB_HW_INDEX;
}

// Set the single step bit in the processor status register.
kern_return_t DNBArchImplI386::EnableHardwareSingleStep(bool enable) {
  if (GetGPRState(false) == KERN_SUCCESS) {
    const uint32_t trace_bit = 0x100u;
    if (enable)
      m_state.context.gpr.__eflags |= trace_bit;
    else
      m_state.context.gpr.__eflags &= ~trace_bit;
    return SetGPRState();
  }
  return m_state.GetError(e_regSetGPR, Read);
}

//----------------------------------------------------------------------
// Register information definitions
//----------------------------------------------------------------------

#define DEFINE_GPR_PSEUDO_16(reg16, reg32)                                     \
  {                                                                            \
    e_regSetGPR, gpr_##reg16, #reg16, NULL, Uint, Hex, 2, 0,                   \
        INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM,            \
        INVALID_NUB_REGNUM, g_contained_##reg32, g_invalidate_##reg32          \
  }
#define DEFINE_GPR_PSEUDO_8H(reg8, reg32)                                      \
  {                                                                            \
    e_regSetGPR, gpr_##reg8, #reg8, NULL, Uint, Hex, 1, 1, INVALID_NUB_REGNUM, \
        INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM,            \
        g_contained_##reg32, g_invalidate_##reg32                              \
  }
#define DEFINE_GPR_PSEUDO_8L(reg8, reg32)                                      \
  {                                                                            \
    e_regSetGPR, gpr_##reg8, #reg8, NULL, Uint, Hex, 1, 0, INVALID_NUB_REGNUM, \
        INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM,            \
        g_contained_##reg32, g_invalidate_##reg32                              \
  }

#define GPR_OFFSET(reg) (offsetof(DNBArchImplI386::GPR, __##reg))
#define FPU_OFFSET(reg)                                                        \
  (offsetof(DNBArchImplI386::FPU, __fpu_##reg) +                               \
   offsetof(DNBArchImplI386::Context, fpu.no_avx))
#define AVX_OFFSET(reg)                                                        \
  (offsetof(DNBArchImplI386::AVX, __fpu_##reg) +                               \
   offsetof(DNBArchImplI386::Context, fpu.avx))
#define EXC_OFFSET(reg)                                                        \
  (offsetof(DNBArchImplI386::EXC, __##reg) +                                   \
   offsetof(DNBArchImplI386::Context, exc))

#define GPR_SIZE(reg) (sizeof(((DNBArchImplI386::GPR *)NULL)->__##reg))
#define FPU_SIZE_UINT(reg) (sizeof(((DNBArchImplI386::FPU *)NULL)->__fpu_##reg))
#define FPU_SIZE_MMST(reg)                                                     \
  (sizeof(((DNBArchImplI386::FPU *)NULL)->__fpu_##reg.__mmst_reg))
#define FPU_SIZE_XMM(reg)                                                      \
  (sizeof(((DNBArchImplI386::FPU *)NULL)->__fpu_##reg.__xmm_reg))
#define FPU_SIZE_YMM(reg) (32)
#define EXC_SIZE(reg) (sizeof(((DNBArchImplI386::EXC *)NULL)->__##reg))

// This does not accurately identify the location of ymm0...7 in
// Context.fpu.avx.  That is because there is a bunch of padding
// in Context.fpu.avx that we don't need.  Offset macros lay out
// the register state that Debugserver transmits to the debugger
// -- not to interpret the thread_get_state info.
#define AVX_OFFSET_YMM(n) (AVX_OFFSET(xmm7) + FPU_SIZE_XMM(xmm7) + (32 * n))

// These macros will auto define the register name, alt name, register size,
// register offset, encoding, format and native register. This ensures that
// the register state structures are defined correctly and have the correct
// sizes and offsets.

const char *g_contained_eax[] = {"eax", NULL};
const char *g_contained_ebx[] = {"ebx", NULL};
const char *g_contained_ecx[] = {"ecx", NULL};
const char *g_contained_edx[] = {"edx", NULL};
const char *g_contained_edi[] = {"edi", NULL};
const char *g_contained_esi[] = {"esi", NULL};
const char *g_contained_ebp[] = {"ebp", NULL};
const char *g_contained_esp[] = {"esp", NULL};

const char *g_invalidate_eax[] = {"eax", "ax", "ah", "al", NULL};
const char *g_invalidate_ebx[] = {"ebx", "bx", "bh", "bl", NULL};
const char *g_invalidate_ecx[] = {"ecx", "cx", "ch", "cl", NULL};
const char *g_invalidate_edx[] = {"edx", "dx", "dh", "dl", NULL};
const char *g_invalidate_edi[] = {"edi", "di", "dil", NULL};
const char *g_invalidate_esi[] = {"esi", "si", "sil", NULL};
const char *g_invalidate_ebp[] = {"ebp", "bp", "bpl", NULL};
const char *g_invalidate_esp[] = {"esp", "sp", "spl", NULL};

// General purpose registers for 64 bit
const DNBRegisterInfo DNBArchImplI386::g_gpr_registers[] = {
    {e_regSetGPR, gpr_eax, "eax", NULL, Uint, Hex, GPR_SIZE(eax),
     GPR_OFFSET(eax), ehframe_eax, dwarf_eax, INVALID_NUB_REGNUM,
     debugserver_eax, NULL, g_invalidate_eax},
    {e_regSetGPR, gpr_ebx, "ebx", NULL, Uint, Hex, GPR_SIZE(ebx),
     GPR_OFFSET(ebx), ehframe_ebx, dwarf_ebx, INVALID_NUB_REGNUM,
     debugserver_ebx, NULL, g_invalidate_ebx},
    {e_regSetGPR, gpr_ecx, "ecx", NULL, Uint, Hex, GPR_SIZE(ecx),
     GPR_OFFSET(ecx), ehframe_ecx, dwarf_ecx, INVALID_NUB_REGNUM,
     debugserver_ecx, NULL, g_invalidate_ecx},
    {e_regSetGPR, gpr_edx, "edx", NULL, Uint, Hex, GPR_SIZE(edx),
     GPR_OFFSET(edx), ehframe_edx, dwarf_edx, INVALID_NUB_REGNUM,
     debugserver_edx, NULL, g_invalidate_edx},
    {e_regSetGPR, gpr_edi, "edi", NULL, Uint, Hex, GPR_SIZE(edi),
     GPR_OFFSET(edi), ehframe_edi, dwarf_edi, INVALID_NUB_REGNUM,
     debugserver_edi, NULL, g_invalidate_edi},
    {e_regSetGPR, gpr_esi, "esi", NULL, Uint, Hex, GPR_SIZE(esi),
     GPR_OFFSET(esi), ehframe_esi, dwarf_esi, INVALID_NUB_REGNUM,
     debugserver_esi, NULL, g_invalidate_esi},
    {e_regSetGPR, gpr_ebp, "ebp", "fp", Uint, Hex, GPR_SIZE(ebp),
     GPR_OFFSET(ebp), ehframe_ebp, dwarf_ebp, GENERIC_REGNUM_FP,
     debugserver_ebp, NULL, g_invalidate_ebp},
    {e_regSetGPR, gpr_esp, "esp", "sp", Uint, Hex, GPR_SIZE(esp),
     GPR_OFFSET(esp), ehframe_esp, dwarf_esp, GENERIC_REGNUM_SP,
     debugserver_esp, NULL, g_invalidate_esp},
    {e_regSetGPR, gpr_ss, "ss", NULL, Uint, Hex, GPR_SIZE(ss), GPR_OFFSET(ss),
     INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, debugserver_ss,
     NULL, NULL},
    {e_regSetGPR, gpr_eflags, "eflags", "flags", Uint, Hex, GPR_SIZE(eflags),
     GPR_OFFSET(eflags), ehframe_eflags, dwarf_eflags, GENERIC_REGNUM_FLAGS,
     debugserver_eflags, NULL, NULL},
    {e_regSetGPR, gpr_eip, "eip", "pc", Uint, Hex, GPR_SIZE(eip),
     GPR_OFFSET(eip), ehframe_eip, dwarf_eip, GENERIC_REGNUM_PC,
     debugserver_eip, NULL, NULL},
    {e_regSetGPR, gpr_cs, "cs", NULL, Uint, Hex, GPR_SIZE(cs), GPR_OFFSET(cs),
     INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, debugserver_cs,
     NULL, NULL},
    {e_regSetGPR, gpr_ds, "ds", NULL, Uint, Hex, GPR_SIZE(ds), GPR_OFFSET(ds),
     INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, debugserver_ds,
     NULL, NULL},
    {e_regSetGPR, gpr_es, "es", NULL, Uint, Hex, GPR_SIZE(es), GPR_OFFSET(es),
     INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, debugserver_es,
     NULL, NULL},
    {e_regSetGPR, gpr_fs, "fs", NULL, Uint, Hex, GPR_SIZE(fs), GPR_OFFSET(fs),
     INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, debugserver_fs,
     NULL, NULL},
    {e_regSetGPR, gpr_gs, "gs", NULL, Uint, Hex, GPR_SIZE(gs), GPR_OFFSET(gs),
     INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, debugserver_gs,
     NULL, NULL},
    DEFINE_GPR_PSEUDO_16(ax, eax),
    DEFINE_GPR_PSEUDO_16(bx, ebx),
    DEFINE_GPR_PSEUDO_16(cx, ecx),
    DEFINE_GPR_PSEUDO_16(dx, edx),
    DEFINE_GPR_PSEUDO_16(di, edi),
    DEFINE_GPR_PSEUDO_16(si, esi),
    DEFINE_GPR_PSEUDO_16(bp, ebp),
    DEFINE_GPR_PSEUDO_16(sp, esp),
    DEFINE_GPR_PSEUDO_8H(ah, eax),
    DEFINE_GPR_PSEUDO_8H(bh, ebx),
    DEFINE_GPR_PSEUDO_8H(ch, ecx),
    DEFINE_GPR_PSEUDO_8H(dh, edx),
    DEFINE_GPR_PSEUDO_8L(al, eax),
    DEFINE_GPR_PSEUDO_8L(bl, ebx),
    DEFINE_GPR_PSEUDO_8L(cl, ecx),
    DEFINE_GPR_PSEUDO_8L(dl, edx),
    DEFINE_GPR_PSEUDO_8L(dil, edi),
    DEFINE_GPR_PSEUDO_8L(sil, esi),
    DEFINE_GPR_PSEUDO_8L(bpl, ebp),
    DEFINE_GPR_PSEUDO_8L(spl, esp)};

const DNBRegisterInfo DNBArchImplI386::g_fpu_registers_no_avx[] = {
    {e_regSetFPU, fpu_fcw, "fctrl", NULL, Uint, Hex, FPU_SIZE_UINT(fcw),
     FPU_OFFSET(fcw), INVALID_NUB_REGNUM, INVALID_NUB_REGNUM,
     INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, NULL, NULL},
    {e_regSetFPU, fpu_fsw, "fstat", NULL, Uint, Hex, FPU_SIZE_UINT(fsw),
     FPU_OFFSET(fsw), INVALID_NUB_REGNUM, INVALID_NUB_REGNUM,
     INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, NULL, NULL},
    {e_regSetFPU, fpu_ftw, "ftag", NULL, Uint, Hex, 2 /* sizeof __fpu_ftw + sizeof __fpu_rsrv1 */,
     FPU_OFFSET(ftw), INVALID_NUB_REGNUM, INVALID_NUB_REGNUM,
     INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, NULL, NULL},
    {e_regSetFPU, fpu_fop, "fop", NULL, Uint, Hex, FPU_SIZE_UINT(fop),
     FPU_OFFSET(fop), INVALID_NUB_REGNUM, INVALID_NUB_REGNUM,
     INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, NULL, NULL},
    {e_regSetFPU, fpu_ip, "fioff", NULL, Uint, Hex, FPU_SIZE_UINT(ip),
     FPU_OFFSET(ip), INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM,
     INVALID_NUB_REGNUM, NULL, NULL},
    {e_regSetFPU, fpu_cs, "fiseg", NULL, Uint, Hex, FPU_SIZE_UINT(cs),
     FPU_OFFSET(cs), INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM,
     INVALID_NUB_REGNUM, NULL, NULL},
    {e_regSetFPU, fpu_dp, "fooff", NULL, Uint, Hex, FPU_SIZE_UINT(dp),
     FPU_OFFSET(dp), INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM,
     INVALID_NUB_REGNUM, NULL, NULL},
    {e_regSetFPU, fpu_ds, "foseg", NULL, Uint, Hex, FPU_SIZE_UINT(ds),
     FPU_OFFSET(ds), INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM,
     INVALID_NUB_REGNUM, NULL, NULL},
    {e_regSetFPU, fpu_mxcsr, "mxcsr", NULL, Uint, Hex, FPU_SIZE_UINT(mxcsr),
     FPU_OFFSET(mxcsr), INVALID_NUB_REGNUM, INVALID_NUB_REGNUM,
     INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, NULL, NULL},
    {e_regSetFPU, fpu_mxcsrmask, "mxcsrmask", NULL, Uint, Hex,
     FPU_SIZE_UINT(mxcsrmask), FPU_OFFSET(mxcsrmask), INVALID_NUB_REGNUM,
     INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, NULL, NULL},

    {e_regSetFPU, fpu_stmm0, "stmm0", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_MMST(stmm0), FPU_OFFSET(stmm0), INVALID_NUB_REGNUM, dwarf_stmm0,
     INVALID_NUB_REGNUM, debugserver_stmm0, NULL, NULL},
    {e_regSetFPU, fpu_stmm1, "stmm1", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_MMST(stmm1), FPU_OFFSET(stmm1), INVALID_NUB_REGNUM, dwarf_stmm1,
     INVALID_NUB_REGNUM, debugserver_stmm1, NULL, NULL},
    {e_regSetFPU, fpu_stmm2, "stmm2", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_MMST(stmm2), FPU_OFFSET(stmm2), INVALID_NUB_REGNUM, dwarf_stmm2,
     INVALID_NUB_REGNUM, debugserver_stmm2, NULL, NULL},
    {e_regSetFPU, fpu_stmm3, "stmm3", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_MMST(stmm3), FPU_OFFSET(stmm3), INVALID_NUB_REGNUM, dwarf_stmm3,
     INVALID_NUB_REGNUM, debugserver_stmm3, NULL, NULL},
    {e_regSetFPU, fpu_stmm4, "stmm4", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_MMST(stmm4), FPU_OFFSET(stmm4), INVALID_NUB_REGNUM, dwarf_stmm4,
     INVALID_NUB_REGNUM, debugserver_stmm4, NULL, NULL},
    {e_regSetFPU, fpu_stmm5, "stmm5", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_MMST(stmm5), FPU_OFFSET(stmm5), INVALID_NUB_REGNUM, dwarf_stmm5,
     INVALID_NUB_REGNUM, debugserver_stmm5, NULL, NULL},
    {e_regSetFPU, fpu_stmm6, "stmm6", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_MMST(stmm6), FPU_OFFSET(stmm6), INVALID_NUB_REGNUM, dwarf_stmm6,
     INVALID_NUB_REGNUM, debugserver_stmm6, NULL, NULL},
    {e_regSetFPU, fpu_stmm7, "stmm7", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_MMST(stmm7), FPU_OFFSET(stmm7), INVALID_NUB_REGNUM, dwarf_stmm7,
     INVALID_NUB_REGNUM, debugserver_stmm7, NULL, NULL},

    {e_regSetFPU, fpu_xmm0, "xmm0", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_XMM(xmm0), FPU_OFFSET(xmm0), INVALID_NUB_REGNUM, dwarf_xmm0,
     INVALID_NUB_REGNUM, debugserver_xmm0, NULL, NULL},
    {e_regSetFPU, fpu_xmm1, "xmm1", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_XMM(xmm1), FPU_OFFSET(xmm1), INVALID_NUB_REGNUM, dwarf_xmm1,
     INVALID_NUB_REGNUM, debugserver_xmm1, NULL, NULL},
    {e_regSetFPU, fpu_xmm2, "xmm2", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_XMM(xmm2), FPU_OFFSET(xmm2), INVALID_NUB_REGNUM, dwarf_xmm2,
     INVALID_NUB_REGNUM, debugserver_xmm2, NULL, NULL},
    {e_regSetFPU, fpu_xmm3, "xmm3", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_XMM(xmm3), FPU_OFFSET(xmm3), INVALID_NUB_REGNUM, dwarf_xmm3,
     INVALID_NUB_REGNUM, debugserver_xmm3, NULL, NULL},
    {e_regSetFPU, fpu_xmm4, "xmm4", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_XMM(xmm4), FPU_OFFSET(xmm4), INVALID_NUB_REGNUM, dwarf_xmm4,
     INVALID_NUB_REGNUM, debugserver_xmm4, NULL, NULL},
    {e_regSetFPU, fpu_xmm5, "xmm5", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_XMM(xmm5), FPU_OFFSET(xmm5), INVALID_NUB_REGNUM, dwarf_xmm5,
     INVALID_NUB_REGNUM, debugserver_xmm5, NULL, NULL},
    {e_regSetFPU, fpu_xmm6, "xmm6", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_XMM(xmm6), FPU_OFFSET(xmm6), INVALID_NUB_REGNUM, dwarf_xmm6,
     INVALID_NUB_REGNUM, debugserver_xmm6, NULL, NULL},
    {e_regSetFPU, fpu_xmm7, "xmm7", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_XMM(xmm7), FPU_OFFSET(xmm7), INVALID_NUB_REGNUM, dwarf_xmm7,
     INVALID_NUB_REGNUM, debugserver_xmm7, NULL, NULL}};

static const char *g_contained_ymm0[] = {"ymm0", NULL};
static const char *g_contained_ymm1[] = {"ymm1", NULL};
static const char *g_contained_ymm2[] = {"ymm2", NULL};
static const char *g_contained_ymm3[] = {"ymm3", NULL};
static const char *g_contained_ymm4[] = {"ymm4", NULL};
static const char *g_contained_ymm5[] = {"ymm5", NULL};
static const char *g_contained_ymm6[] = {"ymm6", NULL};
static const char *g_contained_ymm7[] = {"ymm7", NULL};

const DNBRegisterInfo DNBArchImplI386::g_fpu_registers_avx[] = {
    {e_regSetFPU, fpu_fcw, "fctrl", NULL, Uint, Hex, FPU_SIZE_UINT(fcw),
     AVX_OFFSET(fcw), INVALID_NUB_REGNUM, INVALID_NUB_REGNUM,
     INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, NULL, NULL},
    {e_regSetFPU, fpu_fsw, "fstat", NULL, Uint, Hex, FPU_SIZE_UINT(fsw),
     AVX_OFFSET(fsw), INVALID_NUB_REGNUM, INVALID_NUB_REGNUM,
     INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, NULL, NULL},
    {e_regSetFPU, fpu_ftw, "ftag", NULL, Uint, Hex, 2 /* sizeof __fpu_ftw + sizeof __fpu_rsrv1 */,
     AVX_OFFSET(ftw), INVALID_NUB_REGNUM, INVALID_NUB_REGNUM,
     INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, NULL, NULL},
    {e_regSetFPU, fpu_fop, "fop", NULL, Uint, Hex, FPU_SIZE_UINT(fop),
     AVX_OFFSET(fop), INVALID_NUB_REGNUM, INVALID_NUB_REGNUM,
     INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, NULL, NULL},
    {e_regSetFPU, fpu_ip, "fioff", NULL, Uint, Hex, FPU_SIZE_UINT(ip),
     AVX_OFFSET(ip), INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM,
     INVALID_NUB_REGNUM, NULL, NULL},
    {e_regSetFPU, fpu_cs, "fiseg", NULL, Uint, Hex, FPU_SIZE_UINT(cs),
     AVX_OFFSET(cs), INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM,
     INVALID_NUB_REGNUM, NULL, NULL},
    {e_regSetFPU, fpu_dp, "fooff", NULL, Uint, Hex, FPU_SIZE_UINT(dp),
     AVX_OFFSET(dp), INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM,
     INVALID_NUB_REGNUM, NULL, NULL},
    {e_regSetFPU, fpu_ds, "foseg", NULL, Uint, Hex, FPU_SIZE_UINT(ds),
     AVX_OFFSET(ds), INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM,
     INVALID_NUB_REGNUM, NULL, NULL},
    {e_regSetFPU, fpu_mxcsr, "mxcsr", NULL, Uint, Hex, FPU_SIZE_UINT(mxcsr),
     AVX_OFFSET(mxcsr), INVALID_NUB_REGNUM, INVALID_NUB_REGNUM,
     INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, NULL, NULL},
    {e_regSetFPU, fpu_mxcsrmask, "mxcsrmask", NULL, Uint, Hex,
     FPU_SIZE_UINT(mxcsrmask), AVX_OFFSET(mxcsrmask), INVALID_NUB_REGNUM,
     INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, NULL, NULL},

    {e_regSetFPU, fpu_stmm0, "stmm0", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_MMST(stmm0), AVX_OFFSET(stmm0), INVALID_NUB_REGNUM, dwarf_stmm0,
     INVALID_NUB_REGNUM, debugserver_stmm0, NULL, NULL},
    {e_regSetFPU, fpu_stmm1, "stmm1", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_MMST(stmm1), AVX_OFFSET(stmm1), INVALID_NUB_REGNUM, dwarf_stmm1,
     INVALID_NUB_REGNUM, debugserver_stmm1, NULL, NULL},
    {e_regSetFPU, fpu_stmm2, "stmm2", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_MMST(stmm2), AVX_OFFSET(stmm2), INVALID_NUB_REGNUM, dwarf_stmm2,
     INVALID_NUB_REGNUM, debugserver_stmm2, NULL, NULL},
    {e_regSetFPU, fpu_stmm3, "stmm3", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_MMST(stmm3), AVX_OFFSET(stmm3), INVALID_NUB_REGNUM, dwarf_stmm3,
     INVALID_NUB_REGNUM, debugserver_stmm3, NULL, NULL},
    {e_regSetFPU, fpu_stmm4, "stmm4", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_MMST(stmm4), AVX_OFFSET(stmm4), INVALID_NUB_REGNUM, dwarf_stmm4,
     INVALID_NUB_REGNUM, debugserver_stmm4, NULL, NULL},
    {e_regSetFPU, fpu_stmm5, "stmm5", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_MMST(stmm5), AVX_OFFSET(stmm5), INVALID_NUB_REGNUM, dwarf_stmm5,
     INVALID_NUB_REGNUM, debugserver_stmm5, NULL, NULL},
    {e_regSetFPU, fpu_stmm6, "stmm6", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_MMST(stmm6), AVX_OFFSET(stmm6), INVALID_NUB_REGNUM, dwarf_stmm6,
     INVALID_NUB_REGNUM, debugserver_stmm6, NULL, NULL},
    {e_regSetFPU, fpu_stmm7, "stmm7", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_MMST(stmm7), AVX_OFFSET(stmm7), INVALID_NUB_REGNUM, dwarf_stmm7,
     INVALID_NUB_REGNUM, debugserver_stmm7, NULL, NULL},

    {e_regSetFPU, fpu_ymm0, "ymm0", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_YMM(ymm0), AVX_OFFSET_YMM(0), INVALID_NUB_REGNUM, dwarf_ymm0,
     INVALID_NUB_REGNUM, debugserver_ymm0, NULL, NULL},
    {e_regSetFPU, fpu_ymm1, "ymm1", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_YMM(ymm1), AVX_OFFSET_YMM(1), INVALID_NUB_REGNUM, dwarf_ymm1,
     INVALID_NUB_REGNUM, debugserver_ymm1, NULL, NULL},
    {e_regSetFPU, fpu_ymm2, "ymm2", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_YMM(ymm2), AVX_OFFSET_YMM(2), INVALID_NUB_REGNUM, dwarf_ymm2,
     INVALID_NUB_REGNUM, debugserver_ymm2, NULL, NULL},
    {e_regSetFPU, fpu_ymm3, "ymm3", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_YMM(ymm3), AVX_OFFSET_YMM(3), INVALID_NUB_REGNUM, dwarf_ymm3,
     INVALID_NUB_REGNUM, debugserver_ymm3, NULL, NULL},
    {e_regSetFPU, fpu_ymm4, "ymm4", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_YMM(ymm4), AVX_OFFSET_YMM(4), INVALID_NUB_REGNUM, dwarf_ymm4,
     INVALID_NUB_REGNUM, debugserver_ymm4, NULL, NULL},
    {e_regSetFPU, fpu_ymm5, "ymm5", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_YMM(ymm5), AVX_OFFSET_YMM(5), INVALID_NUB_REGNUM, dwarf_ymm5,
     INVALID_NUB_REGNUM, debugserver_ymm5, NULL, NULL},
    {e_regSetFPU, fpu_ymm6, "ymm6", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_YMM(ymm6), AVX_OFFSET_YMM(6), INVALID_NUB_REGNUM, dwarf_ymm6,
     INVALID_NUB_REGNUM, debugserver_ymm6, NULL, NULL},
    {e_regSetFPU, fpu_ymm7, "ymm7", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_YMM(ymm7), AVX_OFFSET_YMM(7), INVALID_NUB_REGNUM, dwarf_ymm7,
     INVALID_NUB_REGNUM, debugserver_ymm7, NULL, NULL},

    {e_regSetFPU, fpu_xmm0, "xmm0", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_XMM(xmm0), 0, INVALID_NUB_REGNUM, dwarf_xmm0, INVALID_NUB_REGNUM,
     debugserver_xmm0, g_contained_ymm0, NULL},
    {e_regSetFPU, fpu_xmm1, "xmm1", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_XMM(xmm1), 0, INVALID_NUB_REGNUM, dwarf_xmm1, INVALID_NUB_REGNUM,
     debugserver_xmm1, g_contained_ymm1, NULL},
    {e_regSetFPU, fpu_xmm2, "xmm2", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_XMM(xmm2), 0, INVALID_NUB_REGNUM, dwarf_xmm2, INVALID_NUB_REGNUM,
     debugserver_xmm2, g_contained_ymm2, NULL},
    {e_regSetFPU, fpu_xmm3, "xmm3", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_XMM(xmm3), 0, INVALID_NUB_REGNUM, dwarf_xmm3, INVALID_NUB_REGNUM,
     debugserver_xmm3, g_contained_ymm3, NULL},
    {e_regSetFPU, fpu_xmm4, "xmm4", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_XMM(xmm4), 0, INVALID_NUB_REGNUM, dwarf_xmm4, INVALID_NUB_REGNUM,
     debugserver_xmm4, g_contained_ymm4, NULL},
    {e_regSetFPU, fpu_xmm5, "xmm5", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_XMM(xmm5), 0, INVALID_NUB_REGNUM, dwarf_xmm5, INVALID_NUB_REGNUM,
     debugserver_xmm5, g_contained_ymm5, NULL},
    {e_regSetFPU, fpu_xmm6, "xmm6", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_XMM(xmm6), 0, INVALID_NUB_REGNUM, dwarf_xmm6, INVALID_NUB_REGNUM,
     debugserver_xmm6, g_contained_ymm6, NULL},
    {e_regSetFPU, fpu_xmm7, "xmm7", NULL, Vector, VectorOfUInt8,
     FPU_SIZE_XMM(xmm7), 0, INVALID_NUB_REGNUM, dwarf_xmm7, INVALID_NUB_REGNUM,
     debugserver_xmm7, g_contained_ymm7, NULL},

};

const DNBRegisterInfo DNBArchImplI386::g_exc_registers[] = {
    {e_regSetEXC, exc_trapno, "trapno", NULL, Uint, Hex, EXC_SIZE(trapno),
     EXC_OFFSET(trapno), INVALID_NUB_REGNUM, INVALID_NUB_REGNUM,
     INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, NULL, NULL},
    {e_regSetEXC, exc_err, "err", NULL, Uint, Hex, EXC_SIZE(err),
     EXC_OFFSET(err), INVALID_NUB_REGNUM, INVALID_NUB_REGNUM,
     INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, NULL, NULL},
    {e_regSetEXC, exc_faultvaddr, "faultvaddr", NULL, Uint, Hex,
     EXC_SIZE(faultvaddr), EXC_OFFSET(faultvaddr), INVALID_NUB_REGNUM,
     INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, NULL, NULL}};

// Number of registers in each register set
const size_t DNBArchImplI386::k_num_gpr_registers =
    sizeof(g_gpr_registers) / sizeof(DNBRegisterInfo);
const size_t DNBArchImplI386::k_num_fpu_registers_no_avx =
    sizeof(g_fpu_registers_no_avx) / sizeof(DNBRegisterInfo);
const size_t DNBArchImplI386::k_num_fpu_registers_avx =
    sizeof(g_fpu_registers_avx) / sizeof(DNBRegisterInfo);
const size_t DNBArchImplI386::k_num_exc_registers =
    sizeof(g_exc_registers) / sizeof(DNBRegisterInfo);
const size_t DNBArchImplI386::k_num_all_registers_no_avx =
    k_num_gpr_registers + k_num_fpu_registers_no_avx + k_num_exc_registers;
const size_t DNBArchImplI386::k_num_all_registers_avx =
    k_num_gpr_registers + k_num_fpu_registers_avx + k_num_exc_registers;

//----------------------------------------------------------------------
// Register set definitions. The first definitions at register set index
// of zero is for all registers, followed by other registers sets. The
// register information for the all register set need not be filled in.
//----------------------------------------------------------------------
const DNBRegisterSetInfo DNBArchImplI386::g_reg_sets_no_avx[] = {
    {"i386 Registers", NULL, k_num_all_registers_no_avx},
    {"General Purpose Registers", g_gpr_registers, k_num_gpr_registers},
    {"Floating Point Registers", g_fpu_registers_no_avx,
     k_num_fpu_registers_no_avx},
    {"Exception State Registers", g_exc_registers, k_num_exc_registers}};

const DNBRegisterSetInfo DNBArchImplI386::g_reg_sets_avx[] = {
    {"i386 Registers", NULL, k_num_all_registers_avx},
    {"General Purpose Registers", g_gpr_registers, k_num_gpr_registers},
    {"Floating Point Registers", g_fpu_registers_avx, k_num_fpu_registers_avx},
    {"Exception State Registers", g_exc_registers, k_num_exc_registers}};

// Total number of register sets for this architecture
const size_t DNBArchImplI386::k_num_register_sets =
    sizeof(g_reg_sets_no_avx) / sizeof(DNBRegisterSetInfo);

DNBArchProtocol *DNBArchImplI386::Create(MachThread *thread) {
  DNBArchImplI386 *obj = new DNBArchImplI386(thread);
  return obj;
}

const uint8_t *DNBArchImplI386::SoftwareBreakpointOpcode(nub_size_t byte_size) {
  static const uint8_t g_breakpoint_opcode[] = {0xCC};
  if (byte_size == 1)
    return g_breakpoint_opcode;
  return NULL;
}

const DNBRegisterSetInfo *
DNBArchImplI386::GetRegisterSetInfo(nub_size_t *num_reg_sets) {
  *num_reg_sets = k_num_register_sets;
  if (CPUHasAVX() || FORCE_AVX_REGS)
    return g_reg_sets_avx;
  else
    return g_reg_sets_no_avx;
}

void DNBArchImplI386::Initialize() {
  DNBArchPluginInfo arch_plugin_info = {
      CPU_TYPE_I386, DNBArchImplI386::Create,
      DNBArchImplI386::GetRegisterSetInfo,
      DNBArchImplI386::SoftwareBreakpointOpcode};

  // Register this arch plug-in with the main protocol class
  DNBArchProtocol::RegisterArchPlugin(arch_plugin_info);
}

bool DNBArchImplI386::GetRegisterValue(uint32_t set, uint32_t reg,
                                       DNBRegisterValue *value) {
  if (set == REGISTER_SET_GENERIC) {
    switch (reg) {
    case GENERIC_REGNUM_PC: // Program Counter
      set = e_regSetGPR;
      reg = gpr_eip;
      break;

    case GENERIC_REGNUM_SP: // Stack Pointer
      set = e_regSetGPR;
      reg = gpr_esp;
      break;

    case GENERIC_REGNUM_FP: // Frame Pointer
      set = e_regSetGPR;
      reg = gpr_ebp;
      break;

    case GENERIC_REGNUM_FLAGS: // Processor flags register
      set = e_regSetGPR;
      reg = gpr_eflags;
      break;

    case GENERIC_REGNUM_RA: // Return Address
    default:
      return false;
    }
  }

  if (GetRegisterState(set, false) != KERN_SUCCESS)
    return false;

  const DNBRegisterInfo *regInfo = m_thread->GetRegisterInfo(set, reg);
  if (regInfo) {
    value->info = *regInfo;
    switch (set) {
    case e_regSetGPR:
      if (reg < k_num_gpr_registers) {
        value->value.uint32 = ((uint32_t *)(&m_state.context.gpr))[reg];
        return true;
      }
      break;

    case e_regSetFPU:
      if (reg > fpu_xmm7 && !(CPUHasAVX() || FORCE_AVX_REGS))
        return false;
      switch (reg) {
      case fpu_fcw:
        value->value.uint16 =
            *((uint16_t *)(&m_state.context.fpu.no_avx.__fpu_fcw));
        return true;
      case fpu_fsw:
        value->value.uint16 =
            *((uint16_t *)(&m_state.context.fpu.no_avx.__fpu_fsw));
        return true;
      case fpu_ftw:
        memcpy (&value->value.uint16, &m_state.context.fpu.no_avx.__fpu_ftw, 2);
        return true;
      case fpu_fop:
        value->value.uint16 = m_state.context.fpu.no_avx.__fpu_fop;
        return true;
      case fpu_ip:
        value->value.uint32 = m_state.context.fpu.no_avx.__fpu_ip;
        return true;
      case fpu_cs:
        value->value.uint16 = m_state.context.fpu.no_avx.__fpu_cs;
        return true;
      case fpu_dp:
        value->value.uint32 = m_state.context.fpu.no_avx.__fpu_dp;
        return true;
      case fpu_ds:
        value->value.uint16 = m_state.context.fpu.no_avx.__fpu_ds;
        return true;
      case fpu_mxcsr:
        value->value.uint32 = m_state.context.fpu.no_avx.__fpu_mxcsr;
        return true;
      case fpu_mxcsrmask:
        value->value.uint32 = m_state.context.fpu.no_avx.__fpu_mxcsrmask;
        return true;

      case fpu_stmm0:
        memcpy(&value->value.uint8,
               m_state.context.fpu.no_avx.__fpu_stmm0.__mmst_reg, 10);
        return true;
      case fpu_stmm1:
        memcpy(&value->value.uint8,
               m_state.context.fpu.no_avx.__fpu_stmm1.__mmst_reg, 10);
        return true;
      case fpu_stmm2:
        memcpy(&value->value.uint8,
               m_state.context.fpu.no_avx.__fpu_stmm2.__mmst_reg, 10);
        return true;
      case fpu_stmm3:
        memcpy(&value->value.uint8,
               m_state.context.fpu.no_avx.__fpu_stmm3.__mmst_reg, 10);
        return true;
      case fpu_stmm4:
        memcpy(&value->value.uint8,
               m_state.context.fpu.no_avx.__fpu_stmm4.__mmst_reg, 10);
        return true;
      case fpu_stmm5:
        memcpy(&value->value.uint8,
               m_state.context.fpu.no_avx.__fpu_stmm5.__mmst_reg, 10);
        return true;
      case fpu_stmm6:
        memcpy(&value->value.uint8,
               m_state.context.fpu.no_avx.__fpu_stmm6.__mmst_reg, 10);
        return true;
      case fpu_stmm7:
        memcpy(&value->value.uint8,
               m_state.context.fpu.no_avx.__fpu_stmm7.__mmst_reg, 10);
        return true;

      case fpu_xmm0:
        memcpy(&value->value.uint8,
               m_state.context.fpu.no_avx.__fpu_xmm0.__xmm_reg, 16);
        return true;
      case fpu_xmm1:
        memcpy(&value->value.uint8,
               m_state.context.fpu.no_avx.__fpu_xmm1.__xmm_reg, 16);
        return true;
      case fpu_xmm2:
        memcpy(&value->value.uint8,
               m_state.context.fpu.no_avx.__fpu_xmm2.__xmm_reg, 16);
        return true;
      case fpu_xmm3:
        memcpy(&value->value.uint8,
               m_state.context.fpu.no_avx.__fpu_xmm3.__xmm_reg, 16);
        return true;
      case fpu_xmm4:
        memcpy(&value->value.uint8,
               m_state.context.fpu.no_avx.__fpu_xmm4.__xmm_reg, 16);
        return true;
      case fpu_xmm5:
        memcpy(&value->value.uint8,
               m_state.context.fpu.no_avx.__fpu_xmm5.__xmm_reg, 16);
        return true;
      case fpu_xmm6:
        memcpy(&value->value.uint8,
               m_state.context.fpu.no_avx.__fpu_xmm6.__xmm_reg, 16);
        return true;
      case fpu_xmm7:
        memcpy(&value->value.uint8,
               m_state.context.fpu.no_avx.__fpu_xmm7.__xmm_reg, 16);
        return true;

#define MEMCPY_YMM(n)                                                          \
  memcpy(&value->value.uint8, m_state.context.fpu.avx.__fpu_xmm##n.__xmm_reg,  \
         16);                                                                  \
  memcpy((&value->value.uint8) + 16,                                           \
         m_state.context.fpu.avx.__fpu_ymmh##n.__xmm_reg, 16);
      case fpu_ymm0:
        MEMCPY_YMM(0);
        return true;
      case fpu_ymm1:
        MEMCPY_YMM(1);
        return true;
      case fpu_ymm2:
        MEMCPY_YMM(2);
        return true;
      case fpu_ymm3:
        MEMCPY_YMM(3);
        return true;
      case fpu_ymm4:
        MEMCPY_YMM(4);
        return true;
      case fpu_ymm5:
        MEMCPY_YMM(5);
        return true;
      case fpu_ymm6:
        MEMCPY_YMM(6);
        return true;
      case fpu_ymm7:
        MEMCPY_YMM(7);
        return true;
#undef MEMCPY_YMM
      }
      break;

    case e_regSetEXC:
      if (reg < k_num_exc_registers) {
        value->value.uint32 = (&m_state.context.exc.__trapno)[reg];
        return true;
      }
      break;
    }
  }
  return false;
}

bool DNBArchImplI386::SetRegisterValue(uint32_t set, uint32_t reg,
                                       const DNBRegisterValue *value) {
  if (set == REGISTER_SET_GENERIC) {
    switch (reg) {
    case GENERIC_REGNUM_PC: // Program Counter
      set = e_regSetGPR;
      reg = gpr_eip;
      break;

    case GENERIC_REGNUM_SP: // Stack Pointer
      set = e_regSetGPR;
      reg = gpr_esp;
      break;

    case GENERIC_REGNUM_FP: // Frame Pointer
      set = e_regSetGPR;
      reg = gpr_ebp;
      break;

    case GENERIC_REGNUM_FLAGS: // Processor flags register
      set = e_regSetGPR;
      reg = gpr_eflags;
      break;

    case GENERIC_REGNUM_RA: // Return Address
    default:
      return false;
    }
  }

  if (GetRegisterState(set, false) != KERN_SUCCESS)
    return false;

  bool success = false;
  const DNBRegisterInfo *regInfo = m_thread->GetRegisterInfo(set, reg);
  if (regInfo) {
    switch (set) {
    case e_regSetGPR:
      if (reg < k_num_gpr_registers) {
        ((uint32_t *)(&m_state.context.gpr))[reg] = value->value.uint32;
        success = true;
      }
      break;

    case e_regSetFPU:
      if (reg > fpu_xmm7 && !(CPUHasAVX() || FORCE_AVX_REGS))
        return false;
      switch (reg) {
      case fpu_fcw:
        *((uint16_t *)(&m_state.context.fpu.no_avx.__fpu_fcw)) =
            value->value.uint16;
        success = true;
        break;
      case fpu_fsw:
        *((uint16_t *)(&m_state.context.fpu.no_avx.__fpu_fsw)) =
            value->value.uint16;
        success = true;
        break;
      case fpu_ftw:
        memcpy (&m_state.context.fpu.no_avx.__fpu_ftw, &value->value.uint16, 2);
        success = true;
        break;
      case fpu_fop:
        m_state.context.fpu.no_avx.__fpu_fop = value->value.uint16;
        success = true;
        break;
      case fpu_ip:
        m_state.context.fpu.no_avx.__fpu_ip = value->value.uint32;
        success = true;
        break;
      case fpu_cs:
        m_state.context.fpu.no_avx.__fpu_cs = value->value.uint16;
        success = true;
        break;
      case fpu_dp:
        m_state.context.fpu.no_avx.__fpu_dp = value->value.uint32;
        success = true;
        break;
      case fpu_ds:
        m_state.context.fpu.no_avx.__fpu_ds = value->value.uint16;
        success = true;
        break;
      case fpu_mxcsr:
        m_state.context.fpu.no_avx.__fpu_mxcsr = value->value.uint32;
        success = true;
        break;
      case fpu_mxcsrmask:
        m_state.context.fpu.no_avx.__fpu_mxcsrmask = value->value.uint32;
        success = true;
        break;

      case fpu_stmm0:
        memcpy(m_state.context.fpu.no_avx.__fpu_stmm0.__mmst_reg,
               &value->value.uint8, 10);
        success = true;
        break;
      case fpu_stmm1:
        memcpy(m_state.context.fpu.no_avx.__fpu_stmm1.__mmst_reg,
               &value->value.uint8, 10);
        success = true;
        break;
      case fpu_stmm2:
        memcpy(m_state.context.fpu.no_avx.__fpu_stmm2.__mmst_reg,
               &value->value.uint8, 10);
        success = true;
        break;
      case fpu_stmm3:
        memcpy(m_state.context.fpu.no_avx.__fpu_stmm3.__mmst_reg,
               &value->value.uint8, 10);
        success = true;
        break;
      case fpu_stmm4:
        memcpy(m_state.context.fpu.no_avx.__fpu_stmm4.__mmst_reg,
               &value->value.uint8, 10);
        success = true;
        break;
      case fpu_stmm5:
        memcpy(m_state.context.fpu.no_avx.__fpu_stmm5.__mmst_reg,
               &value->value.uint8, 10);
        success = true;
        break;
      case fpu_stmm6:
        memcpy(m_state.context.fpu.no_avx.__fpu_stmm6.__mmst_reg,
               &value->value.uint8, 10);
        success = true;
        break;
      case fpu_stmm7:
        memcpy(m_state.context.fpu.no_avx.__fpu_stmm7.__mmst_reg,
               &value->value.uint8, 10);
        success = true;
        break;

      case fpu_xmm0:
        memcpy(m_state.context.fpu.no_avx.__fpu_xmm0.__xmm_reg,
               &value->value.uint8, 16);
        success = true;
        break;
      case fpu_xmm1:
        memcpy(m_state.context.fpu.no_avx.__fpu_xmm1.__xmm_reg,
               &value->value.uint8, 16);
        success = true;
        break;
      case fpu_xmm2:
        memcpy(m_state.context.fpu.no_avx.__fpu_xmm2.__xmm_reg,
               &value->value.uint8, 16);
        success = true;
        break;
      case fpu_xmm3:
        memcpy(m_state.context.fpu.no_avx.__fpu_xmm3.__xmm_reg,
               &value->value.uint8, 16);
        success = true;
        break;
      case fpu_xmm4:
        memcpy(m_state.context.fpu.no_avx.__fpu_xmm4.__xmm_reg,
               &value->value.uint8, 16);
        success = true;
        break;
      case fpu_xmm5:
        memcpy(m_state.context.fpu.no_avx.__fpu_xmm5.__xmm_reg,
               &value->value.uint8, 16);
        success = true;
        break;
      case fpu_xmm6:
        memcpy(m_state.context.fpu.no_avx.__fpu_xmm6.__xmm_reg,
               &value->value.uint8, 16);
        success = true;
        break;
      case fpu_xmm7:
        memcpy(m_state.context.fpu.no_avx.__fpu_xmm7.__xmm_reg,
               &value->value.uint8, 16);
        success = true;
        break;

#define MEMCPY_YMM(n)                                                          \
  memcpy(m_state.context.fpu.avx.__fpu_xmm##n.__xmm_reg, &value->value.uint8,  \
         16);                                                                  \
  memcpy(m_state.context.fpu.avx.__fpu_ymmh##n.__xmm_reg,                      \
         (&value->value.uint8) + 16, 16);
      case fpu_ymm0:
        MEMCPY_YMM(0);
        return true;
      case fpu_ymm1:
        MEMCPY_YMM(1);
        return true;
      case fpu_ymm2:
        MEMCPY_YMM(2);
        return true;
      case fpu_ymm3:
        MEMCPY_YMM(3);
        return true;
      case fpu_ymm4:
        MEMCPY_YMM(4);
        return true;
      case fpu_ymm5:
        MEMCPY_YMM(5);
        return true;
      case fpu_ymm6:
        MEMCPY_YMM(6);
        return true;
      case fpu_ymm7:
        MEMCPY_YMM(7);
        return true;
#undef MEMCPY_YMM
      }
      break;

    case e_regSetEXC:
      if (reg < k_num_exc_registers) {
        (&m_state.context.exc.__trapno)[reg] = value->value.uint32;
        success = true;
      }
      break;
    }
  }

  if (success)
    return SetRegisterState(set) == KERN_SUCCESS;
  return false;
}

uint32_t DNBArchImplI386::GetRegisterContextSize() {
  static uint32_t g_cached_size = 0;
  if (g_cached_size == 0) {
    if (CPUHasAVX() || FORCE_AVX_REGS) {
      for (size_t i = 0; i < k_num_fpu_registers_avx; ++i) {
        if (g_fpu_registers_avx[i].value_regs == NULL)
          g_cached_size += g_fpu_registers_avx[i].size;
      }
    } else {
      for (size_t i = 0; i < k_num_fpu_registers_no_avx; ++i) {
        if (g_fpu_registers_no_avx[i].value_regs == NULL)
          g_cached_size += g_fpu_registers_no_avx[i].size;
      }
    }
    DNBLogThreaded("DNBArchImplX86_64::GetRegisterContextSize() - GPR = %zu, "
                   "FPU = %u, EXC = %zu",
                   sizeof(GPR), g_cached_size, sizeof(EXC));
    g_cached_size += sizeof(GPR);
    g_cached_size += sizeof(EXC);
    DNBLogThreaded(
        "DNBArchImplX86_64::GetRegisterContextSize() - GPR + FPU + EXC = %u",
        g_cached_size);
  }
  return g_cached_size;
}

nub_size_t DNBArchImplI386::GetRegisterContext(void *buf, nub_size_t buf_len) {
  uint32_t size = GetRegisterContextSize();

  if (buf && buf_len) {
    if (size > buf_len)
      size = static_cast<uint32_t>(buf_len);

    bool force = false;
    kern_return_t kret;
    if ((kret = GetGPRState(force)) != KERN_SUCCESS) {
      DNBLogThreadedIf(LOG_THREAD, "DNBArchImplI386::GetRegisterContext (buf = "
                                   "%p, len = %llu) error: GPR regs failed to "
                                   "read: %u ",
                       buf, (uint64_t)buf_len, kret);
      size = 0;
    } else if ((kret = GetFPUState(force)) != KERN_SUCCESS) {
      DNBLogThreadedIf(
          LOG_THREAD, "DNBArchImplI386::GetRegisterContext (buf = %p, len = "
                      "%llu) error: %s regs failed to read: %u",
          buf, (uint64_t)buf_len, CPUHasAVX() ? "AVX" : "FPU", kret);
      size = 0;
    } else if ((kret = GetEXCState(force)) != KERN_SUCCESS) {
      DNBLogThreadedIf(LOG_THREAD, "DNBArchImplI386::GetRegisterContext (buf = "
                                   "%p, len = %llu) error: EXC regs failed to "
                                   "read: %u",
                       buf, (uint64_t)buf_len, kret);
      size = 0;
    } else {
      uint8_t *p = (uint8_t *)buf;
      // Copy the GPR registers
      memcpy(p, &m_state.context.gpr, sizeof(GPR));
      p += sizeof(GPR);

      // Walk around the gaps in the FPU regs
      memcpy(p, &m_state.context.fpu.no_avx.__fpu_fcw, 5);
      p += 5;
      memcpy(p, &m_state.context.fpu.no_avx.__fpu_fop, 8);
      p += 8;
      memcpy(p, &m_state.context.fpu.no_avx.__fpu_dp, 6);
      p += 6;
      memcpy(p, &m_state.context.fpu.no_avx.__fpu_mxcsr, 8);
      p += 8;

      // Work around the padding between the stmm registers as they are 16
      // byte structs with 10 bytes of the value in each
      for (size_t i = 0; i < 8; ++i) {
        memcpy(p, &m_state.context.fpu.no_avx.__fpu_stmm0 + i, 10);
        p += 10;
      }

      if (CPUHasAVX() || FORCE_AVX_REGS) {
        // Interleave the XMM and YMMH registers to make the YMM registers
        for (size_t i = 0; i < 8; ++i) {
          memcpy(p, &m_state.context.fpu.avx.__fpu_xmm0 + i, 16);
          p += 16;
          memcpy(p, &m_state.context.fpu.avx.__fpu_ymmh0 + i, 16);
          p += 16;
        }
      } else {
        // Copy the XMM registers in a single block
        memcpy(p, &m_state.context.fpu.no_avx.__fpu_xmm0, 8 * 16);
        p += 8 * 16;
      }

      // Copy the exception registers
      memcpy(p, &m_state.context.exc, sizeof(EXC));
      p += sizeof(EXC);

      // make sure we end up with exactly what we think we should have
      size_t bytes_written = p - (uint8_t *)buf;
      UNUSED_IF_ASSERT_DISABLED(bytes_written);
      assert(bytes_written == size);
    }
  }
  DNBLogThreadedIf(
      LOG_THREAD,
      "DNBArchImplI386::GetRegisterContext (buf = %p, len = %llu) => %llu", buf,
      (uint64_t)buf_len, (uint64_t)size);
  // Return the size of the register context even if NULL was passed in
  return size;
}

nub_size_t DNBArchImplI386::SetRegisterContext(const void *buf,
                                               nub_size_t buf_len) {
  nub_size_t size = sizeof(m_state.context);
  if (buf == NULL || buf_len == 0)
    size = 0;

  if (size) {
    if (size > buf_len)
      size = buf_len;

    uint8_t *p = (uint8_t *)buf;
    // Copy the GPR registers
    memcpy(&m_state.context.gpr, p, sizeof(GPR));
    p += sizeof(GPR);

    // Copy fcw through mxcsrmask as there is no padding
    memcpy(&m_state.context.fpu.no_avx.__fpu_fcw, p, 5);
    p += 5;
    memcpy(&m_state.context.fpu.no_avx.__fpu_fop, p, 8);
    p += 8;
    memcpy(&m_state.context.fpu.no_avx.__fpu_dp, p, 6);
    p += 6;
    memcpy(&m_state.context.fpu.no_avx.__fpu_mxcsr, p, 8);
    p += 8;

    // Work around the padding between the stmm registers as they are 16
    // byte structs with 10 bytes of the value in each
    for (size_t i = 0; i < 8; ++i) {
      memcpy(&m_state.context.fpu.no_avx.__fpu_stmm0 + i, p, 10);
      p += 10;
    }

    if (CPUHasAVX() || FORCE_AVX_REGS) {
      // Interleave the XMM and YMMH registers to make the YMM registers
      for (size_t i = 0; i < 8; ++i) {
        memcpy(&m_state.context.fpu.avx.__fpu_xmm0 + i, p, 16);
        p += 16;
        memcpy(&m_state.context.fpu.avx.__fpu_ymmh0 + i, p, 16);
        p += 16;
      }
    } else {
      // Copy the XMM registers in a single block
      memcpy(&m_state.context.fpu.no_avx.__fpu_xmm0, p, 8 * 16);
      p += 8 * 16;
    }

    // Copy the exception registers
    memcpy(&m_state.context.exc, p, sizeof(EXC));
    p += sizeof(EXC);

    // make sure we end up with exactly what we think we should have
    size_t bytes_written = p - (uint8_t *)buf;
    UNUSED_IF_ASSERT_DISABLED(bytes_written);
    assert(bytes_written == size);
    kern_return_t kret;
    if ((kret = SetGPRState()) != KERN_SUCCESS)
      DNBLogThreadedIf(LOG_THREAD, "DNBArchImplI386::SetRegisterContext (buf = "
                                   "%p, len = %llu) error: GPR regs failed to "
                                   "write: %u",
                       buf, (uint64_t)buf_len, kret);
    if ((kret = SetFPUState()) != KERN_SUCCESS)
      DNBLogThreadedIf(
          LOG_THREAD, "DNBArchImplI386::SetRegisterContext (buf = %p, len = "
                      "%llu) error: %s regs failed to write: %u",
          buf, (uint64_t)buf_len, CPUHasAVX() ? "AVX" : "FPU", kret);
    if ((kret = SetEXCState()) != KERN_SUCCESS)
      DNBLogThreadedIf(LOG_THREAD, "DNBArchImplI386::SetRegisterContext (buf = "
                                   "%p, len = %llu) error: EXP regs failed to "
                                   "write: %u",
                       buf, (uint64_t)buf_len, kret);
  }
  DNBLogThreadedIf(
      LOG_THREAD,
      "DNBArchImplI386::SetRegisterContext (buf = %p, len = %llu) => %llu", buf,
      (uint64_t)buf_len, (uint64_t)size);
  return size;
}

uint32_t DNBArchImplI386::SaveRegisterState() {
  kern_return_t kret = ::thread_abort_safely(m_thread->MachPortNumber());
  DNBLogThreadedIf(
      LOG_THREAD, "thread = 0x%4.4x calling thread_abort_safely (tid) => %u "
                  "(SetGPRState() for stop_count = %u)",
      m_thread->MachPortNumber(), kret, m_thread->Process()->StopCount());

  bool force = true;

  if ((kret = GetGPRState(force)) != KERN_SUCCESS) {
    DNBLogThreadedIf(LOG_THREAD, "DNBArchImplI386::SaveRegisterState () error: "
                                 "GPR regs failed to read: %u ",
                     kret);
  } else if ((kret = GetFPUState(force)) != KERN_SUCCESS) {
    DNBLogThreadedIf(LOG_THREAD, "DNBArchImplI386::SaveRegisterState () error: "
                                 "%s regs failed to read: %u",
                     CPUHasAVX() ? "AVX" : "FPU", kret);
  } else {
    const uint32_t save_id = GetNextRegisterStateSaveID();
    m_saved_register_states[save_id] = m_state.context;
    return save_id;
  }
  return 0;
}
bool DNBArchImplI386::RestoreRegisterState(uint32_t save_id) {
  SaveRegisterStates::iterator pos = m_saved_register_states.find(save_id);
  if (pos != m_saved_register_states.end()) {
    m_state.context.gpr = pos->second.gpr;
    m_state.context.fpu = pos->second.fpu;
    m_state.context.exc = pos->second.exc;
    m_state.SetError(e_regSetGPR, Read, 0);
    m_state.SetError(e_regSetFPU, Read, 0);
    m_state.SetError(e_regSetEXC, Read, 0);
    kern_return_t kret;
    bool success = true;
    if ((kret = SetGPRState()) != KERN_SUCCESS) {
      DNBLogThreadedIf(LOG_THREAD, "DNBArchImplI386::RestoreRegisterState "
                                   "(save_id = %u) error: GPR regs failed to "
                                   "write: %u",
                       save_id, kret);
      success = false;
    } else if ((kret = SetFPUState()) != KERN_SUCCESS) {
      DNBLogThreadedIf(LOG_THREAD, "DNBArchImplI386::RestoreRegisterState "
                                   "(save_id = %u) error: %s regs failed to "
                                   "write: %u",
                       save_id, CPUHasAVX() ? "AVX" : "FPU", kret);
      success = false;
    }
    m_saved_register_states.erase(pos);
    return success;
  }
  return false;
}

kern_return_t DNBArchImplI386::GetRegisterState(int set, bool force) {
  switch (set) {
  case e_regSetALL:
    return GetGPRState(force) | GetFPUState(force) | GetEXCState(force);
  case e_regSetGPR:
    return GetGPRState(force);
  case e_regSetFPU:
    return GetFPUState(force);
  case e_regSetEXC:
    return GetEXCState(force);
  default:
    break;
  }
  return KERN_INVALID_ARGUMENT;
}

kern_return_t DNBArchImplI386::SetRegisterState(int set) {
  // Make sure we have a valid context to set.
  if (RegisterSetStateIsValid(set)) {
    switch (set) {
    case e_regSetALL:
      return SetGPRState() | SetFPUState() | SetEXCState();
    case e_regSetGPR:
      return SetGPRState();
    case e_regSetFPU:
      return SetFPUState();
    case e_regSetEXC:
      return SetEXCState();
    default:
      break;
    }
  }
  return KERN_INVALID_ARGUMENT;
}

bool DNBArchImplI386::RegisterSetStateIsValid(int set) const {
  return m_state.RegsAreValid(set);
}

#endif // #if defined (__i386__)
