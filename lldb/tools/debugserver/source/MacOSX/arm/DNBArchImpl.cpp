//===-- DNBArchImpl.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 6/25/07.
//
//===----------------------------------------------------------------------===//

#if defined(__arm__) || defined(__arm64__) || defined(__aarch64__)

#include "MacOSX/arm/DNBArchImpl.h"
#include "ARM_DWARF_Registers.h"
#include "ARM_ehframe_Registers.h"
#include "DNB.h"
#include "DNBBreakpoint.h"
#include "DNBLog.h"
#include "DNBRegisterInfo.h"
#include "MacOSX/MachProcess.h"
#include "MacOSX/MachThread.h"

#include <cinttypes>
#include <sys/sysctl.h>

// BCR address match type
#define BCR_M_IMVA_MATCH ((uint32_t)(0u << 21))
#define BCR_M_CONTEXT_ID_MATCH ((uint32_t)(1u << 21))
#define BCR_M_IMVA_MISMATCH ((uint32_t)(2u << 21))
#define BCR_M_RESERVED ((uint32_t)(3u << 21))

// Link a BVR/BCR or WVR/WCR pair to another
#define E_ENABLE_LINKING ((uint32_t)(1u << 20))

// Byte Address Select
#define BAS_IMVA_PLUS_0 ((uint32_t)(1u << 5))
#define BAS_IMVA_PLUS_1 ((uint32_t)(1u << 6))
#define BAS_IMVA_PLUS_2 ((uint32_t)(1u << 7))
#define BAS_IMVA_PLUS_3 ((uint32_t)(1u << 8))
#define BAS_IMVA_0_1 ((uint32_t)(3u << 5))
#define BAS_IMVA_2_3 ((uint32_t)(3u << 7))
#define BAS_IMVA_ALL ((uint32_t)(0xfu << 5))

// Break only in privileged or user mode
#define S_RSVD ((uint32_t)(0u << 1))
#define S_PRIV ((uint32_t)(1u << 1))
#define S_USER ((uint32_t)(2u << 1))
#define S_PRIV_USER ((S_PRIV) | (S_USER))

#define BCR_ENABLE ((uint32_t)(1u))
#define WCR_ENABLE ((uint32_t)(1u))

// Watchpoint load/store
#define WCR_LOAD ((uint32_t)(1u << 3))
#define WCR_STORE ((uint32_t)(1u << 4))

// Definitions for the Debug Status and Control Register fields:
// [5:2] => Method of debug entry
//#define WATCHPOINT_OCCURRED     ((uint32_t)(2u))
// I'm seeing this, instead.
#define WATCHPOINT_OCCURRED ((uint32_t)(10u))

// 0xE120BE70
static const uint8_t g_arm_breakpoint_opcode[] = {0x70, 0xBE, 0x20, 0xE1};
static const uint8_t g_thumb_breakpoint_opcode[] = {0x70, 0xBE};

// A watchpoint may need to be implemented using two watchpoint registers.
// e.g. watching an 8-byte region when the device can only watch 4-bytes.
//
// This stores the lo->hi mappings.  It's safe to initialize to all 0's
// since hi > lo and therefore LoHi[i] cannot be 0.
static uint32_t LoHi[16] = {0};

// ARM constants used during decoding
#define REG_RD 0
#define LDM_REGLIST 1
#define PC_REG 15
#define PC_REGLIST_BIT 0x8000

// ARM conditions
#define COND_EQ 0x0
#define COND_NE 0x1
#define COND_CS 0x2
#define COND_HS 0x2
#define COND_CC 0x3
#define COND_LO 0x3
#define COND_MI 0x4
#define COND_PL 0x5
#define COND_VS 0x6
#define COND_VC 0x7
#define COND_HI 0x8
#define COND_LS 0x9
#define COND_GE 0xA
#define COND_LT 0xB
#define COND_GT 0xC
#define COND_LE 0xD
#define COND_AL 0xE
#define COND_UNCOND 0xF

#define MASK_CPSR_T (1u << 5)
#define MASK_CPSR_J (1u << 24)

#define MNEMONIC_STRING_SIZE 32
#define OPERAND_STRING_SIZE 128

#if !defined(__arm64__) && !defined(__aarch64__)
// Returns true if the first 16 bit opcode of a thumb instruction indicates
// the instruction will be a 32 bit thumb opcode
static bool IsThumb32Opcode(uint16_t opcode) {
  if (((opcode & 0xE000) == 0xE000) && (opcode & 0x1800))
    return true;
  return false;
}
#endif

void DNBArchMachARM::Initialize() {
  DNBArchPluginInfo arch_plugin_info = {
      CPU_TYPE_ARM, DNBArchMachARM::Create, DNBArchMachARM::GetRegisterSetInfo,
      DNBArchMachARM::SoftwareBreakpointOpcode};

  // Register this arch plug-in with the main protocol class
  DNBArchProtocol::RegisterArchPlugin(arch_plugin_info);
}

DNBArchProtocol *DNBArchMachARM::Create(MachThread *thread) {
  DNBArchMachARM *obj = new DNBArchMachARM(thread);
  return obj;
}

const uint8_t *DNBArchMachARM::SoftwareBreakpointOpcode(nub_size_t byte_size) {
  switch (byte_size) {
  case 2:
    return g_thumb_breakpoint_opcode;
  case 4:
    return g_arm_breakpoint_opcode;
  }
  return NULL;
}

uint32_t DNBArchMachARM::GetCPUType() { return CPU_TYPE_ARM; }

uint64_t DNBArchMachARM::GetPC(uint64_t failValue) {
  // Get program counter
  if (GetGPRState(false) == KERN_SUCCESS)
    return m_state.context.gpr.__pc;
  return failValue;
}

kern_return_t DNBArchMachARM::SetPC(uint64_t value) {
  // Get program counter
  kern_return_t err = GetGPRState(false);
  if (err == KERN_SUCCESS) {
    m_state.context.gpr.__pc = (uint32_t)value;
    err = SetGPRState();
  }
  return err == KERN_SUCCESS;
}

uint64_t DNBArchMachARM::GetSP(uint64_t failValue) {
  // Get stack pointer
  if (GetGPRState(false) == KERN_SUCCESS)
    return m_state.context.gpr.__sp;
  return failValue;
}

kern_return_t DNBArchMachARM::GetGPRState(bool force) {
  int set = e_regSetGPR;
  // Check if we have valid cached registers
  if (!force && m_state.GetError(set, Read) == KERN_SUCCESS)
    return KERN_SUCCESS;

  // Read the registers from our thread
  mach_msg_type_number_t count = ARM_THREAD_STATE_COUNT;
  kern_return_t kret =
      ::thread_get_state(m_thread->MachPortNumber(), ARM_THREAD_STATE,
                         (thread_state_t)&m_state.context.gpr, &count);
  uint32_t *r = &m_state.context.gpr.__r[0];
  DNBLogThreadedIf(
      LOG_THREAD, "thread_get_state(0x%4.4x, %u, &gpr, %u) => 0x%8.8x (count = "
                  "%u) regs r0=%8.8x r1=%8.8x r2=%8.8x r3=%8.8x r4=%8.8x "
                  "r5=%8.8x r6=%8.8x r7=%8.8x r8=%8.8x r9=%8.8x r10=%8.8x "
                  "r11=%8.8x s12=%8.8x sp=%8.8x lr=%8.8x pc=%8.8x cpsr=%8.8x",
      m_thread->MachPortNumber(), ARM_THREAD_STATE, ARM_THREAD_STATE_COUNT,
      kret, count, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9],
      r[10], r[11], r[12], r[13], r[14], r[15], r[16]);
  m_state.SetError(set, Read, kret);
  return kret;
}

kern_return_t DNBArchMachARM::GetVFPState(bool force) {
  int set = e_regSetVFP;
  // Check if we have valid cached registers
  if (!force && m_state.GetError(set, Read) == KERN_SUCCESS)
    return KERN_SUCCESS;

  kern_return_t kret;

#if defined(__arm64__) || defined(__aarch64__)
  // Read the registers from our thread
  mach_msg_type_number_t count = ARM_NEON_STATE_COUNT;
  kret = ::thread_get_state(m_thread->MachPortNumber(), ARM_NEON_STATE,
                            (thread_state_t)&m_state.context.vfp, &count);
  if (DNBLogEnabledForAny(LOG_THREAD)) {
    DNBLogThreaded(
        "thread_get_state(0x%4.4x, %u, &vfp, %u) => 0x%8.8x (count = %u) regs"
        "\n   q0  = 0x%16.16llx%16.16llx"
        "\n   q1  = 0x%16.16llx%16.16llx"
        "\n   q2  = 0x%16.16llx%16.16llx"
        "\n   q3  = 0x%16.16llx%16.16llx"
        "\n   q4  = 0x%16.16llx%16.16llx"
        "\n   q5  = 0x%16.16llx%16.16llx"
        "\n   q6  = 0x%16.16llx%16.16llx"
        "\n   q7  = 0x%16.16llx%16.16llx"
        "\n   q8  = 0x%16.16llx%16.16llx"
        "\n   q9  = 0x%16.16llx%16.16llx"
        "\n   q10 = 0x%16.16llx%16.16llx"
        "\n   q11 = 0x%16.16llx%16.16llx"
        "\n   q12 = 0x%16.16llx%16.16llx"
        "\n   q13 = 0x%16.16llx%16.16llx"
        "\n   q14 = 0x%16.16llx%16.16llx"
        "\n   q15 = 0x%16.16llx%16.16llx"
        "\n  fpsr = 0x%8.8x"
        "\n  fpcr = 0x%8.8x\n\n",
        m_thread->MachPortNumber(), ARM_NEON_STATE, ARM_NEON_STATE_COUNT, kret,
        count, ((uint64_t *)&m_state.context.vfp.__v[0])[0],
        ((uint64_t *)&m_state.context.vfp.__v[0])[1],
        ((uint64_t *)&m_state.context.vfp.__v[1])[0],
        ((uint64_t *)&m_state.context.vfp.__v[1])[1],
        ((uint64_t *)&m_state.context.vfp.__v[2])[0],
        ((uint64_t *)&m_state.context.vfp.__v[2])[1],
        ((uint64_t *)&m_state.context.vfp.__v[3])[0],
        ((uint64_t *)&m_state.context.vfp.__v[3])[1],
        ((uint64_t *)&m_state.context.vfp.__v[4])[0],
        ((uint64_t *)&m_state.context.vfp.__v[4])[1],
        ((uint64_t *)&m_state.context.vfp.__v[5])[0],
        ((uint64_t *)&m_state.context.vfp.__v[5])[1],
        ((uint64_t *)&m_state.context.vfp.__v[6])[0],
        ((uint64_t *)&m_state.context.vfp.__v[6])[1],
        ((uint64_t *)&m_state.context.vfp.__v[7])[0],
        ((uint64_t *)&m_state.context.vfp.__v[7])[1],
        ((uint64_t *)&m_state.context.vfp.__v[8])[0],
        ((uint64_t *)&m_state.context.vfp.__v[8])[1],
        ((uint64_t *)&m_state.context.vfp.__v[9])[0],
        ((uint64_t *)&m_state.context.vfp.__v[9])[1],
        ((uint64_t *)&m_state.context.vfp.__v[10])[0],
        ((uint64_t *)&m_state.context.vfp.__v[10])[1],
        ((uint64_t *)&m_state.context.vfp.__v[11])[0],
        ((uint64_t *)&m_state.context.vfp.__v[11])[1],
        ((uint64_t *)&m_state.context.vfp.__v[12])[0],
        ((uint64_t *)&m_state.context.vfp.__v[12])[1],
        ((uint64_t *)&m_state.context.vfp.__v[13])[0],
        ((uint64_t *)&m_state.context.vfp.__v[13])[1],
        ((uint64_t *)&m_state.context.vfp.__v[14])[0],
        ((uint64_t *)&m_state.context.vfp.__v[14])[1],
        ((uint64_t *)&m_state.context.vfp.__v[15])[0],
        ((uint64_t *)&m_state.context.vfp.__v[15])[1],
        m_state.context.vfp.__fpsr, m_state.context.vfp.__fpcr);
  }
#else
  // Read the registers from our thread
  mach_msg_type_number_t count = ARM_VFP_STATE_COUNT;
  kret = ::thread_get_state(m_thread->MachPortNumber(), ARM_VFP_STATE,
                            (thread_state_t)&m_state.context.vfp, &count);

  if (DNBLogEnabledForAny(LOG_THREAD)) {
    uint32_t *r = &m_state.context.vfp.__r[0];
    DNBLogThreaded(
        "thread_get_state(0x%4.4x, %u, &gpr, %u) => 0x%8.8x (count => %u)",
        m_thread->MachPortNumber(), ARM_THREAD_STATE, ARM_THREAD_STATE_COUNT,
        kret, count);
    DNBLogThreaded("   s0=%8.8x  s1=%8.8x  s2=%8.8x  s3=%8.8x  s4=%8.8x  "
                   "s5=%8.8x  s6=%8.8x  s7=%8.8x",
                   r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
    DNBLogThreaded("   s8=%8.8x  s9=%8.8x s10=%8.8x s11=%8.8x s12=%8.8x "
                   "s13=%8.8x s14=%8.8x s15=%8.8x",
                   r[8], r[9], r[10], r[11], r[12], r[13], r[14], r[15]);
    DNBLogThreaded("  s16=%8.8x s17=%8.8x s18=%8.8x s19=%8.8x s20=%8.8x "
                   "s21=%8.8x s22=%8.8x s23=%8.8x",
                   r[16], r[17], r[18], r[19], r[20], r[21], r[22], r[23]);
    DNBLogThreaded("  s24=%8.8x s25=%8.8x s26=%8.8x s27=%8.8x s28=%8.8x "
                   "s29=%8.8x s30=%8.8x s31=%8.8x",
                   r[24], r[25], r[26], r[27], r[28], r[29], r[30], r[31]);
    DNBLogThreaded("  s32=%8.8x s33=%8.8x s34=%8.8x s35=%8.8x s36=%8.8x "
                   "s37=%8.8x s38=%8.8x s39=%8.8x",
                   r[32], r[33], r[34], r[35], r[36], r[37], r[38], r[39]);
    DNBLogThreaded("  s40=%8.8x s41=%8.8x s42=%8.8x s43=%8.8x s44=%8.8x "
                   "s45=%8.8x s46=%8.8x s47=%8.8x",
                   r[40], r[41], r[42], r[43], r[44], r[45], r[46], r[47]);
    DNBLogThreaded("  s48=%8.8x s49=%8.8x s50=%8.8x s51=%8.8x s52=%8.8x "
                   "s53=%8.8x s54=%8.8x s55=%8.8x",
                   r[48], r[49], r[50], r[51], r[52], r[53], r[54], r[55]);
    DNBLogThreaded("  s56=%8.8x s57=%8.8x s58=%8.8x s59=%8.8x s60=%8.8x "
                   "s61=%8.8x s62=%8.8x s63=%8.8x fpscr=%8.8x",
                   r[56], r[57], r[58], r[59], r[60], r[61], r[62], r[63],
                   r[64]);
  }

#endif
  m_state.SetError(set, Read, kret);
  return kret;
}

kern_return_t DNBArchMachARM::GetEXCState(bool force) {
  int set = e_regSetEXC;
  // Check if we have valid cached registers
  if (!force && m_state.GetError(set, Read) == KERN_SUCCESS)
    return KERN_SUCCESS;

  // Read the registers from our thread
  mach_msg_type_number_t count = ARM_EXCEPTION_STATE_COUNT;
  kern_return_t kret =
      ::thread_get_state(m_thread->MachPortNumber(), ARM_EXCEPTION_STATE,
                         (thread_state_t)&m_state.context.exc, &count);
  m_state.SetError(set, Read, kret);
  return kret;
}

#if 0
static void DumpDBGState(const DNBArchMachARM::DBG &dbg) {
  uint32_t i = 0;
  for (i = 0; i < 16; i++) {
    DNBLogThreadedIf(LOG_STEP, "BVR%-2u/BCR%-2u = { 0x%8.8x, 0x%8.8x } "
                               "WVR%-2u/WCR%-2u = { 0x%8.8x, 0x%8.8x }",
                     i, i, dbg.__bvr[i], dbg.__bcr[i], i, i, dbg.__wvr[i],
                     dbg.__wcr[i]);
  }
}
#endif

kern_return_t DNBArchMachARM::GetDBGState(bool force) {
  int set = e_regSetDBG;

  // Check if we have valid cached registers
  if (!force && m_state.GetError(set, Read) == KERN_SUCCESS)
    return KERN_SUCCESS;

// Read the registers from our thread
#if defined(ARM_DEBUG_STATE32) && (defined(__arm64__) || defined(__aarch64__))
  mach_msg_type_number_t count = ARM_DEBUG_STATE32_COUNT;
  kern_return_t kret =
      ::thread_get_state(m_thread->MachPortNumber(), ARM_DEBUG_STATE32,
                         (thread_state_t)&m_state.dbg, &count);
#else
  mach_msg_type_number_t count = ARM_DEBUG_STATE_COUNT;
  kern_return_t kret =
      ::thread_get_state(m_thread->MachPortNumber(), ARM_DEBUG_STATE,
                         (thread_state_t)&m_state.dbg, &count);
#endif
  m_state.SetError(set, Read, kret);

  return kret;
}

kern_return_t DNBArchMachARM::SetGPRState() {
  int set = e_regSetGPR;
  kern_return_t kret = ::thread_set_state(
      m_thread->MachPortNumber(), ARM_THREAD_STATE,
      (thread_state_t)&m_state.context.gpr, ARM_THREAD_STATE_COUNT);
  m_state.SetError(set, Write,
                   kret); // Set the current write error for this register set
  m_state.InvalidateRegisterSetState(set); // Invalidate the current register
                                           // state in case registers are read
                                           // back differently
  return kret;                             // Return the error code
}

kern_return_t DNBArchMachARM::SetVFPState() {
  int set = e_regSetVFP;
  kern_return_t kret;
  mach_msg_type_number_t count;

#if defined(__arm64__) || defined(__aarch64__)
  count = ARM_NEON_STATE_COUNT;
  kret = ::thread_set_state(m_thread->MachPortNumber(), ARM_NEON_STATE,
                            (thread_state_t)&m_state.context.vfp, count);
#else
  count = ARM_VFP_STATE_COUNT;
  kret = ::thread_set_state(m_thread->MachPortNumber(), ARM_VFP_STATE,
                            (thread_state_t)&m_state.context.vfp, count);
#endif

#if defined(__arm64__) || defined(__aarch64__)
  if (DNBLogEnabledForAny(LOG_THREAD)) {
    DNBLogThreaded(
        "thread_set_state(0x%4.4x, %u, &vfp, %u) => 0x%8.8x (count = %u) regs"
        "\n   q0  = 0x%16.16llx%16.16llx"
        "\n   q1  = 0x%16.16llx%16.16llx"
        "\n   q2  = 0x%16.16llx%16.16llx"
        "\n   q3  = 0x%16.16llx%16.16llx"
        "\n   q4  = 0x%16.16llx%16.16llx"
        "\n   q5  = 0x%16.16llx%16.16llx"
        "\n   q6  = 0x%16.16llx%16.16llx"
        "\n   q7  = 0x%16.16llx%16.16llx"
        "\n   q8  = 0x%16.16llx%16.16llx"
        "\n   q9  = 0x%16.16llx%16.16llx"
        "\n   q10 = 0x%16.16llx%16.16llx"
        "\n   q11 = 0x%16.16llx%16.16llx"
        "\n   q12 = 0x%16.16llx%16.16llx"
        "\n   q13 = 0x%16.16llx%16.16llx"
        "\n   q14 = 0x%16.16llx%16.16llx"
        "\n   q15 = 0x%16.16llx%16.16llx"
        "\n  fpsr = 0x%8.8x"
        "\n  fpcr = 0x%8.8x\n\n",
        m_thread->MachPortNumber(), ARM_NEON_STATE, ARM_NEON_STATE_COUNT, kret,
        count, ((uint64_t *)&m_state.context.vfp.__v[0])[0],
        ((uint64_t *)&m_state.context.vfp.__v[0])[1],
        ((uint64_t *)&m_state.context.vfp.__v[1])[0],
        ((uint64_t *)&m_state.context.vfp.__v[1])[1],
        ((uint64_t *)&m_state.context.vfp.__v[2])[0],
        ((uint64_t *)&m_state.context.vfp.__v[2])[1],
        ((uint64_t *)&m_state.context.vfp.__v[3])[0],
        ((uint64_t *)&m_state.context.vfp.__v[3])[1],
        ((uint64_t *)&m_state.context.vfp.__v[4])[0],
        ((uint64_t *)&m_state.context.vfp.__v[4])[1],
        ((uint64_t *)&m_state.context.vfp.__v[5])[0],
        ((uint64_t *)&m_state.context.vfp.__v[5])[1],
        ((uint64_t *)&m_state.context.vfp.__v[6])[0],
        ((uint64_t *)&m_state.context.vfp.__v[6])[1],
        ((uint64_t *)&m_state.context.vfp.__v[7])[0],
        ((uint64_t *)&m_state.context.vfp.__v[7])[1],
        ((uint64_t *)&m_state.context.vfp.__v[8])[0],
        ((uint64_t *)&m_state.context.vfp.__v[8])[1],
        ((uint64_t *)&m_state.context.vfp.__v[9])[0],
        ((uint64_t *)&m_state.context.vfp.__v[9])[1],
        ((uint64_t *)&m_state.context.vfp.__v[10])[0],
        ((uint64_t *)&m_state.context.vfp.__v[10])[1],
        ((uint64_t *)&m_state.context.vfp.__v[11])[0],
        ((uint64_t *)&m_state.context.vfp.__v[11])[1],
        ((uint64_t *)&m_state.context.vfp.__v[12])[0],
        ((uint64_t *)&m_state.context.vfp.__v[12])[1],
        ((uint64_t *)&m_state.context.vfp.__v[13])[0],
        ((uint64_t *)&m_state.context.vfp.__v[13])[1],
        ((uint64_t *)&m_state.context.vfp.__v[14])[0],
        ((uint64_t *)&m_state.context.vfp.__v[14])[1],
        ((uint64_t *)&m_state.context.vfp.__v[15])[0],
        ((uint64_t *)&m_state.context.vfp.__v[15])[1],
        m_state.context.vfp.__fpsr, m_state.context.vfp.__fpcr);
  }
#else
  if (DNBLogEnabledForAny(LOG_THREAD)) {
    uint32_t *r = &m_state.context.vfp.__r[0];
    DNBLogThreaded(
        "thread_get_state(0x%4.4x, %u, &gpr, %u) => 0x%8.8x (count => %u)",
        m_thread->MachPortNumber(), ARM_THREAD_STATE, ARM_THREAD_STATE_COUNT,
        kret, count);
    DNBLogThreaded("   s0=%8.8x  s1=%8.8x  s2=%8.8x  s3=%8.8x  s4=%8.8x  "
                   "s5=%8.8x  s6=%8.8x  s7=%8.8x",
                   r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
    DNBLogThreaded("   s8=%8.8x  s9=%8.8x s10=%8.8x s11=%8.8x s12=%8.8x "
                   "s13=%8.8x s14=%8.8x s15=%8.8x",
                   r[8], r[9], r[10], r[11], r[12], r[13], r[14], r[15]);
    DNBLogThreaded("  s16=%8.8x s17=%8.8x s18=%8.8x s19=%8.8x s20=%8.8x "
                   "s21=%8.8x s22=%8.8x s23=%8.8x",
                   r[16], r[17], r[18], r[19], r[20], r[21], r[22], r[23]);
    DNBLogThreaded("  s24=%8.8x s25=%8.8x s26=%8.8x s27=%8.8x s28=%8.8x "
                   "s29=%8.8x s30=%8.8x s31=%8.8x",
                   r[24], r[25], r[26], r[27], r[28], r[29], r[30], r[31]);
    DNBLogThreaded("  s32=%8.8x s33=%8.8x s34=%8.8x s35=%8.8x s36=%8.8x "
                   "s37=%8.8x s38=%8.8x s39=%8.8x",
                   r[32], r[33], r[34], r[35], r[36], r[37], r[38], r[39]);
    DNBLogThreaded("  s40=%8.8x s41=%8.8x s42=%8.8x s43=%8.8x s44=%8.8x "
                   "s45=%8.8x s46=%8.8x s47=%8.8x",
                   r[40], r[41], r[42], r[43], r[44], r[45], r[46], r[47]);
    DNBLogThreaded("  s48=%8.8x s49=%8.8x s50=%8.8x s51=%8.8x s52=%8.8x "
                   "s53=%8.8x s54=%8.8x s55=%8.8x",
                   r[48], r[49], r[50], r[51], r[52], r[53], r[54], r[55]);
    DNBLogThreaded("  s56=%8.8x s57=%8.8x s58=%8.8x s59=%8.8x s60=%8.8x "
                   "s61=%8.8x s62=%8.8x s63=%8.8x fpscr=%8.8x",
                   r[56], r[57], r[58], r[59], r[60], r[61], r[62], r[63],
                   r[64]);
  }
#endif

  m_state.SetError(set, Write,
                   kret); // Set the current write error for this register set
  m_state.InvalidateRegisterSetState(set); // Invalidate the current register
                                           // state in case registers are read
                                           // back differently
  return kret;                             // Return the error code
}

kern_return_t DNBArchMachARM::SetEXCState() {
  int set = e_regSetEXC;
  kern_return_t kret = ::thread_set_state(
      m_thread->MachPortNumber(), ARM_EXCEPTION_STATE,
      (thread_state_t)&m_state.context.exc, ARM_EXCEPTION_STATE_COUNT);
  m_state.SetError(set, Write,
                   kret); // Set the current write error for this register set
  m_state.InvalidateRegisterSetState(set); // Invalidate the current register
                                           // state in case registers are read
                                           // back differently
  return kret;                             // Return the error code
}

kern_return_t DNBArchMachARM::SetDBGState(bool also_set_on_task) {
  int set = e_regSetDBG;
#if defined(ARM_DEBUG_STATE32) && (defined(__arm64__) || defined(__aarch64__))
  kern_return_t kret =
      ::thread_set_state(m_thread->MachPortNumber(), ARM_DEBUG_STATE32,
                         (thread_state_t)&m_state.dbg, ARM_DEBUG_STATE32_COUNT);
  if (also_set_on_task) {
    kern_return_t task_kret = ::task_set_state(
        m_thread->Process()->Task().TaskPort(), ARM_DEBUG_STATE32,
        (thread_state_t)&m_state.dbg, ARM_DEBUG_STATE32_COUNT);
    if (task_kret != KERN_SUCCESS)
      DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchMachARM::SetDBGState failed to "
                                        "set debug control register state: "
                                        "0x%8.8x.",
                       kret);
  }
#else
  kern_return_t kret =
      ::thread_set_state(m_thread->MachPortNumber(), ARM_DEBUG_STATE,
                         (thread_state_t)&m_state.dbg, ARM_DEBUG_STATE_COUNT);
  if (also_set_on_task) {
    kern_return_t task_kret = ::task_set_state(
        m_thread->Process()->Task().TaskPort(), ARM_DEBUG_STATE,
        (thread_state_t)&m_state.dbg, ARM_DEBUG_STATE_COUNT);
    if (task_kret != KERN_SUCCESS)
      DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchMachARM::SetDBGState failed to "
                                        "set debug control register state: "
                                        "0x%8.8x.",
                       kret);
  }
#endif

  m_state.SetError(set, Write,
                   kret); // Set the current write error for this register set
  m_state.InvalidateRegisterSetState(set); // Invalidate the current register
                                           // state in case registers are read
                                           // back differently
  return kret;                             // Return the error code
}

void DNBArchMachARM::ThreadWillResume() {
  // Do we need to step this thread? If so, let the mach thread tell us so.
  if (m_thread->IsStepping()) {
    // This is the primary thread, let the arch do anything it needs
    if (NumSupportedHardwareBreakpoints() > 0) {
      if (EnableHardwareSingleStep(true) != KERN_SUCCESS) {
        DNBLogThreaded("DNBArchMachARM::ThreadWillResume() failed to enable "
                       "hardware single step");
      }
    }
  }

  // Disable the triggered watchpoint temporarily before we resume.
  // Plus, we try to enable hardware single step to execute past the instruction
  // which triggered our watchpoint.
  if (m_watchpoint_did_occur) {
    if (m_watchpoint_hw_index >= 0) {
      kern_return_t kret = GetDBGState(false);
      if (kret == KERN_SUCCESS &&
          !IsWatchpointEnabled(m_state.dbg, m_watchpoint_hw_index)) {
        // The watchpoint might have been disabled by the user.  We don't need
        // to do anything at all
        // to enable hardware single stepping.
        m_watchpoint_did_occur = false;
        m_watchpoint_hw_index = -1;
        return;
      }

      DisableHardwareWatchpoint(m_watchpoint_hw_index, false);
      DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchMachARM::ThreadWillResume() "
                                        "DisableHardwareWatchpoint(%d) called",
                       m_watchpoint_hw_index);

      // Enable hardware single step to move past the watchpoint-triggering
      // instruction.
      m_watchpoint_resume_single_step_enabled =
          (EnableHardwareSingleStep(true) == KERN_SUCCESS);

      // If we are not able to enable single step to move past the
      // watchpoint-triggering instruction,
      // at least we should reset the two watchpoint member variables so that
      // the next time around
      // this callback function is invoked, the enclosing logical branch is
      // skipped.
      if (!m_watchpoint_resume_single_step_enabled) {
        // Reset the two watchpoint member variables.
        m_watchpoint_did_occur = false;
        m_watchpoint_hw_index = -1;
        DNBLogThreadedIf(
            LOG_WATCHPOINTS,
            "DNBArchMachARM::ThreadWillResume() failed to enable single step");
      } else
        DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchMachARM::ThreadWillResume() "
                                          "succeeded to enable single step");
    }
  }
}

bool DNBArchMachARM::ThreadDidStop() {
  bool success = true;

  m_state.InvalidateRegisterSetState(e_regSetALL);

  if (m_watchpoint_resume_single_step_enabled) {
    // Great!  We now disable the hardware single step as well as re-enable the
    // hardware watchpoint.
    // See also ThreadWillResume().
    if (EnableHardwareSingleStep(false) == KERN_SUCCESS) {
      if (m_watchpoint_did_occur && m_watchpoint_hw_index >= 0) {
        ReenableHardwareWatchpoint(m_watchpoint_hw_index);
        m_watchpoint_resume_single_step_enabled = false;
        m_watchpoint_did_occur = false;
        m_watchpoint_hw_index = -1;
      } else {
        DNBLogError("internal error detected: m_watchpoint_resume_step_enabled "
                    "is true but (m_watchpoint_did_occur && "
                    "m_watchpoint_hw_index >= 0) does not hold!");
      }
    } else {
      DNBLogError("internal error detected: m_watchpoint_resume_step_enabled "
                  "is true but unable to disable single step!");
    }
  }

  // Are we stepping a single instruction?
  if (GetGPRState(true) == KERN_SUCCESS) {
    // We are single stepping, was this the primary thread?
    if (m_thread->IsStepping()) {
      success = EnableHardwareSingleStep(false) == KERN_SUCCESS;
    } else {
      // The MachThread will automatically restore the suspend count
      // in ThreadDidStop(), so we don't need to do anything here if
      // we weren't the primary thread the last time
    }
  }
  return success;
}

bool DNBArchMachARM::NotifyException(MachException::Data &exc) {
  switch (exc.exc_type) {
  default:
    break;
  case EXC_BREAKPOINT:
    if (exc.exc_data.size() == 2 && exc.exc_data[0] == EXC_ARM_DA_DEBUG) {
      // The data break address is passed as exc_data[1].
      nub_addr_t addr = exc.exc_data[1];
      // Find the hardware index with the side effect of possibly massaging the
      // addr to return the starting address as seen from the debugger side.
      uint32_t hw_index = GetHardwareWatchpointHit(addr);
      DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchMachARM::NotifyException "
                                        "watchpoint %d was hit on address "
                                        "0x%llx",
                       hw_index, (uint64_t)addr);
      const uint32_t num_watchpoints = NumSupportedHardwareWatchpoints();
      for (uint32_t i = 0; i < num_watchpoints; i++) {
        if (LoHi[i] != 0 && LoHi[i] == hw_index && LoHi[i] != i &&
            GetWatchpointAddressByIndex(i) != INVALID_NUB_ADDRESS) {
          addr = GetWatchpointAddressByIndex(i);
          DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchMachARM::NotifyException "
                                            "It is a linked watchpoint; "
                                            "rewritten to index %d addr 0x%llx",
                           LoHi[i], (uint64_t)addr);
        }
      }
      if (hw_index != INVALID_NUB_HW_INDEX) {
        m_watchpoint_did_occur = true;
        m_watchpoint_hw_index = hw_index;
        exc.exc_data[1] = addr;
        // Piggyback the hw_index in the exc.data.
        exc.exc_data.push_back(hw_index);
      }

      return true;
    }
    break;
  }
  return false;
}

bool DNBArchMachARM::StepNotComplete() {
  if (m_hw_single_chained_step_addr != INVALID_NUB_ADDRESS) {
    kern_return_t kret = KERN_INVALID_ARGUMENT;
    kret = GetGPRState(false);
    if (kret == KERN_SUCCESS) {
      if (m_state.context.gpr.__pc == m_hw_single_chained_step_addr) {
        DNBLogThreadedIf(LOG_STEP, "Need to step some more at 0x%8.8llx",
                         (uint64_t)m_hw_single_chained_step_addr);
        return true;
      }
    }
  }

  m_hw_single_chained_step_addr = INVALID_NUB_ADDRESS;
  return false;
}

// Set the single step bit in the processor status register.
kern_return_t DNBArchMachARM::EnableHardwareSingleStep(bool enable) {
  DNBError err;
  DNBLogThreadedIf(LOG_STEP, "%s( enable = %d )", __FUNCTION__, enable);

  err = GetGPRState(false);

  if (err.Fail()) {
    err.LogThreaded("%s: failed to read the GPR registers", __FUNCTION__);
    return err.Status();
  }

  err = GetDBGState(false);

  if (err.Fail()) {
    err.LogThreaded("%s: failed to read the DBG registers", __FUNCTION__);
    return err.Status();
  }

// The use of __arm64__ here is not ideal.  If debugserver is running on
// an armv8 device, regardless of whether it was built for arch arm or arch
// arm64,
// it needs to use the MDSCR_EL1 SS bit to single instruction step.

#if defined(__arm64__) || defined(__aarch64__)
  if (enable) {
    DNBLogThreadedIf(LOG_STEP,
                     "%s: Setting MDSCR_EL1 Single Step bit at pc 0x%llx",
                     __FUNCTION__, (uint64_t)m_state.context.gpr.__pc);
    m_state.dbg.__mdscr_el1 |=
        1; // Set bit 0 (single step, SS) in the MDSCR_EL1.
  } else {
    DNBLogThreadedIf(LOG_STEP,
                     "%s: Clearing MDSCR_EL1 Single Step bit at pc 0x%llx",
                     __FUNCTION__, (uint64_t)m_state.context.gpr.__pc);
    m_state.dbg.__mdscr_el1 &=
        ~(1ULL); // Clear bit 0 (single step, SS) in the MDSCR_EL1.
  }
#else
  const uint32_t i = 0;
  if (enable) {
    m_hw_single_chained_step_addr = INVALID_NUB_ADDRESS;

    // Save our previous state
    m_dbg_save = m_state.dbg;
    // Set a breakpoint that will stop when the PC doesn't match the current
    // one!
    m_state.dbg.__bvr[i] =
        m_state.context.gpr.__pc &
        0xFFFFFFFCu; // Set the current PC as the breakpoint address
    m_state.dbg.__bcr[i] = BCR_M_IMVA_MISMATCH | // Stop on address mismatch
                           S_USER |              // Stop only in user mode
                           BCR_ENABLE;           // Enable this breakpoint
    if (m_state.context.gpr.__cpsr & 0x20) {
      // Thumb breakpoint
      if (m_state.context.gpr.__pc & 2)
        m_state.dbg.__bcr[i] |= BAS_IMVA_2_3;
      else
        m_state.dbg.__bcr[i] |= BAS_IMVA_0_1;

      uint16_t opcode;
      if (sizeof(opcode) ==
          m_thread->Process()->Task().ReadMemory(m_state.context.gpr.__pc,
                                                 sizeof(opcode), &opcode)) {
        if (IsThumb32Opcode(opcode)) {
          // 32 bit thumb opcode...
          if (m_state.context.gpr.__pc & 2) {
            // We can't take care of a 32 bit thumb instruction single step
            // with just IVA mismatching. We will need to chain an extra
            // hardware single step in order to complete this single step...
            m_hw_single_chained_step_addr = m_state.context.gpr.__pc + 2;
          } else {
            // Extend the number of bits to ignore for the mismatch
            m_state.dbg.__bcr[i] |= BAS_IMVA_ALL;
          }
        }
      }
    } else {
      // ARM breakpoint
      m_state.dbg.__bcr[i] |= BAS_IMVA_ALL; // Stop when any address bits change
    }

    DNBLogThreadedIf(LOG_STEP, "%s: BVR%u=0x%8.8x  BCR%u=0x%8.8x", __FUNCTION__,
                     i, m_state.dbg.__bvr[i], i, m_state.dbg.__bcr[i]);

    for (uint32_t j = i + 1; j < 16; ++j) {
      // Disable all others
      m_state.dbg.__bvr[j] = 0;
      m_state.dbg.__bcr[j] = 0;
    }
  } else {
    // Just restore the state we had before we did single stepping
    m_state.dbg = m_dbg_save;
  }
#endif

  return SetDBGState(false);
}

// return 1 if bit "BIT" is set in "value"
static inline uint32_t bit(uint32_t value, uint32_t bit) {
  return (value >> bit) & 1u;
}

// return the bitfield "value[msbit:lsbit]".
static inline uint32_t bits(uint32_t value, uint32_t msbit, uint32_t lsbit) {
  assert(msbit >= lsbit);
  uint32_t shift_left = sizeof(value) * 8 - 1 - msbit;
  value <<=
      shift_left; // shift anything above the msbit off of the unsigned edge
  value >>= (shift_left + lsbit); // shift it back again down to the lsbit
                                  // (including undoing any shift from above)
  return value;                   // return our result
}

bool DNBArchMachARM::ConditionPassed(uint8_t condition, uint32_t cpsr) {
  uint32_t cpsr_n = bit(cpsr, 31); // Negative condition code flag
  uint32_t cpsr_z = bit(cpsr, 30); // Zero condition code flag
  uint32_t cpsr_c = bit(cpsr, 29); // Carry condition code flag
  uint32_t cpsr_v = bit(cpsr, 28); // Overflow condition code flag

  switch (condition) {
  case COND_EQ: // (0x0)
    if (cpsr_z == 1)
      return true;
    break;
  case COND_NE: // (0x1)
    if (cpsr_z == 0)
      return true;
    break;
  case COND_CS: // (0x2)
    if (cpsr_c == 1)
      return true;
    break;
  case COND_CC: // (0x3)
    if (cpsr_c == 0)
      return true;
    break;
  case COND_MI: // (0x4)
    if (cpsr_n == 1)
      return true;
    break;
  case COND_PL: // (0x5)
    if (cpsr_n == 0)
      return true;
    break;
  case COND_VS: // (0x6)
    if (cpsr_v == 1)
      return true;
    break;
  case COND_VC: // (0x7)
    if (cpsr_v == 0)
      return true;
    break;
  case COND_HI: // (0x8)
    if ((cpsr_c == 1) && (cpsr_z == 0))
      return true;
    break;
  case COND_LS: // (0x9)
    if ((cpsr_c == 0) || (cpsr_z == 1))
      return true;
    break;
  case COND_GE: // (0xA)
    if (cpsr_n == cpsr_v)
      return true;
    break;
  case COND_LT: // (0xB)
    if (cpsr_n != cpsr_v)
      return true;
    break;
  case COND_GT: // (0xC)
    if ((cpsr_z == 0) && (cpsr_n == cpsr_v))
      return true;
    break;
  case COND_LE: // (0xD)
    if ((cpsr_z == 1) || (cpsr_n != cpsr_v))
      return true;
    break;
  default:
    return true;
    break;
  }

  return false;
}

uint32_t DNBArchMachARM::NumSupportedHardwareBreakpoints() {
  // Set the init value to something that will let us know that we need to
  // autodetect how many breakpoints are supported dynamically...
  static uint32_t g_num_supported_hw_breakpoints = UINT_MAX;
  if (g_num_supported_hw_breakpoints == UINT_MAX) {
    // Set this to zero in case we can't tell if there are any HW breakpoints
    g_num_supported_hw_breakpoints = 0;

    size_t len;
    uint32_t n = 0;
    len = sizeof(n);
    if (::sysctlbyname("hw.optional.breakpoint", &n, &len, NULL, 0) == 0) {
      g_num_supported_hw_breakpoints = n;
      DNBLogThreadedIf(LOG_THREAD, "hw.optional.breakpoint=%u", n);
    } else {
#if !defined(__arm64__) && !defined(__aarch64__)
      // Read the DBGDIDR to get the number of available hardware breakpoints
      // However, in some of our current armv7 processors, hardware
      // breakpoints/watchpoints were not properly connected. So detect those
      // cases using a field in a sysctl. For now we are using "hw.cpusubtype"
      // field to distinguish CPU architectures. This is a hack until we can
      // get <rdar://problem/6372672> fixed, at which point we will switch to
      // using a different sysctl string that will tell us how many BRPs
      // are available to us directly without having to read DBGDIDR.
      uint32_t register_DBGDIDR;

      asm("mrc p14, 0, %0, c0, c0, 0" : "=r"(register_DBGDIDR));
      uint32_t numBRPs = bits(register_DBGDIDR, 27, 24);
      // Zero is reserved for the BRP count, so don't increment it if it is zero
      if (numBRPs > 0)
        numBRPs++;
      DNBLogThreadedIf(LOG_THREAD, "DBGDIDR=0x%8.8x (number BRP pairs = %u)",
                       register_DBGDIDR, numBRPs);

      if (numBRPs > 0) {
        uint32_t cpusubtype;
        len = sizeof(cpusubtype);
        // TODO: remove this hack and change to using hw.optional.xx when
        // implmented
        if (::sysctlbyname("hw.cpusubtype", &cpusubtype, &len, NULL, 0) == 0) {
          DNBLogThreadedIf(LOG_THREAD, "hw.cpusubtype=%d", cpusubtype);
          if (cpusubtype == CPU_SUBTYPE_ARM_V7)
            DNBLogThreadedIf(LOG_THREAD, "Hardware breakpoints disabled for "
                                         "armv7 (rdar://problem/6372672)");
          else
            g_num_supported_hw_breakpoints = numBRPs;
        }
      }
#endif
    }
  }
  return g_num_supported_hw_breakpoints;
}

uint32_t DNBArchMachARM::NumSupportedHardwareWatchpoints() {
  // Set the init value to something that will let us know that we need to
  // autodetect how many watchpoints are supported dynamically...
  static uint32_t g_num_supported_hw_watchpoints = UINT_MAX;
  if (g_num_supported_hw_watchpoints == UINT_MAX) {
    // Set this to zero in case we can't tell if there are any HW breakpoints
    g_num_supported_hw_watchpoints = 0;

    size_t len;
    uint32_t n = 0;
    len = sizeof(n);
    if (::sysctlbyname("hw.optional.watchpoint", &n, &len, NULL, 0) == 0) {
      g_num_supported_hw_watchpoints = n;
      DNBLogThreadedIf(LOG_THREAD, "hw.optional.watchpoint=%u", n);
    } else {
#if !defined(__arm64__) && !defined(__aarch64__)
      // Read the DBGDIDR to get the number of available hardware breakpoints
      // However, in some of our current armv7 processors, hardware
      // breakpoints/watchpoints were not properly connected. So detect those
      // cases using a field in a sysctl. For now we are using "hw.cpusubtype"
      // field to distinguish CPU architectures. This is a hack until we can
      // get <rdar://problem/6372672> fixed, at which point we will switch to
      // using a different sysctl string that will tell us how many WRPs
      // are available to us directly without having to read DBGDIDR.

      uint32_t register_DBGDIDR;
      asm("mrc p14, 0, %0, c0, c0, 0" : "=r"(register_DBGDIDR));
      uint32_t numWRPs = bits(register_DBGDIDR, 31, 28) + 1;
      DNBLogThreadedIf(LOG_THREAD, "DBGDIDR=0x%8.8x (number WRP pairs = %u)",
                       register_DBGDIDR, numWRPs);

      if (numWRPs > 0) {
        uint32_t cpusubtype;
        size_t len;
        len = sizeof(cpusubtype);
        // TODO: remove this hack and change to using hw.optional.xx when
        // implmented
        if (::sysctlbyname("hw.cpusubtype", &cpusubtype, &len, NULL, 0) == 0) {
          DNBLogThreadedIf(LOG_THREAD, "hw.cpusubtype=0x%d", cpusubtype);

          if (cpusubtype == CPU_SUBTYPE_ARM_V7)
            DNBLogThreadedIf(LOG_THREAD, "Hardware watchpoints disabled for "
                                         "armv7 (rdar://problem/6372672)");
          else
            g_num_supported_hw_watchpoints = numWRPs;
        }
      }
#endif
    }
  }
  return g_num_supported_hw_watchpoints;
}

uint32_t DNBArchMachARM::EnableHardwareBreakpoint(nub_addr_t addr,
                                                  nub_size_t size,
                                                  bool also_set_on_task) {
  // Make sure our address isn't bogus
  if (addr & 1)
    return INVALID_NUB_HW_INDEX;

  kern_return_t kret = GetDBGState(false);

  if (kret == KERN_SUCCESS) {
    const uint32_t num_hw_breakpoints = NumSupportedHardwareBreakpoints();
    uint32_t i;
    for (i = 0; i < num_hw_breakpoints; ++i) {
      if ((m_state.dbg.__bcr[i] & BCR_ENABLE) == 0)
        break; // We found an available hw breakpoint slot (in i)
    }

    // See if we found an available hw breakpoint slot above
    if (i < num_hw_breakpoints) {
      // Make sure bits 1:0 are clear in our address
      m_state.dbg.__bvr[i] = addr & ~((nub_addr_t)3);

      if (size == 2 || addr & 2) {
        uint32_t byte_addr_select = (addr & 2) ? BAS_IMVA_2_3 : BAS_IMVA_0_1;

        // We have a thumb breakpoint
        // We have an ARM breakpoint
        m_state.dbg.__bcr[i] =
            BCR_M_IMVA_MATCH | // Stop on address match
            byte_addr_select | // Set the correct byte address select so we only
                               // trigger on the correct opcode
            S_USER |           // Which modes should this breakpoint stop in?
            BCR_ENABLE;        // Enable this hardware breakpoint
        DNBLogThreadedIf(LOG_BREAKPOINTS,
                         "DNBArchMachARM::EnableHardwareBreakpoint( addr = "
                         "0x%8.8llx, size = %llu ) - BVR%u/BCR%u = 0x%8.8x / "
                         "0x%8.8x (Thumb)",
                         (uint64_t)addr, (uint64_t)size, i, i,
                         m_state.dbg.__bvr[i], m_state.dbg.__bcr[i]);
      } else if (size == 4) {
        // We have an ARM breakpoint
        m_state.dbg.__bcr[i] =
            BCR_M_IMVA_MATCH | // Stop on address match
            BAS_IMVA_ALL | // Stop on any of the four bytes following the IMVA
            S_USER |       // Which modes should this breakpoint stop in?
            BCR_ENABLE;    // Enable this hardware breakpoint
        DNBLogThreadedIf(LOG_BREAKPOINTS,
                         "DNBArchMachARM::EnableHardwareBreakpoint( addr = "
                         "0x%8.8llx, size = %llu ) - BVR%u/BCR%u = 0x%8.8x / "
                         "0x%8.8x (ARM)",
                         (uint64_t)addr, (uint64_t)size, i, i,
                         m_state.dbg.__bvr[i], m_state.dbg.__bcr[i]);
      }

      kret = SetDBGState(false);
      DNBLogThreadedIf(LOG_BREAKPOINTS, "DNBArchMachARM::"
                                        "EnableHardwareBreakpoint() "
                                        "SetDBGState() => 0x%8.8x.",
                       kret);

      if (kret == KERN_SUCCESS)
        return i;
    } else {
      DNBLogThreadedIf(LOG_BREAKPOINTS,
                       "DNBArchMachARM::EnableHardwareBreakpoint(addr = "
                       "0x%8.8llx, size = %llu) => all hardware breakpoint "
                       "resources are being used.",
                       (uint64_t)addr, (uint64_t)size);
    }
  }

  return INVALID_NUB_HW_INDEX;
}

bool DNBArchMachARM::DisableHardwareBreakpoint(uint32_t hw_index,
                                               bool also_set_on_task) {
  kern_return_t kret = GetDBGState(false);

  const uint32_t num_hw_points = NumSupportedHardwareBreakpoints();
  if (kret == KERN_SUCCESS) {
    if (hw_index < num_hw_points) {
      m_state.dbg.__bcr[hw_index] = 0;
      DNBLogThreadedIf(LOG_BREAKPOINTS, "DNBArchMachARM::SetHardwareBreakpoint("
                                        " %u ) - BVR%u = 0x%8.8x  BCR%u = "
                                        "0x%8.8x",
                       hw_index, hw_index, m_state.dbg.__bvr[hw_index],
                       hw_index, m_state.dbg.__bcr[hw_index]);

      kret = SetDBGState(false);

      if (kret == KERN_SUCCESS)
        return true;
    }
  }
  return false;
}

// ARM v7 watchpoints may be either word-size or double-word-size.
// It's implementation defined which they can handle.  It looks like on an
// armv8 device, armv7 processes can watch dwords.  But on a genuine armv7
// device I tried, only word watchpoints are supported.

#if defined(__arm64__) || defined(__aarch64__)
#define WATCHPOINTS_ARE_DWORD 1
#else
#undef WATCHPOINTS_ARE_DWORD
#endif

uint32_t DNBArchMachARM::EnableHardwareWatchpoint(nub_addr_t addr,
                                                  nub_size_t size, bool read,
                                                  bool write,
                                                  bool also_set_on_task) {

  DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchMachARM::EnableHardwareWatchpoint("
                                    "addr = 0x%8.8llx, size = %zu, read = %u, "
                                    "write = %u)",
                   (uint64_t)addr, size, read, write);

  const uint32_t num_hw_watchpoints = NumSupportedHardwareWatchpoints();

  // Can't watch zero bytes
  if (size == 0)
    return INVALID_NUB_HW_INDEX;

  // We must watch for either read or write
  if (read == false && write == false)
    return INVALID_NUB_HW_INDEX;

  // Otherwise, can't watch more than 8 bytes per WVR/WCR pair
  if (size > 8)
    return INVALID_NUB_HW_INDEX;

// Treat arm watchpoints as having an 8-byte alignment requirement.  You can put
// a watchpoint on a 4-byte
// offset address but you can only watch 4 bytes with that watchpoint.

// arm watchpoints on an 8-byte (double word) aligned addr can watch any bytes
// in that
// 8-byte long region of memory.  They can watch the 1st byte, the 2nd byte, 3rd
// byte, etc, or any
// combination therein by setting the bits in the BAS [12:5] (Byte Address
// Select) field of
// the DBGWCRn_EL1 reg for the watchpoint.

// If the MASK [28:24] bits in the DBGWCRn_EL1 allow a single watchpoint to
// monitor a larger region
// of memory (16 bytes, 32 bytes, or 2GB) but the Byte Address Select bitfield
// then selects a larger
// range of bytes, instead of individual bytes.  See the ARMv8 Debug
// Architecture manual for details.
// This implementation does not currently use the MASK bits; the largest single
// region watched by a single
// watchpoint right now is 8-bytes.

#if defined(WATCHPOINTS_ARE_DWORD)
  nub_addr_t aligned_wp_address = addr & ~0x7;
  uint32_t addr_dword_offset = addr & 0x7;
  const int max_watchpoint_size = 8;
#else
  nub_addr_t aligned_wp_address = addr & ~0x3;
  uint32_t addr_dword_offset = addr & 0x3;
  const int max_watchpoint_size = 4;
#endif

  DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchMachARM::EnableHardwareWatchpoint "
                                    "aligned_wp_address is 0x%llx and "
                                    "addr_dword_offset is 0x%x",
                   (uint64_t)aligned_wp_address, addr_dword_offset);

  // Do we need to split up this logical watchpoint into two hardware watchpoint
  // registers?
  // e.g. a watchpoint of length 4 on address 6.  We need do this with
  //   one watchpoint on address 0 with bytes 6 & 7 being monitored
  //   one watchpoint on address 8 with bytes 0, 1, 2, 3 being monitored

  if (addr_dword_offset + size > max_watchpoint_size) {
    DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchMachARM::"
                                      "EnableHardwareWatchpoint(addr = "
                                      "0x%8.8llx, size = %zu) needs two "
                                      "hardware watchpoints slots to monitor",
                     (uint64_t)addr, size);
    int low_watchpoint_size = max_watchpoint_size - addr_dword_offset;
    int high_watchpoint_size = addr_dword_offset + size - max_watchpoint_size;

    uint32_t lo = EnableHardwareWatchpoint(addr, low_watchpoint_size, read,
                                           write, also_set_on_task);
    if (lo == INVALID_NUB_HW_INDEX)
      return INVALID_NUB_HW_INDEX;
    uint32_t hi = EnableHardwareWatchpoint(
        aligned_wp_address + max_watchpoint_size, high_watchpoint_size, read,
        write, also_set_on_task);
    if (hi == INVALID_NUB_HW_INDEX) {
      DisableHardwareWatchpoint(lo, also_set_on_task);
      return INVALID_NUB_HW_INDEX;
    }
    // Tag this lo->hi mapping in our database.
    LoHi[lo] = hi;
    return lo;
  }

  // At this point
  //  1 aligned_wp_address is the requested address rounded down to 8-byte
  //  alignment
  //  2 addr_dword_offset is the offset into that double word (8-byte) region
  //  that we are watching
  //  3 size is the number of bytes within that 8-byte region that we are
  //  watching

  // Set the Byte Address Selects bits DBGWCRn_EL1 bits [12:5] based on the
  // above.
  // The bit shift and negation operation will give us 0b11 for 2, 0b1111 for 4,
  // etc, up to 0b11111111 for 8.
  // then we shift those bits left by the offset into this dword that we are
  // interested in.
  // e.g. if we are watching bytes 4,5,6,7 in a dword we want a BAS of
  // 0b11110000.
  uint32_t byte_address_select = ((1 << size) - 1) << addr_dword_offset;

  // Read the debug state
  kern_return_t kret = GetDBGState(true);

  if (kret == KERN_SUCCESS) {
    // Check to make sure we have the needed hardware support
    uint32_t i = 0;

    for (i = 0; i < num_hw_watchpoints; ++i) {
      if ((m_state.dbg.__wcr[i] & WCR_ENABLE) == 0)
        break; // We found an available hw watchpoint slot (in i)
    }

    // See if we found an available hw watchpoint slot above
    if (i < num_hw_watchpoints) {
      // DumpDBGState(m_state.dbg);

      // Clear any previous LoHi joined-watchpoint that may have been in use
      LoHi[i] = 0;

      // shift our Byte Address Select bits up to the correct bit range for the
      // DBGWCRn_EL1
      byte_address_select = byte_address_select << 5;

      // Make sure bits 1:0 are clear in our address
      m_state.dbg.__wvr[i] = aligned_wp_address;   // DVA (Data Virtual Address)
      m_state.dbg.__wcr[i] = byte_address_select | // Which bytes that follow
                                                   // the DVA that we will watch
                             S_USER |              // Stop only in user mode
                             (read ? WCR_LOAD : 0) |   // Stop on read access?
                             (write ? WCR_STORE : 0) | // Stop on write access?
                             WCR_ENABLE; // Enable this watchpoint;

      DNBLogThreadedIf(
          LOG_WATCHPOINTS, "DNBArchMachARM::EnableHardwareWatchpoint() adding "
                           "watchpoint on address 0x%llx with control register "
                           "value 0x%x",
          (uint64_t)m_state.dbg.__wvr[i], (uint32_t)m_state.dbg.__wcr[i]);

      // The kernel will set the MDE_ENABLE bit in the MDSCR_EL1 for us
      // automatically, don't need to do it here.

      kret = SetDBGState(also_set_on_task);
      // DumpDBGState(m_state.dbg);

      DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchMachARM::"
                                        "EnableHardwareWatchpoint() "
                                        "SetDBGState() => 0x%8.8x.",
                       kret);

      if (kret == KERN_SUCCESS)
        return i;
    } else {
      DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchMachARM::"
                                        "EnableHardwareWatchpoint(): All "
                                        "hardware resources (%u) are in use.",
                       num_hw_watchpoints);
    }
  }
  return INVALID_NUB_HW_INDEX;
}

bool DNBArchMachARM::ReenableHardwareWatchpoint(uint32_t hw_index) {
  // If this logical watchpoint # is actually implemented using
  // two hardware watchpoint registers, re-enable both of them.

  if (hw_index < NumSupportedHardwareWatchpoints() && LoHi[hw_index]) {
    return ReenableHardwareWatchpoint_helper(hw_index) &&
           ReenableHardwareWatchpoint_helper(LoHi[hw_index]);
  } else {
    return ReenableHardwareWatchpoint_helper(hw_index);
  }
}

bool DNBArchMachARM::ReenableHardwareWatchpoint_helper(uint32_t hw_index) {
  kern_return_t kret = GetDBGState(false);
  if (kret != KERN_SUCCESS)
    return false;
  const uint32_t num_hw_points = NumSupportedHardwareWatchpoints();
  if (hw_index >= num_hw_points)
    return false;

  m_state.dbg.__wvr[hw_index] = m_disabled_watchpoints[hw_index].addr;
  m_state.dbg.__wcr[hw_index] = m_disabled_watchpoints[hw_index].control;

  DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchMachARM::EnableHardwareWatchpoint( "
                                    "%u ) - WVR%u = 0x%8.8llx  WCR%u = "
                                    "0x%8.8llx",
                   hw_index, hw_index, (uint64_t)m_state.dbg.__wvr[hw_index],
                   hw_index, (uint64_t)m_state.dbg.__wcr[hw_index]);

  // The kernel will set the MDE_ENABLE bit in the MDSCR_EL1 for us
  // automatically, don't need to do it here.

  kret = SetDBGState(false);

  return (kret == KERN_SUCCESS);
}

bool DNBArchMachARM::DisableHardwareWatchpoint(uint32_t hw_index,
                                               bool also_set_on_task) {
  if (hw_index < NumSupportedHardwareWatchpoints() && LoHi[hw_index]) {
    return DisableHardwareWatchpoint_helper(hw_index, also_set_on_task) &&
           DisableHardwareWatchpoint_helper(LoHi[hw_index], also_set_on_task);
  } else {
    return DisableHardwareWatchpoint_helper(hw_index, also_set_on_task);
  }
}

bool DNBArchMachARM::DisableHardwareWatchpoint_helper(uint32_t hw_index,
                                                      bool also_set_on_task) {
  kern_return_t kret = GetDBGState(false);
  if (kret != KERN_SUCCESS)
    return false;

  const uint32_t num_hw_points = NumSupportedHardwareWatchpoints();
  if (hw_index >= num_hw_points)
    return false;

  m_disabled_watchpoints[hw_index].addr = m_state.dbg.__wvr[hw_index];
  m_disabled_watchpoints[hw_index].control = m_state.dbg.__wcr[hw_index];

  m_state.dbg.__wvr[hw_index] = 0;
  m_state.dbg.__wcr[hw_index] = 0;
  DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchMachARM::DisableHardwareWatchpoint("
                                    " %u ) - WVR%u = 0x%8.8llx  WCR%u = "
                                    "0x%8.8llx",
                   hw_index, hw_index, (uint64_t)m_state.dbg.__wvr[hw_index],
                   hw_index, (uint64_t)m_state.dbg.__wcr[hw_index]);

  kret = SetDBGState(also_set_on_task);

  return (kret == KERN_SUCCESS);
}

// Returns -1 if the trailing bit patterns are not one of:
// { 0b???1, 0b??10, 0b?100, 0b1000 }.
static inline int32_t LowestBitSet(uint32_t val) {
  for (unsigned i = 0; i < 4; ++i) {
    if (bit(val, i))
      return i;
  }
  return -1;
}

// Iterate through the debug registers; return the index of the first watchpoint
// whose address matches.
// As a side effect, the starting address as understood by the debugger is
// returned which could be
// different from 'addr' passed as an in/out argument.
uint32_t DNBArchMachARM::GetHardwareWatchpointHit(nub_addr_t &addr) {
  // Read the debug state
  kern_return_t kret = GetDBGState(true);
  // DumpDBGState(m_state.dbg);
  DNBLogThreadedIf(
      LOG_WATCHPOINTS,
      "DNBArchMachARM::GetHardwareWatchpointHit() GetDBGState() => 0x%8.8x.",
      kret);
  DNBLogThreadedIf(LOG_WATCHPOINTS,
                   "DNBArchMachARM::GetHardwareWatchpointHit() addr = 0x%llx",
                   (uint64_t)addr);

// This is the watchpoint value to match against, i.e., word address.
#if defined(WATCHPOINTS_ARE_DWORD)
  nub_addr_t wp_val = addr & ~((nub_addr_t)7);
#else
  nub_addr_t wp_val = addr & ~((nub_addr_t)3);
#endif
  if (kret == KERN_SUCCESS) {
    DBG &debug_state = m_state.dbg;
    uint32_t i, num = NumSupportedHardwareWatchpoints();
    for (i = 0; i < num; ++i) {
      nub_addr_t wp_addr = GetWatchAddress(debug_state, i);
      DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchMachARM::"
                                        "GetHardwareWatchpointHit() slot: %u "
                                        "(addr = 0x%llx).",
                       i, (uint64_t)wp_addr);
      if (wp_val == wp_addr) {
#if defined(WATCHPOINTS_ARE_DWORD)
        uint32_t byte_mask = bits(debug_state.__wcr[i], 12, 5);
#else
        uint32_t byte_mask = bits(debug_state.__wcr[i], 8, 5);
#endif

        // Sanity check the byte_mask, first.
        if (LowestBitSet(byte_mask) < 0)
          continue;

        // Compute the starting address (from the point of view of the
        // debugger).
        addr = wp_addr + LowestBitSet(byte_mask);
        return i;
      }
    }
  }
  return INVALID_NUB_HW_INDEX;
}

nub_addr_t DNBArchMachARM::GetWatchpointAddressByIndex(uint32_t hw_index) {
  kern_return_t kret = GetDBGState(true);
  if (kret != KERN_SUCCESS)
    return INVALID_NUB_ADDRESS;
  const uint32_t num = NumSupportedHardwareWatchpoints();
  if (hw_index >= num)
    return INVALID_NUB_ADDRESS;
  if (IsWatchpointEnabled(m_state.dbg, hw_index))
    return GetWatchAddress(m_state.dbg, hw_index);
  return INVALID_NUB_ADDRESS;
}

bool DNBArchMachARM::IsWatchpointEnabled(const DBG &debug_state,
                                         uint32_t hw_index) {
  // Watchpoint Control Registers, bitfield definitions
  // ...
  // Bits    Value    Description
  // [0]     0        Watchpoint disabled
  //         1        Watchpoint enabled.
  return (debug_state.__wcr[hw_index] & 1u);
}

nub_addr_t DNBArchMachARM::GetWatchAddress(const DBG &debug_state,
                                           uint32_t hw_index) {
  // Watchpoint Value Registers, bitfield definitions
  // Bits        Description
  // [31:2]      Watchpoint value (word address, i.e., 4-byte aligned)
  // [1:0]       RAZ/SBZP
  return bits(debug_state.__wvr[hw_index], 31, 0);
}

// Register information definitions for 32 bit ARMV7.
enum gpr_regnums {
  gpr_r0 = 0,
  gpr_r1,
  gpr_r2,
  gpr_r3,
  gpr_r4,
  gpr_r5,
  gpr_r6,
  gpr_r7,
  gpr_r8,
  gpr_r9,
  gpr_r10,
  gpr_r11,
  gpr_r12,
  gpr_sp,
  gpr_lr,
  gpr_pc,
  gpr_cpsr
};

enum {
  vfp_s0 = 0,
  vfp_s1,
  vfp_s2,
  vfp_s3,
  vfp_s4,
  vfp_s5,
  vfp_s6,
  vfp_s7,
  vfp_s8,
  vfp_s9,
  vfp_s10,
  vfp_s11,
  vfp_s12,
  vfp_s13,
  vfp_s14,
  vfp_s15,
  vfp_s16,
  vfp_s17,
  vfp_s18,
  vfp_s19,
  vfp_s20,
  vfp_s21,
  vfp_s22,
  vfp_s23,
  vfp_s24,
  vfp_s25,
  vfp_s26,
  vfp_s27,
  vfp_s28,
  vfp_s29,
  vfp_s30,
  vfp_s31,
  vfp_d0,
  vfp_d1,
  vfp_d2,
  vfp_d3,
  vfp_d4,
  vfp_d5,
  vfp_d6,
  vfp_d7,
  vfp_d8,
  vfp_d9,
  vfp_d10,
  vfp_d11,
  vfp_d12,
  vfp_d13,
  vfp_d14,
  vfp_d15,
  vfp_d16,
  vfp_d17,
  vfp_d18,
  vfp_d19,
  vfp_d20,
  vfp_d21,
  vfp_d22,
  vfp_d23,
  vfp_d24,
  vfp_d25,
  vfp_d26,
  vfp_d27,
  vfp_d28,
  vfp_d29,
  vfp_d30,
  vfp_d31,
  vfp_q0,
  vfp_q1,
  vfp_q2,
  vfp_q3,
  vfp_q4,
  vfp_q5,
  vfp_q6,
  vfp_q7,
  vfp_q8,
  vfp_q9,
  vfp_q10,
  vfp_q11,
  vfp_q12,
  vfp_q13,
  vfp_q14,
  vfp_q15,
#if defined(__arm64__) || defined(__aarch64__)
  vfp_fpsr,
  vfp_fpcr,
#else
  vfp_fpscr
#endif
};

enum {
  exc_exception,
  exc_fsr,
  exc_far,
};

#define GPR_OFFSET_IDX(idx) (offsetof(DNBArchMachARM::GPR, __r[idx]))
#define GPR_OFFSET_NAME(reg) (offsetof(DNBArchMachARM::GPR, __##reg))

#define EXC_OFFSET(reg)                                                        \
  (offsetof(DNBArchMachARM::EXC, __##reg) +                                    \
   offsetof(DNBArchMachARM::Context, exc))

// These macros will auto define the register name, alt name, register size,
// register offset, encoding, format and native register. This ensures that
// the register state structures are defined correctly and have the correct
// sizes and offsets.
#define DEFINE_GPR_IDX(idx, reg, alt, gen)                                     \
  {                                                                            \
    e_regSetGPR, gpr_##reg, #reg, alt, Uint, Hex, 4, GPR_OFFSET_IDX(idx),      \
        ehframe_##reg, dwarf_##reg, gen, INVALID_NUB_REGNUM, NULL, NULL        \
  }
#define DEFINE_GPR_NAME(reg, alt, gen, inval)                                  \
  {                                                                            \
    e_regSetGPR, gpr_##reg, #reg, alt, Uint, Hex, 4, GPR_OFFSET_NAME(reg),     \
        ehframe_##reg, dwarf_##reg, gen, INVALID_NUB_REGNUM, NULL, inval       \
  }

// In case we are debugging to a debug target that the ability to
// change into the protected modes with folded registers (ABT, IRQ,
// FIQ, SYS, USR, etc..), we should invalidate r8-r14 if the CPSR
// gets modified.

const char *g_invalidate_cpsr[] = {"r8",  "r9", "r10", "r11",
                                   "r12", "sp", "lr",  NULL};

// General purpose registers
const DNBRegisterInfo DNBArchMachARM::g_gpr_registers[] = {
    DEFINE_GPR_IDX(0, r0, "arg1", GENERIC_REGNUM_ARG1),
    DEFINE_GPR_IDX(1, r1, "arg2", GENERIC_REGNUM_ARG2),
    DEFINE_GPR_IDX(2, r2, "arg3", GENERIC_REGNUM_ARG3),
    DEFINE_GPR_IDX(3, r3, "arg4", GENERIC_REGNUM_ARG4),
    DEFINE_GPR_IDX(4, r4, NULL, INVALID_NUB_REGNUM),
    DEFINE_GPR_IDX(5, r5, NULL, INVALID_NUB_REGNUM),
    DEFINE_GPR_IDX(6, r6, NULL, INVALID_NUB_REGNUM),
    DEFINE_GPR_IDX(7, r7, "fp", GENERIC_REGNUM_FP),
    DEFINE_GPR_IDX(8, r8, NULL, INVALID_NUB_REGNUM),
    DEFINE_GPR_IDX(9, r9, NULL, INVALID_NUB_REGNUM),
    DEFINE_GPR_IDX(10, r10, NULL, INVALID_NUB_REGNUM),
    DEFINE_GPR_IDX(11, r11, NULL, INVALID_NUB_REGNUM),
    DEFINE_GPR_IDX(12, r12, NULL, INVALID_NUB_REGNUM),
    DEFINE_GPR_NAME(sp, "r13", GENERIC_REGNUM_SP, NULL),
    DEFINE_GPR_NAME(lr, "r14", GENERIC_REGNUM_RA, NULL),
    DEFINE_GPR_NAME(pc, "r15", GENERIC_REGNUM_PC, NULL),
    DEFINE_GPR_NAME(cpsr, "flags", GENERIC_REGNUM_FLAGS, g_invalidate_cpsr)};

const char *g_contained_q0[]{"q0", NULL};
const char *g_contained_q1[]{"q1", NULL};
const char *g_contained_q2[]{"q2", NULL};
const char *g_contained_q3[]{"q3", NULL};
const char *g_contained_q4[]{"q4", NULL};
const char *g_contained_q5[]{"q5", NULL};
const char *g_contained_q6[]{"q6", NULL};
const char *g_contained_q7[]{"q7", NULL};
const char *g_contained_q8[]{"q8", NULL};
const char *g_contained_q9[]{"q9", NULL};
const char *g_contained_q10[]{"q10", NULL};
const char *g_contained_q11[]{"q11", NULL};
const char *g_contained_q12[]{"q12", NULL};
const char *g_contained_q13[]{"q13", NULL};
const char *g_contained_q14[]{"q14", NULL};
const char *g_contained_q15[]{"q15", NULL};

const char *g_invalidate_q0[]{"q0", "d0", "d1", "s0", "s1", "s2", "s3", NULL};
const char *g_invalidate_q1[]{"q1", "d2", "d3", "s4", "s5", "s6", "s7", NULL};
const char *g_invalidate_q2[]{"q2", "d4", "d5", "s8", "s9", "s10", "s11", NULL};
const char *g_invalidate_q3[]{"q3",  "d6",  "d7",  "s12",
                              "s13", "s14", "s15", NULL};
const char *g_invalidate_q4[]{"q4",  "d8",  "d9",  "s16",
                              "s17", "s18", "s19", NULL};
const char *g_invalidate_q5[]{"q5",  "d10", "d11", "s20",
                              "s21", "s22", "s23", NULL};
const char *g_invalidate_q6[]{"q6",  "d12", "d13", "s24",
                              "s25", "s26", "s27", NULL};
const char *g_invalidate_q7[]{"q7",  "d14", "d15", "s28",
                              "s29", "s30", "s31", NULL};
const char *g_invalidate_q8[]{"q8", "d16", "d17", NULL};
const char *g_invalidate_q9[]{"q9", "d18", "d19", NULL};
const char *g_invalidate_q10[]{"q10", "d20", "d21", NULL};
const char *g_invalidate_q11[]{"q11", "d22", "d23", NULL};
const char *g_invalidate_q12[]{"q12", "d24", "d25", NULL};
const char *g_invalidate_q13[]{"q13", "d26", "d27", NULL};
const char *g_invalidate_q14[]{"q14", "d28", "d29", NULL};
const char *g_invalidate_q15[]{"q15", "d30", "d31", NULL};

#define VFP_S_OFFSET_IDX(idx)                                                  \
  (((idx) % 4) * 4) // offset into q reg: 0, 4, 8, 12
#define VFP_D_OFFSET_IDX(idx) (((idx) % 2) * 8) // offset into q reg: 0, 8
#define VFP_Q_OFFSET_IDX(idx) (VFP_S_OFFSET_IDX((idx)*4))

#define VFP_OFFSET_NAME(reg)                                                   \
  (offsetof(DNBArchMachARM::FPU, __##reg) +                                    \
   offsetof(DNBArchMachARM::Context, vfp))

#define FLOAT_FORMAT Float

#define DEFINE_VFP_S_IDX(idx)                                                  \
  e_regSetVFP, vfp_s##idx, "s" #idx, NULL, IEEE754, FLOAT_FORMAT, 4,           \
      VFP_S_OFFSET_IDX(idx), INVALID_NUB_REGNUM, dwarf_s##idx,                 \
      INVALID_NUB_REGNUM, INVALID_NUB_REGNUM
#define DEFINE_VFP_D_IDX(idx)                                                  \
  e_regSetVFP, vfp_d##idx, "d" #idx, NULL, IEEE754, FLOAT_FORMAT, 8,           \
      VFP_D_OFFSET_IDX(idx), INVALID_NUB_REGNUM, dwarf_d##idx,                 \
      INVALID_NUB_REGNUM, INVALID_NUB_REGNUM
#define DEFINE_VFP_Q_IDX(idx)                                                  \
  e_regSetVFP, vfp_q##idx, "q" #idx, NULL, Vector, VectorOfUInt8, 16,          \
      VFP_Q_OFFSET_IDX(idx), INVALID_NUB_REGNUM, dwarf_q##idx,                 \
      INVALID_NUB_REGNUM, INVALID_NUB_REGNUM

// Floating point registers
const DNBRegisterInfo DNBArchMachARM::g_vfp_registers[] = {
    {DEFINE_VFP_S_IDX(0), g_contained_q0, g_invalidate_q0},
    {DEFINE_VFP_S_IDX(1), g_contained_q0, g_invalidate_q0},
    {DEFINE_VFP_S_IDX(2), g_contained_q0, g_invalidate_q0},
    {DEFINE_VFP_S_IDX(3), g_contained_q0, g_invalidate_q0},
    {DEFINE_VFP_S_IDX(4), g_contained_q1, g_invalidate_q1},
    {DEFINE_VFP_S_IDX(5), g_contained_q1, g_invalidate_q1},
    {DEFINE_VFP_S_IDX(6), g_contained_q1, g_invalidate_q1},
    {DEFINE_VFP_S_IDX(7), g_contained_q1, g_invalidate_q1},
    {DEFINE_VFP_S_IDX(8), g_contained_q2, g_invalidate_q2},
    {DEFINE_VFP_S_IDX(9), g_contained_q2, g_invalidate_q2},
    {DEFINE_VFP_S_IDX(10), g_contained_q2, g_invalidate_q2},
    {DEFINE_VFP_S_IDX(11), g_contained_q2, g_invalidate_q2},
    {DEFINE_VFP_S_IDX(12), g_contained_q3, g_invalidate_q3},
    {DEFINE_VFP_S_IDX(13), g_contained_q3, g_invalidate_q3},
    {DEFINE_VFP_S_IDX(14), g_contained_q3, g_invalidate_q3},
    {DEFINE_VFP_S_IDX(15), g_contained_q3, g_invalidate_q3},
    {DEFINE_VFP_S_IDX(16), g_contained_q4, g_invalidate_q4},
    {DEFINE_VFP_S_IDX(17), g_contained_q4, g_invalidate_q4},
    {DEFINE_VFP_S_IDX(18), g_contained_q4, g_invalidate_q4},
    {DEFINE_VFP_S_IDX(19), g_contained_q4, g_invalidate_q4},
    {DEFINE_VFP_S_IDX(20), g_contained_q5, g_invalidate_q5},
    {DEFINE_VFP_S_IDX(21), g_contained_q5, g_invalidate_q5},
    {DEFINE_VFP_S_IDX(22), g_contained_q5, g_invalidate_q5},
    {DEFINE_VFP_S_IDX(23), g_contained_q5, g_invalidate_q5},
    {DEFINE_VFP_S_IDX(24), g_contained_q6, g_invalidate_q6},
    {DEFINE_VFP_S_IDX(25), g_contained_q6, g_invalidate_q6},
    {DEFINE_VFP_S_IDX(26), g_contained_q6, g_invalidate_q6},
    {DEFINE_VFP_S_IDX(27), g_contained_q6, g_invalidate_q6},
    {DEFINE_VFP_S_IDX(28), g_contained_q7, g_invalidate_q7},
    {DEFINE_VFP_S_IDX(29), g_contained_q7, g_invalidate_q7},
    {DEFINE_VFP_S_IDX(30), g_contained_q7, g_invalidate_q7},
    {DEFINE_VFP_S_IDX(31), g_contained_q7, g_invalidate_q7},

    {DEFINE_VFP_D_IDX(0), g_contained_q0, g_invalidate_q0},
    {DEFINE_VFP_D_IDX(1), g_contained_q0, g_invalidate_q0},
    {DEFINE_VFP_D_IDX(2), g_contained_q1, g_invalidate_q1},
    {DEFINE_VFP_D_IDX(3), g_contained_q1, g_invalidate_q1},
    {DEFINE_VFP_D_IDX(4), g_contained_q2, g_invalidate_q2},
    {DEFINE_VFP_D_IDX(5), g_contained_q2, g_invalidate_q2},
    {DEFINE_VFP_D_IDX(6), g_contained_q3, g_invalidate_q3},
    {DEFINE_VFP_D_IDX(7), g_contained_q3, g_invalidate_q3},
    {DEFINE_VFP_D_IDX(8), g_contained_q4, g_invalidate_q4},
    {DEFINE_VFP_D_IDX(9), g_contained_q4, g_invalidate_q4},
    {DEFINE_VFP_D_IDX(10), g_contained_q5, g_invalidate_q5},
    {DEFINE_VFP_D_IDX(11), g_contained_q5, g_invalidate_q5},
    {DEFINE_VFP_D_IDX(12), g_contained_q6, g_invalidate_q6},
    {DEFINE_VFP_D_IDX(13), g_contained_q6, g_invalidate_q6},
    {DEFINE_VFP_D_IDX(14), g_contained_q7, g_invalidate_q7},
    {DEFINE_VFP_D_IDX(15), g_contained_q7, g_invalidate_q7},
    {DEFINE_VFP_D_IDX(16), g_contained_q8, g_invalidate_q8},
    {DEFINE_VFP_D_IDX(17), g_contained_q8, g_invalidate_q8},
    {DEFINE_VFP_D_IDX(18), g_contained_q9, g_invalidate_q9},
    {DEFINE_VFP_D_IDX(19), g_contained_q9, g_invalidate_q9},
    {DEFINE_VFP_D_IDX(20), g_contained_q10, g_invalidate_q10},
    {DEFINE_VFP_D_IDX(21), g_contained_q10, g_invalidate_q10},
    {DEFINE_VFP_D_IDX(22), g_contained_q11, g_invalidate_q11},
    {DEFINE_VFP_D_IDX(23), g_contained_q11, g_invalidate_q11},
    {DEFINE_VFP_D_IDX(24), g_contained_q12, g_invalidate_q12},
    {DEFINE_VFP_D_IDX(25), g_contained_q12, g_invalidate_q12},
    {DEFINE_VFP_D_IDX(26), g_contained_q13, g_invalidate_q13},
    {DEFINE_VFP_D_IDX(27), g_contained_q13, g_invalidate_q13},
    {DEFINE_VFP_D_IDX(28), g_contained_q14, g_invalidate_q14},
    {DEFINE_VFP_D_IDX(29), g_contained_q14, g_invalidate_q14},
    {DEFINE_VFP_D_IDX(30), g_contained_q15, g_invalidate_q15},
    {DEFINE_VFP_D_IDX(31), g_contained_q15, g_invalidate_q15},

    {DEFINE_VFP_Q_IDX(0), NULL, g_invalidate_q0},
    {DEFINE_VFP_Q_IDX(1), NULL, g_invalidate_q1},
    {DEFINE_VFP_Q_IDX(2), NULL, g_invalidate_q2},
    {DEFINE_VFP_Q_IDX(3), NULL, g_invalidate_q3},
    {DEFINE_VFP_Q_IDX(4), NULL, g_invalidate_q4},
    {DEFINE_VFP_Q_IDX(5), NULL, g_invalidate_q5},
    {DEFINE_VFP_Q_IDX(6), NULL, g_invalidate_q6},
    {DEFINE_VFP_Q_IDX(7), NULL, g_invalidate_q7},
    {DEFINE_VFP_Q_IDX(8), NULL, g_invalidate_q8},
    {DEFINE_VFP_Q_IDX(9), NULL, g_invalidate_q9},
    {DEFINE_VFP_Q_IDX(10), NULL, g_invalidate_q10},
    {DEFINE_VFP_Q_IDX(11), NULL, g_invalidate_q11},
    {DEFINE_VFP_Q_IDX(12), NULL, g_invalidate_q12},
    {DEFINE_VFP_Q_IDX(13), NULL, g_invalidate_q13},
    {DEFINE_VFP_Q_IDX(14), NULL, g_invalidate_q14},
    {DEFINE_VFP_Q_IDX(15), NULL, g_invalidate_q15},

#if defined(__arm64__) || defined(__aarch64__)
    {e_regSetVFP, vfp_fpsr, "fpsr", NULL, Uint, Hex, 4, VFP_OFFSET_NAME(fpsr),
     INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM,
     INVALID_NUB_REGNUM, NULL, NULL},
    {e_regSetVFP, vfp_fpcr, "fpcr", NULL, Uint, Hex, 4, VFP_OFFSET_NAME(fpcr),
     INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM,
     INVALID_NUB_REGNUM, NULL, NULL}
#else
    {e_regSetVFP, vfp_fpscr, "fpscr", NULL, Uint, Hex, 4,
     VFP_OFFSET_NAME(fpscr), INVALID_NUB_REGNUM, INVALID_NUB_REGNUM,
     INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, NULL, NULL}
#endif
};

// Exception registers

const DNBRegisterInfo DNBArchMachARM::g_exc_registers[] = {
    {e_regSetVFP, exc_exception, "exception", NULL, Uint, Hex, 4,
     EXC_OFFSET(exception), INVALID_NUB_REGNUM, INVALID_NUB_REGNUM,
     INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, NULL, NULL},
    {e_regSetVFP, exc_fsr, "fsr", NULL, Uint, Hex, 4, EXC_OFFSET(fsr),
     INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM,
     INVALID_NUB_REGNUM, NULL, NULL},
    {e_regSetVFP, exc_far, "far", NULL, Uint, Hex, 4, EXC_OFFSET(far),
     INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM,
     INVALID_NUB_REGNUM, NULL, NULL}};

// Number of registers in each register set
const size_t DNBArchMachARM::k_num_gpr_registers =
    sizeof(g_gpr_registers) / sizeof(DNBRegisterInfo);
const size_t DNBArchMachARM::k_num_vfp_registers =
    sizeof(g_vfp_registers) / sizeof(DNBRegisterInfo);
const size_t DNBArchMachARM::k_num_exc_registers =
    sizeof(g_exc_registers) / sizeof(DNBRegisterInfo);
const size_t DNBArchMachARM::k_num_all_registers =
    k_num_gpr_registers + k_num_vfp_registers + k_num_exc_registers;

// Register set definitions. The first definitions at register set index
// of zero is for all registers, followed by other registers sets. The
// register information for the all register set need not be filled in.
const DNBRegisterSetInfo DNBArchMachARM::g_reg_sets[] = {
    {"ARM Registers", NULL, k_num_all_registers},
    {"General Purpose Registers", g_gpr_registers, k_num_gpr_registers},
    {"Floating Point Registers", g_vfp_registers, k_num_vfp_registers},
    {"Exception State Registers", g_exc_registers, k_num_exc_registers}};
// Total number of register sets for this architecture
const size_t DNBArchMachARM::k_num_register_sets =
    sizeof(g_reg_sets) / sizeof(DNBRegisterSetInfo);

const DNBRegisterSetInfo *
DNBArchMachARM::GetRegisterSetInfo(nub_size_t *num_reg_sets) {
  *num_reg_sets = k_num_register_sets;
  return g_reg_sets;
}

bool DNBArchMachARM::GetRegisterValue(uint32_t set, uint32_t reg,
                                      DNBRegisterValue *value) {
  if (set == REGISTER_SET_GENERIC) {
    switch (reg) {
    case GENERIC_REGNUM_PC: // Program Counter
      set = e_regSetGPR;
      reg = gpr_pc;
      break;

    case GENERIC_REGNUM_SP: // Stack Pointer
      set = e_regSetGPR;
      reg = gpr_sp;
      break;

    case GENERIC_REGNUM_FP: // Frame Pointer
      set = e_regSetGPR;
      reg = gpr_r7; // is this the right reg?
      break;

    case GENERIC_REGNUM_RA: // Return Address
      set = e_regSetGPR;
      reg = gpr_lr;
      break;

    case GENERIC_REGNUM_FLAGS: // Processor flags register
      set = e_regSetGPR;
      reg = gpr_cpsr;
      break;

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
        value->value.uint32 = m_state.context.gpr.__r[reg];
        return true;
      }
      break;

    case e_regSetVFP:
      // "reg" is an index into the floating point register set at this point.
      // We need to translate it up so entry 0 in the fp reg set is the same as
      // vfp_s0
      // in the enumerated values for case statement below.
      if (reg >= vfp_s0 && reg <= vfp_s31) {
#if defined(__arm64__) || defined(__aarch64__)
        uint32_t *s_reg =
            ((uint32_t *)&m_state.context.vfp.__v[0]) + (reg - vfp_s0);
        memcpy(&value->value.v_uint8, s_reg, 4);
#else
        value->value.uint32 = m_state.context.vfp.__r[reg];
#endif
        return true;
      } else if (reg >= vfp_d0 && reg <= vfp_d31) {
#if defined(__arm64__) || defined(__aarch64__)
        uint64_t *d_reg =
            ((uint64_t *)&m_state.context.vfp.__v[0]) + (reg - vfp_d0);
        memcpy(&value->value.v_uint8, d_reg, 8);
#else
        uint32_t d_reg_idx = reg - vfp_d0;
        uint32_t s_reg_idx = d_reg_idx * 2;
        value->value.v_sint32[0] = m_state.context.vfp.__r[s_reg_idx + 0];
        value->value.v_sint32[1] = m_state.context.vfp.__r[s_reg_idx + 1];
#endif
        return true;
      } else if (reg >= vfp_q0 && reg <= vfp_q15) {
#if defined(__arm64__) || defined(__aarch64__)
        memcpy(&value->value.v_uint8,
               (uint8_t *)&m_state.context.vfp.__v[reg - vfp_q0], 16);
#else
        uint32_t s_reg_idx = (reg - vfp_q0) * 4;
        memcpy(&value->value.v_uint8,
               (uint8_t *)&m_state.context.vfp.__r[s_reg_idx], 16);
#endif
        return true;
      }
#if defined(__arm64__) || defined(__aarch64__)
      else if (reg == vfp_fpsr) {
        value->value.uint32 = m_state.context.vfp.__fpsr;
        return true;
      } else if (reg == vfp_fpcr) {
        value->value.uint32 = m_state.context.vfp.__fpcr;
        return true;
      }
#else
      else if (reg == vfp_fpscr) {
        value->value.uint32 = m_state.context.vfp.__fpscr;
        return true;
      }
#endif
      break;

    case e_regSetEXC:
      if (reg < k_num_exc_registers) {
        value->value.uint32 = (&m_state.context.exc.__exception)[reg];
        return true;
      }
      break;
    }
  }
  return false;
}

bool DNBArchMachARM::SetRegisterValue(uint32_t set, uint32_t reg,
                                      const DNBRegisterValue *value) {
  if (set == REGISTER_SET_GENERIC) {
    switch (reg) {
    case GENERIC_REGNUM_PC: // Program Counter
      set = e_regSetGPR;
      reg = gpr_pc;
      break;

    case GENERIC_REGNUM_SP: // Stack Pointer
      set = e_regSetGPR;
      reg = gpr_sp;
      break;

    case GENERIC_REGNUM_FP: // Frame Pointer
      set = e_regSetGPR;
      reg = gpr_r7;
      break;

    case GENERIC_REGNUM_RA: // Return Address
      set = e_regSetGPR;
      reg = gpr_lr;
      break;

    case GENERIC_REGNUM_FLAGS: // Processor flags register
      set = e_regSetGPR;
      reg = gpr_cpsr;
      break;

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
        m_state.context.gpr.__r[reg] = value->value.uint32;
        success = true;
      }
      break;

    case e_regSetVFP:
      // "reg" is an index into the floating point register set at this point.
      // We need to translate it up so entry 0 in the fp reg set is the same as
      // vfp_s0
      // in the enumerated values for case statement below.
      if (reg >= vfp_s0 && reg <= vfp_s31) {
#if defined(__arm64__) || defined(__aarch64__)
        uint32_t *s_reg =
            ((uint32_t *)&m_state.context.vfp.__v[0]) + (reg - vfp_s0);
        memcpy(s_reg, &value->value.v_uint8, 4);
#else
        m_state.context.vfp.__r[reg] = value->value.uint32;
#endif
        success = true;
      } else if (reg >= vfp_d0 && reg <= vfp_d31) {
#if defined(__arm64__) || defined(__aarch64__)
        uint64_t *d_reg =
            ((uint64_t *)&m_state.context.vfp.__v[0]) + (reg - vfp_d0);
        memcpy(d_reg, &value->value.v_uint8, 8);
#else
        uint32_t d_reg_idx = reg - vfp_d0;
        uint32_t s_reg_idx = d_reg_idx * 2;
        m_state.context.vfp.__r[s_reg_idx + 0] = value->value.v_sint32[0];
        m_state.context.vfp.__r[s_reg_idx + 1] = value->value.v_sint32[1];
#endif
        success = true;
      } else if (reg >= vfp_q0 && reg <= vfp_q15) {
#if defined(__arm64__) || defined(__aarch64__)
        memcpy((uint8_t *)&m_state.context.vfp.__v[reg - vfp_q0],
               &value->value.v_uint8, 16);
#else
        uint32_t s_reg_idx = (reg - vfp_q0) * 4;
        memcpy((uint8_t *)&m_state.context.vfp.__r[s_reg_idx],
               &value->value.v_uint8, 16);
#endif
        success = true;
      }
#if defined(__arm64__) || defined(__aarch64__)
      else if (reg == vfp_fpsr) {
        m_state.context.vfp.__fpsr = value->value.uint32;
        success = true;
      } else if (reg == vfp_fpcr) {
        m_state.context.vfp.__fpcr = value->value.uint32;
        success = true;
      }
#else
      else if (reg == vfp_fpscr) {
        m_state.context.vfp.__fpscr = value->value.uint32;
        success = true;
      }
#endif
      break;

    case e_regSetEXC:
      if (reg < k_num_exc_registers) {
        (&m_state.context.exc.__exception)[reg] = value->value.uint32;
        success = true;
      }
      break;
    }
  }
  if (success)
    return SetRegisterState(set) == KERN_SUCCESS;
  return false;
}

kern_return_t DNBArchMachARM::GetRegisterState(int set, bool force) {
  switch (set) {
  case e_regSetALL:
    return GetGPRState(force) | GetVFPState(force) | GetEXCState(force) |
           GetDBGState(force);
  case e_regSetGPR:
    return GetGPRState(force);
  case e_regSetVFP:
    return GetVFPState(force);
  case e_regSetEXC:
    return GetEXCState(force);
  case e_regSetDBG:
    return GetDBGState(force);
  default:
    break;
  }
  return KERN_INVALID_ARGUMENT;
}

kern_return_t DNBArchMachARM::SetRegisterState(int set) {
  // Make sure we have a valid context to set.
  kern_return_t err = GetRegisterState(set, false);
  if (err != KERN_SUCCESS)
    return err;

  switch (set) {
  case e_regSetALL:
    return SetGPRState() | SetVFPState() | SetEXCState() | SetDBGState(false);
  case e_regSetGPR:
    return SetGPRState();
  case e_regSetVFP:
    return SetVFPState();
  case e_regSetEXC:
    return SetEXCState();
  case e_regSetDBG:
    return SetDBGState(false);
  default:
    break;
  }
  return KERN_INVALID_ARGUMENT;
}

bool DNBArchMachARM::RegisterSetStateIsValid(int set) const {
  return m_state.RegsAreValid(set);
}

nub_size_t DNBArchMachARM::GetRegisterContext(void *buf, nub_size_t buf_len) {
  nub_size_t size = sizeof(m_state.context.gpr) + sizeof(m_state.context.vfp) +
                    sizeof(m_state.context.exc);

  if (buf && buf_len) {
    if (size > buf_len)
      size = buf_len;

    bool force = false;
    if (GetGPRState(force) | GetVFPState(force) | GetEXCState(force))
      return 0;

    // Copy each struct individually to avoid any padding that might be between
    // the structs in m_state.context
    uint8_t *p = (uint8_t *)buf;
    ::memcpy(p, &m_state.context.gpr, sizeof(m_state.context.gpr));
    p += sizeof(m_state.context.gpr);
    ::memcpy(p, &m_state.context.vfp, sizeof(m_state.context.vfp));
    p += sizeof(m_state.context.vfp);
    ::memcpy(p, &m_state.context.exc, sizeof(m_state.context.exc));
    p += sizeof(m_state.context.exc);

    size_t bytes_written = p - (uint8_t *)buf;
    UNUSED_IF_ASSERT_DISABLED(bytes_written);
    assert(bytes_written == size);
  }
  DNBLogThreadedIf(
      LOG_THREAD,
      "DNBArchMachARM::GetRegisterContext (buf = %p, len = %llu) => %llu", buf,
      (uint64_t)buf_len, (uint64_t)size);
  // Return the size of the register context even if NULL was passed in
  return size;
}

nub_size_t DNBArchMachARM::SetRegisterContext(const void *buf,
                                              nub_size_t buf_len) {
  nub_size_t size = sizeof(m_state.context.gpr) + sizeof(m_state.context.vfp) +
                    sizeof(m_state.context.exc);

  if (buf == NULL || buf_len == 0)
    size = 0;

  if (size) {
    if (size > buf_len)
      size = buf_len;

    // Copy each struct individually to avoid any padding that might be between
    // the structs in m_state.context
    uint8_t *p = const_cast<uint8_t*>(reinterpret_cast<const uint8_t *>(buf));
    ::memcpy(&m_state.context.gpr, p, sizeof(m_state.context.gpr));
    p += sizeof(m_state.context.gpr);
    ::memcpy(&m_state.context.vfp, p, sizeof(m_state.context.vfp));
    p += sizeof(m_state.context.vfp);
    ::memcpy(&m_state.context.exc, p, sizeof(m_state.context.exc));
    p += sizeof(m_state.context.exc);

    size_t bytes_written = p - reinterpret_cast<const uint8_t *>(buf);
    UNUSED_IF_ASSERT_DISABLED(bytes_written);
    assert(bytes_written == size);

    if (SetGPRState() | SetVFPState() | SetEXCState())
      return 0;
  }
  DNBLogThreadedIf(
      LOG_THREAD,
      "DNBArchMachARM::SetRegisterContext (buf = %p, len = %llu) => %llu", buf,
      (uint64_t)buf_len, (uint64_t)size);
  return size;
}

uint32_t DNBArchMachARM::SaveRegisterState() {
  kern_return_t kret = ::thread_abort_safely(m_thread->MachPortNumber());
  DNBLogThreadedIf(
      LOG_THREAD, "thread = 0x%4.4x calling thread_abort_safely (tid) => %u "
                  "(SetGPRState() for stop_count = %u)",
      m_thread->MachPortNumber(), kret, m_thread->Process()->StopCount());

  // Always re-read the registers because above we call thread_abort_safely();
  bool force = true;

  if ((kret = GetGPRState(force)) != KERN_SUCCESS) {
    DNBLogThreadedIf(LOG_THREAD, "DNBArchMachARM::SaveRegisterState () error: "
                                 "GPR regs failed to read: %u ",
                     kret);
  } else if ((kret = GetVFPState(force)) != KERN_SUCCESS) {
    DNBLogThreadedIf(LOG_THREAD, "DNBArchMachARM::SaveRegisterState () error: "
                                 "%s regs failed to read: %u",
                     "VFP", kret);
  } else {
    const uint32_t save_id = GetNextRegisterStateSaveID();
    m_saved_register_states[save_id] = m_state.context;
    return save_id;
  }
  return UINT32_MAX;
}

bool DNBArchMachARM::RestoreRegisterState(uint32_t save_id) {
  SaveRegisterStates::iterator pos = m_saved_register_states.find(save_id);
  if (pos != m_saved_register_states.end()) {
    m_state.context.gpr = pos->second.gpr;
    m_state.context.vfp = pos->second.vfp;
    kern_return_t kret;
    bool success = true;
    if ((kret = SetGPRState()) != KERN_SUCCESS) {
      DNBLogThreadedIf(LOG_THREAD, "DNBArchMachARM::RestoreRegisterState "
                                   "(save_id = %u) error: GPR regs failed to "
                                   "write: %u",
                       save_id, kret);
      success = false;
    } else if ((kret = SetVFPState()) != KERN_SUCCESS) {
      DNBLogThreadedIf(LOG_THREAD, "DNBArchMachARM::RestoreRegisterState "
                                   "(save_id = %u) error: %s regs failed to "
                                   "write: %u",
                       save_id, "VFP", kret);
      success = false;
    }
    m_saved_register_states.erase(pos);
    return success;
  }
  return false;
}

#endif // #if defined (__arm__)
