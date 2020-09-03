//===-- DNBArchImpl.h -------------------------------------------*- C++ -*-===//
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

#ifndef LLDB_TOOLS_DEBUGSERVER_SOURCE_MACOSX_ARM_DNBARCHIMPL_H
#define LLDB_TOOLS_DEBUGSERVER_SOURCE_MACOSX_ARM_DNBARCHIMPL_H

#if defined(__arm__) || defined(__arm64__) || defined(__aarch64__)

#include "DNBArch.h"

#include <map>

class MachThread;

class DNBArchMachARM : public DNBArchProtocol {
public:
  enum { kMaxNumThumbITBreakpoints = 4 };

  DNBArchMachARM(MachThread *thread)
      : m_thread(thread), m_state(), m_disabled_watchpoints(),
        m_hw_single_chained_step_addr(INVALID_NUB_ADDRESS),
        m_last_decode_pc(INVALID_NUB_ADDRESS), m_watchpoint_hw_index(-1),
        m_watchpoint_did_occur(false),
        m_watchpoint_resume_single_step_enabled(false),
        m_saved_register_states() {
    m_disabled_watchpoints.resize(16);
    memset(&m_dbg_save, 0, sizeof(m_dbg_save));
#if defined(USE_ARM_DISASSEMBLER_FRAMEWORK)
    ThumbStaticsInit(&m_last_decode_thumb);
#endif
  }

  virtual ~DNBArchMachARM() {}

  static void Initialize();
  static const DNBRegisterSetInfo *GetRegisterSetInfo(nub_size_t *num_reg_sets);

  virtual bool GetRegisterValue(uint32_t set, uint32_t reg,
                                DNBRegisterValue *value);
  virtual bool SetRegisterValue(uint32_t set, uint32_t reg,
                                const DNBRegisterValue *value);
  virtual nub_size_t GetRegisterContext(void *buf, nub_size_t buf_len);
  virtual nub_size_t SetRegisterContext(const void *buf, nub_size_t buf_len);
  virtual uint32_t SaveRegisterState();
  virtual bool RestoreRegisterState(uint32_t save_id);

  virtual kern_return_t GetRegisterState(int set, bool force);
  virtual kern_return_t SetRegisterState(int set);
  virtual bool RegisterSetStateIsValid(int set) const;

  virtual uint64_t GetPC(uint64_t failValue); // Get program counter
  virtual kern_return_t SetPC(uint64_t value);
  virtual uint64_t GetSP(uint64_t failValue); // Get stack pointer
  virtual void ThreadWillResume();
  virtual bool ThreadDidStop();
  virtual bool NotifyException(MachException::Data &exc);

  static DNBArchProtocol *Create(MachThread *thread);
  static const uint8_t *SoftwareBreakpointOpcode(nub_size_t byte_size);
  static uint32_t GetCPUType();

  virtual uint32_t NumSupportedHardwareBreakpoints();
  virtual uint32_t NumSupportedHardwareWatchpoints();
  virtual uint32_t EnableHardwareBreakpoint(nub_addr_t addr, nub_size_t size,
                                            bool also_set_on_task);
  virtual bool DisableHardwareBreakpoint(uint32_t hw_break_index,
                                         bool also_set_on_task);

  virtual uint32_t EnableHardwareWatchpoint(nub_addr_t addr, nub_size_t size,
                                            bool read, bool write,
                                            bool also_set_on_task);
  virtual bool DisableHardwareWatchpoint(uint32_t hw_break_index,
                                         bool also_set_on_task);
  virtual bool DisableHardwareWatchpoint_helper(uint32_t hw_break_index,
                                                bool also_set_on_task);
  virtual bool ReenableHardwareWatchpoint(uint32_t hw_break_index);
  virtual bool ReenableHardwareWatchpoint_helper(uint32_t hw_break_index);

  virtual bool StepNotComplete();
  virtual uint32_t GetHardwareWatchpointHit(nub_addr_t &addr);

#if defined(ARM_DEBUG_STATE32) && (defined(__arm64__) || defined(__aarch64__))
  typedef arm_debug_state32_t DBG;
#else
  typedef arm_debug_state_t DBG;
#endif

protected:
  kern_return_t EnableHardwareSingleStep(bool enable);
  kern_return_t SetSingleStepSoftwareBreakpoints();

  bool ConditionPassed(uint8_t condition, uint32_t cpsr);
#if defined(USE_ARM_DISASSEMBLER_FRAMEWORK)
  bool ComputeNextPC(nub_addr_t currentPC,
                     arm_decoded_instruction_t decodedInstruction,
                     bool currentPCIsThumb, nub_addr_t *targetPC);
  arm_error_t DecodeInstructionUsingDisassembler(
      nub_addr_t curr_pc, uint32_t curr_cpsr,
      arm_decoded_instruction_t *decodedInstruction,
      thumb_static_data_t *thumbStaticData, nub_addr_t *next_pc);
  void DecodeITBlockInstructions(nub_addr_t curr_pc);
#endif
  void EvaluateNextInstructionForSoftwareBreakpointSetup(nub_addr_t currentPC,
                                                         uint32_t cpsr,
                                                         bool currentPCIsThumb,
                                                         nub_addr_t *nextPC,
                                                         bool *nextPCIsThumb);

  enum RegisterSet {
    e_regSetALL = REGISTER_SET_ALL,
    e_regSetGPR, // ARM_THREAD_STATE
    e_regSetVFP, // ARM_VFP_STATE (ARM_NEON_STATE if defined __arm64__)
    e_regSetEXC, // ARM_EXCEPTION_STATE
    e_regSetDBG, // ARM_DEBUG_STATE (ARM_DEBUG_STATE32 if defined __arm64__)
    kNumRegisterSets
  };

  enum { Read = 0, Write = 1, kNumErrors = 2 };

  typedef arm_thread_state_t GPR;
#if defined(__arm64__) || defined(__aarch64__)
  typedef arm_neon_state_t FPU;
#else
  typedef arm_vfp_state_t FPU;
#endif
  typedef arm_exception_state_t EXC;

  static const DNBRegisterInfo g_gpr_registers[];
  static const DNBRegisterInfo g_vfp_registers[];
  static const DNBRegisterInfo g_exc_registers[];
  static const DNBRegisterSetInfo g_reg_sets[];

  static const size_t k_num_gpr_registers;
  static const size_t k_num_vfp_registers;
  static const size_t k_num_exc_registers;
  static const size_t k_num_all_registers;
  static const size_t k_num_register_sets;

  struct Context {
    GPR gpr;
    FPU vfp;
    EXC exc;
  };

  struct State {
    Context context;
    DBG dbg;
    kern_return_t gpr_errs[2]; // Read/Write errors
    kern_return_t vfp_errs[2]; // Read/Write errors
    kern_return_t exc_errs[2]; // Read/Write errors
    kern_return_t dbg_errs[2]; // Read/Write errors
    State() {
      uint32_t i;
      for (i = 0; i < kNumErrors; i++) {
        gpr_errs[i] = -1;
        vfp_errs[i] = -1;
        exc_errs[i] = -1;
        dbg_errs[i] = -1;
      }
    }
    void InvalidateRegisterSetState(int set) { SetError(set, Read, -1); }
    kern_return_t GetError(int set, uint32_t err_idx) const {
      if (err_idx < kNumErrors) {
        switch (set) {
        // When getting all errors, just OR all values together to see if
        // we got any kind of error.
        case e_regSetALL:
          return gpr_errs[err_idx] | vfp_errs[err_idx] | exc_errs[err_idx] |
                 dbg_errs[err_idx];
        case e_regSetGPR:
          return gpr_errs[err_idx];
        case e_regSetVFP:
          return vfp_errs[err_idx];
        case e_regSetEXC:
          return exc_errs[err_idx];
        case e_regSetDBG:
          return dbg_errs[err_idx];
        default:
          break;
        }
      }
      return -1;
    }
    bool SetError(int set, uint32_t err_idx, kern_return_t err) {
      if (err_idx < kNumErrors) {
        switch (set) {
        case e_regSetALL:
          gpr_errs[err_idx] = err;
          vfp_errs[err_idx] = err;
          dbg_errs[err_idx] = err;
          exc_errs[err_idx] = err;
          return true;

        case e_regSetGPR:
          gpr_errs[err_idx] = err;
          return true;

        case e_regSetVFP:
          vfp_errs[err_idx] = err;
          return true;

        case e_regSetEXC:
          exc_errs[err_idx] = err;
          return true;

        case e_regSetDBG:
          dbg_errs[err_idx] = err;
          return true;
        default:
          break;
        }
      }
      return false;
    }
    bool RegsAreValid(int set) const {
      return GetError(set, Read) == KERN_SUCCESS;
    }
  };

  kern_return_t GetGPRState(bool force);
  kern_return_t GetVFPState(bool force);
  kern_return_t GetEXCState(bool force);
  kern_return_t GetDBGState(bool force);

  kern_return_t SetGPRState();
  kern_return_t SetVFPState();
  kern_return_t SetEXCState();
  kern_return_t SetDBGState(bool also_set_on_task);

  bool IsWatchpointEnabled(const DBG &debug_state, uint32_t hw_index);
  nub_addr_t GetWatchpointAddressByIndex(uint32_t hw_index);
  nub_addr_t GetWatchAddress(const DBG &debug_state, uint32_t hw_index);

  class disabled_watchpoint {
  public:
    disabled_watchpoint() {
      addr = 0;
      control = 0;
    }
    nub_addr_t addr;
    uint32_t control;
  };

protected:
  MachThread *m_thread;
  State m_state;
  DBG m_dbg_save;

  // armv8 doesn't keep the disabled watchpoint values in the debug register
  // context like armv7;
  // we need to save them aside when we disable them temporarily.
  std::vector<disabled_watchpoint> m_disabled_watchpoints;

  nub_addr_t m_hw_single_chained_step_addr;
  nub_addr_t m_last_decode_pc;

  // The following member variables should be updated atomically.
  int32_t m_watchpoint_hw_index;
  bool m_watchpoint_did_occur;
  bool m_watchpoint_resume_single_step_enabled;

  typedef std::map<uint32_t, Context> SaveRegisterStates;
  SaveRegisterStates m_saved_register_states;
};

#endif // #if defined (__arm__)
#endif // LLDB_TOOLS_DEBUGSERVER_SOURCE_MACOSX_ARM_DNBARCHIMPL_H
