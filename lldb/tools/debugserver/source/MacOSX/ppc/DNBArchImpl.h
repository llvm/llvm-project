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

#ifndef __DebugNubArchMachPPC_h__
#define __DebugNubArchMachPPC_h__

#if defined(__powerpc__) || defined(__ppc__) || defined(__ppc64__)

#include "DNBArch.h"

class MachThread;

class DNBArchMachPPC : public DNBArchProtocol {
public:
  DNBArchMachPPC(MachThread *thread) : m_thread(thread), m_state() {}

  virtual ~DNBArchMachPPC() {}

  virtual const DNBRegisterSetInfo *
  GetRegisterSetInfo(nub_size_t *num_reg_sets) const;
  virtual bool GetRegisterValue(uint32_t set, uint32_t reg,
                                DNBRegisterValue *value) const;
  virtual kern_return_t GetRegisterState(int set, bool force);
  virtual kern_return_t SetRegisterState(int set);
  virtual bool RegisterSetStateIsValid(int set) const;

  virtual uint64_t GetPC(uint64_t failValue); // Get program counter
  virtual kern_return_t SetPC(uint64_t value);
  virtual uint64_t GetSP(uint64_t failValue); // Get stack pointer
  virtual bool ThreadWillResume();
  virtual bool ThreadDidStop();

  static const uint8_t *SoftwareBreakpointOpcode(nub_size_t byte_size);
  static uint32_t GetCPUType();

protected:
  kern_return_t EnableHardwareSingleStep(bool enable);

  enum RegisterSet {
    e_regSetALL = REGISTER_SET_ALL,
    e_regSetGPR,
    e_regSetFPR,
    e_regSetEXC,
    e_regSetVEC,
    kNumRegisterSets
  };

  typedef enum RegisterSetWordSizeTag {
    e_regSetWordSizeGPR = PPC_THREAD_STATE_COUNT,
    e_regSetWordSizeFPR = PPC_FLOAT_STATE_COUNT,
    e_regSetWordSizeEXC = PPC_EXCEPTION_STATE_COUNT,
    e_regSetWordSizeVEC = PPC_VECTOR_STATE_COUNT
  } RegisterSetWordSize;

  enum { Read = 0, Write = 1, kNumErrors = 2 };

  struct State {
    ppc_thread_state_t gpr;
    ppc_float_state_t fpr;
    ppc_exception_state_t exc;
    ppc_vector_state_t vec;
    kern_return_t gpr_errs[2]; // Read/Write errors
    kern_return_t fpr_errs[2]; // Read/Write errors
    kern_return_t exc_errs[2]; // Read/Write errors
    kern_return_t vec_errs[2]; // Read/Write errors

    State() {
      uint32_t i;
      for (i = 0; i < kNumErrors; i++) {
        gpr_errs[i] = -1;
        fpr_errs[i] = -1;
        exc_errs[i] = -1;
        vec_errs[i] = -1;
      }
    }
    void InvalidateAllRegisterStates() { SetError(e_regSetALL, Read, -1); }
    kern_return_t GetError(int set, uint32_t err_idx) const {
      if (err_idx < kNumErrors) {
        switch (set) {
        // When getting all errors, just OR all values together to see if
        // we got any kind of error.
        case e_regSetALL:
          return gpr_errs[err_idx] | fpr_errs[err_idx] | exc_errs[err_idx] |
                 vec_errs[err_idx];
        case e_regSetGPR:
          return gpr_errs[err_idx];
        case e_regSetFPR:
          return fpr_errs[err_idx];
        case e_regSetEXC:
          return exc_errs[err_idx];
        case e_regSetVEC:
          return vec_errs[err_idx];
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
          gpr_errs[err_idx] = fpr_errs[err_idx] = exc_errs[err_idx] =
              vec_errs[err_idx] = err;
          return true;

        case e_regSetGPR:
          gpr_errs[err_idx] = err;
          return true;

        case e_regSetFPR:
          fpr_errs[err_idx] = err;
          return true;

        case e_regSetEXC:
          exc_errs[err_idx] = err;
          return true;

        case e_regSetVEC:
          vec_errs[err_idx] = err;
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
  kern_return_t GetFPRState(bool force);
  kern_return_t GetEXCState(bool force);
  kern_return_t GetVECState(bool force);

  kern_return_t SetGPRState();
  kern_return_t SetFPRState();
  kern_return_t SetEXCState();
  kern_return_t SetVECState();

protected:
  MachThread *m_thread;
  State m_state;
};

#endif // #if defined (__powerpc__) || defined (__ppc__) || defined (__ppc64__)
#endif // #ifndef __DebugNubArchMachPPC_h__
