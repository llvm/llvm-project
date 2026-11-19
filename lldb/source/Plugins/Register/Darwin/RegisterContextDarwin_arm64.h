//===-- RegisterContextDarwin_arm64.h -----------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_REGISTER_DARWIN_REGISTERCONTEXTDARWIN_ARM64_H
#define LLDB_SOURCE_PLUGINS_REGISTER_DARWIN_REGISTERCONTEXTDARWIN_ARM64_H

#include "lldb/Register/Layouts/Darwin/RegisterLayoutDarwin_arm64.h"
#include "lldb/Target/ThreadRegisterContext.h"
#include "lldb/lldb-private.h"

// Break only in privileged or user mode
#define S_RSVD ((uint32_t)(0u << 1))
#define S_PRIV ((uint32_t)(1u << 1))
#define S_USER ((uint32_t)(2u << 1))
#define S_PRIV_USER ((S_PRIV) | (S_USER))

#define WCR_ENABLE ((uint32_t)(1u))

// Watchpoint load/store
#define WCR_LOAD ((uint32_t)(1u << 3))
#define WCR_STORE ((uint32_t)(1u << 4))

class RegisterContextDarwin_arm64 : public lldb_private::ThreadRegisterContext {
public:
  RegisterContextDarwin_arm64(lldb_private::Thread &thread,
                              uint32_t concrete_frame_idx);

  ~RegisterContextDarwin_arm64() override;

  void InvalidateAllRegisters() override;

  llvm::Expected<lldb_private::RegisterValue>
  ReadRegister(lldb_private::RegisterID id) override;

  llvm::Error
  WriteRegister(lldb_private::RegisterID id,
                     const lldb_private::RegisterValue &reg_value) override;

  bool ReadAllRegisterValues(lldb::WritableDataBufferSP &data_sp) override;

  bool WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

  uint32_t NumSupportedHardwareWatchpoints() override;

  uint32_t SetHardwareWatchpoint(lldb::addr_t addr, size_t size, bool read,
                                 bool write) override;

  bool ClearHardwareWatchpoint(uint32_t hw_index) override;

  using GPR = lldb_private::RegisterLayoutDarwin_arm64::GPR;
  using VReg = lldb_private::RegisterLayoutDarwin_arm64::VReg;
  using FPU = lldb_private::RegisterLayoutDarwin_arm64::FPU;
  using EXC = lldb_private::RegisterLayoutDarwin_arm64::EXC;
  using DBG = lldb_private::RegisterLayoutDarwin_arm64::DBG;

  static void LogDBGRegisters(lldb_private::Log *log, const DBG &dbg);

protected:
  enum {
    GPRRegSet = lldb_private::RegisterLayoutDarwin_arm64::GPRRegSet,
    FPURegSet = lldb_private::RegisterLayoutDarwin_arm64::FPURegSet,
    EXCRegSet = lldb_private::RegisterLayoutDarwin_arm64::EXCRegSet,
    DBGRegSet = lldb_private::RegisterLayoutDarwin_arm64::DBGRegSet
  };

  enum {
    GPRWordCount = sizeof(GPR) / sizeof(uint32_t), // ARM_THREAD_STATE64_COUNT
    FPUWordCount = sizeof(FPU) / sizeof(uint32_t), // ARM_NEON_STATE64_COUNT
    EXCWordCount = sizeof(EXC) /
        sizeof(uint32_t),                         // ARM_EXCEPTION_STATE64_COUNT
    DBGWordCount = sizeof(DBG) / sizeof(uint32_t) // ARM_DEBUG_STATE64_COUNT
  };

  enum { Read = 0, Write = 1, kNumErrors = 2 };

  GPR gpr;
  FPU fpu;
  EXC exc;
  DBG dbg;
  int gpr_errs[2]; // Read/Write errors
  int fpu_errs[2]; // Read/Write errors
  int exc_errs[2]; // Read/Write errors
  int dbg_errs[2]; // Read/Write errors

  void InvalidateAllRegisterStates() {
    SetError(GPRRegSet, Read, -1);
    SetError(FPURegSet, Read, -1);
    SetError(EXCRegSet, Read, -1);
  }

  int GetError(int flavor, uint32_t err_idx) const {
    if (err_idx < kNumErrors) {
      switch (flavor) {
      // When getting all errors, just OR all values together to see if
      // we got any kind of error.
      case GPRRegSet:
        return gpr_errs[err_idx];
      case FPURegSet:
        return fpu_errs[err_idx];
      case EXCRegSet:
        return exc_errs[err_idx];
      case DBGRegSet:
        return dbg_errs[err_idx];
      default:
        break;
      }
    }
    return -1;
  }

  bool SetError(int flavor, uint32_t err_idx, int err) {
    if (err_idx < kNumErrors) {
      switch (flavor) {
      case GPRRegSet:
        gpr_errs[err_idx] = err;
        return true;

      case FPURegSet:
        fpu_errs[err_idx] = err;
        return true;

      case EXCRegSet:
        exc_errs[err_idx] = err;
        return true;

      case DBGRegSet:
        exc_errs[err_idx] = err;
        return true;

      default:
        break;
      }
    }
    return false;
  }

  bool RegisterSetIsCached(int set) const { return GetError(set, Read) == 0; }

  int ReadGPR(bool force);

  int ReadFPU(bool force);

  int ReadEXC(bool force);

  int ReadDBG(bool force);

  int WriteGPR();

  int WriteFPU();

  int WriteEXC();

  int WriteDBG();

  // Subclasses override these to do the actual reading.
  virtual int DoReadGPR(lldb::tid_t tid, int flavor, GPR &gpr) { return -1; }

  virtual int DoReadFPU(lldb::tid_t tid, int flavor, FPU &fpu) = 0;

  virtual int DoReadEXC(lldb::tid_t tid, int flavor, EXC &exc) = 0;

  virtual int DoReadDBG(lldb::tid_t tid, int flavor, DBG &dbg) = 0;

  virtual int DoWriteGPR(lldb::tid_t tid, int flavor, const GPR &gpr) = 0;

  virtual int DoWriteFPU(lldb::tid_t tid, int flavor, const FPU &fpu) = 0;

  virtual int DoWriteEXC(lldb::tid_t tid, int flavor, const EXC &exc) = 0;

  virtual int DoWriteDBG(lldb::tid_t tid, int flavor, const DBG &dbg) = 0;

  int ReadRegisterSet(uint32_t set, bool force);

  int WriteRegisterSet(uint32_t set);

  static uint32_t GetRegisterNumber(uint32_t reg_kind, uint32_t reg_num);
};

#endif // LLDB_SOURCE_PLUGINS_REGISTER_DARWIN_REGISTERCONTEXTDARWIN_ARM64_H
