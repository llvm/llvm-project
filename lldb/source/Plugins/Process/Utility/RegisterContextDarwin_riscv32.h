//===-- RegisterContextDarwin_riscv32.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERCONTEXTDARWIN_RISCV32_H
#define LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERCONTEXTDARWIN_RISCV32_H

#include "lldb/Target/RegisterContext.h"
#include "lldb/lldb-private.h"

class RegisterContextDarwin_riscv32 : public lldb_private::RegisterContext {
public:
  RegisterContextDarwin_riscv32(lldb_private::Thread &thread,
                                uint32_t concrete_frame_idx);

  ~RegisterContextDarwin_riscv32() override;

  void InvalidateAllRegisters() override;

  size_t GetRegisterCount() override;

  const lldb_private::RegisterInfo *GetRegisterInfoAtIndex(size_t reg) override;

  size_t GetRegisterSetCount() override;

  const lldb_private::RegisterSet *GetRegisterSet(size_t set) override;

  bool ReadRegister(const lldb_private::RegisterInfo *reg_info,
                    lldb_private::RegisterValue &value) override;

  bool WriteRegister(const lldb_private::RegisterInfo *reg_info,
                     const lldb_private::RegisterValue &value) override;

  bool ReadAllRegisterValues(lldb::WritableDataBufferSP &data_sp) override;

  bool WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

  uint32_t ConvertRegisterKindToRegisterNumber(lldb::RegisterKind kind,
                                               uint32_t num) override;

  struct GPR {
    uint32_t x0;
    uint32_t x1;
    uint32_t x2;
    uint32_t x3;
    uint32_t x4;
    uint32_t x5;
    uint32_t x6;
    uint32_t x7;
    uint32_t x8;
    uint32_t x9;
    uint32_t x10;
    uint32_t x11;
    uint32_t x12;
    uint32_t x13;
    uint32_t x14;
    uint32_t x15;
    uint32_t x16;
    uint32_t x17;
    uint32_t x18;
    uint32_t x19;
    uint32_t x20;
    uint32_t x21;
    uint32_t x22;
    uint32_t x23;
    uint32_t x24;
    uint32_t x25;
    uint32_t x26;
    uint32_t x27;
    uint32_t x28;
    uint32_t x29;
    uint32_t x30;
    uint32_t x31;
    uint32_t pc;
  };

  struct FPU {
    uint32_t f0;
    uint32_t f1;
    uint32_t f2;
    uint32_t f3;
    uint32_t f4;
    uint32_t f5;
    uint32_t f6;
    uint32_t f7;
    uint32_t f8;
    uint32_t f9;
    uint32_t f10;
    uint32_t f11;
    uint32_t f12;
    uint32_t f13;
    uint32_t f14;
    uint32_t f15;
    uint32_t f16;
    uint32_t f17;
    uint32_t f18;
    uint32_t f19;
    uint32_t f20;
    uint32_t f21;
    uint32_t f22;
    uint32_t f23;
    uint32_t f24;
    uint32_t f25;
    uint32_t f26;
    uint32_t f27;
    uint32_t f28;
    uint32_t f29;
    uint32_t f30;
    uint32_t f31;
    uint32_t fcsr;
  };

  struct EXC {
    uint32_t exception;
    uint32_t fsr;
    uint32_t far;
  };

  struct CSR {
    uint32_t csr[1024];
  };

protected:
  enum {
    GPRRegSet = 2,  // RV32_THREAD_STATE
    EXCRegSet = 3,  // RV32_EXCEPTION_STATE
    FPURegSet = 4,  // RV_FP32_STATE
    CSRRegSet1 = 6, // RV_CSR_STATE1
    CSRRegSet2 = 7, // RV_CSR_STATE2
    CSRRegSet3 = 8, // RV_CSR_STATE3
    CSRRegSet4 = 9, // RV_CSR_STATE4
    CSRRegSet = 10  // full 16kbyte CSR reg bank
  };

  enum {
    GPRWordCount = sizeof(GPR) / sizeof(uint32_t),
    FPUWordCount = sizeof(FPU) / sizeof(uint32_t),
    EXCWordCount = sizeof(EXC) / sizeof(uint32_t),
    CSRWordCount = sizeof(CSR) / sizeof(uint32_t)
  };

  enum { Read = 0, Write = 1, kNumErrors = 2 };

  GPR gpr;
  FPU fpr;
  EXC exc;
  CSR csr;
  int gpr_errs[2]; // Read/Write errors
  int fpr_errs[2]; // Read/Write errors
  int exc_errs[2]; // Read/Write errors
  int csr_errs[2]; // Read/Write errors

  void InvalidateAllRegisterStates() {
    SetError(GPRRegSet, Read, -1);
    SetError(FPURegSet, Read, -1);
    SetError(EXCRegSet, Read, -1);
    SetError(CSRRegSet, Read, -1);
  }

  int GetError(int flavor, uint32_t err_idx) const {
    if (err_idx < kNumErrors) {
      switch (flavor) {
      // When getting all errors, just OR all values together to see if
      // we got any kind of error.
      case GPRRegSet:
        return gpr_errs[err_idx];
      case FPURegSet:
        return fpr_errs[err_idx];
      case EXCRegSet:
        return exc_errs[err_idx];
      case CSRRegSet:
        return csr_errs[err_idx];
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
        fpr_errs[err_idx] = err;
        return true;

      case EXCRegSet:
        exc_errs[err_idx] = err;
        return true;

      case CSRRegSet:
        csr_errs[err_idx] = err;
        return true;

      default:
        break;
      }
    }
    return false;
  }

  bool RegisterSetIsCached(int set) const { return GetError(set, Read) == 0; }

  void LogGPR(lldb_private::Log *log, const char *title);

  int ReadGPR(bool force);

  int ReadFPU(bool force);

  int ReadEXC(bool force);

  int ReadCSR(bool force);

  int WriteGPR();

  int WriteFPU();

  int WriteEXC();

  int WriteCSR();

  // Subclasses override these to do the actual reading.
  virtual int DoReadGPR(lldb::tid_t tid, int flavor, GPR &gpr) = 0;

  virtual int DoReadFPU(lldb::tid_t tid, int flavor, FPU &fpr) = 0;

  virtual int DoReadEXC(lldb::tid_t tid, int flavor, EXC &exc) = 0;

  virtual int DoReadCSR(lldb::tid_t tid, int flavor, CSR &exc) = 0;

  virtual int DoWriteGPR(lldb::tid_t tid, int flavor, const GPR &gpr) = 0;

  virtual int DoWriteFPU(lldb::tid_t tid, int flavor, const FPU &fpr) = 0;

  virtual int DoWriteEXC(lldb::tid_t tid, int flavor, const EXC &exc) = 0;

  virtual int DoWriteCSR(lldb::tid_t tid, int flavor, const CSR &exc) = 0;

  int ReadRegisterSet(uint32_t set, bool force);

  int WriteRegisterSet(uint32_t set);

  static uint32_t GetRegisterNumber(uint32_t reg_kind, uint32_t reg_num);

  static int GetSetForNativeRegNum(int reg_num);

  static size_t GetRegisterInfosCount();

  static const lldb_private::RegisterInfo *GetRegisterInfos();
};

#endif // LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERCONTEXTDARWIN_RISCV32_H
