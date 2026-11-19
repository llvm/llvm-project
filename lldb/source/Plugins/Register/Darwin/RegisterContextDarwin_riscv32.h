//===-- RegisterContextDarwin_riscv32.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_REGISTER_DARWIN_REGISTERCONTEXTDARWIN_RISCV32_H
#define LLDB_SOURCE_PLUGINS_REGISTER_DARWIN_REGISTERCONTEXTDARWIN_RISCV32_H

#include "lldb/Register/Layouts/Darwin/RegisterLayoutDarwin_riscv32.h"
#include "lldb/Target/ThreadRegisterContext.h"
#include "lldb/lldb-private.h"

class RegisterContextDarwin_riscv32
    : public lldb_private::ThreadRegisterContext {
public:
  RegisterContextDarwin_riscv32(lldb_private::Thread &thread,
                                uint32_t concrete_frame_idx);

  ~RegisterContextDarwin_riscv32() override;

  void InvalidateAllRegisters() override;

  llvm::Expected<lldb_private::RegisterValue>
  ReadRegister(lldb_private::RegisterID id) override;

  llvm::Error WriteRegister(lldb_private::RegisterID id,
                     const lldb_private::RegisterValue &value) override;

  bool ReadAllRegisterValues(lldb::WritableDataBufferSP &data_sp) override;

  bool WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

  using GPR = lldb_private::RegisterLayoutDarwin_riscv32::GPR;
  using FPU = lldb_private::RegisterLayoutDarwin_riscv32::FPU;
  using EXC = lldb_private::RegisterLayoutDarwin_riscv32::EXC;
  using CSR = lldb_private::RegisterLayoutDarwin_riscv32::CSR;

protected:
  enum {
    GPRRegSet = lldb_private::RegisterLayoutDarwin_riscv32::GPRRegSet,
    EXCRegSet = lldb_private::RegisterLayoutDarwin_riscv32::EXCRegSet,
    FPURegSet = lldb_private::RegisterLayoutDarwin_riscv32::FPURegSet,
    CSRRegSet1 = lldb_private::RegisterLayoutDarwin_riscv32::CSRRegSet1,
    CSRRegSet2 = lldb_private::RegisterLayoutDarwin_riscv32::CSRRegSet2,
    CSRRegSet3 = lldb_private::RegisterLayoutDarwin_riscv32::CSRRegSet3,
    CSRRegSet4 = lldb_private::RegisterLayoutDarwin_riscv32::CSRRegSet4,
    CSRRegSet = lldb_private::RegisterLayoutDarwin_riscv32::CSRRegSet
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
};

#endif // LLDB_SOURCE_PLUGINS_REGISTER_DARWIN_REGISTERCONTEXTDARWIN_RISCV32_H
