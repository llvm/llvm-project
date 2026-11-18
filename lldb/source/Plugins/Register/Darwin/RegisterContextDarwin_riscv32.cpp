//===-- RegisterContextDarwin_riscv32.cpp
//------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/RegisterValue.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Scalar.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Compiler.h"

#include <cstddef>

#include <memory>

#include "RegisterContextDarwin_riscv32.h"

using namespace lldb;
using namespace lldb_private;

#define REG_CONTEXT_SIZE                                                       \
  (sizeof(RegisterContextDarwin_riscv32::GPR) +                                \
   sizeof(RegisterContextDarwin_riscv32::FPU) +                                \
   sizeof(RegisterContextDarwin_riscv32::EXC) +                                \
   sizeof(RegisterContextDarwin_riscv32::CSR))
/* clang-format on */

RegisterContextDarwin_riscv32::RegisterContextDarwin_riscv32(
    Thread &thread, uint32_t concrete_frame_idx)
    : ThreadRegisterContext(thread, concrete_frame_idx), gpr(), fpr(), exc() {
  SetRegisterLayout(RegisterLayoutDarwin_riscv32::GetShared());

  uint32_t i;
  for (i = 0; i < kNumErrors; i++) {
    gpr_errs[i] = -1;
    fpr_errs[i] = -1;
    exc_errs[i] = -1;
    csr_errs[i] = -1;
  }
}

RegisterContextDarwin_riscv32::~RegisterContextDarwin_riscv32() = default;

void RegisterContextDarwin_riscv32::InvalidateAllRegisters() {
  InvalidateAllRegisterStates();
}

// General purpose registers
void RegisterContextDarwin_riscv32::LogGPR(Log *log, const char *title) {
  if (log) {
    if (title)
      LLDB_LOGF(log, "%s", title);
    const RegisterData bank = {
        GPRRegSet, {reinterpret_cast<const uint8_t *>(&gpr), sizeof(gpr)}};
    for (RegisterID i = 0; i != GPRWordCount; ++i) {
      const auto &reg = *GetRegisterAtIndex(i);
      if (auto value = GetLayout().Read(reg.id, bank))
        LLDB_LOGF(log, "%12s = 0x%4.4x", reg.name.c_str(),
                  value->GetAsUInt32());
      else
        llvm::consumeError(value.takeError());
    }
  }
}

int RegisterContextDarwin_riscv32::ReadGPR(bool force) {
  int set = GPRRegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadGPR(GetThreadID(), set, gpr));
  }
  return GetError(set, Read);
}

int RegisterContextDarwin_riscv32::ReadFPU(bool force) {
  int set = FPURegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadFPU(GetThreadID(), set, fpr));
  }
  return GetError(set, Read);
}

int RegisterContextDarwin_riscv32::ReadEXC(bool force) {
  int set = EXCRegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadEXC(GetThreadID(), set, exc));
  }
  return GetError(set, Read);
}

int RegisterContextDarwin_riscv32::ReadCSR(bool force) {
  int set = CSRRegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadCSR(GetThreadID(), set, csr));
  }
  return GetError(set, Read);
}

int RegisterContextDarwin_riscv32::WriteGPR() {
  int set = GPRRegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return -1;
  }
  SetError(set, Write, DoWriteGPR(GetThreadID(), set, gpr));
  SetError(set, Read, -1);
  return GetError(set, Write);
}

int RegisterContextDarwin_riscv32::WriteFPU() {
  int set = FPURegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return -1;
  }
  SetError(set, Write, DoWriteFPU(GetThreadID(), set, fpr));
  SetError(set, Read, -1);
  return GetError(set, Write);
}

int RegisterContextDarwin_riscv32::WriteEXC() {
  int set = EXCRegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return -1;
  }
  SetError(set, Write, DoWriteEXC(GetThreadID(), set, exc));
  SetError(set, Read, -1);
  return GetError(set, Write);
}

int RegisterContextDarwin_riscv32::WriteCSR() {
  int set = CSRRegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return -1;
  }
  SetError(set, Write, DoWriteCSR(GetThreadID(), set, csr));
  SetError(set, Read, -1);
  return GetError(set, Write);
}

int RegisterContextDarwin_riscv32::ReadRegisterSet(uint32_t set, bool force) {
  switch (set) {
  case GPRRegSet:
    return ReadGPR(force);
  case FPURegSet:
    return ReadFPU(force);
  case EXCRegSet:
    return ReadEXC(force);
  case CSRRegSet:
    return ReadCSR(force);
  default:
    break;
  }
  return -1;
}

int RegisterContextDarwin_riscv32::WriteRegisterSet(uint32_t set) {
  // Make sure we have a valid context to set.
  if (RegisterSetIsCached(set)) {
    switch (set) {
    case GPRRegSet:
      return WriteGPR();
    case FPURegSet:
      return WriteFPU();
    case EXCRegSet:
      return WriteEXC();
    case CSRRegSet:
      return WriteCSR();
    default:
      break;
    }
  }
  return -1;
}

llvm::Expected<RegisterValue>
RegisterContextDarwin_riscv32::ReadRegister(RegisterID id) {
  if (!GetDescription()->Find(id))
    return llvm::createStringError("unknown register ID");

  auto required = GetLayout().GetRequiredBanks(id);
  if (!required)
    return required.takeError();
  for (RegisterBankID bank : *required)
    if (ReadRegisterSet(bank, false) != 0)
      return llvm::createStringError("register value is unavailable");

  const RegisterData banks[] = {
      {GPRRegSet, {reinterpret_cast<const uint8_t *>(&gpr), sizeof(gpr)}},
      {FPURegSet, {reinterpret_cast<const uint8_t *>(&fpr), sizeof(fpr)}},
      {EXCRegSet, {reinterpret_cast<const uint8_t *>(&exc), sizeof(exc)}},
      {CSRRegSet, {reinterpret_cast<const uint8_t *>(&csr), sizeof(csr)}},
  };
  return GetLayout().Read(id, banks);
}

llvm::Error
RegisterContextDarwin_riscv32::WriteRegister(RegisterID id,
                                             const RegisterValue &value) {
  if (!GetDescription()->Find(id))
    return llvm::createStringError("unknown register ID");

  auto required = GetLayout().GetRequiredBanks(id, true);
  if (!required)
    return required.takeError();
  for (RegisterBankID bank : *required)
    if (ReadRegisterSet(bank, false) != 0)
      return llvm::createStringError("register write is unavailable");

  const MutableRegisterData banks[] = {
      {GPRRegSet, {reinterpret_cast<uint8_t *>(&gpr), sizeof(gpr)}},
      {FPURegSet, {reinterpret_cast<uint8_t *>(&fpr), sizeof(fpr)}},
      {EXCRegSet, {reinterpret_cast<uint8_t *>(&exc), sizeof(exc)}},
      {CSRRegSet, {reinterpret_cast<uint8_t *>(&csr), sizeof(csr)}},
  };
  if (auto error = GetLayout().Write(id, value, banks))
    return error;

  for (RegisterBankID bank : *required)
    if (WriteRegisterSet(bank) != 0)
      return llvm::createStringError("register write failed");

  return llvm::Error::success();
}

bool RegisterContextDarwin_riscv32::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  data_sp = std::make_shared<DataBufferHeap>(REG_CONTEXT_SIZE, 0);
  if (ReadGPR(false) == 0 && ReadFPU(false) == 0 && ReadEXC(false) == 0 &&
      ReadCSR(false) == 0) {
    uint8_t *dst = data_sp->GetBytes();
    ::memcpy(dst, &gpr, sizeof(gpr));
    dst += sizeof(gpr);

    ::memcpy(dst, &fpr, sizeof(fpr));
    dst += sizeof(gpr);

    ::memcpy(dst, &exc, sizeof(exc));
    return true;

    ::memcpy(dst, &csr, sizeof(csr));
    return true;
  }
  return false;
}

bool RegisterContextDarwin_riscv32::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  if (data_sp && data_sp->GetByteSize() == REG_CONTEXT_SIZE) {
    const uint8_t *src = data_sp->GetBytes();
    ::memcpy(&gpr, src, sizeof(gpr));
    src += sizeof(gpr);

    ::memcpy(&fpr, src, sizeof(fpr));
    src += sizeof(fpr);

    ::memcpy(&exc, src, sizeof(exc));
    src += sizeof(exc);

    ::memcpy(&csr, src, sizeof(csr));
    uint32_t success_count = 0;

    if (WriteGPR() == 0)
      ++success_count;
    if (WriteFPU() == 0)
      ++success_count;
    if (WriteEXC() == 0)
      ++success_count;
    if (WriteCSR() == 0)
      ++success_count;
    return success_count == 3;
  }
  return false;
}
