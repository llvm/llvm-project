//===-- RegisterContextDarwin_x86_64.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cinttypes>
#include <cstdarg>
#include <cstddef>

#include <memory>

#include "lldb/Register/RegisterValue.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Scalar.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Compiler.h"

#include "RegisterContextDarwin_x86_64.h"

using namespace lldb;
using namespace lldb_private;

#define REG_CONTEXT_SIZE                                                       \
  (sizeof(RegisterContextDarwin_x86_64::GPR) +                                 \
   sizeof(RegisterContextDarwin_x86_64::FPU) +                                 \
   sizeof(RegisterContextDarwin_x86_64::EXC))

// General purpose registers for 64 bit
RegisterContextDarwin_x86_64::RegisterContextDarwin_x86_64(
    Thread &thread, uint32_t concrete_frame_idx)
    : ThreadRegisterContext(thread, concrete_frame_idx), gpr(), fpu(), exc() {
  SetRegisterLayout(RegisterLayoutDarwin_x86_64::GetShared());

  uint32_t i;
  for (i = 0; i < kNumErrors; i++) {
    gpr_errs[i] = -1;
    fpu_errs[i] = -1;
    exc_errs[i] = -1;
  }
}

RegisterContextDarwin_x86_64::~RegisterContextDarwin_x86_64() = default;

void RegisterContextDarwin_x86_64::InvalidateAllRegisters() {
  InvalidateAllRegisterStates();
}

int RegisterContextDarwin_x86_64::ReadGPR(bool force) {
  int set = GPRRegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadGPR(GetThreadID(), set, gpr));
  }
  return GetError(GPRRegSet, Read);
}

int RegisterContextDarwin_x86_64::ReadFPU(bool force) {
  int set = FPURegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadFPU(GetThreadID(), set, fpu));
  }
  return GetError(FPURegSet, Read);
}

int RegisterContextDarwin_x86_64::ReadEXC(bool force) {
  int set = EXCRegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadEXC(GetThreadID(), set, exc));
  }
  return GetError(EXCRegSet, Read);
}

int RegisterContextDarwin_x86_64::WriteGPR() {
  int set = GPRRegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return -1;
  }
  SetError(set, Write, DoWriteGPR(GetThreadID(), set, gpr));
  SetError(set, Read, -1);
  return GetError(set, Write);
}

int RegisterContextDarwin_x86_64::WriteFPU() {
  int set = FPURegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return -1;
  }
  SetError(set, Write, DoWriteFPU(GetThreadID(), set, fpu));
  SetError(set, Read, -1);
  return GetError(set, Write);
}

int RegisterContextDarwin_x86_64::WriteEXC() {
  int set = EXCRegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return -1;
  }
  SetError(set, Write, DoWriteEXC(GetThreadID(), set, exc));
  SetError(set, Read, -1);
  return GetError(set, Write);
}

int RegisterContextDarwin_x86_64::ReadRegisterSet(uint32_t set, bool force) {
  switch (set) {
  case GPRRegSet:
    return ReadGPR(force);
  case FPURegSet:
    return ReadFPU(force);
  case EXCRegSet:
    return ReadEXC(force);
  default:
    break;
  }
  return -1;
}

int RegisterContextDarwin_x86_64::WriteRegisterSet(uint32_t set) {
  // Make sure we have a valid context to set.
  switch (set) {
  case GPRRegSet:
    return WriteGPR();
  case FPURegSet:
    return WriteFPU();
  case EXCRegSet:
    return WriteEXC();
  default:
    break;
  }
  return -1;
}

llvm::Expected<RegisterValue>
RegisterContextDarwin_x86_64::ReadRegister(RegisterID id) {
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
      {FPURegSet, {reinterpret_cast<const uint8_t *>(&fpu), sizeof(fpu)}},
      {EXCRegSet, {reinterpret_cast<const uint8_t *>(&exc), sizeof(exc)}},
  };
  return GetLayout().Read(id, banks);
}

llvm::Error
RegisterContextDarwin_x86_64::WriteRegister(RegisterID id,
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
      {FPURegSet, {reinterpret_cast<uint8_t *>(&fpu), sizeof(fpu)}},
      {EXCRegSet, {reinterpret_cast<uint8_t *>(&exc), sizeof(exc)}},
  };
  if (auto error = GetLayout().Write(id, value, banks))
    return error;

  // Keep the existing policy of rejecting XMM writes before committing the
  // bank.
  if (RegisterLayoutDarwin_x86_64::IsXMM(id))
    return llvm::createStringError("register write is unavailable");

  for (RegisterBankID bank : *required)
    if (WriteRegisterSet(bank) != 0)
      return llvm::createStringError("register write failed");

  return llvm::Error::success();
}

bool RegisterContextDarwin_x86_64::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  data_sp = std::make_shared<DataBufferHeap>(REG_CONTEXT_SIZE, 0);
  if (ReadGPR(false) == 0 && ReadFPU(false) == 0 && ReadEXC(false) == 0) {
    uint8_t *dst = data_sp->GetBytes();
    ::memcpy(dst, &gpr, sizeof(gpr));
    dst += sizeof(gpr);

    ::memcpy(dst, &fpu, sizeof(fpu));
    dst += sizeof(gpr);

    ::memcpy(dst, &exc, sizeof(exc));
    return true;
  }
  return false;
}

bool RegisterContextDarwin_x86_64::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  if (data_sp && data_sp->GetByteSize() == REG_CONTEXT_SIZE) {
    const uint8_t *src = data_sp->GetBytes();
    ::memcpy(&gpr, src, sizeof(gpr));
    src += sizeof(gpr);

    ::memcpy(&fpu, src, sizeof(fpu));
    src += sizeof(gpr);

    ::memcpy(&exc, src, sizeof(exc));
    uint32_t success_count = 0;
    if (WriteGPR() == 0)
      ++success_count;
    if (WriteFPU() == 0)
      ++success_count;
    if (WriteEXC() == 0)
      ++success_count;
    return success_count == 3;
  }
  return false;
}

bool RegisterContextDarwin_x86_64::HardwareSingleStep(bool enable) {
  if (ReadGPR(true) != 0)
    return false;

  const uint64_t trace_bit = 0x100ull;
  if (enable) {

    if (gpr.rflags & trace_bit)
      return true; // trace bit is already set, there is nothing to do
    else
      gpr.rflags |= trace_bit;
  } else {
    if (gpr.rflags & trace_bit)
      gpr.rflags &= ~trace_bit;
    else
      return true; // trace bit is clear, there is nothing to do
  }

  return WriteGPR() == 0;
}
