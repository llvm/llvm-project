//===-- RegisterContextPOSIX_riscv32.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextPOSIX_riscv32.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Scalar.h"
#include "llvm/Support/Compiler.h"

using namespace lldb;
using namespace lldb_private;

RegisterContextPOSIX_riscv32::RegisterContextPOSIX_riscv32(
    lldb_private::Thread &thread,
    std::unique_ptr<RegisterInfoPOSIX_riscv32> register_info)
    : lldb_private::RegisterContext(thread, 0),
      m_register_info_up(std::move(register_info)) {}

RegisterContextPOSIX_riscv32::~RegisterContextPOSIX_riscv32() = default;

void RegisterContextPOSIX_riscv32::invalidate() {}

void RegisterContextPOSIX_riscv32::InvalidateAllRegisters() {}

size_t RegisterContextPOSIX_riscv32::GetRegisterCount() {
  return m_register_info_up->GetRegisterCount();
}

size_t RegisterContextPOSIX_riscv32::GetGPRSize() {
  return m_register_info_up->GetGPRSize();
}

unsigned RegisterContextPOSIX_riscv32::GetRegisterSize(unsigned int reg) {
  return m_register_info_up->GetRegisterInfo()[reg].byte_size;
}

unsigned RegisterContextPOSIX_riscv32::GetRegisterOffset(unsigned int reg) {
  return m_register_info_up->GetRegisterInfo()[reg].byte_offset;
}

const lldb_private::RegisterInfo *
RegisterContextPOSIX_riscv32::GetRegisterInfoAtIndex(size_t reg) {
  if (reg < GetRegisterCount())
    return &GetRegisterInfo()[reg];

  return nullptr;
}

size_t RegisterContextPOSIX_riscv32::GetRegisterSetCount() {
  return m_register_info_up->GetRegisterCount();
}

const lldb_private::RegisterSet *
RegisterContextPOSIX_riscv32::GetRegisterSet(size_t set) {
  return m_register_info_up->GetRegisterSet(set);
}

const lldb_private::RegisterInfo *
RegisterContextPOSIX_riscv32::GetRegisterInfo() {
  return m_register_info_up->GetRegisterInfo();
}

bool RegisterContextPOSIX_riscv32::IsGPR(unsigned int reg) {
  return m_register_info_up->GetRegisterSetFromRegisterIndex(reg) ==
         RegisterInfoPOSIX_riscv32::eRegsetMaskDefault;
}

bool RegisterContextPOSIX_riscv32::IsFPR(unsigned int reg) {
  return m_register_info_up->GetRegisterSetFromRegisterIndex(reg) ==
         RegisterInfoPOSIX_riscv32::eRegsetMaskFP;
}
