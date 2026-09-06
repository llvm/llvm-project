//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cerrno>
#include <cstdint>
#include <cstring>

#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Scalar.h"
#include "llvm/Support/Compiler.h"

#include "RegisterContextCore_arm64.h"

using namespace lldb;
using namespace lldb_private;

bool RegisterContextCore_arm64::IsGPR(unsigned reg) {
  if (m_register_info_up->GetRegisterSetFromRegisterIndex(reg) ==
      RegisterInfo_arm64::GPRegSet)
    return true;
  return false;
}

bool RegisterContextCore_arm64::IsFPR(unsigned reg) {
  if (m_register_info_up->GetRegisterSetFromRegisterIndex(reg) ==
      RegisterInfo_arm64::FPRegSet)
    return true;
  return false;
}

bool RegisterContextCore_arm64::IsSVE(unsigned reg) const {
  return m_register_info_up->IsSVEReg(reg);
}

bool RegisterContextCore_arm64::IsSME(unsigned reg) const {
  return m_register_info_up->IsSMEReg(reg);
}

bool RegisterContextCore_arm64::IsPAuth(unsigned reg) const {
  return m_register_info_up->IsPAuthReg(reg);
}

bool RegisterContextCore_arm64::IsTLS(unsigned reg) const {
  return m_register_info_up->IsTLSReg(reg);
}

bool RegisterContextCore_arm64::IsMTE(unsigned reg) const {
  return m_register_info_up->IsMTEReg(reg);
}

bool RegisterContextCore_arm64::IsFPMR(unsigned reg) const {
  return m_register_info_up->IsFPMRReg(reg);
}

bool RegisterContextCore_arm64::IsGCS(unsigned reg) const {
  return m_register_info_up->IsGCSReg(reg);
}

bool RegisterContextCore_arm64::IsPOE(unsigned reg) const {
  return m_register_info_up->IsPOEReg(reg);
}

RegisterContextCore_arm64::RegisterContextCore_arm64(
    lldb_private::Thread &thread,
    std::unique_ptr<RegisterInfo_arm64> register_info)
    : lldb_private::RegisterContext(thread, 0),
      m_register_info_up(std::move(register_info)) {}

RegisterContextCore_arm64::~RegisterContextCore_arm64() = default;

void RegisterContextCore_arm64::Invalidate() {}

void RegisterContextCore_arm64::InvalidateAllRegisters() {}

unsigned RegisterContextCore_arm64::GetRegisterOffset(unsigned reg) {
  return m_register_info_up->GetRegisterInfo()[reg].byte_offset;
}

unsigned RegisterContextCore_arm64::GetRegisterSize(unsigned reg) {
  return m_register_info_up->GetRegisterInfo()[reg].byte_size;
}

size_t RegisterContextCore_arm64::GetRegisterCount() {
  return m_register_info_up->GetRegisterCount();
}

size_t RegisterContextCore_arm64::GetGPRSize() {
  return m_register_info_up->GetGPRSize();
}

const lldb_private::RegisterInfo *
RegisterContextCore_arm64::GetRegisterInfo() {
  // Commonly, this method is overridden and g_register_infos is copied and
  // specialized. So, use GetRegisterInfo() rather than g_register_infos in
  // this scope.
  return m_register_info_up->GetRegisterInfo();
}

const lldb_private::RegisterInfo *
RegisterContextCore_arm64::GetRegisterInfoAtIndex(size_t reg) {
  if (reg < GetRegisterCount())
    return &GetRegisterInfo()[reg];

  return nullptr;
}

size_t RegisterContextCore_arm64::GetRegisterSetCount() {
  return m_register_info_up->GetRegisterSetCount();
}

const lldb_private::RegisterSet *
RegisterContextCore_arm64::GetRegisterSet(size_t set) {
  return m_register_info_up->GetRegisterSet(set);
}

const char *RegisterContextCore_arm64::GetRegisterName(unsigned reg) {
  if (reg < GetRegisterCount())
    return GetRegisterInfo()[reg].name;
  return nullptr;
}
