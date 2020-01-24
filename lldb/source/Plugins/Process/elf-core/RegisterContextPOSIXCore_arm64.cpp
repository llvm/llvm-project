//===-- RegisterContextPOSIXCore_arm64.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextPOSIXCore_arm64.h"

#include "Plugins/Process/elf-core/RegisterUtilities.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/RegisterValue.h"

#include <memory>

using namespace lldb_private;

RegisterContextCorePOSIX_arm64::RegisterContextCorePOSIX_arm64(
    Thread &thread, RegisterInfoInterface *register_info,
    const DataExtractor &gpregset, llvm::ArrayRef<CoreNote> notes)
    : RegisterContextPOSIX_arm64(thread, 0, register_info) {
  m_gpr_buffer = std::make_shared<DataBufferHeap>(gpregset.GetDataStart(),
                                                  gpregset.GetByteSize());
  m_gpr.SetData(m_gpr_buffer);
  m_gpr.SetByteOrder(gpregset.GetByteOrder());
}

RegisterContextCorePOSIX_arm64::~RegisterContextCorePOSIX_arm64() {}

bool RegisterContextCorePOSIX_arm64::ReadGPR() { return true; }

bool RegisterContextCorePOSIX_arm64::ReadFPR() { return false; }

bool RegisterContextCorePOSIX_arm64::WriteGPR() {
  assert(0);
  return false;
}

bool RegisterContextCorePOSIX_arm64::WriteFPR() {
  assert(0);
  return false;
}

bool RegisterContextCorePOSIX_arm64::ReadRegister(const RegisterInfo *reg_info,
                                                  RegisterValue &value) {
  lldb::offset_t offset = reg_info->byte_offset;
  uint64_t v = m_gpr.GetMaxU64(&offset, reg_info->byte_size);
  if (offset == reg_info->byte_offset + reg_info->byte_size) {
    value = v;
    return true;
  }
  return false;
}

bool RegisterContextCorePOSIX_arm64::ReadAllRegisterValues(
    lldb::DataBufferSP &data_sp) {
  return false;
}

bool RegisterContextCorePOSIX_arm64::WriteRegister(const RegisterInfo *reg_info,
                                                   const RegisterValue &value) {
  return false;
}

bool RegisterContextCorePOSIX_arm64::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  return false;
}

bool RegisterContextCorePOSIX_arm64::HardwareSingleStep(bool enable) {
  return false;
}
