//===-- RegisterContextCorePOSIX_x86_64.cpp ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RegisterContextPOSIXCore_x86_64.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Target/Thread.h"

using namespace lldb_private;

RegisterContextCorePOSIX_x86_64::RegisterContextCorePOSIX_x86_64(
    Thread &thread, RegisterInfoInterface *register_info,
    const DataExtractor &gpregset, const DataExtractor &fpregset)
    : RegisterContextPOSIX_x86(thread, 0, register_info) {
  size_t size, len;

  size = GetGPRSize();
  m_gpregset = new uint8_t[size];
  len = gpregset.ExtractBytes(0, size, lldb::eByteOrderLittle, m_gpregset);
  assert(len == size);
}

RegisterContextCorePOSIX_x86_64::~RegisterContextCorePOSIX_x86_64() {
  delete[] m_gpregset;
}

bool RegisterContextCorePOSIX_x86_64::ReadGPR() { return m_gpregset != NULL; }

bool RegisterContextCorePOSIX_x86_64::ReadFPR() { return false; }

bool RegisterContextCorePOSIX_x86_64::WriteGPR() {
  assert(0);
  return false;
}

bool RegisterContextCorePOSIX_x86_64::WriteFPR() {
  assert(0);
  return false;
}

bool RegisterContextCorePOSIX_x86_64::ReadRegister(const RegisterInfo *reg_info,
                                                   RegisterValue &value) {
  switch (reg_info->byte_size) {
  case 4:
    value = *(uint32_t *)(m_gpregset + reg_info->byte_offset);
    return true;
  case 8:
    value = *(uint64_t *)(m_gpregset + reg_info->byte_offset);
    return true;
  }
  return false;
}

bool RegisterContextCorePOSIX_x86_64::ReadAllRegisterValues(
    lldb::DataBufferSP &data_sp) {
  return false;
}

bool RegisterContextCorePOSIX_x86_64::WriteRegister(
    const RegisterInfo *reg_info, const RegisterValue &value) {
  return false;
}

bool RegisterContextCorePOSIX_x86_64::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  return false;
}

bool RegisterContextCorePOSIX_x86_64::HardwareSingleStep(bool enable) {
  return false;
}
