//===---- NativeRegisterContextAIX.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NativeRegisterContextAIX.h"
#include "Plugins/Process/AIX/NativeProcessAIX.h"

using namespace lldb_private;
using namespace lldb_private::process_aix;

lldb::ByteOrder NativeRegisterContextAIX::GetByteOrder() const {
  return lldb::eByteOrderInvalid;
}

Status NativeRegisterContextAIX::ReadRegisterRaw(uint32_t reg_index,
                                                 RegisterValue &reg_value) {
  return Status("unimplemented");
}

Status
NativeRegisterContextAIX::WriteRegisterRaw(uint32_t reg_index,
                                           const RegisterValue &reg_value) {
  return Status("unimplemented");
}

Status NativeRegisterContextAIX::ReadGPR() { return Status("unimplemented"); }

Status NativeRegisterContextAIX::WriteGPR() { return Status("unimplemented"); }

Status NativeRegisterContextAIX::ReadFPR() { return Status("unimplemented"); }

Status NativeRegisterContextAIX::WriteFPR() { return Status("unimplemented"); }

Status NativeRegisterContextAIX::ReadVMX() { return Status("unimplemented"); }

Status NativeRegisterContextAIX::WriteVMX() { return Status("unimplemented"); }

Status NativeRegisterContextAIX::ReadVSX() { return Status("unimplemented"); }

Status NativeRegisterContextAIX::WriteVSX() { return Status("unimplemented"); }

Status NativeRegisterContextAIX::ReadRegisterSet(void *buf, size_t buf_size,
                                                 unsigned int regset) {
  return Status("unimplemented");
}

Status NativeRegisterContextAIX::WriteRegisterSet(void *buf, size_t buf_size,
                                                  unsigned int regset) {
  return Status("unimplemented");
}
