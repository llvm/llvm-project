//===-- RegisterContextAMDGPU.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_REGISTERCONTEXTAMDGPU_H
#define LLDB_TOOLS_LLDB_SERVER_REGISTERCONTEXTAMDGPU_H

// #include "Plugins/Process/Utility/NativeRegisterContextRegisterInfo.h"
#include "lldb/Host/common/NativeRegisterContext.h"
#include "lldb/lldb-forward.h"

namespace lldb_private {
namespace lldb_server {

class RegisterContextAMDGPU : public NativeRegisterContext {
public:
  RegisterContextAMDGPU(NativeThreadProtocol &native_thread);

  uint32_t GetRegisterCount() const override;

  uint32_t GetUserRegisterCount() const override;

  const RegisterInfo *GetRegisterInfoAtIndex(uint32_t reg) const override;

  uint32_t GetRegisterSetCount() const override;

  const RegisterSet *GetRegisterSet(uint32_t set_index) const override;

  Status ReadRegister(const RegisterInfo *reg_info,
                      RegisterValue &reg_value) override;

  Status WriteRegister(const RegisterInfo *reg_info,
                       const RegisterValue &reg_value) override;

  Status ReadAllRegisterValues(lldb::WritableDataBufferSP &data_sp) override;

  Status WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

  std::vector<uint32_t>
  GetExpeditedRegisters(ExpeditedRegs expType) const override;

  void InvalidateAllRegisters();

private:
  bool InitRegisterInfos();
  void InitRegisters();
  
  Status ReadRegs();
  Status ReadReg(const RegisterInfo *reg_info);

  // All AMD GPU registers are contained in this buffer.
  struct {
    std::vector<uint8_t> data;
  } m_regs;
  std::vector<bool> m_regs_valid;
};

} // namespace lldb_server
} // namespace lldb_private

#endif // #ifndef LLDB_TOOLS_LLDB_SERVER_REGISTERCONTEXTAMDGPU_H
