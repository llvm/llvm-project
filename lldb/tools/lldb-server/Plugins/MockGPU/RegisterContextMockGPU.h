//===-- RegisterContextMockGPU.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_REGISTERCONTEXTMOCKGPU_H
#define LLDB_TOOLS_LLDB_SERVER_REGISTERCONTEXTMOCKGPU_H

// #include "Plugins/Process/Utility/NativeRegisterContextRegisterInfo.h"
#include "lldb/Host/common/NativeRegisterContext.h"
#include "lldb/lldb-forward.h"

namespace lldb_private {
namespace lldb_server {

class RegisterContextMockGPU : public NativeRegisterContext {
public:
  RegisterContextMockGPU(NativeThreadProtocol &native_thread);

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

    // A storage stucture for all registers;
    struct RegisterContext {
      uint64_t R0;
      uint64_t R1;
      uint64_t R2;
      uint64_t R3;
      uint64_t R4;
      uint64_t R5;
      uint64_t R6;
      uint64_t R7;
      uint64_t SP;
      uint64_t FP;
      uint64_t PC;
      uint64_t Flags;
      uint64_t V0;
      uint64_t V1;
      uint64_t V2;
      uint64_t V3;
      uint64_t V4;
      uint64_t V5;
      uint64_t V6;
      uint64_t V7;
    };
  
private:
  void InitRegisters();
  void InvalidateAllRegisters();
  Status ReadRegs();


  // All mock GPU registers are contained in this buffer.
  union {
    /// Allow for indexed access to each register value.
    uint64_t data[sizeof(RegisterContext)/sizeof(uint64_t)];
    /// Allow for direct access to the register values by name.
    RegisterContext regs;
  } m_regs;
  std::vector<bool> m_reg_value_is_valid;
};

} // namespace lldb_server
} // namespace lldb_private

#endif // #ifndef LLDB_TOOLS_LLDB_SERVER_REGISTERCONTEXTMOCKGPU_H
