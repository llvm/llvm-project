//===-- RegisterContextEZH.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextEZH_h_
#define liblldb_RegisterContextEZH_h_

#include "lldb/Target/RegisterContext.h"
#include "lldb/lldb-private.h"
#include "EZHRegisters.h"

class RegisterContextEZH : public lldb_private::RegisterContext {
public:
  RegisterContextEZH(lldb_private::Thread &thread, uint32_t concrete_frame_idx);

  ~RegisterContextEZH() override = default;

  void InvalidateAllRegisters() override { m_reg_values_valid = false; }

  size_t GetRegisterCount() override;

  const lldb_private::RegisterInfo *GetRegisterInfoAtIndex(size_t reg) override;

  size_t GetRegisterSetCount() override { return 1; }

  const lldb_private::RegisterSet *GetRegisterSet(size_t reg_set) override;

  bool ReadRegister(const lldb_private::RegisterInfo *reg_info,
                    lldb_private::RegisterValue &value) override;

  bool WriteRegister(const lldb_private::RegisterInfo *reg_info,
                     const lldb_private::RegisterValue &value) override;

  bool ReadAllRegisterValues(lldb::WritableDataBufferSP &data_sp) override;

  bool WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

  uint32_t ConvertRegisterKindToRegisterNumber(lldb::RegisterKind kind,
                                               uint32_t num) override;

private:
  bool ReadAllRegisterValuesBytes();

  uint32_t m_reg_values[EZH_NUM_REGS] = {};
  bool m_reg_values_valid = false;
};

#endif // liblldb_RegisterContextEZH_h_
