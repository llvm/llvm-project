//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_PLUGINS_ACCELERATOR_MOCK_REGISTERCONTEXTMOCKACCELERATOR_H
#define LLDB_TOOLS_LLDB_SERVER_PLUGINS_ACCELERATOR_MOCK_REGISTERCONTEXTMOCKACCELERATOR_H

#include "lldb/Host/common/NativeRegisterContext.h"

#include <array>

namespace lldb_private {
namespace lldb_server {

/// A minimal register context for the mock accelerator process: a small, fixed
/// register set with constant values, just enough to debug it over GDB remote.
class RegisterContextMockAccelerator : public NativeRegisterContext {
public:
  RegisterContextMockAccelerator(NativeThreadProtocol &native_thread);

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

private:
  /// The registers, indexed by LLDB register number. Each one is 64 bits.
  std::array<uint64_t, 6> m_regs;
};

} // namespace lldb_server
} // namespace lldb_private

#endif // LLDB_TOOLS_LLDB_SERVER_PLUGINS_ACCELERATOR_MOCK_REGISTERCONTEXTMOCKACCELERATOR_H
