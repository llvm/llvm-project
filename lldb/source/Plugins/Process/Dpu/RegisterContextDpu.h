//===-- RegisterContextDpu.h ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_RegisterContextDpu_h
#define lldb_RegisterContextDpu_h

#include "Plugins/Process/Utility/NativeRegisterContextRegisterInfo.h"
#include "Plugins/Process/Utility/lldb-dpu-register-enums.h"
#include "lldb/Host/common/NativeThreadProtocol.h"

namespace lldb_private {
namespace process_dpu {

class ProcessDpu;
class ThreadDpu;

class RegisterContextDpu : public NativeRegisterContextRegisterInfo {
public:
  RegisterContextDpu(ThreadDpu &thread, ProcessDpu &process);

  uint32_t GetRegisterSetCount() const override;

  const RegisterSet *GetRegisterSet(uint32_t set_index) const override;

  uint32_t GetUserRegisterCount() const override;

  Status ReadRegister(const RegisterInfo *reg_info,
                      RegisterValue &reg_value) override;

  Status WriteRegister(const RegisterInfo *reg_info,
                       const RegisterValue &reg_value) override;

  Status ReadAllRegisterValues(lldb::DataBufferSP &data_sp) override;

  Status WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

private:
  uint32_t *m_context_reg;
  uint16_t *m_context_pc;
  bool *m_context_zf;
  bool *m_context_cf;
  bool *m_registers_has_been_modified;
};

} // namespace process_dpu
} // namespace lldb_private

#endif // #ifndef lldb_RegisterContextDpu_h
