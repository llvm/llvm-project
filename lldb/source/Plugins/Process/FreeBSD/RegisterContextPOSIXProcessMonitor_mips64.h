//===-- RegisterContextPOSIXProcessMonitor_mips64.h -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextPOSIXProcessMonitor_mips64_H_
#define liblldb_RegisterContextPOSIXProcessMonitor_mips64_H_

#include "Plugins/Process/Utility/RegisterContextPOSIX_mips64.h"
#include "Plugins/Process/Utility/lldb-mips-freebsd-register-enums.h"
#include "RegisterContextPOSIX.h"

class ProcessMonitor;

class RegisterContextPOSIXProcessMonitor_mips64
    : public RegisterContextPOSIX_mips64,
      public POSIXBreakpointProtocol {
public:
  RegisterContextPOSIXProcessMonitor_mips64(
      lldb_private::Thread &thread, uint32_t concrete_frame_idx,
      lldb_private::RegisterInfoInterface *register_info);

protected:
  bool ReadGPR() override;

  bool ReadFPR() override;

  bool WriteGPR() override;

  bool WriteFPR() override;

  // lldb_private::RegisterContext
  bool ReadRegister(const unsigned reg, lldb_private::RegisterValue &value);

  bool WriteRegister(const unsigned reg,
                     const lldb_private::RegisterValue &value);

  bool ReadRegister(const lldb_private::RegisterInfo *reg_info,
                    lldb_private::RegisterValue &value) override;

  bool WriteRegister(const lldb_private::RegisterInfo *reg_info,
                     const lldb_private::RegisterValue &value) override;

  bool ReadAllRegisterValues(lldb::DataBufferSP &data_sp) override;

  bool WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

  uint32_t SetHardwareWatchpoint(lldb::addr_t addr, size_t size, bool read,
                                 bool write) override;

  bool ClearHardwareWatchpoint(uint32_t hw_index) override;

  bool HardwareSingleStep(bool enable) override;

  // POSIXBreakpointProtocol
  bool UpdateAfterBreakpoint() override;

  unsigned GetRegisterIndexFromOffset(unsigned offset) override;

  bool IsWatchpointHit(uint32_t hw_index) override;

  bool ClearWatchpointHits() override;

  lldb::addr_t GetWatchpointAddress(uint32_t hw_index) override;

  bool IsWatchpointVacant(uint32_t hw_index) override;

  bool SetHardwareWatchpointWithIndex(lldb::addr_t addr, size_t size, bool read,
                                      bool write, uint32_t hw_index) override;

  uint32_t NumSupportedHardwareWatchpoints() override;

private:
  uint64_t 
  m_gpr_mips64[lldb_private::k_num_gpr_registers_mips64]; // general purpose registers.
  ProcessMonitor &GetMonitor();
};

#endif
