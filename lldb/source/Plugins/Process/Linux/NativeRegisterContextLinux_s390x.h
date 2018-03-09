//===-- NativeRegisterContextLinux_s390x.h ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#if defined(__s390x__) && defined(__linux__)

#ifndef lldb_NativeRegisterContextLinux_s390x_h
#define lldb_NativeRegisterContextLinux_s390x_h

#include "Plugins/Process/Linux/NativeRegisterContextLinux.h"
#include "Plugins/Process/Utility/RegisterContext_s390x.h"
#include "Plugins/Process/Utility/lldb-s390x-register-enums.h"

namespace lldb_private {
namespace process_linux {

class NativeProcessLinux;

class NativeRegisterContextLinux_s390x : public NativeRegisterContextLinux {
public:
  NativeRegisterContextLinux_s390x(const ArchSpec &target_arch,
                                   NativeThreadProtocol &native_thread);

  uint32_t GetRegisterSetCount() const override;

  const RegisterSet *GetRegisterSet(uint32_t set_index) const override;

  uint32_t GetUserRegisterCount() const override;

  Status ReadRegister(const RegisterInfo *reg_info,
                      RegisterValue &reg_value) override;

  Status WriteRegister(const RegisterInfo *reg_info,
                       const RegisterValue &reg_value) override;

  Status ReadAllRegisterValues(lldb::DataBufferSP &data_sp) override;

  Status WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

  Status IsWatchpointHit(uint32_t wp_index, bool &is_hit) override;

  Status GetWatchpointHitIndex(uint32_t &wp_index,
                               lldb::addr_t trap_addr) override;

  Status IsWatchpointVacant(uint32_t wp_index, bool &is_vacant) override;

  bool ClearHardwareWatchpoint(uint32_t wp_index) override;

  Status ClearAllHardwareWatchpoints() override;

  uint32_t SetHardwareWatchpoint(lldb::addr_t addr, size_t size,
                                 uint32_t watch_flags) override;

  lldb::addr_t GetWatchpointAddress(uint32_t wp_index) override;

  uint32_t NumSupportedHardwareWatchpoints() override;

protected:
  Status DoReadRegisterValue(uint32_t offset, const char *reg_name,
                             uint32_t size, RegisterValue &value) override;

  Status DoWriteRegisterValue(uint32_t offset, const char *reg_name,
                              const RegisterValue &value) override;

  Status DoReadGPR(void *buf, size_t buf_size) override;

  Status DoWriteGPR(void *buf, size_t buf_size) override;

  Status DoReadFPR(void *buf, size_t buf_size) override;

  Status DoWriteFPR(void *buf, size_t buf_size) override;

private:
  // Info about register ranges.
  struct RegInfo {
    uint32_t num_registers;
    uint32_t num_gpr_registers;
    uint32_t num_fpr_registers;

    uint32_t last_gpr;
    uint32_t first_fpr;
    uint32_t last_fpr;
  };

  // Private member variables.
  RegInfo m_reg_info;
  lldb::addr_t m_watchpoint_addr;

  // Private member methods.
  bool IsRegisterSetAvailable(uint32_t set_index) const;

  bool IsGPR(uint32_t reg_index) const;

  bool IsFPR(uint32_t reg_index) const;

  Status PeekUserArea(uint32_t offset, void *buf, size_t buf_size);

  Status PokeUserArea(uint32_t offset, const void *buf, size_t buf_size);

  Status DoReadRegisterSet(uint32_t regset, void *buf, size_t buf_size);

  Status DoWriteRegisterSet(uint32_t regset, const void *buf, size_t buf_size);
};

} // namespace process_linux
} // namespace lldb_private

#endif // #ifndef lldb_NativeRegisterContextLinux_s390x_h

#endif // defined(__s390x__) && defined(__linux__)
