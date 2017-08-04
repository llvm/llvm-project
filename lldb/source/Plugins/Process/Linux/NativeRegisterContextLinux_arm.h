//===-- NativeRegisterContextLinux_arm.h ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#if defined(__arm__) || defined(__arm64__) || defined(__aarch64__)

#ifndef lldb_NativeRegisterContextLinux_arm_h
#define lldb_NativeRegisterContextLinux_arm_h

#include "Plugins/Process/Linux/NativeRegisterContextLinux.h"
#include "Plugins/Process/Utility/lldb-arm-register-enums.h"

namespace lldb_private {
namespace process_linux {

class NativeProcessLinux;

class NativeRegisterContextLinux_arm : public NativeRegisterContextLinux {
public:
  NativeRegisterContextLinux_arm(const ArchSpec &target_arch,
                                 NativeThreadProtocol &native_thread,
                                 uint32_t concrete_frame_idx);

  uint32_t GetRegisterSetCount() const override;

  const RegisterSet *GetRegisterSet(uint32_t set_index) const override;

  uint32_t GetUserRegisterCount() const override;

  Error ReadRegister(const RegisterInfo *reg_info,
                     RegisterValue &reg_value) override;

  Error WriteRegister(const RegisterInfo *reg_info,
                      const RegisterValue &reg_value) override;

  Error ReadAllRegisterValues(lldb::DataBufferSP &data_sp) override;

  Error WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

  //------------------------------------------------------------------
  // Hardware breakpoints/watchpoint mangement functions
  //------------------------------------------------------------------

  uint32_t SetHardwareBreakpoint(lldb::addr_t addr, size_t size) override;

  bool ClearHardwareBreakpoint(uint32_t hw_idx) override;

  uint32_t NumSupportedHardwareWatchpoints() override;

  uint32_t SetHardwareWatchpoint(lldb::addr_t addr, size_t size,
                                 uint32_t watch_flags) override;

  bool ClearHardwareWatchpoint(uint32_t hw_index) override;

  Error ClearAllHardwareWatchpoints() override;

  Error GetWatchpointHitIndex(uint32_t &wp_index,
                              lldb::addr_t trap_addr) override;

  lldb::addr_t GetWatchpointHitAddress(uint32_t wp_index) override;

  lldb::addr_t GetWatchpointAddress(uint32_t wp_index) override;

  uint32_t GetWatchpointSize(uint32_t wp_index);

  bool WatchpointIsEnabled(uint32_t wp_index);

  // Debug register type select
  enum DREGType { eDREGTypeWATCH = 0, eDREGTypeBREAK };

protected:
  Error DoReadRegisterValue(uint32_t offset, const char *reg_name,
                            uint32_t size, RegisterValue &value) override;

  Error DoWriteRegisterValue(uint32_t offset, const char *reg_name,
                             const RegisterValue &value) override;

  Error DoReadGPR(void *buf, size_t buf_size) override;

  Error DoWriteGPR(void *buf, size_t buf_size) override;

  Error DoReadFPR(void *buf, size_t buf_size) override;

  Error DoWriteFPR(void *buf, size_t buf_size) override;

  void *GetGPRBuffer() override { return &m_gpr_arm; }

  void *GetFPRBuffer() override { return &m_fpr; }

  size_t GetFPRSize() override { return sizeof(m_fpr); }

private:
  struct RegInfo {
    uint32_t num_registers;
    uint32_t num_gpr_registers;
    uint32_t num_fpr_registers;

    uint32_t last_gpr;
    uint32_t first_fpr;
    uint32_t last_fpr;

    uint32_t first_fpr_v;
    uint32_t last_fpr_v;

    uint32_t gpr_flags;
  };

  struct QReg {
    uint8_t bytes[16];
  };

  struct FPU {
    union {
      uint32_t s[32];
      uint64_t d[32];
      QReg q[16]; // the 128-bit NEON registers
    } floats;
    uint32_t fpscr;
  };

  uint32_t m_gpr_arm[k_num_gpr_registers_arm];
  RegInfo m_reg_info;
  FPU m_fpr;

  // Debug register info for hardware breakpoints and watchpoints management.
  struct DREG {
    lldb::addr_t address;  // Breakpoint/watchpoint address value.
    lldb::addr_t hit_addr; // Address at which last watchpoint trigger exception
                           // occurred.
    lldb::addr_t real_addr; // Address value that should cause target to stop.
    uint32_t control;       // Breakpoint/watchpoint control value.
    uint32_t refcount;      // Serves as enable/disable and refernce counter.
  };

  struct DREG m_hbr_regs[16]; // Arm native linux hardware breakpoints
  struct DREG m_hwp_regs[16]; // Arm native linux hardware watchpoints

  uint32_t m_max_hwp_supported;
  uint32_t m_max_hbp_supported;
  bool m_refresh_hwdebug_info;

  bool IsGPR(unsigned reg) const;

  bool IsFPR(unsigned reg) const;

  Error ReadHardwareDebugInfo();

  Error WriteHardwareDebugRegs(int hwbType, int hwb_index);

  uint32_t CalculateFprOffset(const RegisterInfo *reg_info) const;
};

} // namespace process_linux
} // namespace lldb_private

#endif // #ifndef lldb_NativeRegisterContextLinux_arm_h

#endif // defined(__arm__) || defined(__arm64__) || defined(__aarch64__)
