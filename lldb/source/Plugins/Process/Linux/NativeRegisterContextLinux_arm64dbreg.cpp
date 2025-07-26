#include "NativeRegisterContextLinux_arm64dbreg.h"

using namespace lldb_private::process_linux::arm64;

namespace lldb_private {
namespace process_linux {
namespace arm64 {

namespace {
Status ReadHardwareDebugInfoHelper(int regset, ::pid_t tid,
                                   uint32_t &max_supported) {
  struct iovec ioVec;
  struct user_hwdebug_state dreg_state;
  Status error;

  ioVec.iov_base = &dreg_state;
  ioVec.iov_len = sizeof(dreg_state);
  error = NativeProcessLinux::PtraceWrapper(PTRACE_GETREGSET, tid, &regset,
                                            &ioVec, ioVec.iov_len);

  if (error.Fail())
    return error;

  max_supported = dreg_state.dbg_info & 0xff;
  return error;
}
} // namespace

Status ReadHardwareDebugInfo(::pid_t tid, uint32_t &max_hwp_supported,
                             uint32_t &max_hbp_supported) {
  Status error =
      ReadHardwareDebugInfoHelper(NT_ARM_HW_WATCH, tid, max_hwp_supported);

  if (error.Fail())
    return error;

  return ReadHardwareDebugInfoHelper(NT_ARM_HW_BREAK, tid, max_hbp_supported);
}

Status WriteHardwareDebugRegs(
    int hwbType, ::pid_t tid, uint32_t max_supported,
    const std::array<NativeRegisterContextDBReg::DREG, 16> &regs) {
  struct iovec ioVec;
  struct user_hwdebug_state dreg_state;
  int regset = hwbType == NativeRegisterContextDBReg::eDREGTypeWATCH
                   ? NT_ARM_HW_WATCH
                   : NT_ARM_HW_BREAK;
  memset(&dreg_state, 0, sizeof(dreg_state));
  ioVec.iov_base = &dreg_state;
  ioVec.iov_len = sizeof(dreg_state.dbg_info) + sizeof(dreg_state.pad) +
                  (sizeof(dreg_state.dbg_regs[0]) * max_supported);
  for (uint32_t i = 0; i < max_supported; i++) {
    dreg_state.dbg_regs[i].addr = regs[i].address;
    dreg_state.dbg_regs[i].ctrl = regs[i].control;
  }

  return NativeProcessLinux::PtraceWrapper(PTRACE_SETREGSET, tid, &regset,
                                           &ioVec, ioVec.iov_len);
}

} // namespace arm64
} // namespace process_linux
} // namespace lldb_private
