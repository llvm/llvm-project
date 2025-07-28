#include "Plugins/Process/Linux/NativeProcessLinux.h"
#include "Plugins/Process/Utility/NativeRegisterContextDBReg.h"
#include "lldb/Utility/Status.h"
#include <asm/ptrace.h>
#include <cstdint>
#include <elf.h>
#include <sys/ptrace.h>
#include <sys/uio.h>

namespace lldb_private {
namespace process_linux {
namespace arm64 {

Status ReadHardwareDebugInfo(::pid_t tid, uint32_t &max_hwp_supported,
                             uint32_t &max_hbp_supported);

Status WriteHardwareDebugRegs(
    int hwbType, ::pid_t tid, uint32_t max_supported,
    const std::array<NativeRegisterContextDBReg::DREG, 16> &regs);

} // namespace arm64
} // namespace process_linux
} // namespace lldb_private