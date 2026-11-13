//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_PROCESS_MINIDUMP_REGISTERCONTEXTMINIDUMP_X86_H
#define LLDB_PLUGINS_PROCESS_MINIDUMP_REGISTERCONTEXTMINIDUMP_X86_H
#include "lldb/Register/Common/RegisterInfo_x86.h"
#include "lldb/Target/ThreadRegisterContext.h"
namespace lldb_private::minidump {
llvm::Expected<RegisterLayout>
CreateMinidumpX86RegisterLayout(const ArchSpec &arch);
llvm::Expected<RegisterValue>
ReadMinidumpX86Register(const ArchSpec &arch, const RegisterLayout &layout,
                        llvm::ArrayRef<uint8_t> context, RegisterID id);
class RegisterContextMinidump_x86 : public ThreadRegisterContext {
public:
  RegisterContextMinidump_x86(Thread &thread, const ArchSpec &arch,
                              RegisterLayout layout,
                              llvm::ArrayRef<uint8_t> context);
  llvm::Expected<lldb_private::RegisterValue>
  ReadRegister(lldb_private::RegisterID id) override;
  llvm::Error WriteRegister(RegisterID, const RegisterValue &) override {
    return llvm::createStringError("minidump registers are read-only");
  }
  bool ReadAllRegisterValues(lldb::WritableDataBufferSP &) override {
    return false;
  }
  bool WriteAllRegisterValues(const lldb::DataBufferSP &) override {
    return false;
  }
  bool HardwareSingleStep(bool) override { return false; }

private:
  ArchSpec m_arch;
  std::vector<uint8_t> m_context;
};
} // namespace lldb_private::minidump
#endif
