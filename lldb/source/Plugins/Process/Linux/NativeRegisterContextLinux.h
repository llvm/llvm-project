//===-- NativeRegisterContextLinux.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef lldb_NativeRegisterContextLinux_h
#define lldb_NativeRegisterContextLinux_h

#include "Plugins/Process/Utility/NativeRegisterContextRegisterInfo.h"
#include "lldb/Host/common/NativeThreadProtocol.h"

namespace lldb_private {
namespace process_linux {

class NativeRegisterContextLinux : public NativeRegisterContextRegisterInfo {
public:
  NativeRegisterContextLinux(NativeThreadProtocol &native_thread,
                             RegisterInfoInterface *reg_info_interface_p);

  // This function is implemented in the NativeRegisterContextLinux_* subclasses
  // to create a new instance of the host specific NativeRegisterContextLinux.
  // The implementations can't collide as only one NativeRegisterContextLinux_*
  // variant should be compiled into the final executable.
  static std::unique_ptr<NativeRegisterContextLinux>
  CreateHostNativeRegisterContextLinux(const ArchSpec &target_arch,
                                       NativeThreadProtocol &native_thread);

  // Invalidates cached values in register context data structures
  virtual void InvalidateAllRegisters(){}

protected:
  lldb::ByteOrder GetByteOrder() const;

  virtual Status ReadRegisterRaw(uint32_t reg_index, RegisterValue &reg_value);

  virtual Status WriteRegisterRaw(uint32_t reg_index,
                                  const RegisterValue &reg_value);

  virtual Status ReadRegisterSet(void *buf, size_t buf_size,
                                 unsigned int regset);

  virtual Status WriteRegisterSet(void *buf, size_t buf_size,
                                  unsigned int regset);

  virtual Status ReadGPR();

  virtual Status WriteGPR();

  virtual Status ReadFPR();

  virtual Status WriteFPR();

  virtual void *GetGPRBuffer() = 0;

  virtual size_t GetGPRSize() const {
    return GetRegisterInfoInterface().GetGPRSize();
  }

  virtual void *GetFPRBuffer() = 0;

  virtual size_t GetFPRSize() = 0;

  virtual uint32_t GetPtraceOffset(uint32_t reg_index) {
    return GetRegisterInfoAtIndex(reg_index)->byte_offset;
  }

  // The Do*** functions are executed on the privileged thread and can perform
  // ptrace
  // operations directly.
  virtual Status DoReadRegisterValue(uint32_t offset, const char *reg_name,
                                     uint32_t size, RegisterValue &value);

  virtual Status DoWriteRegisterValue(uint32_t offset, const char *reg_name,
                                      const RegisterValue &value);
};

} // namespace process_linux
} // namespace lldb_private

#endif // #ifndef lldb_NativeRegisterContextLinux_h
