//===-- NativeRegisterContextLinux.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

  virtual void *GetGPRBuffer() { return nullptr; }

  virtual size_t GetGPRSize() {
    return GetRegisterInfoInterface().GetGPRSize();
  }

  virtual void *GetFPRBuffer() { return nullptr; }

  virtual size_t GetFPRSize() { return 0; }

  // The Do*** functions are executed on the privileged thread and can perform
  // ptrace
  // operations directly.
  virtual Status DoReadRegisterValue(uint32_t offset, const char *reg_name,
                                     uint32_t size, RegisterValue &value);

  virtual Status DoWriteRegisterValue(uint32_t offset, const char *reg_name,
                                      const RegisterValue &value);

  virtual Status DoReadGPR(void *buf, size_t buf_size);

  virtual Status DoWriteGPR(void *buf, size_t buf_size);

  virtual Status DoReadFPR(void *buf, size_t buf_size);

  virtual Status DoWriteFPR(void *buf, size_t buf_size);
};

} // namespace process_linux
} // namespace lldb_private

#endif // #ifndef lldb_NativeRegisterContextLinux_h
