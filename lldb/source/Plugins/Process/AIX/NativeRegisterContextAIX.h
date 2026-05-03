//===---- NativeRegisterContextAIX.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_AIX_NATIVEREGISTERCONTEXTAIX_H
#define LLDB_SOURCE_PLUGINS_PROCESS_AIX_NATIVEREGISTERCONTEXTAIX_H

#include "Plugins/Process/Utility/NativeRegisterContextRegisterInfo.h"

namespace lldb_private::process_aix {

class NativeThreadAIX;

class NativeRegisterContextAIX
    : public virtual NativeRegisterContextRegisterInfo {
public:
  // This function is implemented in the NativeRegisterContextAIX_ppc64
  // subclasses to create a new instance (for both 32-bit or 64-bit) of the host
  // specific NativeRegisterContextAIX.
  // The appropriate implementation is selected at runtime based on the
  // target process architecture, so only the relevant code path is used.
  static std::unique_ptr<NativeRegisterContextAIX>
  CreateHostNativeRegisterContextAIX(const ArchSpec &target_arch,
                                     NativeThreadAIX &native_thread);

protected:
  NativeRegisterContextAIX(NativeThreadProtocol &thread)
      : NativeRegisterContextRegisterInfo(thread, nullptr) {}

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

  virtual Status ReadVMX();

  virtual Status WriteVMX();

  virtual Status ReadVSX();

  virtual Status WriteVSX();

  virtual void *GetGPRBuffer() = 0;

  virtual size_t GetGPRSize() const {
    return GetRegisterInfoInterface().GetGPRSize();
  }

  virtual void *GetFPRBuffer() = 0;

  virtual size_t GetFPRSize() = 0;
};

} // namespace lldb_private::process_aix

#endif // #ifndef LLDB_SOURCE_PLUGINS_PROCESS_AIX_NATIVEREGISTERCONTEXTAIX_H
