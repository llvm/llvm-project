//===-- NativeRegisterContextAIX.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef lldb_NativeRegisterContextAIX_h
#define lldb_NativeRegisterContextAIX_h

#include "Plugins/Process/Utility/NativeRegisterContextRegisterInfo.h"

namespace lldb_private {
namespace process_aix {

class NativeThreadAIX;

class NativeRegisterContextAIX
    : public virtual NativeRegisterContextRegisterInfo {
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

  virtual size_t GetGPRSize() = 0;

  virtual void *GetFPRBuffer() = 0;

  virtual size_t GetFPRSize() = 0;

  // The Do*** functions are executed on the privileged thread and can perform
  // ptrace operations directly.
  virtual Status DoReadRegisterValue(uint32_t offset, const char *reg_name,
                                     uint32_t size, RegisterValue &value);

  virtual Status DoWriteRegisterValue(uint32_t offset, const char *reg_name,
                                      const RegisterValue &value);
};

} // namespace process_aix
} // namespace lldb_private

#endif // #ifndef lldb_NativeRegisterContextAIX_h
