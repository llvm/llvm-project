//===-- NativeRegisterContextOpenBSD.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_OPENBSD_NATIVEREGISTERCONTEXTOPENBSD_H
#define LLDB_SOURCE_PLUGINS_PROCESS_OPENBSD_NATIVEREGISTERCONTEXTOPENBSD_H

#include "lldb/Host/common/NativeThreadProtocol.h"

#include "Plugins/Process/OpenBSD/NativeProcessOpenBSD.h"
#include "Plugins/Process/Utility/NativeRegisterContextRegisterInfo.h"

namespace lldb_private {
namespace process_openbsd {

class NativeRegisterContextOpenBSD : public NativeRegisterContextRegisterInfo {
public:
  NativeRegisterContextOpenBSD(NativeThreadProtocol &native_thread,
                               RegisterInfoInterface *reg_info_interface_p);

  // This function is implemented in the NativeRegisterContextOpenBSD_*
  // subclasses to create a new instance of the host specific
  // NativeRegisterContextOpenBSD. The implementations can't collide as only one
  // NativeRegisterContextOpenBSD_* variant should be compiled into the final
  // executable.
  static std::unique_ptr<NativeRegisterContextOpenBSD>
  CreateHostNativeRegisterContextOpenBSD(const ArchSpec &target_arch,
                                         NativeThreadProtocol &native_thread);

protected:
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

  virtual Status DoReadGPR(void *buf);
  virtual Status DoWriteGPR(void *buf);

  virtual Status DoReadFPR(void *buf);
  virtual Status DoWriteFPR(void *buf);

  virtual NativeProcessOpenBSD &GetProcess();
  virtual ::pid_t GetProcessPid();
};

} // namespace process_openbsd
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_OPENBSD_NATIVEREGISTERCONTEXTOPENBSD_H
