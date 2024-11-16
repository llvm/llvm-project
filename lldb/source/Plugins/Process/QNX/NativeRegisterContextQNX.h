//===-- NativeRegisterContextQNX.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef lldb_NativeRegisterContextQNX_h
#define lldb_NativeRegisterContextQNX_h

#include "lldb/Host/common/NativeThreadProtocol.h"

#include "Plugins/Process/Utility/NativeRegisterContextRegisterInfo.h"

namespace lldb_private {
namespace process_qnx {

class NativeThreadQNX;

class NativeRegisterContextQNX
    : public virtual NativeRegisterContextRegisterInfo {
public:
  // This function is implemented in the NativeRegisterContextQNX_*
  // subclasses to create a new instance of the host specific
  // NativeRegisterContextQNX. The implementations can't collide as only one
  // NativeRegisterContextQNX_* variant should be compiled into the final
  // executable.
  static NativeRegisterContextQNX *
  CreateHostNativeRegisterContextQNX(const ArchSpec &target_arch,
                                     NativeThreadProtocol &native_thread);
};

} // namespace process_qnx
} // namespace lldb_private

#endif // #ifndef lldb_NativeRegisterContextQNX_h
