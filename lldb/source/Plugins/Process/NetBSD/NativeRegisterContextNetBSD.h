//===-- NativeRegisterContextNetBSD.h ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_NativeRegisterContextNetBSD_h
#define lldb_NativeRegisterContextNetBSD_h

#include "lldb/Host/common/NativeThreadProtocol.h"

#include "Plugins/Process/NetBSD/NativeProcessNetBSD.h"
#include "Plugins/Process/Utility/NativeRegisterContextRegisterInfo.h"

namespace lldb_private {
namespace process_netbsd {

class NativeRegisterContextNetBSD : public NativeRegisterContextRegisterInfo {
public:
  NativeRegisterContextNetBSD(NativeThreadProtocol &native_thread,
                              uint32_t concrete_frame_idx,
                              RegisterInfoInterface *reg_info_interface_p);

  // This function is implemented in the NativeRegisterContextNetBSD_*
  // subclasses to create a new instance of the host specific
  // NativeRegisterContextNetBSD. The implementations can't collide as only one
  // NativeRegisterContextNetBSD_* variant should be compiled into the final
  // executable.
  static NativeRegisterContextNetBSD *
  CreateHostNativeRegisterContextNetBSD(const ArchSpec &target_arch,
                                        NativeThreadProtocol &native_thread,
                                        uint32_t concrete_frame_idx);
};

} // namespace process_netbsd
} // namespace lldb_private

#endif // #ifndef lldb_NativeRegisterContextNetBSD_h
