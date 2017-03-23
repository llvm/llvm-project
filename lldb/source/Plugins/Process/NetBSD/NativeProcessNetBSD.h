//===-- NativeProcessNetBSD.h --------------------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_NativeProcessNetBSD_H_
#define liblldb_NativeProcessNetBSD_H_

// C++ Includes

// Other libraries and framework includes

#include "lldb/Core/ArchSpec.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Utility/FileSpec.h"

#include "NativeThreadNetBSD.h"
#include "lldb/Host/common/NativeProcessProtocol.h"

namespace lldb_private {
namespace process_netbsd {
/// @class NativeProcessNetBSD
/// @brief Manages communication with the inferior (debugee) process.
///
/// Upon construction, this class prepares and launches an inferior process for
/// debugging.
///
/// Changes in the inferior process state are broadcasted.
class NativeProcessNetBSD : public NativeProcessProtocol {
  friend Error NativeProcessProtocol::Launch(
      ProcessLaunchInfo &launch_info, NativeDelegate &native_delegate,
      MainLoop &mainloop, NativeProcessProtocolSP &process_sp);

  friend Error NativeProcessProtocol::Attach(
      lldb::pid_t pid, NativeProcessProtocol::NativeDelegate &native_delegate,
      MainLoop &mainloop, NativeProcessProtocolSP &process_sp);

private:
  NativeProcessNetBSD();
};

} // namespace process_netbsd
} // namespace lldb_private

#endif // #ifndef liblldb_NativeProcessNetBSD_H_
