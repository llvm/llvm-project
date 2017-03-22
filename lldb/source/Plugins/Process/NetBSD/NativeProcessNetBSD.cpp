//===-- NativeProcessNetBSD.cpp ------------------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "NativeProcessNetBSD.h"

// C Includes

// C++ Includes

// Other libraries and framework includes

#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"

// System includes - They have to be included after framework includes because
// they define some
// macros which collide with variable names in other modules

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_netbsd;
using namespace llvm;

// -----------------------------------------------------------------------------
// Public Static Methods
// -----------------------------------------------------------------------------

Error NativeProcessProtocol::Launch(
    ProcessLaunchInfo &launch_info,
    NativeProcessProtocol::NativeDelegate &native_delegate, MainLoop &mainloop,
    NativeProcessProtocolSP &native_process_sp) {
  return Error();
}

Error NativeProcessProtocol::Attach(
    lldb::pid_t pid, NativeProcessProtocol::NativeDelegate &native_delegate,
    MainLoop &mainloop, NativeProcessProtocolSP &native_process_sp) {
  return Error();
}

// -----------------------------------------------------------------------------
// Public Instance Methods
// -----------------------------------------------------------------------------

NativeProcessNetBSD::NativeProcessNetBSD()
    : NativeProcessProtocol(LLDB_INVALID_PROCESS_ID) {}
