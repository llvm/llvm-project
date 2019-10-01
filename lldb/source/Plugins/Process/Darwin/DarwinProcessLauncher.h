//===-- DarwinProcessLauncher.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DarwinProcessLauncher_h
#define DarwinProcessLauncher_h

// C headers
#include <mach/machine.h>
#include <sys/types.h>

// C++ headers
#include <functional>

// LLDB headers
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"

#include "LaunchFlavor.h"

namespace lldb_private {
namespace darwin_process_launcher {
// =============================================================================
/// Launches a process for debugging.
///
/// \param[inout] launch_info
///     Specifies details about the process to launch (e.g. path, architecture,
///     etc.).  On output, includes the launched ProcessID (pid).
///
/// \param[out] pty_master_fd
///     Returns the master side of the pseudo-terminal used to communicate
///     with stdin/stdout from the launched process.  May be nullptr.
///
/// \param[out] launch_flavor
///     Contains the launch flavor used when launching the process.
// =============================================================================
Status
LaunchInferior(ProcessLaunchInfo &launch_info, int *pty_master_fd,
               lldb_private::process_darwin::LaunchFlavor *launch_flavor);

} // darwin_process_launcher
} // lldb_private

#endif /* DarwinProcessLauncher_h */
