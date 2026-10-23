//===-- PthdbWrapper.h ----------------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Wraps the libpthdebug callback table and session-management helpers used
/// by NativeProcessAIX to debug pthreaded AIX inferiors.
///
/// The nine pthdb_callbacks_t functions are implemented in PthdbWrapper.cpp
/// and receive a NativeProcessAIX* packed into the opaque pthdb_user_t handle.
/// NativeProcessAIX calls EnsurePthdbSession() to lazily initialise (and
/// optionally refresh) the session on each stop event.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_PROCESS_AIX_PTHDBWRAPPER_H
#define LLDB_PLUGINS_PROCESS_AIX_PTHDBWRAPPER_H

#include <sys/pthdebug.h>

namespace lldb_private {
namespace process_aix {

class NativeProcessAIX;

/// The single, process-wide pthdb_callbacks_t table passed to
/// pthdb_session_init() and pthdb_session_pthreaded().
extern pthdb_callbacks_t g_pthdb_callbacks;

/// Lazily initialise (and optionally update) the libpthdebug session for
/// \p process.  Mirrors pthreads_active() from pthreaded.c.
///
/// \param process  The owning NativeProcessAIX instance.
/// \param update   When true, also call pthdb_session_update() if the session
///                 is already active.
/// \returns        true when the session is active after the call.
bool EnsurePthdbSession(NativeProcessAIX &process, bool update);

} // namespace process_aix
} // namespace lldb_private

#endif // LLDB_PLUGINS_PROCESS_AIX_PTHDBWRAPPER_H
