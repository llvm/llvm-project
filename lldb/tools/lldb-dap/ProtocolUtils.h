//===-- ProtocolUtils.h ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains Utility function for protocol objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_PROTOCOL_PROTOCOL_UTILS_H
#define LLDB_TOOLS_LLDB_DAP_PROTOCOL_PROTOCOL_UTILS_H

#include "ExceptionBreakpoint.h"
#include "Protocol/ProtocolTypes.h"

#include "lldb/API/SBAddress.h"

namespace lldb_dap {

/// Create a "Source" JSON object as described in the debug adapter definition.
///
/// \param[in] file
///     The SBFileSpec to use when populating out the "Source" object
///
/// \return
///     A "Source" JSON object that follows the formal JSON
///     definition outlined by Microsoft.
protocol::Source CreateSource(const lldb::SBFileSpec &file);

/// Create a "Source" JSON object as described in the debug adapter definition.
///
/// \param[in] address
///     The address to use when populating out the "Source" object.
///
/// \param[in] target
///     The target that has the address.
///
/// \return
///     A "Source" JSON object that follows the formal JSON
///     definition outlined by Microsoft.
protocol::Source CreateSource(lldb::SBAddress address, lldb::SBTarget &target);

/// Checks if the given source is for assembly code.
bool IsAssemblySource(const protocol::Source &source);

/// Get the address as a 16-digit hex string, e.g. "0x0000000000012345"
std::string GetLoadAddressString(const lldb::addr_t addr);

/// Create a "Thread" object for a LLDB thread object.
///
/// This function will fill in the following keys in the returned
/// object:
///   "id" - the thread ID as an integer
///   "name" - the thread name as a string which combines the LLDB
///            thread index ID along with the string name of the thread
///            from the OS if it has a name.
///
/// \param[in] thread
///     The LLDB thread to use when populating out the "Thread"
///     object.
///
/// \param[in] format
///     The LLDB format to use when populating out the "Thread"
///     object.
///
/// \return
///     A "Thread" JSON object with that follows the formal JSON
///     definition outlined by Microsoft.
protocol::Thread CreateThread(lldb::SBThread &thread, lldb::SBFormat &format);

/// Returns the set of threads associated with the process.
std::vector<protocol::Thread> GetThreads(lldb::SBProcess process,
                                         lldb::SBFormat &format);

/// Create a "ExceptionBreakpointsFilter" JSON object as described in
/// the debug adapter definition.
///
/// \param[in] bp
///     The exception breakpoint object to use
///
/// \return
///     A "ExceptionBreakpointsFilter" JSON object with that follows
///     the formal JSON definition outlined by Microsoft.
protocol::ExceptionBreakpointsFilter
CreateExceptionBreakpointFilter(const ExceptionBreakpoint &bp);

} // namespace lldb_dap

#endif
