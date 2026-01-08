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

/// Converts a LLDB module to a DAP protocol module for use in `module events or
/// request.
///
/// \param[in] target
///     The target that has the module
///
/// \param[in] module
///     A LLDB module object to convert into a protocol module
///
/// \param[in] id_only
///     Only initialize the module ID in the return type. This is used when
///     sending a "removed" module event.
///
/// \return
///     A `protocol::Module` that follows the formal Module
///     definition outlined by the DAP protocol.
std::optional<protocol::Module> CreateModule(const lldb::SBTarget &target,
                                             lldb::SBModule &module,
                                             bool id_only = false);

/// Create a "Source" JSON object as described in the debug adapter definition.
///
/// \param[in] file
///     The SBFileSpec to use when populating out the "Source" object
///
/// \return
///     An optional "Source" JSON object that follows the formal JSON
///     definition outlined by Microsoft.
std::optional<protocol::Source> CreateSource(const lldb::SBFileSpec &file);

/// Checks if the given source is for assembly code.
bool IsAssemblySource(const protocol::Source &source);

bool DisplayAssemblySource(lldb::SBDebugger &debugger,
                           lldb::SBLineEntry line_entry);

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

/// Converts a size in bytes to a human-readable string format.
///
/// \param[in] debug_size
///     Size of the debug information in bytes (uint64_t).
///
/// \return
///     A string representing the size in a readable format (e.g., "1 KB",
///     "2 MB").
std::string ConvertDebugInfoSizeToString(uint64_t debug_size);

/// Create a protocol Variable for the given value.
///
/// \param[in] v
///     The LLDB value to use when populating out the "Variable"
///     object.
///
/// \param[in] var_ref
///     The variable reference. Used to identify the value, e.g.
///     in the `variablesReference` or `declarationLocationReference`
///     properties.
///
/// \param[in] format_hex
///     If set to true the variable will be formatted as hex in
///     the "value" key value pair for the value of the variable.
///
/// \param[in] auto_variable_summaries
///     If set to true the variable will create an automatic variable summary.
///
/// \param[in] synthetic_child_debugging
///     Whether to include synthetic children when listing properties of the
///     value.
///
/// \param[in] is_name_duplicated
///     Whether the same variable name appears multiple times within the same
///     context (e.g. locals). This can happen due to shadowed variables in
///     nested blocks.
///
///     As VSCode doesn't render two of more variables with the same name, we
///     apply a suffix to distinguish duplicated variables.
///
/// \param[in] custom_name
///     A provided custom name that is used instead of the SBValue's when
///     creating the JSON representation.
///
/// \return
///     A Variable representing the given value.
protocol::Variable CreateVariable(lldb::SBValue v, int64_t var_ref,
                                  bool format_hex, bool auto_variable_summaries,
                                  bool synthetic_child_debugging,
                                  bool is_name_duplicated,
                                  std::optional<std::string> custom_name = {});

} // namespace lldb_dap

#endif
