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

} // namespace lldb_dap

#endif
