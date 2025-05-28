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

namespace lldb_dap::protocol {

bool IsAssemblySource(const protocol::Source &source);

} // namespace lldb_dap::protocol

#endif
