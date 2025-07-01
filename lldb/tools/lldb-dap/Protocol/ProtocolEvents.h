//===-- ProtocolEvents.h --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains POD structs based on the DAP specification at
// https://microsoft.github.io/debug-adapter-protocol/specification
//
// This is not meant to be a complete implementation, new interfaces are added
// when they're needed.
//
// Each struct has a toJSON and fromJSON function, that converts between
// the struct and a JSON representation. (See JSON.h)
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_PROTOCOL_PROTOCOL_EVENTS_H
#define LLDB_TOOLS_LLDB_DAP_PROTOCOL_PROTOCOL_EVENTS_H

#include "Protocol/ProtocolTypes.h"
#include "llvm/Support/JSON.h"

namespace lldb_dap::protocol {

/// The event indicates that one or more capabilities have changed.
///
/// Since the capabilities are dependent on the client and its UI, it might not
/// be possible to change that at random times (or too late).
///
/// Consequently this event has a hint characteristic: a client can only be
/// expected to make a 'best effort' in honoring individual capabilities but
/// there are no guarantees.
///
/// Only changed capabilities need to be included, all other capabilities keep
/// their values.
struct CapabilitiesEventBody {
  Capabilities capabilities;
};
llvm::json::Value toJSON(const CapabilitiesEventBody &);

} // end namespace lldb_dap::protocol

#endif
