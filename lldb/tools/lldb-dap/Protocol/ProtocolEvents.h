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
#include "lldb/lldb-types.h"
#include "llvm/Support/JSON.h"
#include <cstdint>
#include <optional>
#include <vector>

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

/// The event indicates that some information about a module has changed.
struct ModuleEventBody {
  enum Reason : unsigned { eReasonNew, eReasonChanged, eReasonRemoved };

  /// The new, changed, or removed module. In case of `removed` only the module
  /// id is used.
  Module module;

  /// The reason for the event.
  /// Values: 'new', 'changed', 'removed'
  Reason reason;
};
llvm::json::Value toJSON(const ModuleEventBody::Reason &);
llvm::json::Value toJSON(const ModuleEventBody &);

/// This event signals that some state in the debug adapter has changed and
/// requires that the client needs to re-render the data snapshot previously
/// requested.
///
/// Debug adapters do not have to emit this event for runtime changes like
/// stopped or thread events because in that case the client refetches the new
/// state anyway. But the event can be used for example to refresh the UI after
/// rendering formatting has changed in the debug adapter.
///
/// This event should only be sent if the corresponding capability
/// supportsInvalidatedEvent is true.
struct InvalidatedEventBody {
  enum Area : unsigned { eAreaAll, eAreaStacks, eAreaThreads, eAreaVariables };

  /// Set of logical areas that got invalidated.
  std::vector<Area> areas;

  /// If specified, the client only needs to refetch data related to this
  /// thread.
  std::optional<lldb::tid_t> threadId;

  /// If specified, the client only needs to refetch data related to this stack
  /// frame (and the `threadId` is ignored).
  std::optional<uint64_t> frameId;
};
llvm::json::Value toJSON(const InvalidatedEventBody::Area &);
llvm::json::Value toJSON(const InvalidatedEventBody &);

} // end namespace lldb_dap::protocol

#endif
