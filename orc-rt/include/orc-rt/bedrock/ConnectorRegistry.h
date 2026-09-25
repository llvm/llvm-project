//===- ConnectorRegistry.h - Transport connector registry -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A registry of connectors, keyed by transport name.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_BEDROCK_CONNECTORREGISTRY_H
#define ORC_RT_BEDROCK_CONNECTORREGISTRY_H

#include "orc-rt/bedrock/BootstrapInfo.h"
#include "orc-rt/bedrock/ConnectionSpec.h"
#include "orc-rt/support/Error.h"
#include "orc-rt/support/move_only_function.h"

#include <mutex>
#include <string>
#include <unordered_map>

namespace orc_rt {

class Session;

/// Maps transport names to the connectors that establish them, so that a
/// process can act on a ConnectionSpec without knowing which transports were
/// built into it.
///
/// Connectors are registered explicitly rather than self-registering. This
/// ensures that only requested transport mechanisms are available.
class ConnectorRegistry {
public:
  struct AttachInfo {
    Session &S;
    BootstrapInfo BI;
  };

  /// Supplies the Session and BootstrapInfo to connect.
  using GetAttachInfoFn = move_only_function<Expected<AttachInfo>() noexcept>;

  /// Establishes the connection CS describes and attaches it to the Session
  /// that GetSession returns.
  using ConnectorFn = move_only_function<Error(
      GetAttachInfoFn GetAttachInfo, const ConnectionSpec &) noexcept>;

  /// Registers Connector as the handler for Transport.
  ///
  /// Errors if Transport already has one: two connectors for one name means
  /// the process cannot tell which it is speaking.
  Error registerConnector(std::string Transport,
                          ConnectorFn Connector) noexcept;

  /// Runs the connector registered for CS's transport.
  ///
  /// Fails if no connector is registered for it, which is how a spec naming a
  /// transport this process was not built with is reported.
  Error connect(GetAttachInfoFn GetAttachInfo,
                const ConnectionSpec &CS) noexcept;

private:
  std::mutex M;
  std::unordered_map<std::string, ConnectorFn> Connectors;
};

} // namespace orc_rt

#endif // ORC_RT_BEDROCK_CONNECTORREGISTRY_H
