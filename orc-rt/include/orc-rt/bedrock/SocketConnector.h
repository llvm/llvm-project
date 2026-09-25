//===- SocketConnector.h - Inherited socket connector -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A connector for the "socket" transport.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_BEDROCK_SOCKETCONNECTOR_H
#define ORC_RT_BEDROCK_SOCKETCONNECTOR_H

#include "orc-rt/bedrock/ConnectorRegistry.h"

namespace orc_rt {

/// Registers the connector for the "socket" transport, whose only action is
/// "adopt": a stream socket this process was handed, already connected.
///
/// If the descriptor named by the spec is a socket, the connector takes
/// ownership of it whether or not the connection succeeds. Otherwise it is left
/// untouched.
Error registerSocketConnector(ConnectorRegistry &R) noexcept;

} // namespace orc_rt

#endif // ORC_RT_BEDROCK_SOCKETCONNECTOR_H
