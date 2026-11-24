//===- ConnectionSpec.h - Connection spec parsing ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A ConnectionSpec describes one connection a process should establish with
// its peer, e.g. "tcp:connect=localhost:20000" or "socket:adopt=3".
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SHARED_CONNECTIONSPEC_H
#define LLVM_EXECUTIONENGINE_ORC_SHARED_CONNECTIONSPEC_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include <string>

namespace llvm::orc {

/// The parsed form of a string describing one connection a process should
/// establish with its peer:
///
///   <transport>[:<action>]=<descriptor>
///
///   transport   what kind of thing the descriptor names: "tcp" for a host:port
///               endpoint, "socket" for a handle this process already holds.
///   action      what this process does with it: "connect", "listen", or
///               "adopt" for a handle it was handed.
///   descriptor  the thing itself, in whatever syntax the transport defines.
///
/// E.g. "tcp:connect=localhost:20000", "tcp:listen=[::1]:0", "socket:adopt=3".
///
/// A spec says what the process reading it does, so the two ends of one
/// connection carry different specs: an executor told
/// "tcp:connect=<host>:<port>" pairs with a controller told "tcp:listen=:0".
///
/// Parsing checks punctuation only: transport and action are opaque tokens, the
/// descriptor's syntax belongs to the transport, and all three fields are kept
/// verbatim -- so a caller matching a name against a known set does so
/// case-sensitively. Only the text before the first '=' is searched for the
/// ':', which lets a descriptor hold ':' and '=' unescaped.
class ConnectionSpec {
public:
  /// Parses Spec as <transport>[:<action>]=<descriptor>.
  LLVM_ABI static Expected<ConnectionSpec> parse(StringRef Spec);

  /// The transport name, e.g. "tcp". Never empty.
  StringRef getTransport() const { return Transport; }

  /// The action name, e.g. "connect" or "adopt". May be empty.
  StringRef getAction() const { return Action; }

  /// The thing the transport names, in whatever syntax that transport defines.
  /// Opaque here: the parser neither splits nor validates it. May be empty.
  StringRef getDescriptor() const { return Descriptor; }

  /// Rebuilds the original connection string, e.g. "tcp:connect=host:port".
  std::string str() const {
    std::string S = Transport;
    if (!Action.empty()) {
      S += ':';
      S += Action;
    }
    S += '=';
    S += Descriptor;
    return S;
  }

private:
  ConnectionSpec(std::string Transport, std::string Action,
                 std::string Descriptor)
      : Transport(std::move(Transport)), Action(std::move(Action)),
        Descriptor(std::move(Descriptor)) {}

  std::string Transport;
  std::string Action;
  std::string Descriptor;
};

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_CONNECTIONSPEC_H
