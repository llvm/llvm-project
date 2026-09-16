//===- ConnectionSpec.h - Connection spec parsing ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A ConnectionSpec describes one connection a process should establish with
// its peer, e.g. "tcp:connect=localhost:20000" or "fd=3".
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
/// E.g. "fd=3", "tcp:connect=localhost:20000", "tcp:listen=[::1]:0". Such
/// strings typically reach a process as a command-line argument, but nothing
/// in the grammar or the parser assumes that.
///
/// A spec describes what the process reading it does, so the two ends of one
/// connection carry different specs: an executor told "tcp:listen=:0" pairs
/// with a controller told "tcp:connect=<host>:<port>".
///
/// The parser only checks punctuation: the transport and action names are
/// opaque tokens, and the descriptor's syntax is entirely up to the
/// transport. Splitting the action from the transport is confined to the
/// text before the first '=', which lets a descriptor contain ':' and '='
/// unescaped (e.g. "tcp:listen=[::1]:0", "unix:listen=/tmp/a=b.sock").
///
/// All three fields are preserved verbatim, so callers matching a transport
/// or action name against a known set do so case-sensitively.
class ConnectionSpec {
public:
  /// Parses Spec as <transport>[:<action>]=<descriptor>.
  LLVM_ABI static Expected<ConnectionSpec> parse(StringRef Spec);

  /// The transport name, e.g. "tcp". Never empty.
  StringRef getTransport() const { return Transport; }

  /// The action name, e.g. "listen" or "connect". May be empty: direction is
  /// degenerate for some transports (an inherited socket fd is already
  /// connected), so single-mode transports omit it.
  StringRef getAction() const { return Action; }

  /// The transport-specific address. May be empty (e.g. "fd=").
  StringRef getDescriptor() const { return Descriptor; }

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
