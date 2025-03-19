//===-- Transport.h -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Debug Adapter Protocol transport layer for encoding and decoding protocol
// messages.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_TRANSPORT_H
#define LLDB_TOOLS_LLDB_DAP_TRANSPORT_H

#include "DAPForward.h"
#include "Protocol/ProtocolBase.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <optional>

namespace lldb_dap {

/// A transport class that performs the Debug Adapter Protocol communication
/// with the client.
class Transport {
public:
  Transport(llvm::StringRef client_name, Log *log, lldb::IOObjectSP input,
            lldb::IOObjectSP output);
  ~Transport() = default;

  /// Transport is not copyable.
  /// @{
  Transport(const Transport &rhs) = delete;
  void operator=(const Transport &rhs) = delete;
  /// @}

  /// Writes a Debug Adater Protocol message to the output stream.
  llvm::Error Write(const protocol::Message &M);

  /// Reads the next Debug Adater Protocol message from the input stream.
  ///
  /// \returns Returns the next protocol message or nullopt if EOF is reached.
  llvm::Expected<std::optional<protocol::Message>> Read();

  /// Returns the name of this transport client, for example `stdin/stdout` or
  /// `client_1`.
  llvm::StringRef GetClientName() { return m_client_name; }

private:
  llvm::StringRef m_client_name;
  Log *m_log;
  lldb::IOObjectSP m_input;
  lldb::IOObjectSP m_output;
};

} // namespace lldb_dap

#endif
