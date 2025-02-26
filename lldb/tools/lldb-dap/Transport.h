//===-- Transport.h ---------------------------------------------*- C++ -*-===//
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

#include "Protocol.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringRef.h"

namespace lldb_dap {

/// A transport class that performs the Debug Adapter Protocol communication
/// with the client.
class Transport {
public:
  Transport(llvm::StringRef client_name, lldb::IOObjectSP input,
            lldb::IOObjectSP output);
  ~Transport() = default;

  Transport(const Transport &rhs) = delete;
  void operator=(const Transport &rhs) = delete;

  /// Writes a Debug Adater Protocol message to the output stream.
  lldb_private::Status Write(const protocol::ProtocolMessage &M);

  /// Reads the next Debug Adater Protocol message from the input stream.
  llvm::Expected<protocol::ProtocolMessage> Read();

private:
  llvm::StringRef m_client_name;
  lldb::IOObjectSP m_input;
  lldb::IOObjectSP m_output;
};

} // namespace lldb_dap

#endif
