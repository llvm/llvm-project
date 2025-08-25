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
#include "lldb/Host/JSONTransport.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringRef.h"

namespace lldb_dap {

/// A transport class that performs the Debug Adapter Protocol communication
/// with the client.
class Transport final
    : public lldb_private::HTTPDelimitedJSONTransport<
          protocol::Request, protocol::Response, protocol::Event> {
public:
  Transport(llvm::StringRef client_name, lldb_dap::Log *log,
            lldb::IOObjectSP input, lldb::IOObjectSP output);
  virtual ~Transport() = default;

  void Log(llvm::StringRef message) override;

private:
  llvm::StringRef m_client_name;
  lldb_dap::Log *m_log;
};

} // namespace lldb_dap

#endif
