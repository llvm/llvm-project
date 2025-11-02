//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_PROTOCOL_MCP_RESOURCE_H
#define LLDB_PLUGINS_PROTOCOL_MCP_RESOURCE_H

#include "lldb/Protocol/MCP/Protocol.h"
#include "lldb/Protocol/MCP/Resource.h"
#include "lldb/lldb-forward.h"
#include "lldb/lldb-types.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <cstddef>
#include <vector>

namespace lldb_private::mcp {

class DebuggerResourceProvider : public lldb_protocol::mcp::ResourceProvider {
public:
  using ResourceProvider::ResourceProvider;
  virtual ~DebuggerResourceProvider() = default;

  std::vector<lldb_protocol::mcp::Resource> GetResources() const override;
  llvm::Expected<lldb_protocol::mcp::ReadResourceResult>
  ReadResource(llvm::StringRef uri) const override;

private:
  static lldb_protocol::mcp::Resource GetDebuggerResource(Debugger &debugger);
  static lldb_protocol::mcp::Resource GetTargetResource(size_t target_idx,
                                                        Target &target);

  static llvm::Expected<lldb_protocol::mcp::ReadResourceResult>
  ReadDebuggerResource(llvm::StringRef uri, lldb::user_id_t debugger_id);
  static llvm::Expected<lldb_protocol::mcp::ReadResourceResult>
  ReadTargetResource(llvm::StringRef uri, lldb::user_id_t debugger_id,
                     size_t target_idx);
};

} // namespace lldb_private::mcp

#endif
