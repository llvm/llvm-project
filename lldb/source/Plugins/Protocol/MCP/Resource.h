//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_PROTOCOL_MCP_RESOURCE_H
#define LLDB_PLUGINS_PROTOCOL_MCP_RESOURCE_H

#include "Protocol.h"
#include "lldb/lldb-private.h"
#include <vector>

namespace lldb_private::mcp {

class ResourceProvider {
public:
  ResourceProvider() = default;
  virtual ~ResourceProvider() = default;

  virtual std::vector<protocol::Resource> GetResources() const = 0;
  virtual llvm::Expected<protocol::ResourceResult>
  ReadResource(llvm::StringRef uri) const = 0;
};

class DebuggerResourceProvider : public ResourceProvider {
public:
  using ResourceProvider::ResourceProvider;
  virtual ~DebuggerResourceProvider() = default;

  virtual std::vector<protocol::Resource> GetResources() const override;
  virtual llvm::Expected<protocol::ResourceResult>
  ReadResource(llvm::StringRef uri) const override;

private:
  static protocol::Resource GetDebuggerResource(Debugger &debugger);
  static protocol::Resource GetTargetResource(size_t target_idx,
                                              Target &target);

  static llvm::Expected<protocol::ResourceResult>
  ReadDebuggerResource(llvm::StringRef uri, lldb::user_id_t debugger_id);
  static llvm::Expected<protocol::ResourceResult>
  ReadTargetResource(llvm::StringRef uri, lldb::user_id_t debugger_id,
                     size_t target_idx);
};

} // namespace lldb_private::mcp

#endif
