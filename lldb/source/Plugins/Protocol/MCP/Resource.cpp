// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Resource.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Protocol/MCP/MCPError.h"

using namespace lldb_private;
using namespace lldb_private::mcp;
using namespace lldb_protocol::mcp;

namespace {
struct DebuggerResource {
  uint64_t debugger_id = 0;
  std::string name;
  uint64_t num_targets = 0;
};

llvm::json::Value toJSON(const DebuggerResource &DR) {
  llvm::json::Object Result{{"debugger_id", DR.debugger_id},
                            {"num_targets", DR.num_targets}};
  if (!DR.name.empty())
    Result.insert({"name", DR.name});
  return Result;
}

struct TargetResource {
  size_t debugger_id = 0;
  size_t target_idx = 0;
  bool selected = false;
  bool dummy = false;
  std::string arch;
  std::string path;
  std::string platform;
};

llvm::json::Value toJSON(const TargetResource &TR) {
  llvm::json::Object Result{{"debugger_id", TR.debugger_id},
                            {"target_idx", TR.target_idx},
                            {"selected", TR.selected},
                            {"dummy", TR.dummy}};
  if (!TR.arch.empty())
    Result.insert({"arch", TR.arch});
  if (!TR.path.empty())
    Result.insert({"path", TR.path});
  if (!TR.platform.empty())
    Result.insert({"platform", TR.platform});
  return Result;
}
} // namespace

static constexpr llvm::StringLiteral kMimeTypeJSON = "application/json";

template <typename... Args>
static llvm::Error createStringError(const char *format, Args &&...args) {
  return llvm::createStringError(
      llvm::formatv(format, std::forward<Args>(args)...).str());
}

static llvm::Error createUnsupportedURIError(llvm::StringRef uri) {
  return llvm::make_error<UnsupportedURI>(uri.str());
}

lldb_protocol::mcp::Resource
DebuggerResourceProvider::GetDebuggerResource(Debugger &debugger) {
  const lldb::user_id_t debugger_id = debugger.GetID();

  lldb_protocol::mcp::Resource resource;
  resource.uri = llvm::formatv("lldb://debugger/{0}", debugger_id);
  resource.name = debugger.GetInstanceName();
  resource.description =
      llvm::formatv("Information about debugger instance {0}: {1}", debugger_id,
                    debugger.GetInstanceName());
  resource.mimeType = kMimeTypeJSON;
  return resource;
}

lldb_protocol::mcp::Resource
DebuggerResourceProvider::GetTargetResource(size_t target_idx, Target &target) {
  const size_t debugger_id = target.GetDebugger().GetID();

  std::string target_name = llvm::formatv("target {0}", target_idx);

  if (Module *exe_module = target.GetExecutableModulePointer())
    target_name = exe_module->GetFileSpec().GetFilename().GetString();

  lldb_protocol::mcp::Resource resource;
  resource.uri =
      llvm::formatv("lldb://debugger/{0}/target/{1}", debugger_id, target_idx);
  resource.name = target_name;
  resource.description =
      llvm::formatv("Information about target {0} in debugger instance {1}",
                    target_idx, debugger_id);
  resource.mimeType = kMimeTypeJSON;
  return resource;
}

std::vector<lldb_protocol::mcp::Resource>
DebuggerResourceProvider::GetResources() const {
  std::vector<lldb_protocol::mcp::Resource> resources;

  const size_t num_debuggers = Debugger::GetNumDebuggers();
  for (size_t i = 0; i < num_debuggers; ++i) {
    lldb::DebuggerSP debugger_sp = Debugger::GetDebuggerAtIndex(i);
    if (!debugger_sp)
      continue;
    resources.emplace_back(GetDebuggerResource(*debugger_sp));

    TargetList &target_list = debugger_sp->GetTargetList();
    const size_t num_targets = target_list.GetNumTargets();
    for (size_t j = 0; j < num_targets; ++j) {
      lldb::TargetSP target_sp = target_list.GetTargetAtIndex(j);
      if (!target_sp)
        continue;
      resources.emplace_back(GetTargetResource(j, *target_sp));
    }
  }

  return resources;
}

llvm::Expected<lldb_protocol::mcp::ReadResourceResult>
DebuggerResourceProvider::ReadResource(llvm::StringRef uri) const {

  auto [protocol, path] = uri.split("://");

  if (protocol != "lldb")
    return createUnsupportedURIError(uri);

  llvm::SmallVector<llvm::StringRef, 4> components;
  path.split(components, '/');

  if (components.size() < 2)
    return createUnsupportedURIError(uri);

  if (components[0] != "debugger")
    return createUnsupportedURIError(uri);

  size_t debugger_idx;
  if (components[1].getAsInteger(0, debugger_idx))
    return createStringError("invalid debugger id '{0}': {1}", components[1],
                             path);

  if (components.size() > 3) {
    if (components[2] != "target")
      return createUnsupportedURIError(uri);

    size_t target_idx;
    if (components[3].getAsInteger(0, target_idx))
      return createStringError("invalid target id '{0}': {1}", components[3],
                               path);

    return ReadTargetResource(uri, debugger_idx, target_idx);
  }

  return ReadDebuggerResource(uri, debugger_idx);
}

llvm::Expected<lldb_protocol::mcp::ReadResourceResult>
DebuggerResourceProvider::ReadDebuggerResource(llvm::StringRef uri,
                                               lldb::user_id_t debugger_id) {
  lldb::DebuggerSP debugger_sp = Debugger::FindDebuggerWithID(debugger_id);
  if (!debugger_sp)
    return createStringError("invalid debugger id: {0}", debugger_id);

  DebuggerResource debugger_resource;
  debugger_resource.debugger_id = debugger_id;
  debugger_resource.name = debugger_sp->GetInstanceName();
  debugger_resource.num_targets = debugger_sp->GetTargetList().GetNumTargets();

  lldb_protocol::mcp::TextResourceContents contents;
  contents.uri = uri;
  contents.mimeType = kMimeTypeJSON;
  contents.text = llvm::formatv("{0}", toJSON(debugger_resource));

  lldb_protocol::mcp::ReadResourceResult result;
  result.contents.push_back(contents);
  return result;
}

llvm::Expected<lldb_protocol::mcp::ReadResourceResult>
DebuggerResourceProvider::ReadTargetResource(llvm::StringRef uri,
                                             lldb::user_id_t debugger_id,
                                             size_t target_idx) {

  lldb::DebuggerSP debugger_sp = Debugger::FindDebuggerWithID(debugger_id);
  if (!debugger_sp)
    return createStringError("invalid debugger id: {0}", debugger_id);

  TargetList &target_list = debugger_sp->GetTargetList();
  lldb::TargetSP target_sp = target_list.GetTargetAtIndex(target_idx);
  if (!target_sp)
    return createStringError("invalid target idx: {0}", target_idx);

  TargetResource target_resource;
  target_resource.debugger_id = debugger_id;
  target_resource.target_idx = target_idx;
  target_resource.arch = target_sp->GetArchitecture().GetTriple().str();
  target_resource.dummy = target_sp->IsDummyTarget();
  target_resource.selected = target_sp == debugger_sp->GetSelectedTarget();

  if (Module *exe_module = target_sp->GetExecutableModulePointer())
    target_resource.path = exe_module->GetFileSpec().GetPath();
  if (lldb::PlatformSP platform_sp = target_sp->GetPlatform())
    target_resource.platform = platform_sp->GetName();

  lldb_protocol::mcp::TextResourceContents contents;
  contents.uri = uri;
  contents.mimeType = kMimeTypeJSON;
  contents.text = llvm::formatv("{0}", toJSON(target_resource));

  lldb_protocol::mcp::ReadResourceResult result;
  result.contents.push_back(contents);
  return result;
}
