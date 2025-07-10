// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Resource.h"
#include "MCPError.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"

using namespace lldb_private::mcp;

template <typename... Args>
static llvm::Error createStringError(const char *format, Args &&...args) {
  return llvm::createStringError(
      llvm::formatv(format, std::forward<Args>(args)...).str());
}

static llvm::Error createUnsupportedURIError(llvm::StringRef uri) {
  return llvm::make_error<UnsupportedURI>(uri.str());
}

protocol::Resource
DebuggerResourceProvider::GetDebuggerResource(lldb::user_id_t debugger_id) {
  protocol::Resource resource;
  resource.uri = llvm::formatv("lldb://debugger/{0}", debugger_id);
  resource.name = llvm::formatv("debugger {0}", debugger_id);
  resource.description =
      llvm::formatv("Information about debugger instance {0}", debugger_id);
  resource.mimeType = "application/json";
  return resource;
}

protocol::Resource
DebuggerResourceProvider::GetTargetResource(lldb::user_id_t debugger_id,
                                            lldb::user_id_t target_id) {
  protocol::Resource resource;
  resource.uri =
      llvm::formatv("lldb://debugger/{0}/target/{1}", debugger_id, target_id);
  resource.name = llvm::formatv("target {0}", target_id);
  resource.description =
      llvm::formatv("Information about target {0} in debugger instance {1}",
                    target_id, debugger_id);
  resource.mimeType = "application/json";
  return resource;
}

std::vector<protocol::Resource> DebuggerResourceProvider::GetResources() const {
  std::vector<protocol::Resource> resources;

  const size_t num_debuggers = Debugger::GetNumDebuggers();
  for (size_t i = 0; i < num_debuggers; ++i) {
    lldb::DebuggerSP debugger_sp = Debugger::GetDebuggerAtIndex(i);
    if (!debugger_sp)
      continue;
    resources.emplace_back(GetDebuggerResource(i));

    TargetList &target_list = debugger_sp->GetTargetList();
    const size_t num_targets = target_list.GetNumTargets();
    for (size_t j = 0; j < num_targets; ++j) {
      lldb::TargetSP target_sp = target_list.GetTargetAtIndex(j);
      if (!target_sp)
        continue;
      resources.emplace_back(GetTargetResource(i, j));
    }
  }

  return resources;
}

llvm::Expected<protocol::ResourceResult>
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

  lldb::user_id_t debugger_id;
  if (components[1].getAsInteger(0, debugger_id))
    return createStringError("invalid debugger id '{0}': {1}", components[1],
                             path);

  if (components.size() > 3) {
    if (components[2] != "target")
      return createUnsupportedURIError(uri);

    lldb::user_id_t target_id;
    if (components[3].getAsInteger(0, target_id))
      return createStringError("invalid target id '{0}': {1}", components[3],
                               path);

    return ReadTargetResource(uri, debugger_id, target_id);
  }

  return ReadDebuggerResource(uri, debugger_id);
}

llvm::Expected<protocol::ResourceResult>
DebuggerResourceProvider::ReadDebuggerResource(llvm::StringRef uri,
                                               lldb::user_id_t debugger_id) {
  lldb::DebuggerSP debugger_sp = Debugger::GetDebuggerAtIndex(debugger_id);
  if (!debugger_sp)
    return createStringError("invalid debugger id: {0}", debugger_id);

  TargetList &target_list = debugger_sp->GetTargetList();
  const size_t num_targets = target_list.GetNumTargets();

  llvm::json::Value value = llvm::json::Object{{"debugger_id", debugger_id},
                                               {"num_targets", num_targets}};

  std::string json = llvm::formatv("{0}", value);

  protocol::ResourceContents contents;
  contents.uri = uri;
  contents.mimeType = "application/json";
  contents.text = json;

  protocol::ResourceResult result;
  result.contents.push_back(contents);
  return result;
}

llvm::Expected<protocol::ResourceResult>
DebuggerResourceProvider::ReadTargetResource(llvm::StringRef uri,
                                             lldb::user_id_t debugger_id,
                                             lldb::user_id_t target_id) {

  lldb::DebuggerSP debugger_sp = Debugger::GetDebuggerAtIndex(debugger_id);
  if (!debugger_sp)
    return createStringError("invalid debugger id: {0}", debugger_id);

  TargetList &target_list = debugger_sp->GetTargetList();
  lldb::TargetSP target_sp = target_list.GetTargetAtIndex(target_id);
  if (!target_sp)
    return createStringError("invalid target id: {0}", target_id);

  llvm::json::Object object{
      {"debugger_id", debugger_id},
      {"target_id", target_id},
      {"arch", target_sp->GetArchitecture().GetTriple().str()}};

  if (Module *exe_module = target_sp->GetExecutableModulePointer())
    object.insert({"path", exe_module->GetFileSpec().GetPath()});

  llvm::json::Value value = std::move(object);
  std::string json = llvm::formatv("{0}", value);

  protocol::ResourceContents contents;
  contents.uri = uri;
  contents.mimeType = "application/json";
  contents.text = json;

  protocol::ResourceResult result;
  result.contents.push_back(contents);
  return result;
}
