//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/AcceleratorGDBRemotePackets.h"

using namespace llvm;
using namespace llvm::json;

namespace lldb_private {

bool fromJSON(const Value &value, SymbolValue &data, Path path) {
  ObjectMapper o(value, path);
  return o && o.map("name", data.name) && o.map("value", data.value);
}

json::Value toJSON(const SymbolValue &data) {
  return Object{{"name", data.name}, {"value", data.value}};
}

bool fromJSON(const Value &value, AcceleratorBreakpointByName &data,
              Path path) {
  ObjectMapper o(value, path);
  return o && o.mapOptional("shlib", data.shlib) &&
         o.map("function_name", data.function_name);
}

json::Value toJSON(const AcceleratorBreakpointByName &data) {
  return Object{{"shlib", data.shlib}, {"function_name", data.function_name}};
}

bool fromJSON(const Value &value, AcceleratorBreakpointByAddress &data,
              Path path) {
  ObjectMapper o(value, path);
  return o && o.map("load_address", data.load_address);
}

json::Value toJSON(const AcceleratorBreakpointByAddress &data) {
  return Object{{"load_address", static_cast<int64_t>(data.load_address)}};
}

bool fromJSON(const Value &value, AcceleratorBreakpointInfo &data, Path path) {
  ObjectMapper o(value, path);
  return o && o.map("identifier", data.identifier) &&
         o.mapOptional("by_name", data.by_name) &&
         o.mapOptional("by_address", data.by_address) &&
         o.map("symbol_names", data.symbol_names);
}

json::Value toJSON(const AcceleratorBreakpointInfo &data) {
  return Object{
      {"identifier", data.identifier},
      {"by_name", data.by_name},
      {"by_address", data.by_address},
      {"symbol_names", data.symbol_names},
  };
}

bool fromJSON(const Value &value, AcceleratorBreakpointHitArgs &data,
              Path path) {
  ObjectMapper o(value, path);
  return o && o.map("plugin_name", data.plugin_name) &&
         o.map("breakpoint", data.breakpoint) &&
         o.map("symbol_values", data.symbol_values);
}

json::Value toJSON(const AcceleratorBreakpointHitArgs &data) {
  return Object{
      {"plugin_name", data.plugin_name},
      {"breakpoint", data.breakpoint},
      {"symbol_values", data.symbol_values},
  };
}

std::optional<uint64_t>
AcceleratorBreakpointHitArgs::GetSymbolValue(StringRef symbol_name) const {
  auto it = llvm::find_if(symbol_values, [&](const SymbolValue &symbol) {
    return symbol.name == symbol_name;
  });
  if (it != symbol_values.end())
    return it->value;
  return std::nullopt;
}

bool fromJSON(const Value &value, AcceleratorConnectionInfo &data, Path path) {
  ObjectMapper o(value, path);
  return o && o.mapOptional("exe_path", data.exe_path) &&
         o.mapOptional("platform_name", data.platform_name) &&
         o.mapOptional("triple", data.triple) &&
         o.map("connect_url", data.connect_url);
}

json::Value toJSON(const AcceleratorConnectionInfo &data) {
  return Object{
      {"exe_path", data.exe_path},
      {"platform_name", data.platform_name},
      {"triple", data.triple},
      {"connect_url", data.connect_url},
  };
}

bool fromJSON(const Value &value, AcceleratorActions &data, Path path) {
  ObjectMapper o(value, path);
  return o && o.map("plugin_name", data.plugin_name) &&
         o.map("session_name", data.session_name) &&
         o.map("identifier", data.identifier) &&
         o.map("breakpoints", data.breakpoints) &&
         o.mapOptional("connect_info", data.connect_info);
}

json::Value toJSON(const AcceleratorActions &data) {
  Object obj{
      {"plugin_name", data.plugin_name},
      {"session_name", data.session_name},
      {"identifier", data.identifier},
      {"breakpoints", data.breakpoints},
  };
  if (data.connect_info)
    obj["connect_info"] = *data.connect_info;
  return obj;
}

bool fromJSON(const Value &value, AcceleratorBreakpointHitResponse &data,
              Path path) {
  ObjectMapper o(value, path);
  return o && o.map("disable_bp", data.disable_bp) &&
         o.map("auto_resume_native", data.auto_resume_native) &&
         o.mapOptional("actions", data.actions);
}

json::Value toJSON(const AcceleratorBreakpointHitResponse &data) {
  Object obj{
      {"disable_bp", data.disable_bp},
      {"auto_resume_native", data.auto_resume_native},
  };
  if (data.actions)
    obj["actions"] = *data.actions;
  return obj;
}

} // namespace lldb_private
