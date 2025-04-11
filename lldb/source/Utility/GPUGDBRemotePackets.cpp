//===-- GPUGDBRemotePackets.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/GPUGDBRemotePackets.h"

using namespace llvm;
using namespace llvm::json;

namespace lldb_private {

bool fromJSON(const json::Value &value, SymbolValue &data, Path path) {
  ObjectMapper o(value, path);
  return o && o.map("name", data.name) && o.map("value", data.value);
}

json::Value toJSON(const SymbolValue &data) {
  return json::Value(Object{{"name", data.name}, {"value", data.value}});
}


bool fromJSON(const llvm::json::Value &value, GPUBreakpointInfo &data,
              llvm::json::Path path) {
  ObjectMapper o(value, path);
  return o && 
         o.map("identifier", data.identifier) &&
         o.map("shlib", data.shlib) &&
         o.map("function_name", data.function_name) &&
         o.map("symbol_names", data.symbol_names);
}

llvm::json::Value toJSON(const GPUBreakpointInfo &data) {
  return json::Value(
    Object{{"identifier", data.identifier}, 
           {"shlib", data.shlib},
           {"function_name", data.function_name},
           {"symbol_names", data.symbol_names},
          });
}

bool fromJSON(const llvm::json::Value &value, GPUPluginInfo &data,
              llvm::json::Path path) {
  ObjectMapper o(value, path);
  return o && 
         o.map("name", data.name) &&
         o.map("description", data.description) &&
         o.map("breakpoints", data.breakpoints);      
}

llvm::json::Value toJSON(const GPUPluginInfo &data) {
  return json::Value(
    Object{{"name", data.name}, 
           {"description", data.description},
           {"breakpoints", data.breakpoints},
          });
}

bool fromJSON(const json::Value &value, GPUPluginBreakpointHitArgs &data,
              Path path) {
  ObjectMapper o(value, path);
  return o && 
         o.map("plugin_name", data.plugin_name) &&
         o.map("breakpoint", data.breakpoint) &&
         o.map("symbol_values", data.symbol_values);
}

json::Value toJSON(const GPUPluginBreakpointHitArgs &data) {
  return json::Value(
      Object{{"plugin_name", data.plugin_name}, 
             {"breakpoint", data.breakpoint},
             {"symbol_values", data.symbol_values},
            });
}

bool fromJSON(const llvm::json::Value &value, 
              GPUPluginBreakpointHitResponse &data,
              llvm::json::Path path) {
  ObjectMapper o(value, path);
  return o && 
         o.map("disable_bp", data.disable_bp) &&
         o.mapOptional("breakpoints", data.breakpoints) &&
         o.mapOptional("connect_url", data.connect_url);
}

llvm::json::Value toJSON(const GPUPluginBreakpointHitResponse &data) {
  return json::Value(
    Object{{"disable_bp", data.disable_bp}, 
           {"breakpoints", data.breakpoints},
           {"connect_url", data.connect_url},
          });
}

} // namespace lldb_private
