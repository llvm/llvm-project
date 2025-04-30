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

//------------------------------------------------------------------------------
// SymbolValue
//------------------------------------------------------------------------------

bool fromJSON(const json::Value &value, SymbolValue &data, Path path) {
  ObjectMapper o(value, path);
  return o && o.map("name", data.name) && o.map("value", data.value);
}

json::Value toJSON(const SymbolValue &data) {
  return json::Value(Object{{"name", data.name}, {"value", data.value}});
}

//------------------------------------------------------------------------------
// GPUBreakpointInfo
//------------------------------------------------------------------------------
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

//------------------------------------------------------------------------------
// GPUPluginConnectionInfo
//------------------------------------------------------------------------------
bool fromJSON(const llvm::json::Value &value, GPUPluginConnectionInfo &data,
              llvm::json::Path path) {
  ObjectMapper o(value, path);
  return o && 
         o.mapOptional("exe_path", data.exe_path) &&
         o.mapOptional("platform_name", data.platform_name) &&
         o.mapOptional("triple", data.triple) &&
         o.map("connect_url", data.connect_url);
}

llvm::json::Value toJSON(const GPUPluginConnectionInfo &data) {
  return json::Value(
      Object{{"exe_path", data.exe_path}, 
             {"platform_name", data.platform_name}, 
             {"triple", data.triple},
             {"connect_url", data.connect_url},
            });
}


//------------------------------------------------------------------------------
// GPUPluginBreakpointHitArgs
//------------------------------------------------------------------------------
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

//------------------------------------------------------------------------------
// GPUActions
//------------------------------------------------------------------------------
bool fromJSON(const llvm::json::Value &value, GPUActions &data,
              llvm::json::Path path) {
  ObjectMapper o(value, path);
  return o && 
         o.map("plugin_name", data.plugin_name) &&
         o.mapOptional("breakpoints", data.breakpoints) &&
         o.mapOptional("connect_info", data.connect_info);
}

llvm::json::Value toJSON(const GPUActions &data) {
  return json::Value(
    Object{{"plugin_name", data.plugin_name},
           {"breakpoints", data.breakpoints},
           {"connect_info", data.connect_info},
          });
}

//------------------------------------------------------------------------------
// GPUPluginBreakpointHitResponse
//------------------------------------------------------------------------------


bool fromJSON(const llvm::json::Value &value, 
              GPUPluginBreakpointHitResponse &data,
              llvm::json::Path path) {
  ObjectMapper o(value, path);
  return o && 
         o.map("disable_bp", data.disable_bp) &&
         o.map("actions", data.actions);
}

llvm::json::Value toJSON(const GPUPluginBreakpointHitResponse &data) {
  return json::Value(
    Object{{"disable_bp", data.disable_bp}, 
           {"actions", data.actions},
          });
}

//------------------------------------------------------------------------------
// GPUSectionInfo
//------------------------------------------------------------------------------

bool fromJSON(const llvm::json::Value &value, GPUSectionInfo &data,
              llvm::json::Path path) {
  ObjectMapper o(value, path);
  return o && 
         o.map("name", data.name) &&
         o.mapOptional("load_address", data.load_address) &&
         o.map("children", data.children);
}

llvm::json::Value toJSON(const GPUSectionInfo &data) {
  return json::Value(
    Object{{"name", data.name}, 
           {"load_address", data.load_address},
           {"children", data.children}
          });
}

//------------------------------------------------------------------------------
// GPUDynamicLoaderLibraryInfo
//------------------------------------------------------------------------------

bool fromJSON(const llvm::json::Value &value, GPUDynamicLoaderLibraryInfo &data,
  llvm::json::Path path) {
  ObjectMapper o(value, path);
  return o && 
         o.map("pathname", data.pathname) &&
         o.mapOptional("uuid", data.uuid_str) &&
         o.map("load", data.load) &&
         o.mapOptional("load_address", data.load_address) &&
         o.mapOptional("native_memory_address", data.native_memory_address) &&
         o.mapOptional("native_memory_size", data.native_memory_size) &&
         o.mapOptional("file_offset", data.file_offset) &&
         o.mapOptional("file_size", data.file_size) &&
         o.map("loaded_sections", data.loaded_sections);
}

llvm::json::Value toJSON(const GPUDynamicLoaderLibraryInfo &data) {
return json::Value(
Object{{"pathname", data.pathname}, 
       {"uuid", data.uuid_str},
       {"load", data.load},
       {"native_memory_address", data.native_memory_address},
       {"native_memory_size", data.native_memory_size},
       {"file_offset", data.file_offset},
       {"file_size", data.file_size},
       {"loaded_sections", data.loaded_sections}
      });
}

//------------------------------------------------------------------------------
// GPUDynamicLoaderArgs
//------------------------------------------------------------------------------

bool fromJSON(const llvm::json::Value &value, GPUDynamicLoaderArgs &data,
    llvm::json::Path path) {
  ObjectMapper o(value, path);
  return o && 
         o.map("full", data.full);
}

llvm::json::Value toJSON(const GPUDynamicLoaderArgs &data) {
  return json::Value(Object{{"full", data.full}});
}

//------------------------------------------------------------------------------
// GPUDynamicLoaderResponse
//------------------------------------------------------------------------------
bool fromJSON(const llvm::json::Value &value, GPUDynamicLoaderResponse &data,
              llvm::json::Path path) {
  ObjectMapper o(value, path);
  return o && 
         o.map("library_infos", data.library_infos);
}

llvm::json::Value toJSON(const GPUDynamicLoaderResponse &data) {
  return json::Value(Object{{"library_infos", data.library_infos}});
}

} // namespace lldb_private
