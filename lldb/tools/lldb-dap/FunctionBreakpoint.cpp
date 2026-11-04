//===-- FunctionBreakpoint.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FunctionBreakpoint.h"
#include "DAP.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBFileSpecList.h"
#include "lldb/API/SBModule.h"
#include "lldb/API/SBModuleSpec.h"
#include "lldb/API/SBMutex.h"
#include "lldb/Utility/UUID.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include <mutex>
#include <string>
#include <variant>

namespace lldb_dap {

/// Decodes a module id. Returns false if it isn't a valid UUID.
static bool DecodeModuleId(llvm::StringRef id,
                           llvm::SmallVectorImpl<uint8_t> &uuid_bytes) {
  return lldb_private::UUID::DecodeUUIDBytesFromString(id, uuid_bytes)
             .empty() &&
         !uuid_bytes.empty();
}

/// Module ids are compared by value, so spell them in a single form (upper case
/// hex digits without separators). An id that isn't a valid UUID is kept as is.
static std::string CanonicalizeModuleId(const std::string &id) {
  llvm::SmallVector<uint8_t, 20> uuid_bytes;
  if (!DecodeModuleId(id, uuid_bytes))
    return id;
  return llvm::toHex(uuid_bytes);
}

FunctionBreakpoint::FunctionBreakpoint(
    DAP &d, const protocol::FunctionBreakpoint &breakpoint)
    : Breakpoint(d, breakpoint.condition, breakpoint.hitCondition),
      m_function_name(breakpoint.name) {
  for (const protocol::FunctionBreakpointModule &module : breakpoint.modules) {
    if (const auto *by_name = std::get_if<protocol::ModuleByName>(&module)) {
      if (!by_name->name.empty())
        m_modules.emplace(ModuleRefKind::Name, by_name->name.str());
      continue;
    }
    if (const auto *by_id = std::get_if<protocol::ModuleById>(&module)) {
      if (!by_id->id.empty())
        m_modules.emplace(ModuleRefKind::Id,
                          CanonicalizeModuleId(by_id->id.str()));
      continue;
    }
  }
}

void FunctionBreakpoint::SetBreakpoint() {
  lldb::SBMutex lock = m_dap.GetAPIMutex();
  std::lock_guard<lldb::SBMutex> guard(lock);

  if (m_function_name.empty())
    return;

  m_error.clear();
  lldb::SBFileSpecList module_list;
  std::string not_found;
  for (const auto &[kind, value] : m_modules) {
    if (kind == ModuleRefKind::Name) {
      module_list.AppendIfUnique(lldb::SBFileSpec(value.c_str()));
      continue;
    }
    // The lookup is restricted to the modules with this id that are loaded now,
    // a module that is loaded later is not picked up.
    llvm::SmallVector<uint8_t, 20> uuid_bytes;
    lldb::SBModule module;
    if (DecodeModuleId(value, uuid_bytes)) {
      lldb::SBModuleSpec module_spec;
      module_spec.SetUUIDBytes(uuid_bytes.data(), uuid_bytes.size());
      module = m_dap.target.FindModule(module_spec);
    }
    if (module.IsValid())
      module_list.AppendIfUnique(module.GetFileSpec());
    else
      not_found += (not_found.empty() ? "" : ", ") + value;
  }
  // Don't fall back to an unrestricted breakpoint if none of the requested
  // modules was found.
  if (module_list.GetSize() == 0 && !not_found.empty()) {
    m_error = "module id not found: " + not_found;
    return;
  }

  m_bp = m_dap.target.BreakpointCreateByName(
      m_function_name.c_str(), module_list, lldb::SBFileSpecList());
  Breakpoint::SetBreakpoint();
}

protocol::Breakpoint FunctionBreakpoint::ToProtocolBreakpoint() {
  protocol::Breakpoint breakpoint = Breakpoint::ToProtocolBreakpoint();
  if (!m_error.empty())
    breakpoint.message = m_error;
  return breakpoint;
}

} // namespace lldb_dap
