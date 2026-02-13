//===-- Variables.cpp ---------------------------------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Variables.h"
#include "JSONUtils.h"
#include "Protocol/ProtocolTypes.h"
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBValue.h"
#include "lldb/API/SBValueList.h"
#include <cstdint>
#include <optional>
#include <vector>

using namespace lldb_dap;

namespace lldb_dap {

protocol::Scope CreateScope(const ScopeKind kind, int64_t variablesReference,
                            int64_t namedVariables, bool expensive) {
  protocol::Scope scope;
  scope.variablesReference = variablesReference;
  scope.namedVariables = namedVariables;
  scope.expensive = expensive;

  // TODO: Support "arguments" and "return value" scope.
  // At the moment lldb-dap includes the arguments and return_value  into the
  // "locals" scope.
  // VS Code only expands the first non-expensive scope. This causes friction
  // if we add the arguments above the local scope, as the locals scope will not
  // be expanded if we enter a function with arguments. It becomes more
  // annoying when the scope has arguments, return_value and locals.
  switch (kind) {
  case eScopeKindLocals:
    scope.presentationHint = protocol::Scope::eScopePresentationHintLocals;
    scope.name = "Locals";
    break;
  case eScopeKindGlobals:
    scope.name = "Globals";
    break;
  case eScopeKindRegisters:
    scope.presentationHint = protocol::Scope::eScopePresentationHintRegisters;
    scope.name = "Registers";
    break;
  }

  return scope;
}

std::optional<ScopeData>
Variables::GetTopLevelScope(int64_t variablesReference) {
  auto scope_kind_iter = m_scope_kinds.find(variablesReference);
  if (scope_kind_iter == m_scope_kinds.end())
    return std::nullopt;

  ScopeKind scope_kind = scope_kind_iter->second.first;
  uint64_t dap_frame_id = scope_kind_iter->second.second;

  auto frame_iter = m_frames.find(dap_frame_id);
  if (frame_iter == m_frames.end())
    return std::nullopt;

  lldb::SBValueList *scope = frame_iter->second.GetScope(scope_kind);
  if (scope == nullptr)
    return std::nullopt;

  ScopeData scope_data;
  scope_data.kind = scope_kind;
  scope_data.scope = *scope;
  return scope_data;
}

void Variables::Clear() {
  m_referencedvariables.clear();
  m_scope_kinds.clear();
  m_frames.clear();
  m_next_temporary_var_ref = TemporaryVariableStartIndex;
}

int64_t Variables::GetNewVariableReference(bool is_permanent) {
  if (is_permanent)
    return m_next_permanent_var_ref++;
  return m_next_temporary_var_ref++;
}

bool Variables::IsPermanentVariableReference(int64_t var_ref) {
  return var_ref >= PermanentVariableStartIndex;
}

lldb::SBValue Variables::GetVariable(int64_t var_ref) const {
  if (IsPermanentVariableReference(var_ref)) {
    auto pos = m_referencedpermanent_variables.find(var_ref);
    if (pos != m_referencedpermanent_variables.end())
      return pos->second;
  } else {
    auto pos = m_referencedvariables.find(var_ref);
    if (pos != m_referencedvariables.end())
      return pos->second;
  }
  return lldb::SBValue();
}

int64_t Variables::InsertVariable(lldb::SBValue variable, bool is_permanent) {
  int64_t var_ref = GetNewVariableReference(is_permanent);
  if (is_permanent)
    m_referencedpermanent_variables.insert(std::make_pair(var_ref, variable));
  else
    m_referencedvariables.insert(std::make_pair(var_ref, variable));
  return var_ref;
}

lldb::SBValue Variables::FindVariable(uint64_t variablesReference,
                                      llvm::StringRef name) {
  lldb::SBValue variable;
  if (std::optional<ScopeData> scope_data =
          GetTopLevelScope(variablesReference)) {
    bool is_duplicated_variable_name = name.contains(" @");
    // variablesReference is one of our scopes, not an actual variable it is
    // asking for a variable in locals or globals or registers
    int64_t end_idx = scope_data->scope.GetSize();
    // Searching backward so that we choose the variable in closest scope
    // among variables of the same name.
    for (int64_t i = end_idx - 1; i >= 0; --i) {
      lldb::SBValue curr_variable = scope_data->scope.GetValueAtIndex(i);
      std::string variable_name = CreateUniqueVariableNameForDisplay(
          curr_variable, is_duplicated_variable_name);
      if (variable_name == name) {
        variable = curr_variable;
        break;
      }
    }
  } else {
    // This is not under the globals or locals scope, so there are no
    // duplicated names.

    // We have a named item within an actual variable so we need to find it
    // withing the container variable by name.
    lldb::SBValue container = GetVariable(variablesReference);
    variable = container.GetChildMemberWithName(name.data());
    if (!variable.IsValid()) {
      if (name.starts_with("[")) {
        llvm::StringRef index_str(name.drop_front(1));
        uint64_t index = 0;
        if (!index_str.consumeInteger(0, index)) {
          if (index_str == "]")
            variable = container.GetChildAtIndex(index);
        }
      }
    }
  }
  return variable;
}

lldb::SBValueList *Variables::GetScope(const uint64_t dap_frame_id,
                                       const ScopeKind kind) {

  auto frame = m_frames.find(dap_frame_id);
  if (frame == m_frames.end()) {
    return nullptr;
  }

  return frame->second.GetScope(kind);
}

std::vector<protocol::Scope>
Variables::CreateScopes(const uint64_t dap_frame_id, lldb::SBFrame &frame) {
  auto iter = m_frames.find(dap_frame_id);
  if (iter == m_frames.end()) {
    auto locals = frame.GetVariables(/*arguments=*/true,
                                     /*locals=*/true,
                                     /*statics=*/false,
                                     /*in_scope_only=*/true);

    auto globals = frame.GetVariables(/*arguments=*/false,
                                      /*locals=*/false,
                                      /*statics=*/true,
                                      /*in_scope_only=*/true);

    auto registers = frame.GetRegisters();

    iter =
        m_frames.emplace(dap_frame_id, FrameScopes{locals, globals, registers})
            .first;
  }

  const FrameScopes &frame_scopes = iter->second;

  auto create_scope = [&](ScopeKind kind, uint32_t size) {
    int64_t ref = GetNewVariableReference(false);
    m_scope_kinds.try_emplace(ref, kind, dap_frame_id);
    return CreateScope(kind, ref, size, false);
  };

  return {
      create_scope(eScopeKindLocals, frame_scopes.locals.GetSize()),
      create_scope(eScopeKindGlobals, frame_scopes.globals.GetSize()),
      create_scope(eScopeKindRegisters, frame_scopes.registers.GetSize()),
  };
}

} // namespace lldb_dap
