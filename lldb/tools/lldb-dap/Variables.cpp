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
#include "lldb/API/SBValueList.h"
#include <cstdint>
#include <optional>
#include <vector>

using namespace lldb_dap;

namespace lldb_dap {

protocol::Scope CreateScope(const ScopeKind kind, int64_t variablesReference,
                            int64_t namedVariables, bool expensive) {
  protocol::Scope scope;

  // TODO: Support "arguments" and "return value" scope.
  // At the moment lldb-he arguments and return_value  into the
  // "locals" scope.
  // vscode only expands the first non-expensive scope, this causes friction
  // if we add the arguments above the local scope as the locals scope will not
  // be expanded if we enter a function with arguments. It becomes more
  // annoying when the scope has arguments, return_value and locals.
  switch (kind) {
  case ScopeKind::Locals:
    scope.presentationHint = protocol::Scope::eScopePresentationHintLocals;
    scope.name = "Locals";
    break;
  case ScopeKind::Globals:
    scope.name = "Globals";
    break;
  case ScopeKind::Registers:
    scope.presentationHint = protocol::Scope::eScopePresentationHintRegisters;
    scope.name = "Registers";
    break;
  }

  scope.variablesReference = variablesReference;
  scope.namedVariables = namedVariables;
  scope.expensive = expensive;

  return scope;
}

lldb::SBValueList *Variables::GetTopLevelScope(int64_t variablesReference) {
  auto iter = m_scope_kinds.find(variablesReference);
  if (iter == m_scope_kinds.end()) {
    return nullptr;
  }

  ScopeKind scope_kind = iter->second.first;
  uint32_t frame_id = iter->second.second;

  auto frame_iter = m_frames.find(frame_id);
  if (frame_iter == m_frames.end()) {
    return nullptr;
  }

  switch (scope_kind) {
  case lldb_dap::ScopeKind::Locals:
    return &std::get<0>(frame_iter->second);
  case lldb_dap::ScopeKind::Globals:
    return &std::get<1>(frame_iter->second);
  case lldb_dap::ScopeKind::Registers:
    return &std::get<2>(frame_iter->second);
  }

  return nullptr;
}

void Variables::Clear() {
  m_referencedvariables.clear();
  m_scope_kinds.clear();
  m_frames.clear();
  m_next_temporary_var_ref = VARREF_FIRST_VAR_IDX;
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
  if (lldb::SBValueList *top_scope = GetTopLevelScope(variablesReference)) {
    bool is_duplicated_variable_name = name.contains(" @");
    // variablesReference is one of our scopes, not an actual variable it is
    // asking for a variable in locals or globals or registers
    int64_t end_idx = top_scope->GetSize();
    // Searching backward so that we choose the variable in closest scope
    // among variables of the same name.
    for (int64_t i = end_idx - 1; i >= 0; --i) {
      lldb::SBValue curr_variable = top_scope->GetValueAtIndex(i);
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

std::optional<ScopeData>
Variables::GetScopeKind(const int64_t variablesReference) {
  auto scope_kind_iter = m_scope_kinds.find(variablesReference);
  if (scope_kind_iter == m_scope_kinds.end()) {
    return std::nullopt;
  }

  auto scope_iter = m_frames.find(scope_kind_iter->second.second);
  if (scope_iter == m_frames.end()) {
    return std::nullopt;
  }

  ScopeData scope_data = ScopeData();
  scope_data.kind = scope_kind_iter->second.first;

  switch (scope_kind_iter->second.first) {
  case lldb_dap::ScopeKind::Locals:
    scope_data.scope = std::get<0>(scope_iter->second);
    return scope_data;
  case lldb_dap::ScopeKind::Globals:
    scope_data.scope = std::get<1>(scope_iter->second);
    return scope_data;
  case lldb_dap::ScopeKind::Registers:
    scope_data.scope = std::get<2>(scope_iter->second);
    return scope_data;
  }

  return std::nullopt;
}

lldb::SBValueList *Variables::GetScope(const uint32_t frame_id,
                                       const ScopeKind kind) {

  auto frame = m_frames.find(frame_id);
  if (m_frames.find(frame_id) == m_frames.end()) {
    return nullptr;
  }

  switch (kind) {
  case ScopeKind::Locals:
    return &std::get<0>(frame->second);
  case ScopeKind::Globals:
    return &std::get<1>(frame->second);
  case ScopeKind::Registers:
    return &std::get<2>(frame->second);
  }

  return nullptr;
}

std::vector<protocol::Scope> Variables::ReadyFrame(uint32_t frame_id,
                                                   lldb::SBFrame &frame) {

  if (m_frames.find(frame_id) == m_frames.end()) {

    auto locals = frame.GetVariables(/*arguments=*/true,
                                     /*locals=*/true,
                                     /*statics=*/false,
                                     /*in_scope_only=*/true);

    auto globals = frame.GetVariables(/*arguments=*/false,
                                      /*locals=*/false,
                                      /*statics=*/true,
                                      /*in_scope_only=*/true);

    auto registers = frame.GetRegisters();

    m_frames.insert(
        std::make_pair(frame_id, std::make_tuple(locals, globals, registers)));
  }

  std::vector<protocol::Scope> scopes = {};

  int64_t locals_ref = GetNewVariableReference(false);

  scopes.push_back(CreateScope(ScopeKind::Locals, locals_ref,
                               GetScope(frame_id, ScopeKind::Locals)->GetSize(),
                               false));

  m_scope_kinds[locals_ref] = std::make_pair(ScopeKind::Locals, frame_id);

  int64_t globals_ref = GetNewVariableReference(false);
  scopes.push_back(
      CreateScope(ScopeKind::Globals, globals_ref,
                  GetScope(frame_id, ScopeKind::Globals)->GetSize(), false));
  m_scope_kinds[globals_ref] = std::make_pair(ScopeKind::Globals, frame_id);

  int64_t registers_ref = GetNewVariableReference(false);
  scopes.push_back(
      CreateScope(ScopeKind::Registers, registers_ref,
                  GetScope(frame_id, ScopeKind::Registers)->GetSize(), false));

  m_scope_kinds[registers_ref] = std::make_pair(ScopeKind::Registers, frame_id);

  return scopes;
}

} // namespace lldb_dap
