//===-- Variables.cpp ---------------------------------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Variables.h"
#include "DAPLog.h"
#include "JSONUtils.h"
#include "Protocol/DAPTypes.h"
#include "Protocol/ProtocolRequests.h"
#include "Protocol/ProtocolTypes.h"
#include "ProtocolUtils.h"
#include "SBAPIExtras.h"
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBValue.h"
#include "lldb/API/SBValueList.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>
#include <optional>
#include <vector>

using namespace lldb_dap;
using namespace lldb_dap::protocol;

namespace lldb_dap {

protocol::Scope CreateScope(ScopeKind kind, var_ref_t variablesReference,
                            bool expensive) {
  protocol::Scope scope;
  scope.variablesReference = variablesReference;
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

std::vector<Variable>
ScopeStore::GetVariables(VariableReferenceStorage &storage,
                         const Configuration &config,
                         const VariablesArguments &args) {
  LoadVariables();
  if (m_kind == lldb_dap::eScopeKindRegisters)
    SetRegistersFormat();

  const bool format_hex = args.format ? args.format->hex : false;
  std::vector<Variable> variables;
  if (m_kind == eScopeKindLocals)
    AddReturnValue(storage, config, variables, format_hex);

  const uint64_t count = args.count;
  const uint32_t start_idx = 0;
  const uint32_t num_children = m_children.GetSize();
  const uint32_t end_idx = start_idx + ((count == 0) ? num_children : count);

  // We first find out which variable names are duplicated.
  std::map<llvm::StringRef, uint32_t> variable_name_counts;
  for (auto i = start_idx; i < end_idx; ++i) {
    lldb::SBValue variable = m_children.GetValueAtIndex(i);
    if (!variable.IsValid())
      break;
    variable_name_counts[GetNonNullVariableName(variable)]++;
  }

  // Now we construct the result with unique display variable names.
  for (auto i = start_idx; i < end_idx; ++i) {
    lldb::SBValue variable = m_children.GetValueAtIndex(i);

    if (!variable.IsValid())
      break;

    const var_ref_t frame_var_ref =
        storage.InsertVariable(variable, /*is_permanent=*/false);
    if (LLVM_UNLIKELY(frame_var_ref.AsUInt32() >=
                      var_ref_t::k_variables_reference_threshold)) {
      DAP_LOG(storage.log,
              "warning: scopes variablesReference threshold reached. "
              "current: {} threshold: {}, maximum {}.",
              frame_var_ref.AsUInt32(),
              var_ref_t::k_variables_reference_threshold,
              var_ref_t::k_max_variables_references);
    }

    if (LLVM_UNLIKELY(frame_var_ref.Kind() == eReferenceKindInvalid))
      break;

    variables.emplace_back(CreateVariable(
        variable, frame_var_ref, format_hex, config.enableAutoVariableSummaries,
        config.enableSyntheticChildDebugging,
        variable_name_counts[GetNonNullVariableName(variable)] > 1));
  }
  return variables;
}

lldb::SBValue ScopeStore::FindVariable(llvm::StringRef name) {
  LoadVariables();

  lldb::SBValue variable;
  const bool is_name_duplicated = name.contains(" @");
  // variablesReference is one of our scopes, not an actual variable it is
  // asking for a variable in locals or globals or registers.
  const uint32_t end_idx = m_children.GetSize();
  // Searching backward so that we choose the variable in closest scope
  // among variables of the same name.
  for (const uint32_t i : llvm::reverse(llvm::seq<uint32_t>(0, end_idx))) {
    lldb::SBValue curr_variable = m_children.GetValueAtIndex(i);
    std::string variable_name =
        CreateUniqueVariableNameForDisplay(curr_variable, is_name_duplicated);
    if (variable_name == name) {
      variable = curr_variable;
      break;
    }
  }
  return variable;
}

void ScopeStore::LoadVariables() {
  if (m_variables_loaded)
    return;

  m_variables_loaded = true;
  switch (m_kind) {
  case eScopeKindLocals:
    m_children = m_frame.GetVariables(/*arguments=*/true,
                                      /*locals=*/true,
                                      /*statics=*/false,
                                      /*in_scope_only=*/true);
    break;
  case eScopeKindGlobals:
    m_children = m_frame.GetVariables(/*arguments=*/false,
                                      /*locals=*/false,
                                      /*statics=*/true,
                                      /*in_scope_only=*/true);
    break;
  case eScopeKindRegisters:
    m_children = m_frame.GetRegisters();
  }
}

void ScopeStore::SetRegistersFormat() {
  // Change the default format of any pointer sized registers in the first
  // register set to be the lldb::eFormatAddressInfo so we show the pointer
  // and resolve what the pointer resolves to. Only change the format if the
  // format was set to the default format or if it was hex as some registers
  // have formats set for them.
  const uint32_t addr_size =
      m_frame.GetThread().GetProcess().GetAddressByteSize();
  for (lldb::SBValue reg : m_children.GetValueAtIndex(0)) {
    const lldb::Format format = reg.GetFormat();
    if (format == lldb::eFormatDefault || format == lldb::eFormatHex) {
      if (reg.GetByteSize() == addr_size)
        reg.SetFormat(lldb::eFormatAddressInfo);
    }
  }
}

void ScopeStore::AddReturnValue(VariableReferenceStorage &storage,
                                const Configuration &config,
                                std::vector<Variable> &variables,
                                bool format_hex) {
  assert(m_kind == eScopeKindLocals &&
         "Return Value Should only be in local scope");
  if (m_children.GetSize() == 0) {
    // Check for an error in the SBValueList that might explain why we don't
    // have locals. If we have an error display it as the sole value in the
    // the locals.

    // "error" owns the error string so we must keep it alive as long as we
    // want to use the returns "const char *".
    lldb::SBError error = m_children.GetError();
    if (const char *var_err = error.GetCString()) {
      // Create a fake variable named "error" to explain why variables were
      // not available. This new error will help let users know when there was
      // a problem that kept variables from being available for display and
      // allow users to fix this issue instead of seeing no variables. The
      // errors are only set when there is a problem that the user could
      // fix, so no error will show up when you have no debug info, only when
      // we do have debug info and something that is fixable can be done.
      Variable err_var;
      err_var.name = "<error>";
      err_var.type = "const char *";
      err_var.value = var_err;
      variables.push_back(std::move(err_var));
    }
    return;
  }

  // Show return value if there is any ( in the local top frame )
  lldb::SBThread selected_thread = m_frame.GetThread();
  lldb::SBValue stop_return_value = selected_thread.GetStopReturnValue();

  if (stop_return_value.IsValid() &&
      (selected_thread.GetSelectedFrame().GetFrameID() == 0)) {
    auto renamed_return_value = stop_return_value.Clone("(Return Value)");
    var_ref_t return_var_ref{var_ref_t::k_no_child};

    if (stop_return_value.MightHaveChildren() ||
        stop_return_value.IsSynthetic()) {
      return_var_ref = storage.InsertVariable(stop_return_value,
                                              /*is_permanent=*/false);
    }
    variables.emplace_back(
        CreateVariable(renamed_return_value, return_var_ref, format_hex,
                       config.enableAutoVariableSummaries,
                       config.enableSyntheticChildDebugging, false));
  }
}

std::vector<Variable>
ExpandableValueStore::GetVariables(VariableReferenceStorage &storage,
                                   const Configuration &config,
                                   const VariablesArguments &args) {
  const var_ref_t var_ref = args.variablesReference;
  const uint64_t count = args.count;
  const uint64_t start = args.start;
  const bool hex = args.format ? args.format->hex : false;

  lldb::SBValue variable = storage.GetVariable(var_ref);
  if (!variable)
    return {};

  // We are expanding a variable that has children, so we will return its
  // children.
  std::vector<Variable> variables;
  const bool is_permanent = var_ref.Kind() == eReferenceKindPermanent;
  auto addChild = [&](lldb::SBValue child,
                      std::optional<llvm::StringRef> custom_name = {}) {
    if (!child.IsValid())
      return;

    const var_ref_t child_var_ref = storage.InsertVariable(child, is_permanent);
    if (LLVM_UNLIKELY(child_var_ref.AsUInt32() ==
                      var_ref_t::k_variables_reference_threshold)) {
      DAP_LOG(storage.log,
              "warning: {} variablesReference threshold reached. "
              "current: {} threshold: {}, maximum {}.",
              (is_permanent ? "permanent" : "temporary"),
              child_var_ref.AsUInt32(),
              var_ref_t::k_variables_reference_threshold,
              var_ref_t::k_max_variables_references);
    }

    if (LLVM_UNLIKELY(child_var_ref.Kind() == eReferenceKindInvalid)) {
      DAP_LOG(storage.log,
              "error: invalid variablesReference created for {} variable {}.",
              (is_permanent ? "permanent" : "temporary"), variable.GetName());
      return;
    }

    variables.emplace_back(CreateVariable(
        child, child_var_ref, hex, config.enableAutoVariableSummaries,
        config.enableSyntheticChildDebugging,
        /*is_name_duplicated=*/false, custom_name));
  };

  const uint32_t num_children = variable.GetNumChildren();
  const uint32_t end_idx = start + ((count == 0) ? num_children : count);
  uint32_t i = start;
  for (; i < end_idx && i < num_children; ++i)
    addChild(variable.GetChildAtIndex(i));

  // If we haven't filled the count quota from the request, we insert a new
  // "[raw]" child that can be used to inspect the raw version of a
  // synthetic member. That eliminates the need for the user to go to the
  // debug console and type `frame var <variable> to get these values.
  if (config.enableSyntheticChildDebugging && variable.IsSynthetic() &&
      i == num_children)
    addChild(variable.GetNonSyntheticValue(), "[raw]");

  return variables;
}

lldb::SBValue ExpandableValueStore::FindVariable(llvm::StringRef name) {
  lldb::SBValue variable = m_value.GetChildMemberWithName(name.data());
  if (variable.IsValid())
    return variable;

  if (name.consume_front('[') && name.consume_back("]")) {
    uint64_t index = 0;
    if (!name.consumeInteger(0, index)) {
      variable = m_value.GetChildAtIndex(index);
    }
  }
  return variable;
}

lldb::SBValue VariableReferenceStorage::GetVariable(var_ref_t var_ref) {
  const ReferenceKind kind = var_ref.Kind();

  if (kind == eReferenceKindTemporary) {
    if (auto *store = m_temporary_kind_pool.GetVariableStore(var_ref))
      return store->GetVariable();
  }

  if (kind == eReferenceKindPermanent) {
    if (auto *store = m_permanent_kind_pool.GetVariableStore(var_ref))
      return store->GetVariable();
  }

  return {};
}

var_ref_t
VariableReferenceStorage::InsertVariable(const lldb::SBValue &variable,
                                         bool is_permanent) {
  if (is_permanent)
    return m_permanent_kind_pool.Add(variable);

  return m_temporary_kind_pool.Add(variable);
}

lldb::SBValue VariableReferenceStorage::FindVariable(var_ref_t var_ref,
                                                     llvm::StringRef name) {
  if (VariableStore *store = GetVariableStore(var_ref))
    return store->FindVariable(name);

  return {};
}

std::vector<protocol::Scope>
VariableReferenceStorage::CreateScopes(lldb::SBFrame &frame) {
  auto create_scope = [&](ScopeKind kind) {
    const var_ref_t var_ref = m_scope_kind_pool.Add(kind, frame);
    const bool is_expensive = kind != eScopeKindLocals;
    return CreateScope(kind, var_ref, is_expensive);
  };

  return {create_scope(eScopeKindLocals), create_scope(eScopeKindGlobals),
          create_scope(eScopeKindRegisters)};
}

VariableStore *VariableReferenceStorage::GetVariableStore(var_ref_t var_ref) {
  const ReferenceKind kind = var_ref.Kind();
  switch (kind) {
  case eReferenceKindPermanent:
    return m_permanent_kind_pool.GetVariableStore(var_ref);
  case eReferenceKindTemporary:
    return m_temporary_kind_pool.GetVariableStore(var_ref);
  case eReferenceKindScope:
    return m_scope_kind_pool.GetVariableStore(var_ref);
  default:
    return nullptr;
  }
  llvm_unreachable("Unknown reference kind.");
}

} // namespace lldb_dap
