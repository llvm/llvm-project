//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Variables.h"
#include "JSONUtils.h"
#include "LLDBUtils.h"
#include "Protocol/DAPTypes.h"
#include "Protocol/ProtocolRequests.h"
#include "Protocol/ProtocolTypes.h"
#include "SBAPIExtras.h"
#include "lldb/API/SBDeclaration.h"
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBValue.h"
#include "lldb/API/SBValueList.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>
#include <optional>
#include <vector>

using namespace llvm;
using namespace lldb_dap;
using namespace lldb_dap::protocol;

namespace {

bool HasInnerVarref(lldb::SBValue &v) {
  return v.MightHaveChildren() || ValuePointsToCode(v) ||
         v.GetDeclaration().IsValid();
}

template <typename T> StringMap<uint32_t> DistinctNames(T &container) {
  StringMap<uint32_t> variable_name_counts;
  for (auto variable : container) {
    if (!variable.IsValid())
      break;
    variable_name_counts[GetNonNullVariableName(variable)]++;
  }
  return variable_name_counts;
}

protocol::Scope MakeScope(ScopeKind kind, var_ref_t variablesReference,
                          bool expensive) {
  protocol::Scope scope;
  scope.variablesReference = variablesReference;
  scope.expensive = expensive;

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

std::optional<VariablePresentationHint>
MakeVariablePresentationHints(bool is_readonly, bool is_internal) {
  if (!is_readonly && !is_internal)
    return std::nullopt;

  VariablePresentationHint hint;

  if (is_readonly)
    hint.attributes.push_back("readOnly");
  if (is_internal)
    hint.visibility = "internal";

  return hint;
}

bool IsReservedName(llvm::StringRef name) {
  if (name == "[raw]" || name.starts_with("std::__"))
    return true;
  auto last_namespace_component = name.rfind("::");
  if (last_namespace_component != llvm::StringRef::npos)
    name = name.substr(last_namespace_component + 2);
  return /* c/c++ std reserves prefixes __ or _[A-Z] for internal use */
      name.starts_with("__") ||
      (name.size() >= 2 && name[0] == '_' && isUpper(name[1]));
}

class VariableStoreImpl : public VariableStore {
public:
  using VariableStore::VariableStore;
  Variable CreateVariable(lldb::SBValue v, bool format_hex,
                          bool is_name_duplicated,
                          std::optional<llvm::StringRef> custom_name) {
    VariableDescription desc(v, m_storage.config.enableAutoVariableSummaries,
                             format_hex, is_name_duplicated, custom_name);
    Variable var;
    var.name = std::move(desc.name);
    var.value = std::move(desc.display_value);
    var.type = std::move(desc.display_type_name);

    if (!desc.evaluate_name.empty())
      var.evaluateName = std::move(desc.evaluate_name);

    // If we have a type with many children, we would like to be able to
    // give a hint to the IDE that the type has indexed children so that the
    // request can be broken up in grabbing only a few children at a time. We
    // want to be careful and only call "v.GetNumChildren()" if we have an array
    // type or if we have a synthetic child provider producing indexed children.
    // We don't want to call "v.GetNumChildren()" on all objects as class,
    // struct and union types don't need to be completed if they are never
    // expanded. So we want to avoid calling this to only cases where we it
    // makes sense to keep performance high during normal debugging.

    // If we have an array type, say that it is indexed and provide the number
    // of children in case we have a huge array. If we don't do this, then we
    // might take a while to produce all children at onces which can delay your
    // debug session.
    if (desc.type_obj.IsArrayType()) {
      var.indexedVariables = v.GetNumChildren();
    } else if (v.IsSynthetic()) {
      // For a type with a synthetic child provider, the SBType of "v" won't
      // tell us anything about what might be displayed. Instead, we check if
      // the first child's name is "[0]" and then say it is indexed. We call
      // GetNumChildren() only if the child name matches to avoid a potentially
      // expensive operation.
      if (lldb::SBValue first_child = v.GetChildAtIndex(0)) {
        llvm::StringRef first_child_name = first_child.GetName();
        if (first_child_name == "[0]") {
          size_t num_children = v.GetNumChildren();
          // If we are creating a "[raw]" fake child for each synthetic type, we
          // have to account for it when returning indexed variables.
          if (m_storage.config.enableSyntheticChildDebugging)
            ++num_children;
          var.indexedVariables = num_children;
        }
      }
    }

    const bool is_internal = IsReservedName(var.name) || m_is_internal;
    const bool is_readonly = is_internal || v.GetType().IsAggregateType() ||
                             v.GetValueType() == lldb::eValueTypeRegisterSet ||
                             var.name == "(Return Value)";

    var.presentationHint =
        MakeVariablePresentationHints(is_readonly, is_internal);

    const var_ref_t var_ref =
        HasInnerVarref(v)
            ? m_storage.Insert(v, /*is_permanent=*/m_is_permanent, is_internal)
            : var_ref_t(var_ref_t::k_no_child);

    if (var.indexedVariables || v.MightHaveChildren())
      var.variablesReference = var_ref;

    if (v.GetDeclaration().IsValid())
      var.declarationLocationReference =
          PackLocation(var_ref.AsUInt32(), false);

    if (ValuePointsToCode(v))
      var.valueLocationReference = PackLocation(var_ref.AsUInt32(), true);

    if (lldb::addr_t addr = v.GetLoadAddress(); addr != LLDB_INVALID_ADDRESS)
      var.memoryReference = addr;

    return var;
  }

  template <typename T>
  std::vector<Variable> MakeVariables(
      const VariablesArguments &args, T &container,
      const std::map<lldb::user_id_t, std::string> &name_overrides = {}) {
    std::vector<Variable> variables;

    // We first find out which variable names are duplicated.
    StringMap<uint32_t> variable_name_counts = DistinctNames(container);

    const bool format_hex = args.format ? args.format->hex : false;
    auto start_it = begin(container) + args.start;
    auto end_it = args.count == 0 ? end(container) : start_it + args.count;

    // Now we construct the result with unique display variable names.
    for (; start_it != end_it; start_it++) {
      lldb::SBValue variable = *start_it;
      if (!variable.IsValid())
        break;

      std::optional<std::string> custom_name;
      auto name_it = name_overrides.find(variable.GetID());
      if (name_it != name_overrides.end())
        custom_name = name_it->second;

      variables.emplace_back(CreateVariable(
          variable, format_hex,
          variable_name_counts[GetNonNullVariableName(variable)] > 1,
          custom_name));
    }

    return variables;
  }
};

/// A Variable store for fetching variables within a specific scope (locals,
/// globals, or registers) for a given stack frame.
class ScopeStore final : public VariableStoreImpl {
public:
  explicit ScopeStore(VariableReferenceStorage &storage, ScopeKind kind,
                      const lldb::SBFrame &frame)
      : VariableStoreImpl(storage, /*is_permanent=*/false,
                          /*is_internal=*/frame.IsArtificial()),
        m_frame(frame), m_kind(kind) {}

  Expected<std::vector<Variable>>
  GetVariables(const VariablesArguments &args) override {
    LoadVariables();
    if (m_error.Fail())
      return ToError(m_error);
    return MakeVariables(args, m_children, m_names);
  }

  lldb::SBValue FindVariable(llvm::StringRef name) override {
    LoadVariables();

    lldb::SBValue variable;
    const bool is_name_duplicated = name.contains(" @");
    // variablesReference is one of our scopes, not an actual variable it is
    // asking for a variable in locals or globals or registers.
    const uint32_t end_idx = m_children.GetSize();
    // Searching backward so that we choose the variable in closest scope
    // among variables of the same name.
    for (const uint32_t i : reverse(seq<uint32_t>(0, end_idx))) {
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

private:
  void LoadVariables() {
    if (m_variables_loaded)
      return;

    m_variables_loaded = true;

    // TODO: Support "arguments" and "return value" scope.
    // At the moment lldb-dap includes the arguments and return_value  into the
    // "locals" scope.
    // VS Code only expands the first non-expensive scope. This causes friction
    // if we add the arguments above the local scope, as the locals scope will
    // not be expanded if we enter a function with arguments. It becomes more
    // annoying when the scope has arguments, return_value and locals.
    switch (m_kind) {
    case eScopeKindLocals: {
      // Show return value if there is any (in the local top frame)
      lldb::SBValue stop_return_value;
      if (m_frame.GetFrameID() == 0 &&
          ((stop_return_value = m_frame.GetThread().GetStopReturnValue()))) {
        // FIXME: Cloning this value seems to change the type summary, see
        // https://github.com/llvm/llvm-project/issues/183578
        // m_children.Append(stop_return_value.Clone("(Return Value)"));
        m_names[stop_return_value.GetID()] = "(Return Value)";
        m_children.Append(stop_return_value);
      }

      lldb::SBValueList locals = m_frame.GetVariables(/*arguments=*/true,
                                                      /*locals=*/true,
                                                      /*statics=*/false,
                                                      /*in_scope_only=*/true);
      m_children.Append(locals);
      // Save the error since we cannot insert into the SBValueList
      m_error = locals.GetError();
    } break;
    case eScopeKindGlobals:
      m_children = m_frame.GetVariables(/*arguments=*/false,
                                        /*locals=*/false,
                                        /*statics=*/true,
                                        /*in_scope_only=*/true);
      m_error = m_children.GetError();
      break;
    case eScopeKindRegisters:
      m_children = m_frame.GetRegisters();
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
  }

  lldb::SBFrame m_frame;
  lldb::SBValueList m_children;
  lldb::SBError m_error;
  std::map<lldb::user_id_t, std::string> m_names;
  ScopeKind m_kind;
  bool m_variables_loaded = false;
};

/// Variable store for expandable values.
///
/// Manages children variables of complex types (structs, arrays, pointers,
/// etc.) that can be expanded in the debugger UI.
class ExpandableValueStore final : public VariableStoreImpl {

public:
  explicit ExpandableValueStore(VariableReferenceStorage &storage,
                                bool is_permanent, bool is_internal,
                                const lldb::SBValue &value)
      : VariableStoreImpl(storage, is_permanent, is_internal), m_value(value) {}

  llvm::Expected<std::vector<protocol::Variable>>
  GetVariables(const protocol::VariablesArguments &args) override {
    std::map<lldb::user_id_t, std::string> name_overrides;
    lldb::SBValueList list;
    for (auto inner : m_value)
      list.Append(inner);

    // We insert a new "[raw]" child that can be used to inspect the raw version
    // of a synthetic member. That eliminates the need for the user to go to the
    // debug console and type `frame var <variable> to get these values.
    if (m_storage.config.enableSyntheticChildDebugging &&
        m_value.IsSynthetic()) {
      lldb::SBValue synthetic_value = m_value.GetNonSyntheticValue();
      name_overrides[synthetic_value.GetID()] = "[raw]";
      // FIXME: Cloning the value seems to affect the type summary, see
      // https://github.com/llvm/llvm-project/issues/183578
      // m_value.GetSyntheticValue().Clone("[raw]");
      list.Append(synthetic_value);
    }

    return MakeVariables(args, list, name_overrides);
  }

  lldb::SBValue FindVariable(llvm::StringRef name) override {
    if (name == "[raw]" && m_value.IsSynthetic())
      return m_value.GetNonSyntheticValue();

    // Handle mapped index
    lldb::SBValue variable = m_value.GetChildMemberWithName(name.data());
    if (variable.IsValid())
      return variable;

    // Handle array indexes
    uint64_t index = 0;
    if (name.consume_front('[') && name.consume_back("]") &&
        !name.consumeInteger(0, index))
      variable = m_value.GetChildAtIndex(index);

    return variable;
  }

  [[nodiscard]] lldb::SBValue GetVariable() const override { return m_value; }

private:
  lldb::SBValue m_value;
};

class ExpandableValueListStore final : public VariableStoreImpl {

public:
  explicit ExpandableValueListStore(VariableReferenceStorage &storage,
                                    bool is_permanent, bool is_internal,
                                    const lldb::SBValueList &list)
      : VariableStoreImpl(storage, is_permanent, is_internal),
        m_value_list(list) {}

  llvm::Expected<std::vector<protocol::Variable>>
  GetVariables(const protocol::VariablesArguments &args) override {
    return MakeVariables(args, m_value_list);
  }

  lldb::SBValue FindVariable(llvm::StringRef name) override {
    lldb::SBValue variable = m_value_list.GetFirstValueByName(name.data());
    if (variable.IsValid())
      return variable;

    return lldb::SBValue();
  }

private:
  lldb::SBValueList m_value_list;
};

} // namespace

namespace lldb_dap {

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

var_ref_t VariableReferenceStorage::Insert(const lldb::SBValue &variable,
                                           bool is_permanent,
                                           bool is_internal) {
  if (is_permanent)
    return m_permanent_kind_pool.Add<ExpandableValueStore>(
        *this, is_permanent, is_internal, variable);

  return m_temporary_kind_pool.Add<ExpandableValueStore>(*this, is_permanent,
                                                         is_internal, variable);
}

var_ref_t VariableReferenceStorage::Insert(const lldb::SBValueList &values) {
  return m_permanent_kind_pool.Add<ExpandableValueListStore>(
      *this, /*is_permanent=*/true, /*is_internal=*/false, values);
}

std::vector<protocol::Scope>
VariableReferenceStorage::Insert(const lldb::SBFrame &frame) {
  auto create_scope = [&](ScopeKind kind) {
    const var_ref_t var_ref =
        m_temporary_kind_pool.Add<ScopeStore>(*this, kind, frame);
    const bool is_expensive = kind != eScopeKindLocals;
    return MakeScope(kind, var_ref, is_expensive);
  };

  return {create_scope(eScopeKindLocals), create_scope(eScopeKindGlobals),
          create_scope(eScopeKindRegisters)};
}

lldb::SBValue VariableReferenceStorage::FindVariable(var_ref_t var_ref,
                                                     llvm::StringRef name) {
  if (VariableStore *store = GetVariableStore(var_ref))
    return store->FindVariable(name);

  return {};
}

VariableStore *VariableReferenceStorage::GetVariableStore(var_ref_t var_ref) {
  const ReferenceKind kind = var_ref.Kind();
  switch (kind) {
  case eReferenceKindPermanent:
    return m_permanent_kind_pool.GetVariableStore(var_ref);
  case eReferenceKindTemporary:
    return m_temporary_kind_pool.GetVariableStore(var_ref);
  case eReferenceKindInvalid:
    return nullptr;
  }
  llvm_unreachable("Unknown reference kind.");
}

} // namespace lldb_dap
