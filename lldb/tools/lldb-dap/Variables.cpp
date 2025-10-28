//===-- Variables.cpp ---------------------------------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Variables.h"
#include "JSONUtils.h"

using namespace lldb_dap;

lldb::SBValueList *Variables::GetTopLevelScope(int64_t variablesReference) {
  switch (variablesReference) {
  case VARREF_LOCALS:
    return &locals;
  case VARREF_GLOBALS:
    return &globals;
  case VARREF_REGS:
    return &registers;
  default:
    return nullptr;
  }
}

void Variables::Clear() {
  locals.Clear();
  globals.Clear();
  registers.Clear();
  m_referencedvariables.clear();
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
