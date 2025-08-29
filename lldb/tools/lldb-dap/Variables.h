//===-- Variables.h -----------------------------------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_VARIABLES_H
#define LLDB_TOOLS_LLDB_DAP_VARIABLES_H

#include "lldb/API/SBValue.h"
#include "lldb/API/SBValueList.h"
#include "llvm/ADT/DenseMap.h"
#include <map>
#include <utility>

#define VARREF_FIRST_VAR_IDX (int64_t)1

namespace lldb_dap {

enum ScopeKind { Locals, Globals, Registers };

struct ScopeData {
  ScopeKind kind;
  lldb::SBValueList scope;

  ScopeData(ScopeKind kind, lldb::SBValueList scope)
      : kind(kind), scope(scope) {}
};

struct Variables {
  /// Check if \p var_ref points to a variable that should persist for the
  /// entire duration of the debug session, e.g. repl expandable variables
  static bool IsPermanentVariableReference(int64_t var_ref);

  /// \return a new variableReference.
  /// Specify is_permanent as true for variable that should persist entire
  /// debug session.
  int64_t GetNewVariableReference(bool is_permanent);

  /// \return the expandable variable corresponding with variableReference
  /// value of \p value.
  /// If \p var_ref is invalid an empty SBValue is returned.
  lldb::SBValue GetVariable(int64_t var_ref) const;

  lldb::SBValueList *GetScope(const uint32_t frame_id, const ScopeKind kind);

  /// Insert a new \p variable.
  /// \return variableReference assigned to this expandable variable.
  int64_t InsertVariable(lldb::SBValue variable, bool is_permanent);

  lldb::SBValueList *GetTopLevelScope(int64_t variablesReference);

  lldb::SBValue FindVariable(uint64_t variablesReference, llvm::StringRef name);

  /// Initialize a frame if it hasn't been already, otherwise do nothing
  void ReadyFrame(uint32_t frame_id, lldb::SBFrame &frame);
  std::optional<ScopeData> GetScopeKind(const int64_t variablesReference);

  /// Clear all scope variables and non-permanent expandable variables.
  void Clear();

  void AddScopeKind(int64_t variable_reference, ScopeKind kind,
                    uint32_t frame_id);

private:
  /// Variable_reference start index of permanent expandable variable.
  static constexpr int64_t PermanentVariableStartIndex = (1ll << 32);
  int64_t m_next_temporary_var_ref{VARREF_FIRST_VAR_IDX};

  // Variable Reference,                 frame_id
  std::map<int64_t, std::pair<ScopeKind, uint32_t>> m_scope_kinds;

  /// Variables that are alive in this stop state.
  /// Will be cleared when debuggee resumes.
  llvm::DenseMap<int64_t, lldb::SBValue> m_referencedvariables;

  /// Variables that persist across entire debug session.
  /// These are the variables evaluated from debug console REPL.
  llvm::DenseMap<int64_t, lldb::SBValue> m_referencedpermanent_variables;

  /// Key = frame_id
  /// Value = (locals, globals Registers) scopes
  std::map<uint32_t,
           std::tuple<lldb::SBValueList, lldb::SBValueList, lldb::SBValueList>>
      m_frames;
  int64_t m_next_permanent_var_ref{PermanentVariableStartIndex};
};

} // namespace lldb_dap

#endif
