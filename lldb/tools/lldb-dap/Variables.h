//===-- Variables.h -----------------------------------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_VARIABLES_H
#define LLDB_TOOLS_LLDB_DAP_VARIABLES_H

#include "Protocol/ProtocolTypes.h"
#include "lldb/API/SBValue.h"
#include "lldb/API/SBValueList.h"
#include "llvm/ADT/DenseMap.h"
#include <map>
#include <optional>
#include <utility>

namespace lldb_dap {

enum ScopeKind : unsigned {
  eScopeKindLocals,
  eScopeKindGlobals,
  eScopeKindRegisters
};
/// Creates a `protocol::Scope` struct.
///
/// \param[in] kind
///     The kind of scope to create
///
/// \param[in] variablesReference
///     The value to place into the "variablesReference" key
///
/// \param[in] namedVariables
///     The value to place into the "namedVariables" key
///
/// \param[in] expensive
///     The value to place into the "expensive" key
///
/// \return
///     A `protocol::Scope`
protocol::Scope CreateScope(const ScopeKind kind, int64_t variablesReference,
                            int64_t namedVariables, bool expensive);

struct ScopeData {
  ScopeKind kind;
  lldb::SBValueList scope;
};

/// Stores the three scope variable lists for a single stack frame.
struct FrameScopes {
  lldb::SBValueList locals;
  lldb::SBValueList globals;
  lldb::SBValueList registers;

  /// Returns a pointer to the scope corresponding to the given kind.
  lldb::SBValueList *GetScope(ScopeKind kind) {
    switch (kind) {
    case eScopeKindLocals:
      return &locals;
    case eScopeKindGlobals:
      return &globals;
    case eScopeKindRegisters:
      return &registers;
    }

    llvm_unreachable("unknown scope kind");
  }
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

  lldb::SBValueList *GetScope(const uint64_t dap_frame_id,
                              const ScopeKind kind);

  /// Insert a new \p variable.
  /// \return variableReference assigned to this expandable variable.
  int64_t InsertVariable(lldb::SBValue variable, bool is_permanent);

  std::optional<ScopeData> GetTopLevelScope(int64_t variablesReference);

  lldb::SBValue FindVariable(uint64_t variablesReference, llvm::StringRef name);

  /// Initialize a frame if it hasn't been already, otherwise do nothing
  std::vector<protocol::Scope> CreateScopes(const uint64_t dap_frame_id,
                                            lldb::SBFrame &frame);

  /// Clear all scope variables and non-permanent expandable variables.
  void Clear();

private:
  /// Variable reference start index of temporary variables.
  static constexpr int64_t TemporaryVariableStartIndex = 1;

  /// Variable reference start index of permanent expandable variable.
  static constexpr int64_t PermanentVariableStartIndex = (1ll << 32);

  int64_t m_next_permanent_var_ref{PermanentVariableStartIndex};
  int64_t m_next_temporary_var_ref{TemporaryVariableStartIndex};

  // Variable Reference,                 dap_frame_id
  std::map<int64_t, std::pair<ScopeKind, uint64_t>> m_scope_kinds;

  /// Variables that are alive in this stop state.
  /// Will be cleared when debuggee resumes.
  llvm::DenseMap<int64_t, lldb::SBValue> m_referencedvariables;

  /// Variables that persist across entire debug session.
  /// These are the variables evaluated from debug console REPL.
  llvm::DenseMap<int64_t, lldb::SBValue> m_referencedpermanent_variables;

  /// Key = dap_frame_id (encodes both thread index ID and frame ID)
  /// Value = scopes for the frame (locals, globals, registers)
  std::map<uint64_t, FrameScopes> m_frames;
};

} // namespace lldb_dap

#endif
