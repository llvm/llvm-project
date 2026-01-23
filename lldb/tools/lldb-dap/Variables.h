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

#define VARREF_FIRST_VAR_IDX (int64_t)4
#define VARREF_LOCALS (int64_t)1
#define VARREF_GLOBALS (int64_t)2
#define VARREF_REGS (int64_t)3

namespace lldb_dap {

struct Variables {
  lldb::SBValueList locals;
  lldb::SBValueList globals;
  lldb::SBValueList registers;

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

  /// Insert a new \p variable.
  /// \return variableReference assigned to this expandable variable.
  int64_t InsertVariable(lldb::SBValue variable, bool is_permanent);

  lldb::SBValueList *GetTopLevelScope(int64_t variablesReference);

  lldb::SBValue FindVariable(uint64_t variablesReference, llvm::StringRef name);

  /// Clear all scope variables and non-permanent expandable variables.
  void Clear();

private:
  /// Variable_reference start index of permanent expandable variable.
  static constexpr int64_t PermanentVariableStartIndex = (1ll << 32);

  /// Variables that are alive in this stop state.
  /// Will be cleared when debuggee resumes.
  llvm::DenseMap<int64_t, lldb::SBValue> m_referencedvariables;

  /// Variables that persist across entire debug session.
  /// These are the variables evaluated from debug console REPL.
  llvm::DenseMap<int64_t, lldb::SBValue> m_referencedpermanent_variables;

  int64_t m_next_temporary_var_ref{VARREF_FIRST_VAR_IDX};
  int64_t m_next_permanent_var_ref{PermanentVariableStartIndex};
};

} // namespace lldb_dap

#endif
