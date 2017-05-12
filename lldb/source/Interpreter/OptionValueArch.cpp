//===-- OptionValueArch.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionValueArch.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/State.h"
#include "lldb/DataFormatters/FormatManager.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/CommandCompletions.h"
#include "lldb/Interpreter/CommandInterpreter.h"

using namespace lldb;
using namespace lldb_private;

void OptionValueArch::DumpValue(const ExecutionContext *exe_ctx, Stream &strm,
                                uint32_t dump_mask) {
  if (dump_mask & eDumpOptionType)
    strm.Printf("(%s)", GetTypeAsCString());
  if (dump_mask & eDumpOptionValue) {
    if (dump_mask & eDumpOptionType)
      strm.PutCString(" = ");

    if (m_current_value.IsValid()) {
      const char *arch_name = m_current_value.GetArchitectureName();
      if (arch_name)
        strm.PutCString(arch_name);
    }
  }
}

Status OptionValueArch::SetValueFromString(llvm::StringRef value,
                                           VarSetOperationType op) {
  Status error;
  switch (op) {
  case eVarSetOperationClear:
    Clear();
    NotifyValueChanged();
    break;

  case eVarSetOperationReplace:
  case eVarSetOperationAssign: {
    std::string value_str = value.trim().str();
    if (m_current_value.SetTriple(value_str.c_str())) {
      m_value_was_set = true;
      NotifyValueChanged();
    } else
      error.SetErrorStringWithFormat("unsupported architecture '%s'",
                                     value_str.c_str());
    break;
  }
  case eVarSetOperationInsertBefore:
  case eVarSetOperationInsertAfter:
  case eVarSetOperationRemove:
  case eVarSetOperationAppend:
  case eVarSetOperationInvalid:
    error = OptionValue::SetValueFromString(value, op);
    break;
  }
  return error;
}

lldb::OptionValueSP OptionValueArch::DeepCopy() const {
  return OptionValueSP(new OptionValueArch(*this));
}

size_t OptionValueArch::AutoComplete(CommandInterpreter &interpreter,
                                     llvm::StringRef s, int match_start_point,
                                     int max_return_elements,
                                     bool &word_complete, StringList &matches) {
  word_complete = false;
  matches.Clear();
  CommandCompletions::InvokeCommonCompletionCallbacks(
      interpreter, CommandCompletions::eArchitectureCompletion, s,
      match_start_point, max_return_elements, nullptr, word_complete, matches);
  return matches.GetSize();
}
