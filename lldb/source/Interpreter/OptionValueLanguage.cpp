//===-- OptionValueFormat.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionValueLanguage.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Stream.h"
#include "lldb/DataFormatters/FormatManager.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Target/Language.h"

using namespace lldb;
using namespace lldb_private;

void OptionValueLanguage::DumpValue(const ExecutionContext *exe_ctx,
                                    Stream &strm, uint32_t dump_mask) {
  if (dump_mask & eDumpOptionType)
    strm.Printf("(%s)", GetTypeAsCString());
  if (dump_mask & eDumpOptionValue) {
    if (dump_mask & eDumpOptionType)
      strm.PutCString(" = ");
    strm.PutCString(Language::GetNameForLanguageType(m_current_value));
  }
}

Error OptionValueLanguage::SetValueFromString(llvm::StringRef value,
                                              VarSetOperationType op) {
  Error error;
  switch (op) {
  case eVarSetOperationClear:
    Clear();
    break;

  case eVarSetOperationReplace:
  case eVarSetOperationAssign: {
    ConstString lang_name(value.trim());
    std::set<lldb::LanguageType> languages_for_types;
    std::set<lldb::LanguageType> languages_for_expressions;
    Language::GetLanguagesSupportingTypeSystems(languages_for_types,
                                                languages_for_expressions);

    LanguageType new_type =
        Language::GetLanguageTypeFromString(lang_name.GetCString());
    if (new_type && languages_for_types.count(new_type)) {
      m_value_was_set = true;
      m_current_value = new_type;
    } else {
      StreamString error_strm;
      error_strm.Printf("invalid language type '%s', ", value.str().c_str());
      error_strm.Printf("valid values are:\n");
      for (lldb::LanguageType language : languages_for_types) {
        error_strm.Printf("%s%s%s", "    ",
                          Language::GetNameForLanguageType(language), "\n");
      }
      error.SetErrorString(error_strm.GetData());
    }
  } break;

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

lldb::OptionValueSP OptionValueLanguage::DeepCopy() const {
  return OptionValueSP(new OptionValueLanguage(*this));
}
