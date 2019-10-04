//===-- OptionGroupPythonClassWithDict.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionGroupPythonClassWithDict.h"

#include "lldb/Host/OptionParser.h"

using namespace lldb;
using namespace lldb_private;

OptionGroupPythonClassWithDict::OptionGroupPythonClassWithDict
    (const char *class_use,
     int class_option,
     int key_option, 
     int value_option,
     const char *class_long_option,
     const char *key_long_option,
     const char *value_long_option,
     bool required) {
  m_key_usage_text.assign("The key for a key/value pair passed to the class"
                            " that implements a ");
  m_key_usage_text.append(class_use);
  m_key_usage_text.append(".  Pairs can be specified more than once.");
  
  m_value_usage_text.assign("The value for a previous key in the pair passed to"
                            " the class that implements a ");
  m_value_usage_text.append(class_use);
  m_value_usage_text.append(".  Pairs can be specified more than once.");
  
  m_class_usage_text.assign("The name of the class that will manage a ");
  m_class_usage_text.append(class_use);
  m_class_usage_text.append(".");
  
  m_option_definition[0].usage_mask = LLDB_OPT_SET_1;
  m_option_definition[0].required = required;
  m_option_definition[0].long_option = class_long_option;
  m_option_definition[0].short_option = class_option;
  m_option_definition[0].validator = nullptr;
  m_option_definition[0].option_has_arg = OptionParser::eRequiredArgument;
  m_option_definition[0].enum_values = {};
  m_option_definition[0].completion_type = 0;
  m_option_definition[0].argument_type = eArgTypePythonClass;
  m_option_definition[0].usage_text = m_class_usage_text.data();

  m_option_definition[1].usage_mask = LLDB_OPT_SET_1;
  m_option_definition[1].required = required;
  m_option_definition[1].long_option = key_long_option;
  m_option_definition[1].short_option = key_option;
  m_option_definition[1].validator = nullptr;
  m_option_definition[1].option_has_arg = OptionParser::eRequiredArgument;
  m_option_definition[1].enum_values = {};
  m_option_definition[1].completion_type = 0;
  m_option_definition[1].argument_type = eArgTypeNone;
  m_option_definition[1].usage_text = m_key_usage_text.data();

  m_option_definition[2].usage_mask = LLDB_OPT_SET_1;
  m_option_definition[2].required = required;
  m_option_definition[2].long_option = value_long_option;
  m_option_definition[2].short_option = value_option;
  m_option_definition[2].validator = nullptr;
  m_option_definition[2].option_has_arg = OptionParser::eRequiredArgument;
  m_option_definition[2].enum_values = {};
  m_option_definition[2].completion_type = 0;
  m_option_definition[2].argument_type = eArgTypeNone;
  m_option_definition[2].usage_text = m_value_usage_text.data();
}

OptionGroupPythonClassWithDict::~OptionGroupPythonClassWithDict() {}

Status OptionGroupPythonClassWithDict::SetOptionValue(
    uint32_t option_idx,
    llvm::StringRef option_arg,
    ExecutionContext *execution_context) {
  Status error;
  switch (option_idx) {
  case 0: {
    m_class_name.assign(option_arg);
  } break;
  case 1: {
      if (m_current_key.empty())
        m_current_key.assign(option_arg);
      else
        error.SetErrorStringWithFormat("Key: \"%s\" missing value.",
                                        m_current_key.c_str());
    
  } break;
  case 2: {
      if (!m_current_key.empty()) {
          m_dict_sp->AddStringItem(m_current_key, option_arg);
          m_current_key.clear();
      }
      else
        error.SetErrorStringWithFormat("Value: \"%s\" missing matching key.",
                                       option_arg.str().c_str());
  } break;
  default:
    llvm_unreachable("Unimplemented option");
  }
  return error;
}

void OptionGroupPythonClassWithDict::OptionParsingStarting(
  ExecutionContext *execution_context) {
  m_current_key.erase();
  m_dict_sp = std::make_shared<StructuredData::Dictionary>();
}

Status OptionGroupPythonClassWithDict::OptionParsingFinished(
  ExecutionContext *execution_context) {
  Status error;
  // If we get here and there's contents in the m_current_key, somebody must
  // have provided a key but no value.
  if (!m_current_key.empty())
      error.SetErrorStringWithFormat("Key: \"%s\" missing value.",
                                     m_current_key.c_str());
  return error;
}

