//===-- OptionGroupString.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionGroupString.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

using namespace lldb;
using namespace lldb_private;

OptionGroupString::OptionGroupString (uint32_t usage_mask,
                                      bool required,
                                      const char *long_option,
                                      char short_option,
                                      uint32_t completion_type,
                                      lldb::CommandArgumentType argument_type,
                                      const char *usage_text,
                                      const char *default_value) :
    m_value (default_value, default_value)
{
    m_option_definition.usage_mask = usage_mask;
    m_option_definition.required = required;
    m_option_definition.long_option = long_option;
    m_option_definition.short_option = short_option;
    m_option_definition.option_has_arg = required_argument;
    m_option_definition.enum_values = NULL;
    m_option_definition.completion_type = completion_type;
    m_option_definition.argument_type = argument_type;
    m_option_definition.usage_text = usage_text;
}

OptionGroupString::~OptionGroupString ()
{
}

Error
OptionGroupString::SetOptionValue (CommandInterpreter &interpreter,
                                   uint32_t option_idx,
                                   const char *option_arg)
{
    Error error (m_value.SetValueFromCString (option_arg));
    return error;
}

void
OptionGroupString::OptionParsingStarting (CommandInterpreter &interpreter)
{
    m_value.Clear();
}
