//===-- OptionGroupBoolean.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionGroupBoolean.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

using namespace lldb;
using namespace lldb_private;

OptionGroupBoolean::OptionGroupBoolean (uint32_t usage_mask,
                                        bool required,
                                        const char *long_option, 
                                        int short_option,
                                        const char *usage_text,
                                        bool default_value,
                                        bool no_argument_toggle_default) :
    m_value (default_value, default_value)
{
    m_option_definition.usage_mask = usage_mask;
    m_option_definition.required = required;
    m_option_definition.long_option = long_option;
    m_option_definition.short_option = short_option;
    m_option_definition.option_has_arg = no_argument_toggle_default ? no_argument : required_argument;
    m_option_definition.enum_values = NULL;
    m_option_definition.completion_type = 0;
    m_option_definition.argument_type = eArgTypeBoolean;
    m_option_definition.usage_text = usage_text;
}

OptionGroupBoolean::~OptionGroupBoolean ()
{
}

Error
OptionGroupBoolean::SetOptionValue (CommandInterpreter &interpreter,
                                    uint32_t option_idx,
                                    const char *option_arg)
{
    Error error;
    if (m_option_definition.option_has_arg == no_argument)
    {
        // Not argument, toggle the default value and mark the option as having been set
        m_value.SetCurrentValue (!m_value.GetDefaultValue());
        m_value.SetOptionWasSet ();
    }
    else
    {
        error = m_value.SetValueFromCString (option_arg);
    }
    return error;
}

void
OptionGroupBoolean::OptionParsingStarting (CommandInterpreter &interpreter)
{
    m_value.Clear();
}
