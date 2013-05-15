//===-- OptionValueString.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionValueString.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Stream.h"
#include "lldb/Interpreter/Args.h"

using namespace lldb;
using namespace lldb_private;

void
OptionValueString::DumpValue (const ExecutionContext *exe_ctx, Stream &strm, uint32_t dump_mask)
{
    if (dump_mask & eDumpOptionType)
        strm.Printf ("(%s)", GetTypeAsCString ());
    if (dump_mask & eDumpOptionValue)
    {
        if (dump_mask & eDumpOptionType)
            strm.PutCString (" = ");
        if (!m_current_value.empty() || m_value_was_set)
        {
            if (m_options.Test (eOptionEncodeCharacterEscapeSequences))
            {
                std::string expanded_escape_value;
                Args::ExpandEscapedCharacters(m_current_value.c_str(), expanded_escape_value);
                if (dump_mask & eDumpOptionRaw)
                    strm.Printf ("%s", expanded_escape_value.c_str());
                else
                    strm.Printf ("\"%s\"", expanded_escape_value.c_str());                
            }
            else
            {
                if (dump_mask & eDumpOptionRaw)
                    strm.Printf ("%s", m_current_value.c_str());
                else
                    strm.Printf ("\"%s\"", m_current_value.c_str());
            }
        }
    }
}

Error
OptionValueString::SetValueFromCString (const char *value_cstr,
                                        VarSetOperationType op)
{
    Error error;

    std::string value_str_no_quotes;
    if (value_cstr)
    {
        switch (value_cstr[0])
        {
        case '"':
        case '\'':
            {
                size_t len = strlen(value_cstr);
                if (len <= 1 || value_cstr[len-1] != value_cstr[0])
                {
                    error.SetErrorString("mismatched quotes");
                    return error;
                }
                value_str_no_quotes.assign (value_cstr + 1, len - 2);
                value_cstr = value_str_no_quotes.c_str();
            }
            break;
        }
    }

    switch (op)
    {
    case eVarSetOperationInvalid:
    case eVarSetOperationInsertBefore:
    case eVarSetOperationInsertAfter:
    case eVarSetOperationRemove:
        if (m_validator)
        {
            error = m_validator(value_cstr,m_validator_baton);
            if (error.Fail())
                return error;
        }
        error = OptionValue::SetValueFromCString (value_cstr, op);
        break;

    case eVarSetOperationAppend:
        {
        std::string new_value(m_current_value);
        if (value_cstr && value_cstr[0])
        {
            if (m_options.Test (eOptionEncodeCharacterEscapeSequences))
            {
                std::string str;
                Args::EncodeEscapeSequences (value_cstr, str);
                new_value.append(str);
            }
            else
                new_value.append(value_cstr);
        }
        if (m_validator)
        {
            error = m_validator(new_value.c_str(),m_validator_baton);
            if (error.Fail())
                return error;
        }
        m_current_value.assign(new_value);
        }
        break;

    case eVarSetOperationClear:
        Clear ();
        break;

    case eVarSetOperationReplace:
    case eVarSetOperationAssign:
        if (m_validator)
        {
            error = m_validator(value_cstr,m_validator_baton);
            if (error.Fail())
                return error;
        }
        m_value_was_set = true;
        if (m_options.Test (eOptionEncodeCharacterEscapeSequences))
        {
            Args::EncodeEscapeSequences (value_cstr, m_current_value);
        }
        else
        {
            SetCurrentValue (value_cstr);
        }
        break;
    }
    return error;
}


lldb::OptionValueSP
OptionValueString::DeepCopy () const
{
    return OptionValueSP(new OptionValueString(*this));
}

Error
OptionValueString::SetCurrentValue (const char *value)
{
    if (m_validator)
    {
        Error error(m_validator(value,m_validator_baton));
        if (error.Fail())
            return error;
    }
    if (value && value[0])
        m_current_value.assign (value);
    else
        m_current_value.clear();
    return Error();
}

Error
OptionValueString::AppendToCurrentValue (const char *value)
{
    if (value && value[0])
    {
        if (m_validator)
        {
            std::string new_value(m_current_value);
            new_value.append(value);
            Error error(m_validator(value,m_validator_baton));
            if (error.Fail())
                return error;
            m_current_value.assign(new_value);
        }
        else
            m_current_value.append (value);
    }
    return Error();
}
