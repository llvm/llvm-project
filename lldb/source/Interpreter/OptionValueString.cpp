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
    switch (op)
    {
    case eVarSetOperationInvalid:
    case eVarSetOperationInsertBefore:
    case eVarSetOperationInsertAfter:
    case eVarSetOperationRemove:
        error = OptionValue::SetValueFromCString (value_cstr, op);
        break;

    case eVarSetOperationAppend:
        if (value_cstr && value_cstr[0])
        {
            if (m_options.Test (eOptionEncodeCharacterEscapeSequences))
            {
                std::string str;
                Args::EncodeEscapeSequences (value_cstr, str);
                m_current_value += str;
            }
            else
                m_current_value += value_cstr;
        }
        break;

    case eVarSetOperationClear:
        Clear ();
        break;

    case eVarSetOperationReplace:
    case eVarSetOperationAssign:
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
