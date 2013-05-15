//===-- OptionValueBoolean.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionValueBoolean.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Stream.h"
#include "lldb/Core/StringList.h"
#include "lldb/Interpreter/Args.h"

using namespace lldb;
using namespace lldb_private;

void
OptionValueBoolean::DumpValue (const ExecutionContext *exe_ctx, Stream &strm, uint32_t dump_mask)
{
    if (dump_mask & eDumpOptionType)
        strm.Printf ("(%s)", GetTypeAsCString ());
//    if (dump_mask & eDumpOptionName)
//        DumpQualifiedName (strm);
    if (dump_mask & eDumpOptionValue)
    {
        if (dump_mask & eDumpOptionType)
            strm.PutCString (" = ");
        strm.PutCString (m_current_value ? "true" : "false");
    }
}

Error
OptionValueBoolean::SetValueFromCString (const char *value_cstr,
                                         VarSetOperationType op)
{
    Error error;
    switch (op)
    {
    case eVarSetOperationClear:
        Clear();
        break;

    case eVarSetOperationReplace:
    case eVarSetOperationAssign:
        {
            bool success = false;
            bool value = Args::StringToBoolean(value_cstr, false, &success);
            if (success)
            {
                m_value_was_set = true;
                m_current_value = value;
            }
            else
            {
                if (value_cstr == NULL)
                    error.SetErrorString ("invalid boolean string value: NULL");
                else if (value_cstr[0] == '\0')
                    error.SetErrorString ("invalid boolean string value <empty>");
                else
                    error.SetErrorStringWithFormat ("invalid boolean string value: '%s'", value_cstr);
            }
        }
        break;

    case eVarSetOperationInsertBefore:
    case eVarSetOperationInsertAfter:
    case eVarSetOperationRemove:
    case eVarSetOperationAppend:
    case eVarSetOperationInvalid:
        error = OptionValue::SetValueFromCString (value_cstr, op);
        break;
    }
    return error;
}

lldb::OptionValueSP
OptionValueBoolean::DeepCopy () const
{
    return OptionValueSP(new OptionValueBoolean(*this));
}

size_t
OptionValueBoolean::AutoComplete (CommandInterpreter &interpreter,
                                  const char *s,
                                  int match_start_point,
                                  int max_return_elements,
                                  bool &word_complete,
                                  StringList &matches)
{
    word_complete = false;
    matches.Clear();
    struct StringEntry {
        const char *string;
        const size_t length;
    };
    static const StringEntry g_autocomplete_entries[] =
    {
        { "true" , 4 },
        { "false", 5 },
        { "on"   , 2 },
        { "off"  , 3 },
        { "yes"  , 3 },
        { "no"   , 2 },
        { "1"    , 1 },
        { "0"    , 1 },
    };
    const size_t k_num_autocomplete_entries = sizeof(g_autocomplete_entries)/sizeof(StringEntry);
    
    if (s && s[0])
    {
        const size_t s_len = strlen(s);
        for (size_t i=0; i<k_num_autocomplete_entries; ++i)
        {
            if (s_len <= g_autocomplete_entries[i].length)
                if (::strncasecmp(s, g_autocomplete_entries[i].string, s_len) == 0)
                    matches.AppendString(g_autocomplete_entries[i].string);
        }
    }
    else
    {
        // only suggest "true" or "false" by default
        for (size_t i=0; i<2; ++i)
            matches.AppendString(g_autocomplete_entries[i].string);
    }
    return matches.GetSize();
}



