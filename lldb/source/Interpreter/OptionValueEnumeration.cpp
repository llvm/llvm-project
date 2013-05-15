//===-- OptionValueEnumeration.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionValueEnumeration.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/StringList.h"

using namespace lldb;
using namespace lldb_private;

OptionValueEnumeration::OptionValueEnumeration (const OptionEnumValueElement *enumerators,
                                                enum_type value) :
    OptionValue(),
    m_current_value (value),
    m_default_value (value),
    m_enumerations ()
{
    SetEnumerations(enumerators);
}

OptionValueEnumeration::~OptionValueEnumeration()
{
}

void
OptionValueEnumeration::DumpValue (const ExecutionContext *exe_ctx, Stream &strm, uint32_t dump_mask)
{
    if (dump_mask & eDumpOptionType)
        strm.Printf ("(%s)", GetTypeAsCString ());
    if (dump_mask & eDumpOptionValue)
    {
        if (dump_mask & eDumpOptionType)
            strm.PutCString (" = ");
        const size_t count = m_enumerations.GetSize ();
        for (size_t i=0; i<count; ++i)
        {
            if (m_enumerations.GetValueAtIndexUnchecked(i).value == m_current_value)
            {
                strm.PutCString(m_enumerations.GetCStringAtIndex(i));
                return;
            }
        }
        strm.Printf("%" PRIu64, (uint64_t)m_current_value);
    }
}

Error
OptionValueEnumeration::SetValueFromCString (const char *value, VarSetOperationType op)
{
    Error error;
    switch (op)
    {
        case eVarSetOperationClear:
            Clear ();
            break;
            
        case eVarSetOperationReplace:
        case eVarSetOperationAssign:
            if (value && value[0])
            {
                ConstString const_enumerator_name(value);
                const EnumerationMapEntry *enumerator_entry = m_enumerations.FindFirstValueForName (const_enumerator_name.GetCString());
                if (enumerator_entry)
                {
                    m_current_value = enumerator_entry->value.value;
                }
                else
                {
                    StreamString error_strm;
                    error_strm.Printf("invalid enumeration value '%s'", value);
                    const size_t count = m_enumerations.GetSize ();
                    if (count)
                    {
                        error_strm.Printf(", valid values are: %s", m_enumerations.GetCStringAtIndex(0));
                        for (size_t i=1; i<count; ++i)
                        {
                            error_strm.Printf (", %s", m_enumerations.GetCStringAtIndex(i));
                        }
                    }
                    error.SetErrorString(error_strm.GetData());
                }
            }
            else
            {
                error.SetErrorString("invalid enumeration value");
            }
            break;
            
        case eVarSetOperationInsertBefore:
        case eVarSetOperationInsertAfter:
        case eVarSetOperationRemove:
        case eVarSetOperationAppend:
        case eVarSetOperationInvalid:
            error = OptionValue::SetValueFromCString (value, op);
            break;
    }
    return error;
}

void
OptionValueEnumeration::SetEnumerations (const OptionEnumValueElement *enumerators)
{
    m_enumerations.Clear();
    if (enumerators)
    {
        for (size_t i=0; enumerators[i].string_value != NULL; ++i)
        {
            ConstString const_enumerator_name(enumerators[i].string_value);
            EnumeratorInfo enumerator_info = { enumerators[i].value, enumerators[i].usage };
            m_enumerations.Append (const_enumerator_name.GetCString(), enumerator_info);
        }
        m_enumerations.Sort();
    }
}


lldb::OptionValueSP
OptionValueEnumeration::DeepCopy () const
{
    return OptionValueSP(new OptionValueEnumeration(*this));
}

size_t
OptionValueEnumeration::AutoComplete (CommandInterpreter &interpreter,
                                      const char *s,
                                      int match_start_point,
                                      int max_return_elements,
                                      bool &word_complete,
                                      StringList &matches)
{
    word_complete = false;
    matches.Clear();
    
    const uint32_t num_enumerators = m_enumerations.GetSize();
    if (s && s[0])
    {
        const size_t s_len = strlen(s);
        for (size_t i=0; i<num_enumerators; ++i)
        {
            const char *name = m_enumerations.GetCStringAtIndex(i);
            if (::strncmp(s, name, s_len) == 0)
                matches.AppendString(name);
        }
    }
    else
    {
        // only suggest "true" or "false" by default
        for (size_t i=0; i<num_enumerators; ++i)
            matches.AppendString(m_enumerations.GetCStringAtIndex(i));
    }
    return matches.GetSize();
}




