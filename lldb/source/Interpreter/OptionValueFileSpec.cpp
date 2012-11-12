//===-- OptionValueFileSpec.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionValueFileSpec.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/FormatManager.h"
#include "lldb/Core/State.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/CommandCompletions.h"

using namespace lldb;
using namespace lldb_private;


OptionValueFileSpec::OptionValueFileSpec () :
    OptionValue(),
    m_current_value (),
    m_default_value (),
    m_data_sp(),
    m_completion_mask (CommandCompletions::eDiskFileCompletion)
{
}

OptionValueFileSpec::OptionValueFileSpec (const FileSpec &value) :
    OptionValue(),
    m_current_value (value),
    m_default_value (value),
    m_data_sp(),
    m_completion_mask (CommandCompletions::eDiskFileCompletion)
{
}

OptionValueFileSpec::OptionValueFileSpec (const FileSpec &current_value,
                                          const FileSpec &default_value) :
    OptionValue(),
    m_current_value (current_value),
    m_default_value (default_value),
    m_data_sp(),
    m_completion_mask (CommandCompletions::eDiskFileCompletion)
{
}

void
OptionValueFileSpec::DumpValue (const ExecutionContext *exe_ctx, Stream &strm, uint32_t dump_mask)
{
    if (dump_mask & eDumpOptionType)
        strm.Printf ("(%s)", GetTypeAsCString ());
    if (dump_mask & eDumpOptionValue)
    {
        if (dump_mask & eDumpOptionType)
            strm.PutCString (" = ");

        if (m_current_value)
        {
            if (m_current_value.GetDirectory())
            {
                strm << '"' << m_current_value.GetDirectory();
                if (m_current_value.GetFilename())
                    strm << '/' << m_current_value.GetFilename();
                strm << '"';
            }
            else
            {
                strm << '"' << m_current_value.GetFilename() << '"';
            }
        }
    }
}

Error
OptionValueFileSpec::SetValueFromCString (const char *value_cstr,
                                          VarSetOperationType op)
{
    Error error;
    switch (op)
    {
    case eVarSetOperationClear:
        Clear ();
        break;
        
    case eVarSetOperationReplace:
    case eVarSetOperationAssign:
        if (value_cstr && value_cstr[0])
        {
            m_value_was_set = true;
            m_current_value.SetFile(value_cstr, value_cstr[0] == '~');
        }
        else
        {
            error.SetErrorString("invalid value string");
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
OptionValueFileSpec::DeepCopy () const
{
    return OptionValueSP(new OptionValueFileSpec(*this));
}


size_t
OptionValueFileSpec::AutoComplete (CommandInterpreter &interpreter,
                                   const char *s,
                                   int match_start_point,
                                   int max_return_elements,
                                   bool &word_complete,
                                   StringList &matches)
{
    word_complete = false;
    matches.Clear();
    CommandCompletions::InvokeCommonCompletionCallbacks (interpreter,
                                                         m_completion_mask,
                                                         s,
                                                         match_start_point,
                                                         max_return_elements,
                                                         NULL,
                                                         word_complete,
                                                         matches);
    return matches.GetSize();
}



const lldb::DataBufferSP &
OptionValueFileSpec::GetFileContents(bool null_terminate)
{
    if (!m_data_sp && m_current_value)
    {
        if (null_terminate)
            m_data_sp = m_current_value.ReadFileContentsAsCString();
        else
            m_data_sp = m_current_value.ReadFileContents();
    }
    return m_data_sp;
}


