//===-- OptionValueFormat.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "lldb/Interpreter/OptionValueFormat.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Stream.h"
#include "lldb/DataFormatters/FormatManager.h"
#include "lldb/Interpreter/Args.h"

using namespace lldb;
using namespace lldb_private;

void
OptionValueFormat::DumpValue (const ExecutionContext *exe_ctx, Stream &strm, uint32_t dump_mask)
{
    if (dump_mask & eDumpOptionType)
        strm.Printf ("(%s)", GetTypeAsCString ());
    if (dump_mask & eDumpOptionValue)
    {
        if (dump_mask & eDumpOptionType)
            strm.PutCString (" = ");
        strm.PutCString (FormatManager::GetFormatAsCString (m_current_value));
    }
}

Error
OptionValueFormat::SetValueFromCString (const char *value_cstr, VarSetOperationType op)
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
            Format new_format;
            error = Args::StringToFormat (value_cstr, new_format, NULL);
            if (error.Success())
            {
                m_value_was_set = true;
                m_current_value = new_format;
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
OptionValueFormat::DeepCopy () const
{
    return OptionValueSP(new OptionValueFormat(*this));
}

