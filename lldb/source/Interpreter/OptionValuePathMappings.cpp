//===-- OptionValuePathMappings.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionValuePathMappings.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Stream.h"
#include "lldb/Interpreter/Args.h"

using namespace lldb;
using namespace lldb_private;

void
OptionValuePathMappings::DumpValue (const ExecutionContext *exe_ctx, Stream &strm, uint32_t dump_mask)
{
    if (dump_mask & eDumpOptionType)
        strm.Printf ("(%s)", GetTypeAsCString ());
    if (dump_mask & eDumpOptionValue)
    {
        if (dump_mask & eDumpOptionType)
            strm.Printf (" =%s", (m_path_mappings.GetSize() > 0) ? "\n" : "");
        m_path_mappings.Dump(&strm);
    }
}

Error
OptionValuePathMappings::SetValueFromCString (const char *value, VarSetOperationType op)
{
    Error error;
    Args args(value);
    const size_t argc = args.GetArgumentCount();

    switch (op)
    {
        case eVarSetOperationClear:
            Clear ();
            break;
            
        case eVarSetOperationReplace:
            // Must be at least one index + 1 pair of paths, and the pair count must be even
            if (argc >= 3 && (((argc - 1) & 1) == 0))
            {
                uint32_t idx = Args::StringToUInt32(args.GetArgumentAtIndex(0), UINT32_MAX);
                const uint32_t count = m_path_mappings.GetSize();
                if (idx > count)
                {
                    error.SetErrorStringWithFormat("invalid file list index %u, index must be 0 through %u", idx, count);
                }
                else
                {
                    for (size_t i=1; i<argc; i += 2, ++idx)
                    {
                        ConstString a(args.GetArgumentAtIndex(i));
                        ConstString b(args.GetArgumentAtIndex(i+1));
                        if (!m_path_mappings.Replace (a, b, idx, m_notify_changes))
                            m_path_mappings.Append(a, b, m_notify_changes);
                    }
                }
            }
            else
            {
                error.SetErrorString("replace operation takes an array index followed by one or more path pairs");
            }
            break;
            
            
            
        case eVarSetOperationAssign:
            if (argc < 2 || (argc & 1))
            {
                error.SetErrorString("assign operation takes one or more path pairs");
                break;
            }
            m_path_mappings.Clear(m_notify_changes);
            // Fall through to append case
        case eVarSetOperationAppend:
            if (argc < 2 || (argc & 1))
            {
                error.SetErrorString("append operation takes one or more path pairs");
                break;
            }
            else
            {
                for (size_t i=0; i<argc; i += 2)
                {
                    ConstString a(args.GetArgumentAtIndex(i));
                    ConstString b(args.GetArgumentAtIndex(i+1));
                    m_path_mappings.Append(a, b, m_notify_changes);
                    m_value_was_set = true;
                }
            }
            break;
            
        case eVarSetOperationInsertBefore:
        case eVarSetOperationInsertAfter:
            // Must be at least one index + 1 pair of paths, and the pair count must be even
            if (argc >= 3 && (((argc - 1) & 1) == 0))
            {
                uint32_t idx = Args::StringToUInt32(args.GetArgumentAtIndex(0), UINT32_MAX);
                const uint32_t count = m_path_mappings.GetSize();
                if (idx > count)
                {
                    error.SetErrorStringWithFormat("invalid file list index %u, index must be 0 through %u", idx, count);
                }
                else
                {
                    if (op == eVarSetOperationInsertAfter)
                        ++idx;
                    for (size_t i=1; i<argc; i += 2, ++idx)
                    {
                        ConstString a(args.GetArgumentAtIndex(i));
                        ConstString b(args.GetArgumentAtIndex(i+1));
                        m_path_mappings.Insert (a, b, idx, m_notify_changes);
                    }
                }
            }
            else
            {
                error.SetErrorString("insert operation takes an array index followed by one or more path pairs");
            }
            break;
            
        case eVarSetOperationRemove:
            if (argc > 0)
            {
                std::vector<int> remove_indexes;
                bool all_indexes_valid = true;
                size_t i;
                for (i=0; all_indexes_valid && i<argc; ++i)
                {
                    const int idx = Args::StringToSInt32(args.GetArgumentAtIndex(i), INT32_MAX);
                    if (idx == INT32_MAX)
                        all_indexes_valid = false;
                    else
                        remove_indexes.push_back(idx);
                }
                
                if (all_indexes_valid)
                {
                    size_t num_remove_indexes = remove_indexes.size();
                    if (num_remove_indexes)
                    {
                        // Sort and then erase in reverse so indexes are always valid
                        std::sort(remove_indexes.begin(), remove_indexes.end());
                        for (int j=num_remove_indexes-1; j<num_remove_indexes; ++j)
                        {
                            m_path_mappings.Remove (j, m_notify_changes);
                        }
                    }
                }
                else
                {
                    error.SetErrorStringWithFormat("invalid array index '%s', aborting remove operation", args.GetArgumentAtIndex(i));
                }
            }
            else
            {
                error.SetErrorString("remove operation takes one or more array index");
            }
            break;

        case eVarSetOperationInvalid:
            error = OptionValue::SetValueFromCString (value, op);
            break;
    }
    return error;

    m_value_was_set = true;
    return Error();
}

lldb::OptionValueSP
OptionValuePathMappings::DeepCopy () const
{
    return OptionValueSP(new OptionValuePathMappings(*this));
}
