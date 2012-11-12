//===-- AuxVector.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

// C++ Includes
// Other libraries and framework includes
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Log.h"
#include "lldb/Target/Process.h"

#include "AuxVector.h"

using namespace lldb;
using namespace lldb_private;

static bool
GetMaxU64(DataExtractor &data,
          uint32_t *offset, uint64_t *value, unsigned int byte_size)
{
    uint32_t saved_offset = *offset;
    *value = data.GetMaxU64(offset, byte_size);
    return *offset != saved_offset;
}

static bool
ParseAuxvEntry(DataExtractor &data, AuxVector::Entry &entry, 
               uint32_t *offset, unsigned int byte_size)
{
    if (!GetMaxU64(data, offset, &entry.type, byte_size))
        return false;

    if (!GetMaxU64(data, offset, &entry.value, byte_size))
        return false;

    return true;
}

DataBufferSP
AuxVector::GetAuxvData()
{

    return lldb_private::Host::GetAuxvData(m_process);
}

void
AuxVector::ParseAuxv(DataExtractor &data)
{
    const unsigned int byte_size  = m_process->GetAddressByteSize();
    uint32_t offset = 0;

    for (;;) 
    {
        Entry entry;

        if (!ParseAuxvEntry(data, entry, &offset, byte_size))
            break;

        if (entry.type == AT_NULL)
            break;

        if (entry.type == AT_IGNORE)
            continue;

        m_auxv.push_back(entry);
    }
}

AuxVector::AuxVector(Process *process)
    : m_process(process)
{
    DataExtractor data;
    LogSP log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_DYNAMIC_LOADER));

    data.SetData(GetAuxvData());
    data.SetByteOrder(m_process->GetByteOrder());
    data.SetAddressByteSize(m_process->GetAddressByteSize());
    
    ParseAuxv(data);

    if (log)
        DumpToLog(log);
}

AuxVector::iterator 
AuxVector::FindEntry(EntryType type) const
{
    for (iterator I = begin(); I != end(); ++I)
    {
        if (I->type == static_cast<uint64_t>(type))
            return I;
    }

    return end();
}

void
AuxVector::DumpToLog(LogSP log) const
{
    if (!log)
        return;

    log->PutCString("AuxVector: ");
    for (iterator I = begin(); I != end(); ++I)
    {
        log->Printf("   %s [%llu]: %llx", GetEntryName(*I), I->type, I->value);
    }
}

const char *
AuxVector::GetEntryName(EntryType type)
{
    const char *name;

#define ENTRY_NAME(_type) _type: name = #_type
    switch (type) 
    {
    default:
        name = "unkown";
        break;

    case ENTRY_NAME(AT_NULL);   break;
    case ENTRY_NAME(AT_IGNORE); break;
    case ENTRY_NAME(AT_EXECFD); break;
    case ENTRY_NAME(AT_PHDR);   break;
    case ENTRY_NAME(AT_PHENT);  break;
    case ENTRY_NAME(AT_PHNUM);  break;
    case ENTRY_NAME(AT_PAGESZ); break;
    case ENTRY_NAME(AT_BASE);   break;
    case ENTRY_NAME(AT_FLAGS);  break;
    case ENTRY_NAME(AT_ENTRY);  break;
    case ENTRY_NAME(AT_NOTELF); break;
    case ENTRY_NAME(AT_UID);    break;
    case ENTRY_NAME(AT_EUID);   break;
    case ENTRY_NAME(AT_GID);    break;
    case ENTRY_NAME(AT_EGID);   break;
    case ENTRY_NAME(AT_CLKTCK); break;
    }
#undef ENTRY_NAME

    return name;
}

