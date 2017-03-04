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
#include "lldb/Target/Process.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Log.h"

#if defined(__linux__) || defined(__FreeBSD__)
#include "Plugins/Process/elf-core/ProcessElfCore.h"
#endif

#include "AuxVector.h"

using namespace lldb;
using namespace lldb_private;

static bool GetMaxU64(DataExtractor &data, lldb::offset_t *offset_ptr,
                      uint64_t *value, unsigned int byte_size) {
  lldb::offset_t saved_offset = *offset_ptr;
  *value = data.GetMaxU64(offset_ptr, byte_size);
  return *offset_ptr != saved_offset;
}

static bool ParseAuxvEntry(DataExtractor &data, AuxVector::Entry &entry,
                           lldb::offset_t *offset_ptr, unsigned int byte_size) {
  if (!GetMaxU64(data, offset_ptr, &entry.type, byte_size))
    return false;

  if (!GetMaxU64(data, offset_ptr, &entry.value, byte_size))
    return false;

  return true;
}

DataBufferSP AuxVector::GetAuxvData() {
  if (m_process)
    return m_process->GetAuxvData();
  else
    return DataBufferSP();
}

void AuxVector::ParseAuxv(DataExtractor &data) {
  const unsigned int byte_size = m_process->GetAddressByteSize();
  lldb::offset_t offset = 0;

  for (;;) {
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

AuxVector::AuxVector(Process *process) : m_process(process) {
  DataExtractor data;
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_DYNAMIC_LOADER));

  data.SetData(GetAuxvData());
  data.SetByteOrder(m_process->GetByteOrder());
  data.SetAddressByteSize(m_process->GetAddressByteSize());

  ParseAuxv(data);

  if (log)
    DumpToLog(log);
}

AuxVector::iterator AuxVector::FindEntry(EntryType type) const {
  for (iterator I = begin(); I != end(); ++I) {
    if (I->type == static_cast<uint64_t>(type))
      return I;
  }

  return end();
}

void AuxVector::DumpToLog(Log *log) const {
  if (!log)
    return;

  log->PutCString("AuxVector: ");
  for (iterator I = begin(); I != end(); ++I) {
    log->Printf("   %s [%" PRIu64 "]: %" PRIx64, GetEntryName(*I), I->type,
                I->value);
  }
}

const char *AuxVector::GetEntryName(EntryType type) {
  const char *name = "AT_???";

#define ENTRY_NAME(_type)                                                      \
  _type:                                                                       \
  name = #_type
  switch (type) {
    case ENTRY_NAME(AT_NULL);           break;
    case ENTRY_NAME(AT_IGNORE);         break;
    case ENTRY_NAME(AT_EXECFD);         break;
    case ENTRY_NAME(AT_PHDR);           break;
    case ENTRY_NAME(AT_PHENT);          break;
    case ENTRY_NAME(AT_PHNUM);          break;
    case ENTRY_NAME(AT_PAGESZ);         break;
    case ENTRY_NAME(AT_BASE);           break;
    case ENTRY_NAME(AT_FLAGS);          break;
    case ENTRY_NAME(AT_ENTRY);          break;
    case ENTRY_NAME(AT_NOTELF);         break;
    case ENTRY_NAME(AT_UID);            break;
    case ENTRY_NAME(AT_EUID);           break;
    case ENTRY_NAME(AT_GID);            break;
    case ENTRY_NAME(AT_EGID);           break;
    case ENTRY_NAME(AT_CLKTCK);         break;
    case ENTRY_NAME(AT_PLATFORM);       break;
    case ENTRY_NAME(AT_HWCAP);          break;
    case ENTRY_NAME(AT_FPUCW);          break;
    case ENTRY_NAME(AT_DCACHEBSIZE);    break;
    case ENTRY_NAME(AT_ICACHEBSIZE);    break;
    case ENTRY_NAME(AT_UCACHEBSIZE);    break;
    case ENTRY_NAME(AT_IGNOREPPC);      break;
    case ENTRY_NAME(AT_SECURE);         break;
    case ENTRY_NAME(AT_BASE_PLATFORM);  break;
    case ENTRY_NAME(AT_RANDOM);         break;
    case ENTRY_NAME(AT_EXECFN);         break;
    case ENTRY_NAME(AT_SYSINFO);        break;
    case ENTRY_NAME(AT_SYSINFO_EHDR);   break;
    case ENTRY_NAME(AT_L1I_CACHESHAPE); break;
    case ENTRY_NAME(AT_L1D_CACHESHAPE); break;
    case ENTRY_NAME(AT_L2_CACHESHAPE);  break;
    case ENTRY_NAME(AT_L3_CACHESHAPE);  break;
    }
#undef ENTRY_NAME

    return name;
}
