//===-- LocateSymbolFile.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/LocateSymbolFile.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/Progress.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/DataBuffer.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/Timer.h"
#include "lldb/Utility/UUID.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ThreadPool.h"

// From MacOSX system header "mach/machine.h"
typedef int cpu_type_t;
typedef int cpu_subtype_t;

using namespace lldb;
using namespace lldb_private;

void Symbols::DownloadSymbolFileAsync(const UUID &uuid) {
  if (!ModuleList::GetGlobalModuleListProperties().GetEnableBackgroundLookup())
    return;

  static llvm::SmallSet<UUID, 8> g_seen_uuids;
  static std::mutex g_mutex;
  Debugger::GetThreadPool().async([=]() {
    {
      std::lock_guard<std::mutex> guard(g_mutex);
      if (g_seen_uuids.count(uuid))
        return;
      g_seen_uuids.insert(uuid);
    }

    Status error;
    ModuleSpec module_spec;
    module_spec.GetUUID() = uuid;
    if (!Symbols::DownloadObjectAndSymbolFile(module_spec, error,
                                              /*force_lookup=*/true,
                                              /*copy_executable=*/false))
      return;

    if (error.Fail())
      return;

    Debugger::ReportSymbolChange(module_spec);
  });
}

#if !defined(__APPLE__)

bool Symbols::DownloadObjectAndSymbolFile(ModuleSpec &module_spec,
                                          Status &error, bool force_lookup,
                                          bool copy_executable) {
  // Fill in the module_spec.GetFileSpec() for the object file and/or the
  // module_spec.GetSymbolFileSpec() for the debug symbols file.
  return false;
}

#endif
