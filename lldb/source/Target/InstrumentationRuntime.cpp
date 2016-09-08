//===-- InstrumentationRuntime.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/InstrumentationRuntime.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/Process.h"
#include "lldb/lldb-private.h"

using namespace lldb;
using namespace lldb_private;

void InstrumentationRuntime::ModulesDidLoad(
    lldb_private::ModuleList &module_list, lldb_private::Process *process,
    InstrumentationRuntimeCollection &runtimes) {
  InstrumentationRuntimeCreateInstance create_callback = nullptr;
  InstrumentationRuntimeGetType get_type_callback;
  for (uint32_t idx = 0;; ++idx) {
    create_callback =
        PluginManager::GetInstrumentationRuntimeCreateCallbackAtIndex(idx);
    if (create_callback == nullptr)
      break;
    get_type_callback =
        PluginManager::GetInstrumentationRuntimeGetTypeCallbackAtIndex(idx);
    InstrumentationRuntimeType type = get_type_callback();

    InstrumentationRuntimeCollection::iterator pos;
    pos = runtimes.find(type);
    if (pos == runtimes.end()) {
      runtimes[type] = create_callback(process->shared_from_this());
    }
  }
}

void InstrumentationRuntime::ModulesDidLoad(
    lldb_private::ModuleList &module_list) {}

bool InstrumentationRuntime::IsActive() { return false; }

lldb::ThreadCollectionSP
InstrumentationRuntime::GetBacktracesFromExtendedStopInfo(
    StructuredData::ObjectSP info) {
  return ThreadCollectionSP(new ThreadCollection());
}
