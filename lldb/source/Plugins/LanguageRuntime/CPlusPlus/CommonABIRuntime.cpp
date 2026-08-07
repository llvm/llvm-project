//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommonABIRuntime.h"

#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "lldb/Core/Module.h"
#include "lldb/Utility/LLDBLog.h"

using namespace lldb;
using namespace lldb_private;

CommonABIRuntime::CommonABIRuntime(Process *process) : m_process(process) {}

lldb::TypeSP CommonABIRuntime::LookupTypeByName(llvm::StringRef type_name,
                                                lldb::ModuleSP preferred_module,
                                                bool &any_found) const {
  Log *log = GetLog(LLDBLog::Object);

  any_found = false;

  ConstString const_lookup_name(type_name);
  TypeList class_types;
  // First look in the module that the vtable symbol came from and
  // look for a single exact match.
  TypeResults results;
  TypeQuery query(const_lookup_name.GetStringRef(),
                  TypeQueryOptions::e_exact_match |
                      TypeQueryOptions::e_strict_namespaces |
                      TypeQueryOptions::e_find_one);
  if (preferred_module) {
    preferred_module->FindTypes(query, results);
    TypeSP type_sp = results.GetFirstType();
    if (type_sp)
      class_types.Insert(type_sp);
  }

  // If we didn't find a symbol, then move on to the entire module
  // list in the target and get as many unique matches as possible
  if (class_types.Empty()) {
    query.SetFindOne(false);
    m_process->GetTarget().GetImages().FindTypes(nullptr, query, results);
    for (const auto &type_sp : results.GetTypeMap().Types())
      class_types.Insert(type_sp);
  }

  lldb::TypeSP type_sp;
  if (class_types.Empty()) {
    LLDB_LOG(log, "Failed to find '{0}'", type_name);
    return {};
  }
  any_found = true;

  if (class_types.GetSize() == 1) {
    type_sp = class_types.GetTypeAtIndex(0);
    if (!type_sp)
      return {};
    if (!TypeSystemClang::IsCXXClassType(type_sp->GetForwardCompilerType()))
      return {};

    return type_sp;
  }

  if (log) {
    LLDB_LOG(log,
             "'{0}' has multiple matching dynamic "
             "types:",
             type_name);
    for (size_t i = 0; i < class_types.GetSize(); i++) {
      type_sp = class_types.GetTypeAtIndex(i);
      if (type_sp) {
        LLDB_LOG(log, "[{0}]: uid={1:x}, type-name='{2}'", i, type_sp->GetID(),
                 type_sp->GetName());
      }
    }
  }

  for (size_t i = 0; i < class_types.GetSize(); i++) {
    type_sp = class_types.GetTypeAtIndex(i);
    if (type_sp) {
      if (TypeSystemClang::IsCXXClassType(type_sp->GetForwardCompilerType())) {
        LLDB_LOG(log,
                 "'{0}' has multiple matching dynamic types, "
                 "picking this one: [{1}] uid={2:x}, type-name='{3}'\n",
                 type_name, i, type_sp->GetID(), type_sp->GetName());
        return type_sp;
      }
    }
  }

  LLDB_LOG(log,
           "'{0}' has multiple matching dynamic types, didn't find a C++ match",
           type_name);
  return {};
}
