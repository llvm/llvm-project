//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_CPLUSPLUS_COMMONABIRUNTIME_H
#define LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_CPLUSPLUS_COMMONABIRUNTIME_H

#include "lldb/Target/Process.h"

#include <mutex>

namespace lldb_private {

class CommonABIRuntime {
public:
  virtual ~CommonABIRuntime() = default;

protected:
  CommonABIRuntime(Process *process);

  /// Find a type by its name, preferably in `preferred_module`.
  ///
  /// `any_found` will be set to `true` if any type with the name is found.
  /// Even if a type with the name was found, this function may return an empty
  /// `TypeSP` if the type is not a C++ type.
  lldb::TypeSP LookupTypeByName(llvm::StringRef type_name,
                                lldb::ModuleSP preferred_module,
                                bool &any_found) const;

protected:
  Process *m_process;
  std::mutex m_mutex;
};

} // namespace lldb_private

#endif
