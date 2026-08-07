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

  lldb::TypeSP LookupTypeByName(llvm::StringRef type_name,
                                lldb::ModuleSP preferred_module) const;

protected:
  Process *m_process;
  std::mutex m_mutex;
};

} // namespace lldb_private

#endif
