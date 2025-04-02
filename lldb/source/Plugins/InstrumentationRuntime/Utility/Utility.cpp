//===-- Utility.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Utility.h"

#include "lldb/Core/Module.h"
#include "lldb/Target/Target.h"

namespace lldb_private {

lldb::ModuleSP GetPreferredAsanModule(const Target &target) {
  // Currently only supported on Darwin.
  if (!target.GetArchitecture().GetTriple().isOSDarwin())
    return nullptr;

  lldb::ModuleSP module;
  llvm::Regex pattern(R"(libclang_rt\.asan_.*_dynamic\.dylib)");
  target.GetImages().ForEach([&](const lldb::ModuleSP &m) {
    if (pattern.match(m->GetFileSpec().GetFilename().GetStringRef())) {
      module = m;
      return false;
    }

    return true;
  });

  return module;
}

} // namespace lldb_private
