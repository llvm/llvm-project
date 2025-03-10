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

///< On Darwin, if LLDB loaded libclang_rt, it's coming from a locally built
///< compiler-rt, and we should prefer it in favour of the system sanitizers.
///< This helper searches the target for such a dylib. Returns nullptr if no
///< such dylib was found.
lldb::ModuleSP GetPreferredAsanModule(const Target &target) {
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
