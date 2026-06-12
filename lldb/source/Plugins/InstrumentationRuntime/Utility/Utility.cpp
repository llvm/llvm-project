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

std::tuple<lldb::ModuleSP, HistoryPCType>
GetPreferredAsanModule(const Target &target) {
  // Currently only Darwin provides ASan runtime support as part of the OS
  // (libsanitizers).
  if (!target.GetArchitecture().GetTriple().isOSDarwin())
    return {nullptr, HistoryPCType::Calls};

  lldb::ModuleSP module;
  llvm::Regex pattern(R"(libclang_rt\.asan_.*_dynamic\.dylib)");
  target.GetImages().ForEach([&](const lldb::ModuleSP &m) {
    if (pattern.match(m->GetFileSpec().GetFilename().GetStringRef())) {
      module = m;
      return IterationAction::Stop;
    }

    return IterationAction::Continue;
  });

  // `Calls` - The ASan compiler-rt runtime already massages the return
  //   addresses into call addresses, so we don't want LLDB's unwinder to try to
  //   locate the previous instruction again as this might lead to us reporting
  //   a different line.
  // `ReturnsNoZerothFrame` - Darwin, but not ASan compiler-rt implies
  //   libsanitizers which collects return addresses.  It also discards a few
  //   non-user frames at the top of the stack.
  auto pc_type =
      (module ? HistoryPCType::Calls : HistoryPCType::ReturnsNoZerothFrame);
  return {module, pc_type};
}

} // namespace lldb_private
