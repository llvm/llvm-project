//===-- Utility.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_INSTRUMENTATIONRUNTIME_UTILITY_UTILITY_H
#define LLDB_SOURCE_PLUGINS_INSTRUMENTATIONRUNTIME_UTILITY_UTILITY_H

#include <tuple>

#include "lldb/lldb-forward.h"
#include "lldb/lldb-private-enumerations.h"

namespace lldb_private {

/// On Darwin, if LLDB loaded libclang_rt, it's coming from a locally built
/// compiler-rt, and we should prefer it in favour of the system sanitizers
/// when running InstrumentationRuntime utility expressions that use symbols
/// from the sanitizer libraries. This helper searches the target for such a
/// dylib. Returns nullptr if no such dylib was found.
std::tuple<lldb::ModuleSP, HistoryPCType>
GetPreferredAsanModule(const Target &target);

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_INSTRUMENTATIONRUNTIME_UTILITY_UTILITY_H
