//===- DataAccessProfGenerator.h - Data access profile ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Convert a `perf report -D` dump of hardware memory samples into a MemProf
// data-access profile for static data partitioning.
//
// https://discourse.llvm.org/t/rfc-profile-guided-static-data-partitioning/83744
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_PROFGEN_DATAACCESSPROFGENERATOR_H
#define LLVM_TOOLS_LLVM_PROFGEN_DATAACCESSPROFGENERATOR_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

namespace llvm {
namespace sampleprof {
class ProfiledBinary;
}

// Build a MemProf data-access profile from \p Binary and a `perf report -D`
// dump in \p PerfDumpPath, and write indexed MemProf v4 to \p OS. If
// \p PIDFilter is set, only samples of that process are used; mappings of
// every process are still tracked, so a forked child inherits its parent's.
Error generateDataAccessProf(sampleprof::ProfiledBinary &Binary,
                             StringRef PerfDumpPath, raw_fd_ostream &OS,
                             std::optional<int32_t> PIDFilter);

} // namespace llvm

#endif // LLVM_TOOLS_LLVM_PROFGEN_DATAACCESSPROFGENERATOR_H
