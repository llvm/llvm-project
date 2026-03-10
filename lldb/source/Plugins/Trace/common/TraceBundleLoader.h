//===-- TraceBundleLoader.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TRACE_COMMON_TRACEBUNDLELOADER_H
#define LLDB_SOURCE_PLUGINS_TRACE_COMMON_TRACEBUNDLELOADER_H

#include "TraceJSONStructs.h"
#include "lldb/lldb-types.h"
#include "llvm/Support/JSON.h"

namespace lldb_private {

class TraceBundleLoader {
public:
  /// Helper struct holding the objects created when parsing a process
  struct ParsedProcess {
    lldb::TargetSP target_sp;
    std::vector<lldb::ThreadPostMortemTraceSP> threads;
  };

  /// \param[in] debugger
  ///   The debugger that will own the targets to create.
  ///
  /// \param[in] bundle_dir
  ///   The folder where the trace bundle is located.
  TraceBundleLoader(Debugger &debugger, llvm::StringRef bundle_dir)
      : m_debugger(debugger), m_bundle_dir(bundle_dir) {}

protected:
  /// Resolve non-absolute paths relative to the bundle folder.
  FileSpec NormalizePath(const std::string &path);

  /// Create an empty Process object with given pid and target.
  llvm::Expected<ParsedProcess> CreateEmptyProcess(lldb::pid_t pid,
                                                   llvm::StringRef triple);

  /// Create a module associated with the given \p target using the definition
  /// from \p module.
  llvm::Error ParseModule(Target &target, const JSONModule &module);

  Debugger &m_debugger;
  const std::string m_bundle_dir;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TRACE_COMMON_TRACEBUNDLELOADER_H
