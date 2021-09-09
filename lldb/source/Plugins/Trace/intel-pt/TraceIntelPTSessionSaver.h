//===-- TraceIntelPTSessionSaver.h ---------------------------*- C++ //-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPTSESSIONSAVER_H
#define LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPTSESSIONSAVER_H

#include "TraceIntelPT.h"

#include "../common/TraceJSONStructs.h"

namespace lldb_private {
namespace trace_intel_pt {

class TraceIntelPT;

class TraceIntelPTSessionSaver {

public:
  /// Save the Intel PT trace of a live process to the specified directory,
  /// which will be created if needed. This will also create a file
  /// \a <directory>/trace.json with the main properties of the trace
  /// session, along with others files which contain the actual trace data.
  /// The trace.json file can be used later as input for the "trace load"
  /// command to load the trace in LLDB.
  ///
  /// \param[in] trace_ipt
  ///     The Intel PT trace to be saved to disk.
  ///
  /// \param[in] directory
  ///     The directory where the trace files will be saved.
  ///
  /// \return
  ///     \a llvm::success if the operation was successful, or an \a llvm::Error
  ///     otherwise.
  llvm::Error SaveToDisk(TraceIntelPT &trace_ipt, FileSpec directory);

private:
  /// Build trace section of the intel-pt trace session description file.
  ///
  /// \param[in] trace_ipt
  ///     The Intel PT trace.
  ///
  /// \return
  ///     The trace section  an \a llvm::Error in case of failures.
  llvm::Expected<JSONTraceIntelPTTrace>
  BuildTraceSection(TraceIntelPT &trace_ipt);
};

} // namespace trace_intel_pt
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPTSESSIONSAVER_H
