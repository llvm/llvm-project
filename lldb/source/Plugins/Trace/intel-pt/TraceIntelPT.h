//===-- TraceIntelPT.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPT_H
#define LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPT_H

#include "intel-pt.h"
#include "llvm/ADT/Optional.h"

#include "TraceIntelPTSessionFileParser.h"
#include "lldb/Target/Trace.h"
#include "lldb/lldb-private.h"

namespace lldb_private {
namespace trace_intel_pt {

class TraceIntelPT : public Trace {
public:
  void Dump(Stream *s) const override;

  ~TraceIntelPT() override = default;

  /// PluginInterface protocol
  /// \{
  ConstString GetPluginName() override;

  static void Initialize();

  static void Terminate();

  /// Create an instance of this class.
  ///
  /// \param[in] trace_session_file
  ///     The contents of the trace session file. See \a Trace::FindPlugin.
  ///
  /// \param[in] session_file_dir
  ///     The path to the directory that contains the session file. It's used to
  ///     resolved relative paths in the session file.
  ///
  /// \param[in] debugger
  ///     The debugger instance where new Targets will be created as part of the
  ///     JSON data parsing.
  ///
  /// \return
  ///     A trace instance or an error in case of failures.
  static llvm::Expected<lldb::TraceSP>
  CreateInstance(const llvm::json::Value &trace_session_file,
                 llvm::StringRef session_file_dir, Debugger &debugger);

  static ConstString GetPluginNameStatic();

  uint32_t GetPluginVersion() override;
  /// \}

  llvm::StringRef GetSchema() override;

  TraceIntelPT(const pt_cpu &pt_cpu, const std::vector<lldb::TargetSP> &targets)
      : m_pt_cpu(pt_cpu) {
    for (const lldb::TargetSP &target_sp : targets)
      m_targets.push_back(target_sp);
  }

private:
  pt_cpu m_pt_cpu;
  std::vector<std::weak_ptr<Target>> m_targets;
};

} // namespace trace_intel_pt
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPT_H
