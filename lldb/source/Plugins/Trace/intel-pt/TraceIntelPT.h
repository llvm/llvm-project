//===-- TraceIntelPT.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_TraceIntelPT_h_
#define liblldb_TraceIntelPT_h_

#include "intel-pt.h"
#include "llvm/ADT/Optional.h"

#include "TraceIntelPTSettingsParser.h"
#include "lldb/Target/Trace.h"
#include "lldb/lldb-private.h"

class TraceIntelPT : public lldb_private::Trace {
public:
  void Dump(lldb_private::Stream *s) const override;

  /// PluginInterface protocol
  /// \{
  lldb_private::ConstString GetPluginName() override;

  static void Initialize();

  static void Terminate();

  static lldb::TraceSP CreateInstance();

  static lldb_private::ConstString GetPluginNameStatic();

  uint32_t GetPluginVersion() override;
  /// \}

protected:
  TraceIntelPT() : Trace() {}

  std::unique_ptr<lldb_private::TraceSettingsParser> CreateParser() override;

private:
  friend class TraceIntelPTSettingsParser;
  pt_cpu m_pt_cpu;
};

#endif // liblldb_TraceIntelPT_h_
