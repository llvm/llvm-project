//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_BUGREPORTER_NONE_BUGREPORTERNONE_H
#define LLDB_SOURCE_PLUGINS_BUGREPORTER_NONE_BUGREPORTERNONE_H

#include "lldb/Core/BugReporter.h"

namespace lldb_private {

/// The fallback when no bug tracker is configured. Its File() returns an error
/// rather than succeeding, so the report command surfaces it to the user.
class BugReporterNone : public BugReporter {
public:
  static void Initialize();
  static void Terminate();
  static llvm::StringRef GetPluginNameStatic() { return "none"; }
  static std::unique_ptr<BugReporter> CreateInstance();

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  llvm::Error File(const Diagnostics::Report &report) override;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_BUGREPORTER_NONE_BUGREPORTERNONE_H
