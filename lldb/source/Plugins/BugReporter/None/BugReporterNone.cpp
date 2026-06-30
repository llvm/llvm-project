//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/BugReporter/None/BugReporterNone.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/Config.h"

using namespace lldb_private;

LLDB_PLUGIN_DEFINE(BugReporterNone)

void BugReporterNone::Initialize() {
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(), "Fallback used when no bug tracker is configured.",
      &BugReporterNone::CreateInstance);
}

void BugReporterNone::Terminate() {
  PluginManager::UnregisterPlugin(&BugReporterNone::CreateInstance);
}

std::unique_ptr<BugReporter> BugReporterNone::CreateInstance() {
  return std::make_unique<BugReporterNone>();
}

llvm::Error BugReporterNone::File(const Diagnostics::Report &) {
  return llvm::createStringError("no bug tracker is configured: please file a "
                                 "bug manually on " LLDB_BUG_REPORT_URL);
}
