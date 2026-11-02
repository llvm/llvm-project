//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_BUGREPORTER_GITHUB_GITHUBREPORTER_H
#define LLDB_SOURCE_PLUGINS_BUGREPORTER_GITHUB_GITHUBREPORTER_H

#include "lldb/Core/BugReporter.h"

namespace lldb_private {

/// Opens a pre-filled github.com/llvm/llvm-project "new issue" page. The body
/// carries a short summary and points at the on-disk bundle to attach, since
/// large artifacts cannot travel in the URL.
class GitHubReporter : public BugReporter {
public:
  static void Initialize();
  static void Terminate();
  static llvm::StringRef GetPluginNameStatic() { return "github"; }
  static std::unique_ptr<BugReporter> CreateInstance();

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  llvm::Error File(const Diagnostics::Report &report) override;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_BUGREPORTER_GITHUB_GITHUBREPORTER_H
