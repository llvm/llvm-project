//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/BugReporter/GitHub/GitHubReporter.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/Host.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using namespace lldb_private;

LLDB_PLUGIN_DEFINE(GitHubReporter)

// A GitHub "new issue" URL is fetched with GET, so its length is bounded (the
// server rejects very long URLs). Keep the pre-filled body well under that so
// the rest of the URL always fits. The full payload lives in the bundle.
static constexpr size_t g_max_body_size = 6000;

void GitHubReporter::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                "File a bug as a GitHub issue.",
                                &GitHubReporter::CreateInstance);
}

void GitHubReporter::Terminate() {
  PluginManager::UnregisterPlugin(&GitHubReporter::CreateInstance);
}

std::unique_ptr<BugReporter> GitHubReporter::CreateInstance() {
  return std::make_unique<GitHubReporter>();
}

llvm::Error GitHubReporter::File(const Diagnostics::Report &report) {
  std::string body;
  llvm::raw_string_ostream os(body);
  os << "### LLDB version\n" << report.version << "\n\n";
  os << "### Host\n" << report.os << "\n\n";
  if (!report.invocation.empty())
    os << "### Invocation\n`" << report.invocation << "`\n\n";
  os << "### Diagnostics\n"
     << "Full diagnostics were written to:\n`" << report.attachments.directory
     << "`\nPlease attach the contents of that directory to this issue.\n";

  if (body.size() > g_max_body_size) {
    // Back up off any UTF-8 continuation bytes so truncation lands on a
    // character boundary rather than splitting a multi-byte sequence.
    size_t cut = g_max_body_size;
    while (cut > 0 && (static_cast<unsigned char>(body[cut]) & 0xC0) == 0x80)
      --cut;
    body.resize(cut);
    body += "\n\n...(truncated, see the attached diagnostics directory)";
  }

  std::string url = llvm::formatv("https://github.com/llvm/llvm-project/issues/"
                                  "new?title={0}&body={1}&labels=lldb",
                                  Host::URLEncode("[lldb] Bug report"),
                                  Host::URLEncode(body));

  return Host::OpenURL(url);
}
