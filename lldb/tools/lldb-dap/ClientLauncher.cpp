//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClientLauncher.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FormatVariadic.h"

using namespace lldb_dap;

std::optional<ClientLauncher::Client>
ClientLauncher::GetClientFrom(llvm::StringRef str) {
  return llvm::StringSwitch<std::optional<ClientLauncher::Client>>(str.lower())
      .Case("vscode", ClientLauncher::VSCode)
      .Case("vscode-url", ClientLauncher::VSCodeURL)
      .Default(std::nullopt);
}

std::unique_ptr<ClientLauncher>
ClientLauncher::GetLauncher(ClientLauncher::Client client) {
  switch (client) {
  case ClientLauncher::VSCode:
    return std::make_unique<VSCodeLauncher>();
  case ClientLauncher::VSCodeURL:
    return std::make_unique<VSCodeURLPrinter>();
  }
  return nullptr;
}

std::string VSCodeLauncher::URLEncode(llvm::StringRef str) {
  std::string out;
  llvm::raw_string_ostream os(out);
  for (char c : str) {
    if (std::isalnum(c) || llvm::StringRef("-_.~").contains(c))
      os << c;
    else
      os << '%' << llvm::utohexstr(c, false, 2);
  }
  return os.str();
}

std::string
VSCodeLauncher::GetLaunchURL(const std::vector<llvm::StringRef> args) const {
  assert(!args.empty() && "empty launch args");

  std::vector<std::string> encoded_launch_args;
  for (llvm::StringRef arg : args)
    encoded_launch_args.push_back(URLEncode(arg));

  const std::string args_str = llvm::join(encoded_launch_args, "&args=");
  return llvm::formatv(
             "vscode://llvm-vs-code-extensions.lldb-dap/start?program={0}",
             args_str)
      .str();
}

llvm::Error VSCodeLauncher::Launch(const std::vector<llvm::StringRef> args) {
  const std::string launch_url = GetLaunchURL(args);
  const std::string command =
      llvm::formatv("code --open-url {0}", launch_url).str();

  std::system(command.c_str());
  return llvm::Error::success();
}

llvm::Error VSCodeURLPrinter::Launch(const std::vector<llvm::StringRef> args) {
  llvm::outs() << GetLaunchURL(args) << '\n';
  return llvm::Error::success();
}
