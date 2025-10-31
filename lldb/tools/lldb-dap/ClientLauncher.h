//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_CLIENTLAUNCHER_H
#define LLDB_TOOLS_LLDB_DAP_CLIENTLAUNCHER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <vector>

namespace lldb_dap {

class ClientLauncher {
public:
  enum Client {
    VSCode,
  };

  virtual ~ClientLauncher() = default;
  virtual llvm::Error Launch(const std::vector<llvm::StringRef> args) = 0;

  static std::optional<Client> GetClientFrom(llvm::StringRef str);
  static std::unique_ptr<ClientLauncher> GetLauncher(Client client);
};

class VSCodeLauncher final : public ClientLauncher {
public:
  using ClientLauncher::ClientLauncher;

  llvm::Error Launch(const std::vector<llvm::StringRef> args) override;
};

} // namespace lldb_dap

#endif
