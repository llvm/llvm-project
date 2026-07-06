//===-- ModulesRequestHandler.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "ProtocolUtils.h"
#include "RequestHandler.h"

using namespace lldb_dap::protocol;
namespace lldb_dap {

/// Modules can be retrieved from the debug adapter with this request which can
/// either return all modules or a range of modules to support paging.
///
/// Clients should only call this request if the corresponding capability
/// `supportsModulesRequest` is true.
llvm::Expected<ModulesResponseBody>
ModulesRequestHandler::Run(const std::optional<ModulesArguments> &args) const {
  ModulesResponseBody response;

  std::vector<Module> &modules = response.modules;
  std::lock_guard<std::mutex> guard(dap.modules_mutex);
  const uint32_t total_modules = dap.target.GetNumModules();
  response.totalModules = total_modules;

  modules.reserve(total_modules);
  for (uint32_t i = 0; i < total_modules; i++) {
    lldb::SBModule module = dap.target.GetModuleAtIndex(i);

    std::optional<Module> result = CreateModule(dap.target, module);
    if (result && !result->id.empty()) {
      dap.modules.insert(result->id);
      modules.emplace_back(std::move(result).value());
    }
  }

  return response;
}

} // namespace lldb_dap
