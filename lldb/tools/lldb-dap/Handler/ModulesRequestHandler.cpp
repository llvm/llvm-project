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
#include <algorithm>

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

  const uint32_t start_module = args ? args->startModule : 0;
  if (start_module >= total_modules)
    return response;

  const uint32_t module_count = args ? args->moduleCount : 0;
  const uint32_t end_module =
      module_count == 0 ? total_modules
                        : std::min(total_modules, start_module + module_count);

  modules.reserve(end_module - start_module);
  for (uint32_t i = start_module; i < end_module; ++i) {
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
