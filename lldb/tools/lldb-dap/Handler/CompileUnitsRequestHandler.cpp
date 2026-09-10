//===-- CompileUnitsRequestHandler.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "EventHelper.h"
#include "Protocol/ProtocolRequests.h"
#include "ProtocolUtils.h"
#include "RequestHandler.h"

using namespace lldb_dap;
using namespace lldb_dap::protocol;

/// The `compileUnits` request returns the compile units of the module named by
/// `moduleId`, narrowed to `compileUnitIds` when specified.
llvm::Expected<CompileUnitsResponseBody>
CompileUnitsRequestHandler::Run(const CompileUnitsArguments &args) const {
  std::vector<CompileUnit> units;

  int num_modules = dap.target.GetNumModules();
  for (int i = 0; i < num_modules; i++) {
    lldb::SBModule curr_module = dap.target.GetModuleAtIndex(i);
    if (args.moduleId != curr_module.GetUUIDString())
      continue;

    if (args.compileUnitIds.empty()) {
      const uint32_t num_units = curr_module.GetNumCompileUnits();
      for (uint32_t j = 0; j < num_units; j++) {
        if (std::optional<CompileUnit> unit =
                CreateCompileUnit(curr_module.GetCompileUnitAtIndex(j)))
          units.emplace_back(std::move(*unit));
      }
    } else {
      for (const uint32_t id : args.compileUnitIds) {
        if (std::optional<CompileUnit> unit =
                CreateCompileUnit(curr_module.GetCompileUnitAtIndex(id)))
          units.emplace_back(std::move(*unit));
      }
    }
    break;
  }
  return CompileUnitsResponseBody{std::move(units)};
}
