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
#include "RequestHandler.h"
#include "lldb/API/SBCompileUnit.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/Host/PosixApi.h" // Adds PATH_MAX for windows

using namespace lldb_dap;
using namespace lldb_dap::protocol;

static std::optional<CompileUnit>
CreateCompileUnit(const lldb::SBCompileUnit &unit) {
  const lldb::SBFileSpec file_spec = unit.GetFileSpec();
  if (!file_spec.IsValid())
    return std::nullopt;

  std::array<char, PATH_MAX> path_buffer{};
  const uint32_t path_size =
      file_spec.GetPath(path_buffer.data(), path_buffer.size());

  CompileUnit result;
  result.id = unit.GetIDInModule();
  result.compileUnitPath = std::string(path_buffer.data(), path_size);
  return result;
}

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
      units.reserve(num_units);
      for (uint32_t j = 0; j < num_units; j++) {
        if (std::optional<CompileUnit> unit =
                CreateCompileUnit(curr_module.GetCompileUnitAtIndex(j)))
          units.emplace_back(std::move(*unit));
      }
    } else {
      units.reserve(args.compileUnitIds.size());
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
