//===-- DAPGetModuleSymbolsRequestHandler.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "DAPError.h"
#include "Protocol/DAPTypes.h"
#include "RequestHandler.h"
#include "lldb/API/SBAddress.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBModule.h"
#include "lldb/API/SBModuleSpec.h"
#include "lldb/Utility/UUID.h"
#include "llvm/Support/Error.h"
#include <cstddef>

using namespace lldb_dap::protocol;
namespace lldb_dap {

llvm::Expected<ModuleSymbolsResponseBody>
ModuleSymbolsRequestHandler::Run(const ModuleSymbolsArguments &args) const {
  ModuleSymbolsResponseBody response;

  lldb::SBModuleSpec module_spec;
  if (args.moduleId) {
    llvm::SmallVector<uint8_t, 20> uuid_bytes;
    if (!lldb_private::UUID::DecodeUUIDBytesFromString(*args.moduleId,
                                                       uuid_bytes)
             .empty())
      return llvm::make_error<DAPError>("Invalid module ID");

    module_spec.SetUUIDBytes(uuid_bytes.data(), uuid_bytes.size());
  }

  if (args.moduleName) {
    lldb::SBFileSpec file_spec;
    file_spec.SetFilename(args.moduleName->c_str());
    module_spec.SetFileSpec(file_spec);
  }

  // Empty request, return empty response.
  // We use it in the client to check if the lldb-dap server supports this
  // request.
  if (!module_spec.IsValid())
    return response;

  std::vector<Symbol> &symbols = response.symbols;
  lldb::SBModule module = dap.target.FindModule(module_spec);
  if (!module.IsValid())
    return llvm::make_error<DAPError>("Module not found");

  size_t num_symbols = module.GetNumSymbols();
  for (size_t i = 0; i < num_symbols; ++i) {
    lldb::SBSymbol symbol = module.GetSymbolAtIndex(i);
    if (!symbol.IsValid())
      continue;

    Symbol dap_symbol;
    dap_symbol.userId = symbol.GetID();
    dap_symbol.type = symbol.GetType();
    dap_symbol.isDebug = symbol.IsDebug();
    dap_symbol.isSynthetic = symbol.IsSynthetic();
    dap_symbol.isExternal = symbol.IsExternal();

    lldb::SBAddress start_address = symbol.GetStartAddress();
    if (start_address.IsValid()) {
      lldb::addr_t file_address = start_address.GetFileAddress();
      if (file_address != LLDB_INVALID_ADDRESS)
        dap_symbol.fileAddress = file_address;

      lldb::addr_t load_address = start_address.GetLoadAddress(dap.target);
      if (load_address != LLDB_INVALID_ADDRESS)
        dap_symbol.loadAddress = load_address;
    }

    dap_symbol.size = symbol.GetSize();
    dap_symbol.name = symbol.GetName();
    symbols.push_back(std::move(dap_symbol));
  }

  return response;
}

} // namespace lldb_dap
