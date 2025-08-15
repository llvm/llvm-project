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
#include "lldb/lldb-enumerations.h"
#include "llvm/Support/Error.h"
#include <cstddef>

using namespace lldb_dap::protocol;
namespace lldb_dap {

static std::string SymbolTypeToString(lldb::SymbolType symbol_type) {
  switch (symbol_type) {
  case lldb::eSymbolTypeInvalid:
    return "Invalid";
  case lldb::eSymbolTypeAbsolute:
    return "Absolute";
  case lldb::eSymbolTypeCode:
    return "Code";
  case lldb::eSymbolTypeResolver:
    return "Resolver";
  case lldb::eSymbolTypeData:
    return "Data";
  case lldb::eSymbolTypeTrampoline:
    return "Trampoline";
  case lldb::eSymbolTypeRuntime:
    return "Runtime";
  case lldb::eSymbolTypeException:
    return "Exception";
  case lldb::eSymbolTypeSourceFile:
    return "SourceFile";
  case lldb::eSymbolTypeHeaderFile:
    return "HeaderFile";
  case lldb::eSymbolTypeObjectFile:
    return "ObjectFile";
  case lldb::eSymbolTypeCommonBlock:
    return "CommonBlock";
  case lldb::eSymbolTypeBlock:
    return "Block";
  case lldb::eSymbolTypeLocal:
    return "Local";
  case lldb::eSymbolTypeParam:
    return "Param";
  case lldb::eSymbolTypeVariable:
    return "Variable";
  case lldb::eSymbolTypeVariableType:
    return "VariableType";
  case lldb::eSymbolTypeLineEntry:
    return "LineEntry";
  case lldb::eSymbolTypeLineHeader:
    return "LineHeader";
  case lldb::eSymbolTypeScopeBegin:
    return "ScopeBegin";
  case lldb::eSymbolTypeScopeEnd:
    return "ScopeEnd";
  case lldb::eSymbolTypeAdditional:
    return "Additional";
  case lldb::eSymbolTypeCompiler:
    return "Compiler";
  case lldb::eSymbolTypeInstrumentation:
    return "Instrumentation";
  case lldb::eSymbolTypeUndefined:
    return "Undefined";
  case lldb::eSymbolTypeObjCClass:
    return "ObjCClass";
  case lldb::eSymbolTypeObjCMetaClass:
    return "ObjCMetaClass";
  case lldb::eSymbolTypeObjCIVar:
    return "ObjCIVar";
  case lldb::eSymbolTypeReExported:
    return "ReExported";
  }

  llvm_unreachable("unhandled symbol type.");
}

/// Modules can be retrieved from the debug adapter with this request which can
/// either return all modules or a range of modules to support paging.
///
/// Clients should only call this request if the corresponding capability
/// `supportsModulesRequest` is true.
llvm::Expected<DAPGetModuleSymbolsResponseBody>
DAPGetModuleSymbolsRequestHandler::Run(
    const DAPGetModuleSymbolsArguments &args) const {
  DAPGetModuleSymbolsResponseBody response;

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

  std::vector<DAPSymbol> &symbols = response.symbols;
  lldb::SBModule module = dap.target.FindModule(module_spec);
  if (!module.IsValid())
    return llvm::make_error<DAPError>("Module not found");

  size_t num_symbols = module.GetNumSymbols();
  for (size_t i = 0; i < num_symbols; ++i) {
    lldb::SBSymbol symbol = module.GetSymbolAtIndex(i);
    if (!symbol.IsValid())
      continue;

    DAPSymbol dap_symbol;
    dap_symbol.userId = symbol.GetID();
    dap_symbol.type = SymbolTypeToString(symbol.GetType());
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
