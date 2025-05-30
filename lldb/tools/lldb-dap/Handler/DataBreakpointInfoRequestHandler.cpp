//===-- DataBreakpointInfoRequestHandler.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "Protocol/ProtocolTypes.h"
#include "RequestHandler.h"
#include "lldb/API/SBMemoryRegionInfo.h"
#include "llvm/ADT/StringExtras.h"
#include <optional>

namespace lldb_dap {

namespace {
std::vector<protocol::DataBreakpointAccessType>
GetBreakpointAccessTypes(lldb::SBMemoryRegionInfo region) {
  std::vector<protocol::DataBreakpointAccessType> types;
  if (region.IsReadable())
    types.emplace_back(protocol::eDataBreakpointAccessTypeRead);
  if (region.IsWritable())
    types.emplace_back(protocol::eDataBreakpointAccessTypeWrite);
  if (region.IsReadable() && region.IsWritable())
    types.emplace_back(protocol::eDataBreakpointAccessTypeReadWrite);

  return types;
}

llvm::Expected<protocol::DataBreakpointInfoResponseBody>
HandleDataBreakpointBytes(DAP &dap,
                          const protocol::DataBreakpointInfoArguments &args) {
  const llvm::StringRef raw_address = args.name;

  lldb::addr_t load_addr = LLDB_INVALID_ADDRESS;
  if (raw_address.getAsInteger<lldb::addr_t>(0, load_addr)) {
    return llvm::make_error<DAPError>(llvm::formatv("invalid address"),
                                      llvm::inconvertibleErrorCode(), false);
  }

  if (lldb::SBAddress address(load_addr, dap.target); !address.IsValid()) {
    return llvm::make_error<DAPError>(
        llvm::formatv("address {:x} does not exist in the debuggee", load_addr),
        llvm::inconvertibleErrorCode(), false);
  }

  const uint32_t byte_size =
      args.bytes.value_or(dap.target.GetAddressByteSize());

  lldb::SBMemoryRegionInfo region;
  lldb::SBError err =
      dap.target.GetProcess().GetMemoryRegionInfo(load_addr, region);
  std::vector<protocol::DataBreakpointAccessType> access_types =
      GetBreakpointAccessTypes(region);

  protocol::DataBreakpointInfoResponseBody response;
  if (err.Fail()) {
    response.dataId = std::nullopt;
    response.description = err.GetCString();
    return response;
  }

  if (access_types.empty()) {
    response.dataId = std::nullopt;
    response.description = llvm::formatv(
        "memory region for address {} has no read or write permissions",
        load_addr);
    return response;
  }

  response.dataId = llvm::formatv("{:x-}/{}", load_addr, byte_size);
  response.description =
      llvm::formatv("{} bytes at {:x}", byte_size, load_addr);
  response.accessTypes = std::move(access_types);

  return response;
}
} // namespace

/// Obtains information on a possible data breakpoint that could be set on an
/// expression or variable. Clients should only call this request if the
/// corresponding capability supportsDataBreakpoints is true.
llvm::Expected<protocol::DataBreakpointInfoResponseBody>
DataBreakpointInfoRequestHandler::Run(
    const protocol::DataBreakpointInfoArguments &args) const {

  if (args.asAddress.value_or(false))
    return HandleDataBreakpointBytes(dap, args);

  protocol::DataBreakpointInfoResponseBody response;
  lldb::SBFrame frame = dap.GetLLDBFrame(args.frameId.value_or(UINT64_MAX));
  lldb::SBValue variable = dap.variables.FindVariable(
      args.variablesReference.value_or(0), args.name);
  std::string addr, size;

  bool is_data_ok = true;
  if (variable.IsValid()) {
    lldb::addr_t load_addr = variable.GetLoadAddress();
    size_t byte_size = variable.GetByteSize();
    if (load_addr == LLDB_INVALID_ADDRESS) {
      is_data_ok = false;
      response.description = "does not exist in memory, its location is " +
                             std::string(variable.GetLocation());
    } else if (byte_size == 0) {
      is_data_ok = false;
      response.description = "variable size is 0";
    } else {
      addr = llvm::utohexstr(load_addr, /*lowerCase=*/true);
      size = llvm::utostr(byte_size);
    }
  } else if (args.variablesReference.value_or(0) == 0 && frame.IsValid()) {
    lldb::SBValue value = frame.EvaluateExpression(args.name.c_str());
    if (value.GetError().Fail()) {
      lldb::SBError error = value.GetError();
      const char *error_cstr = error.GetCString();
      is_data_ok = false;
      response.description = error_cstr && error_cstr[0]
                                 ? std::string(error_cstr)
                                 : "evaluation failed";
    } else {
      uint64_t load_addr = value.GetValueAsUnsigned();
      lldb::SBData data = value.GetPointeeData();
      if (data.IsValid()) {
        size = llvm::utostr(data.GetByteSize());
        addr = llvm::utohexstr(load_addr, /*lowerCase=*/true);
        lldb::SBMemoryRegionInfo region;
        lldb::SBError err =
            dap.target.GetProcess().GetMemoryRegionInfo(load_addr, region);
        // Only lldb-server supports "qMemoryRegionInfo". So, don't fail this
        // request if SBProcess::GetMemoryRegionInfo returns error.
        if (err.Success()) {
          if (!(region.IsReadable() || region.IsWritable())) {
            is_data_ok = false;
            response.description = "memory region for address " + addr +
                                   " has no read or write permissions";
          }
        }
      } else {
        is_data_ok = false;
        response.description =
            "unable to get byte size for expression: " + args.name;
      }
    }
  } else {
    is_data_ok = false;
    response.description = "variable not found: " + args.name;
  }

  if (is_data_ok) {
    response.dataId = addr + "/" + size;
    response.accessTypes = {protocol::eDataBreakpointAccessTypeRead,
                            protocol::eDataBreakpointAccessTypeWrite,
                            protocol::eDataBreakpointAccessTypeReadWrite};
    response.description = size + " bytes at 0x" + addr + " " + args.name;
  }

  return response;
}

} // namespace lldb_dap
