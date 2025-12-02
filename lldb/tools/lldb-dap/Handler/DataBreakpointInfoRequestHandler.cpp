//===-- DataBreakpointInfoRequestHandler.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "DAPError.h"
#include "EventHelper.h"
#include "Protocol/ProtocolTypes.h"
#include "RequestHandler.h"
#include "lldb/API/SBAddress.h"
#include "lldb/API/SBMemoryRegionInfo.h"
#include "llvm/ADT/StringExtras.h"
#include <optional>

namespace lldb_dap {

static bool IsRW(DAP &dap, lldb::addr_t load_addr) {
  if (!lldb::SBAddress(load_addr, dap.target).IsValid())
    return false;
  lldb::SBMemoryRegionInfo region;
  lldb::SBError err =
      dap.target.GetProcess().GetMemoryRegionInfo(load_addr, region);
  // Only lldb-server supports "qMemoryRegionInfo". So, don't fail this
  // request if SBProcess::GetMemoryRegionInfo returns error.
  if (err.Success()) {
    if (!(region.IsReadable() || region.IsWritable())) {
      return false;
    }
  }
  return true;
}

/// Obtains information on a possible data breakpoint that could be set on an
/// expression or variable. Clients should only call this request if the
/// corresponding capability supportsDataBreakpoints is true.
llvm::Expected<protocol::DataBreakpointInfoResponseBody>
DataBreakpointInfoRequestHandler::Run(
    const protocol::DataBreakpointInfoArguments &args) const {
  protocol::DataBreakpointInfoResponseBody response;
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
      addr = llvm::utohexstr(load_addr);
      size = llvm::utostr(byte_size);
    }
  } else if (lldb::SBFrame frame = dap.GetLLDBFrame(args.frameId);
             args.variablesReference.value_or(0) == 0 && frame.IsValid()) {
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
        addr = llvm::utohexstr(load_addr);
        if (!IsRW(dap, load_addr)) {
          is_data_ok = false;
          response.description = "memory region for address " + addr +
                                 " has no read or write permissions";
        }
      } else {
        is_data_ok = false;
        response.description =
            "unable to get byte size for expression: " + args.name;
      }
    }
  } else if (args.asAddress) {
    size = llvm::utostr(args.bytes.value_or(dap.target.GetAddressByteSize()));
    lldb::addr_t load_addr = LLDB_INVALID_ADDRESS;
    if (llvm::StringRef(args.name).getAsInteger<lldb::addr_t>(0, load_addr))
      return llvm::make_error<DAPError>(args.name + " is not a valid address",
                                        llvm::inconvertibleErrorCode(), false);
    addr = llvm::utohexstr(load_addr);
    if (!IsRW(dap, load_addr))
      return llvm::make_error<DAPError>("memory region for address " + addr +
                                            " has no read or write permissions",
                                        llvm::inconvertibleErrorCode(), false);
  } else {
    is_data_ok = false;
    response.description = "variable not found: " + args.name;
  }

  if (is_data_ok) {
    response.dataId = addr + "/" + size;
    response.accessTypes = {protocol::eDataBreakpointAccessTypeRead,
                            protocol::eDataBreakpointAccessTypeWrite,
                            protocol::eDataBreakpointAccessTypeReadWrite};
    if (args.asAddress)
      response.description = size + " bytes at " + addr;
    else
      response.description = size + " bytes at " + addr + " " + args.name;
  }

  return response;
}

} // namespace lldb_dap
