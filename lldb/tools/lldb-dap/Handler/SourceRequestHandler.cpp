//===-- SourceRequestHandler.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "Handler/RequestHandler.h"
#include "LLDBUtils.h"
#include "Protocol/ProtocolRequests.h"
#include "Protocol/ProtocolTypes.h"
#include "lldb/API/SBAddress.h"
#include "lldb/API/SBExecutionContext.h"
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBInstructionList.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBSymbol.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBThread.h"
#include "lldb/lldb-types.h"
#include "llvm/Support/Error.h"

namespace lldb_dap {

/// Source request; value of command field is 'source'. The request retrieves
/// the source code for a given source reference.
llvm::Expected<protocol::SourceResponseBody>
SourceRequestHandler::Run(const protocol::SourceArguments &args) const {
  const auto source =
      args.source->sourceReference.value_or(args.sourceReference);

  if (!source)
    return llvm::make_error<DAPError>(
        "invalid arguments, expected source.sourceReference to be set");

  lldb::SBAddress address(source, dap.target);
  if (!address.IsValid())
    return llvm::make_error<DAPError>("source not found");

  lldb::SBSymbol symbol = address.GetSymbol();

  lldb::SBStream stream;
  lldb::SBExecutionContext exe_ctx(dap.target);

  if (symbol.IsValid()) {
    lldb::SBInstructionList insts = symbol.GetInstructions(dap.target);
    insts.GetDescription(stream, exe_ctx);
  } else {
    // No valid symbol, just return the disassembly.
    lldb::SBInstructionList insts = dap.target.ReadInstructions(
        address, dap.k_number_of_assembly_lines_for_nodebug);
    insts.GetDescription(stream, exe_ctx);
  }

  return protocol::SourceResponseBody{/*content=*/stream.GetData(),
                                      /*mimeType=*/"text/x-lldb.disassembly"};
}

} // namespace lldb_dap
