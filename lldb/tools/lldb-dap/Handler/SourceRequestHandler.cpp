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
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBInstructionList.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBThread.h"
#include "llvm/Support/Error.h"

namespace lldb_dap {

/// Source request; value of command field is 'source'. The request retrieves
/// the source code for a given source reference.
llvm::Expected<protocol::SourceResponseBody>
SourceRequestHandler::Run(const protocol::SourceArguments &args) const {
  const auto source =
      args.source->sourceReference.value_or(args.sourceReference);

  if (!source)
    return llvm::createStringError(
        "invalid arguments, expected source.sourceReference to be set");

  lldb::SBProcess process = dap.target.GetProcess();
  // Upper 32 bits is the thread index ID
  lldb::SBThread thread =
      process.GetThreadByIndexID(GetLLDBThreadIndexID(source));
  // Lower 32 bits is the frame index
  lldb::SBFrame frame = thread.GetFrameAtIndex(GetLLDBFrameID(source));
  if (!frame.IsValid())
    return llvm::createStringError("source not found");

  lldb::SBInstructionList insts = frame.GetSymbol().GetInstructions(dap.target);
  lldb::SBStream stream;
  insts.GetDescription(stream);

  return protocol::SourceResponseBody{/*content=*/stream.GetData(),
                                      /*mimeType=*/"text/x-lldb.disassembly"};
}

} // namespace lldb_dap
