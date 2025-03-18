//===-- OutputEventHandler.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "Events/EventHandler.h"
#include "Protocol/ProtocolEvents.h"
#include "llvm/ADT/StringRef.h"

using namespace llvm;
using namespace lldb_dap::protocol;

namespace lldb_dap {

void OutputEventHandler::operator()(llvm::StringRef output,
                                    OutputCategory category) const {
  if (output.empty())
    return;

  // Send each line of output as an individual event, including the newline if
  // present.
  size_t idx = 0;
  do {
    size_t end = output.find('\n', idx);
    if (end == llvm::StringRef::npos)
      end = output.size() - 1;

    OutputEventBody body;
    body.output = output.slice(idx, end + 1).str();
    body.category = category;

    dap.Send(protocol::Event{/*event=*/OutputEventHandler::event.str(),
                             /*body*/ body});

    idx = end + 1;
  } while (idx < output.size());
}

} // namespace lldb_dap
