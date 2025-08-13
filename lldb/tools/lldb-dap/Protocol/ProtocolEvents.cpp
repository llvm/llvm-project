//===-- ProtocolEvents.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol/ProtocolEvents.h"
#include "llvm/Support/JSON.h"

using namespace llvm;

namespace lldb_dap::protocol {

json::Value toJSON(const CapabilitiesEventBody &CEB) {
  return json::Object{{"capabilities", CEB.capabilities}};
}

json::Value toJSON(const ModuleEventBody::Reason &MEBR) {
  switch (MEBR) {
  case ModuleEventBody::eReasonNew:
    return "new";
  case ModuleEventBody::eReasonChanged:
    return "changed";
  case ModuleEventBody::eReasonRemoved:
    return "removed";
  }
  llvm_unreachable("unhandled module event reason!.");
}

json::Value toJSON(const ModuleEventBody &MEB) {
  return json::Object{{"reason", MEB.reason}, {"module", MEB.module}};
}

} // namespace lldb_dap::protocol
