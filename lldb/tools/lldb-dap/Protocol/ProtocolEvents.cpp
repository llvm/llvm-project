//===-- ProtocolEvents.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol/ProtocolEvents.h"
#include "Events/EventHandler.h"
#include "llvm/Support/JSON.h"

using namespace llvm;

namespace lldb_dap::protocol {

json::Value toJSON(const ExitedEventBody &EEB) {
  return json::Object{{"exitCode", EEB.exitCode}};
}

json::Value toJSON(const ProcessEventBody::StartMethod &m) {
  switch (m) {
  case ProcessEventBody::StartMethod::launch:
    return "launch";
  case ProcessEventBody::StartMethod::attach:
    return "attach";
  case ProcessEventBody::StartMethod::attachForSuspendedLaunch:
    return "attachForSuspendedLaunch";
  }
}

json::Value toJSON(const ProcessEventBody &PEB) {
  json::Object result{{"name", PEB.name}};

  if (PEB.systemProcessId)
    result.insert({"systemProcessId", PEB.systemProcessId});
  if (PEB.isLocalProcess)
    result.insert({"isLocalProcess", PEB.isLocalProcess});
  if (PEB.startMethod)
    result.insert({"startMethod", PEB.startMethod});
  if (PEB.pointerSize)
    result.insert({"pointerSize", PEB.pointerSize});

  return std::move(result);
}

json::Value toJSON(const OutputEventBody::Category &C) {
  switch (C) {
  case OutputEventBody::Category::Console:
    return "console";
  case OutputEventBody::Category::Important:
    return "important";
  case OutputEventBody::Category::Stdout:
    return "stdout";
  case OutputEventBody::Category::Stderr:
    return "stderr";
  case OutputEventBody::Category::Telemetry:
    return "telemetry";
  }
}

json::Value toJSON(const OutputEventBody::Group &G) {
  switch (G) {
  case OutputEventBody::Group::start:
    return "start";
  case OutputEventBody::Group::startCollapsed:
    return "startCollapsed";
  case OutputEventBody::Group::end:
    return "end";
  }
}

json::Value toJSON(const OutputEventBody &OEB) {
  json::Object result{{"output", OEB.output}};

  if (OEB.category)
    result.insert({"category", *OEB.category});
  if (OEB.group)
    result.insert({"group", *OEB.group});
  if (OEB.variablesReference)
    result.insert({"variablesReference", *OEB.variablesReference});
  if (OEB.source)
    result.insert({"source", *OEB.source});
  if (OEB.line)
    result.insert({"line", *OEB.line});
  if (OEB.column)
    result.insert({"column", *OEB.column});
  if (OEB.data)
    result.insert({"data", *OEB.data});
  if (OEB.locationReference)
    result.insert({"locationReference", *OEB.locationReference});

  return std::move(result);
}

} // namespace lldb_dap::protocol
