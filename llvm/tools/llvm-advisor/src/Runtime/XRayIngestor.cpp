//===------------------- XRayIngestor.cpp - LLVM Advisor -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of XRayIngestor in Runtime
//
//===----------------------------------------------------------------------===//
#include "Runtime/XRayIngestor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/XRay/Trace.h"

using namespace llvm;
using namespace llvm::advisor;

static StringRef xrayTypeName(xray::RecordTypes Type) {
  switch (Type) {
  case xray::RecordTypes::ENTER:
    return "enter";
  case xray::RecordTypes::EXIT:
    return "exit";
  case xray::RecordTypes::TAIL_EXIT:
    return "tail_exit";
  case xray::RecordTypes::ENTER_ARG:
    return "enter_arg";
  case xray::RecordTypes::CUSTOM_EVENT:
    return "custom_event";
  case xray::RecordTypes::TYPED_EVENT:
    return "typed_event";
  default:
    llvm_unreachable("unknown xray record type");
  }
}

Expected<json::Value> XRayIngestor::load(StringRef Path) {
  if (Path.empty())
    return createStringError(inconvertibleErrorCode(), "empty xray path");
  Expected<xray::Trace> Trace = xray::loadTraceFile(Path, true);
  if (!Trace)
    return Trace.takeError();

  DenseMap<int32_t, uint64_t> FunctionEvents;
  DenseMap<uint32_t, uint64_t> ThreadEvents;
  uint64_t Enters = 0;
  uint64_t Exits = 0;
  json::Array Events;

  for (const xray::XRayRecord &Record : *Trace) {
    ++FunctionEvents[Record.FuncId];
    ++ThreadEvents[Record.TId];
    if (Record.Type == xray::RecordTypes::ENTER ||
        Record.Type == xray::RecordTypes::ENTER_ARG)
      ++Enters;
    if (Record.Type == xray::RecordTypes::EXIT ||
        Record.Type == xray::RecordTypes::TAIL_EXIT)
      ++Exits;
    if (Events.size() < 512)
      Events.push_back(
          json::Object{{"type", xrayTypeName(Record.Type)},
                       {"function_id", Record.FuncId},
                       {"thread_id", static_cast<int64_t>(Record.TId)},
                       {"cpu", Record.CPU},
                       {"tsc", static_cast<int64_t>(Record.TSC)}});
  }

  const xray::XRayFileHeader &Header = Trace->getFileHeader();
  return json::Object{
      {"kind", "xray-trace"},
      {"format", "xray"},
      {"path", Path},
      {"version", Header.Version},
      {"trace_type", Header.Type},
      {"constant_tsc", Header.ConstantTSC},
      {"nonstop_tsc", Header.NonstopTSC},
      {"cycle_frequency", static_cast<int64_t>(Header.CycleFrequency)},
      {"event_count", static_cast<int64_t>(Trace->size())},
      {"function_count", static_cast<int64_t>(FunctionEvents.size())},
      {"thread_count", static_cast<int64_t>(ThreadEvents.size())},
      {"enter_count", static_cast<int64_t>(Enters)},
      {"exit_count", static_cast<int64_t>(Exits)},
      {"events", std::move(Events)}};
}
