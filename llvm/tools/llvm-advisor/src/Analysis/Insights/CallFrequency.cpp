//===--- CallFrequency.cpp - LLVM Advisor --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Insights/CallFrequency.h"
#include "llvm/Support/JSON.h"

using namespace llvm;
using namespace llvm::advisor;

namespace {

struct NodeEntry {
  std::string Name;
  int64_t OutgoingCalls;
  int64_t IncomingCalls;
};

json::Array BuildArray(ArrayRef<const NodeEntry *> Sorted) {
  json::Array A;
  int Count = 0;
  for (const NodeEntry *E : Sorted) {
    if (Count++ >= DefaultTopN)
      break;
    A.push_back(json::Object{
        {"name", E->Name},
        {"incoming_calls", E->IncomingCalls},
        {"outgoing_calls", E->OutgoingCalls},
    });
  }
  return A;
}

} // namespace

Expected<InsightOutput>
CallFrequencyInsight::analyze(const InsightInput &Input) const {
  const json::Object &D = *Input.PrimaryData;

  const json::Array *Nodes = D.getArray("nodes");
  if (!Nodes)
    Nodes = D.getArray("functions");
  if (!Nodes || Nodes->empty())
    return noDataError();

  SmallVector<NodeEntry, 64> Entries;
  int64_t TotalEdges = 0;

  for (const json::Value &V : *Nodes) {
    const json::Object *N = V.getAsObject();
    if (!N)
      continue;
    NodeEntry E;
    E.Name = N->getString("name").value_or("?").str();
    if (const json::Array *Callees = N->getArray("callees"))
      E.OutgoingCalls = static_cast<int64_t>(Callees->size());
    else
      E.OutgoingCalls = getInt(*N, "outgoing_calls");
    if (const json::Array *Callers = N->getArray("callers"))
      E.IncomingCalls = static_cast<int64_t>(Callers->size());
    else
      E.IncomingCalls = getInt(*N, "incoming_calls");
    TotalEdges += E.OutgoingCalls;
    Entries.push_back(std::move(E));
  }

  SmallVector<const NodeEntry *, 64> ByFanIn;
  SmallVector<const NodeEntry *, 64> ByFanOut;
  for (const auto &E : Entries) {
    ByFanIn.push_back(&E);
    ByFanOut.push_back(&E);
  }

  llvm::sort(ByFanIn, [](const NodeEntry *A, const NodeEntry *B) {
    return A->IncomingCalls > B->IncomingCalls;
  });
  llvm::sort(ByFanOut, [](const NodeEntry *A, const NodeEntry *B) {
    return A->OutgoingCalls > B->OutgoingCalls;
  });

  SmallVector<const NodeEntry *, 32> HubEntries;
  for (const auto &E : Entries)
    if (E.IncomingCalls >= 3 && E.OutgoingCalls >= 3)
      HubEntries.push_back(&E);
  llvm::sort(HubEntries, [](const NodeEntry *A, const NodeEntry *B) {
    return (A->IncomingCalls + A->OutgoingCalls) >
           (B->IncomingCalls + B->OutgoingCalls);
  });

  json::Array Hubs;
  int HubCount = 0;
  for (const NodeEntry *E : HubEntries) {
    if (HubCount++ >= DefaultTopN)
      break;
    Hubs.push_back(json::Object{
        {"name", E->Name},
        {"incoming_calls", E->IncomingCalls},
        {"outgoing_calls", E->OutgoingCalls},
    });
  }

  InsightOutput Out;
  Out.Kind = getKind();
  Out.Name = getName().str();
  Out.Data = json::Object{
      {"total_functions", static_cast<int64_t>(Entries.size())},
      {"total_call_edges", TotalEdges},
      {"top_callers_by_fan_in", BuildArray(ByFanIn)},
      {"top_callees_by_fan_out", BuildArray(ByFanOut)},
      {"hub_functions", std::move(Hubs)},
  };
  return Out;
}
