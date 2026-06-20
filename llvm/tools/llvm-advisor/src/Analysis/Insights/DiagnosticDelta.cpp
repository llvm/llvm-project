//===--- DiagnosticDelta.cpp - LLVM Advisor ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Insights/DiagnosticDelta.h"
#include "llvm/Support/JSON.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<InsightOutput>
DiagnosticDeltaInsight::analyze(const InsightInput &Input) const {
  const json::Object &Primary = *Input.PrimaryData;
  const json::Object &Baseline = *Input.BaselineData;

  int64_t PErrors = getInt(Primary, "errors");
  int64_t PWarnings = getInt(Primary, "warnings");
  int64_t PNotes = getInt(Primary, "notes");

  int64_t BErrors = getInt(Baseline, "errors");
  int64_t BWarnings = getInt(Baseline, "warnings");
  int64_t BNotes = getInt(Baseline, "notes");

  // NOTE: Deduplication is by message text only. Two diagnostics with identical
  // messages but different locations will be treated as the same diagnostic.
  StringMap<int> BaselineMsgs;
  if (const json::Array *BD = Baseline.getArray("diagnostics"))
    for (const json::Value &V : *BD)
      if (const json::Object *D = V.getAsObject())
        if (auto Msg = D->getString("message"))
          BaselineMsgs[*Msg]++;

  json::Array NewDiagnostics;
  int64_t NewErrors = 0, NewWarnings = 0;
  if (const json::Array *PD = Primary.getArray("diagnostics")) {
    for (const json::Value &V : *PD) {
      const json::Object *D = V.getAsObject();
      if (!D)
        continue;
      auto Msg = D->getString("message");
      if (!Msg)
        continue;
      auto It = BaselineMsgs.find(*Msg);
      if (It == BaselineMsgs.end() || It->second == 0) {
        StringRef Level = D->getString("level").value_or("unknown");
        if (Level == "error")
          NewErrors++;
        else if (Level == "warning")
          NewWarnings++;
        NewDiagnostics.push_back(json::Object{
            {"level", Level},
            {"message", *Msg},
            {"file", D->getString("file").value_or("")},
            {"line", getInt(*D, "line")},
        });
      } else {
        It->second--;
      }
    }
  }

  InsightOutput Out;
  Out.Kind = getKind();
  Out.Name = getName().str();
  Out.Data = json::Object{
      {"error_delta", PErrors - BErrors},
      {"warning_delta", PWarnings - BWarnings},
      {"note_delta", PNotes - BNotes},
      {"new_errors", NewErrors},
      {"new_warnings", NewWarnings},
      {"new_diagnostics", std::move(NewDiagnostics)},
      {"primary", json::Object{{"errors", PErrors},
                               {"warnings", PWarnings},
                               {"notes", PNotes}}},
      {"baseline", json::Object{{"errors", BErrors},
                                {"warnings", BWarnings},
                                {"notes", BNotes}}},
  };
  return Out;
}
