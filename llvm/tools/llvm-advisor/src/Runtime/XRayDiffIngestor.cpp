//===------------------- XRayDiffIngestor.cpp - LLVM Advisor ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of XRayDiffIngestor in Runtime
//
//===----------------------------------------------------------------------===//
#include "Runtime/XRayDiffIngestor.h"
#include "Runtime/RuntimeUtils.h"
#include "Runtime/XRayIngestor.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<json::Value> XRayDiffIngestor::load(StringRef Before,
                                             StringRef After) {
  if (Before.empty() || After.empty())
    return createStringError(inconvertibleErrorCode(), "empty xray diff path");
  XRayIngestor Ingestor;
  Expected<json::Value> BeforeValue = Ingestor.load(Before);
  if (!BeforeValue)
    return BeforeValue.takeError();
  Expected<json::Value> AfterValue = Ingestor.load(After);
  if (!AfterValue)
    return AfterValue.takeError();

  const json::Object *BeforeObject = BeforeValue->getAsObject();
  const json::Object *AfterObject = AfterValue->getAsObject();
  if (!BeforeObject || !AfterObject)
    return createStringError(inconvertibleErrorCode(),
                             "xray summaries are not objects");

  int64_t BeforeEvents = getInteger(*BeforeObject, "event_count");
  int64_t AfterEvents = getInteger(*AfterObject, "event_count");
  return json::Object{{"kind", "xray-trace-diff"},
                      {"before", Before},
                      {"after", After},
                      {"before_events", BeforeEvents},
                      {"after_events", AfterEvents},
                      {"event_delta", AfterEvents - BeforeEvents},
                      {"before_summary", std::move(*BeforeValue)},
                      {"after_summary", std::move(*AfterValue)}};
}
