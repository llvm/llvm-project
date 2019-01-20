//===-- Gauge.cpp -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Gauge.h"
#include "lldb/lldb-forward.h"

template <>
lldb_perf::Results::ResultSP lldb_perf::GetResult(const char *description,
                                                  double value) {
  if (description && description[0]) {
    std::unique_ptr<Results::Dictionary> value_dict_ap(
        new Results::Dictionary());
    value_dict_ap->AddString("description", NULL, description);
    value_dict_ap->AddDouble("value", NULL, value);
    return Results::ResultSP(value_dict_ap.release());
  }
  return Results::ResultSP(new Results::Double(NULL, NULL, value));
}

template <>
lldb_perf::Results::ResultSP lldb_perf::GetResult(const char *description,
                                                  uint64_t value) {
  if (description && description[0]) {
    std::unique_ptr<Results::Dictionary> value_dict_ap(
        new Results::Dictionary());
    value_dict_ap->AddString("description", NULL, description);
    value_dict_ap->AddUnsigned("value", NULL, value);
    return Results::ResultSP(value_dict_ap.release());
  }
  return Results::ResultSP(new Results::Unsigned(NULL, NULL, value));
}

template <>
lldb_perf::Results::ResultSP lldb_perf::GetResult(const char *description,
                                                  std::string value) {
  if (description && description[0]) {
    std::unique_ptr<Results::Dictionary> value_dict_ap(
        new Results::Dictionary());
    value_dict_ap->AddString("description", NULL, description);
    value_dict_ap->AddString("value", NULL, value.c_str());
    return Results::ResultSP(value_dict_ap.release());
  }
  return Results::ResultSP(new Results::String(NULL, NULL, value.c_str()));
}
