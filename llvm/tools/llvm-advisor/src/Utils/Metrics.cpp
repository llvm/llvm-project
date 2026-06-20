//===------------------- Metrics.cpp - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Prometheus-compatible metrics counters for observability.
// Tracks performance and operational metrics.
//
//===----------------------------------------------------------------------===//
#include "Utils/Metrics.h"

using namespace llvm;
using namespace llvm::advisor;

void Metrics::increment(StringRef Name, uint64_t Delta) {
  Counters[Name] += Delta;
}

uint64_t Metrics::get(StringRef Name) const {
  StringMap<uint64_t>::const_iterator I = Counters.find(Name);
  if (I == Counters.end())
    return 0;
  return I->second;
}

std::string Metrics::toText() const {
  std::string Storage;
  raw_string_ostream OS(Storage);
  for (const StringMapEntry<uint64_t> &Entry : Counters)
    OS << Entry.first() << ' ' << Entry.second << '\n';
  return OS.str();
}
