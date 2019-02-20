//===-- Gauge.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PerfTestDriver_Gauge_h
#define PerfTestDriver_Gauge_h

#include <functional>
#include <string>

#include "Results.h"

namespace lldb_perf {

template <class T> class Gauge {
public:
  typedef T ValueType;

  Gauge() {}

  virtual ~Gauge() {}

  virtual void Start() = 0;

  virtual ValueType Stop() = 0;

  virtual ValueType GetStartValue() const = 0;

  virtual ValueType GetStopValue() const = 0;

  virtual ValueType GetDeltaValue() const = 0;
};

template <class T>
Results::ResultSP GetResult(const char *description, T value);

template <> Results::ResultSP GetResult(const char *description, double value);

template <>
Results::ResultSP GetResult(const char *description, uint64_t value);

template <>
Results::ResultSP GetResult(const char *description, std::string value);
}

#endif
