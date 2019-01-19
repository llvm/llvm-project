//===-- Timer.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __PerfTestDriver__Timer__
#define __PerfTestDriver__Timer__

#include "Gauge.h"

#include <chrono>

using namespace std::chrono;

namespace lldb_perf {
class TimeGauge : public Gauge<double> {
public:
  TimeGauge();

  virtual ~TimeGauge() {}

  void Start();

  double Stop();

  virtual double GetStartValue() const;

  virtual double GetStopValue() const;

  virtual double GetDeltaValue() const;

private:
  enum class State { eNeverUsed, eCounting, eStopped };

  typedef high_resolution_clock::time_point TimeType;
  TimeType m_start;
  TimeType m_stop;
  double m_delta;
  State m_state;

  TimeType Now();
};
}

#endif /* defined(__PerfTestDriver__Timer__) */
