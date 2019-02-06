//===-- Measurement.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __PerfTestDriver__Measurement__
#define __PerfTestDriver__Measurement__

#include "Gauge.h"
#include "MemoryGauge.h"
#include "Metric.h"
#include "Timer.h"

namespace lldb_perf {
template <typename GaugeType, typename Callable> class Measurement {
public:
  Measurement() : m_gauge(), m_callable(), m_metric() {}

  Measurement(Callable callable, const char *name, const char *desc)
      : m_gauge(), m_callable(callable),
        m_metric(Metric<typename GaugeType::ValueType>(name, desc)) {}

  Measurement(const char *name, const char *desc)
      : m_gauge(), m_callable(),
        m_metric(Metric<typename GaugeType::ValueType>(name, desc)) {}

  template <typename GaugeType_Rhs, typename Callable_Rhs>
  Measurement(const Measurement<GaugeType_Rhs, Callable_Rhs> &rhs)
      : m_gauge(rhs.GetGauge()), m_callable(rhs.GetCallable()),
        m_metric(rhs.GetMetric()) {}

  template <typename... Args> void operator()(Args... args) {
    m_gauge.Start();
    m_callable(args...);
    m_metric.Append(m_gauge.Stop());
  }

  virtual const Callable &GetCallable() const { return m_callable; }

  virtual const GaugeType &GetGauge() const { return m_gauge; }

  virtual const Metric<typename GaugeType::ValueType> &GetMetric() const {
    return m_metric;
  }

  void Start() { m_gauge.Start(); }

  typename GaugeType::ValueType Stop() {
    auto value = m_gauge.Stop();
    m_metric.Append(value);
    return value;
  }

  void WriteStartValue(Results &results) {
    auto metric = GetMetric();
    results.GetDictionary().Add(
        metric.GetName(), metric.GetDescription(),
        lldb_perf::GetResult<typename GaugeType::ValueType>(
            NULL, metric.GetStartValue()));
  }

  void WriteStopValue(Results &results) {
    auto metric = GetMetric();
    results.GetDictionary().Add(
        metric.GetName(), metric.GetDescription(),
        lldb_perf::GetResult<typename GaugeType::ValueType>(
            NULL, metric.GetStopValue()));
  }

  void WriteAverageValue(Results &results) {
    auto metric = GetMetric();
    results.GetDictionary().Add(
        metric.GetName(), metric.GetDescription(),
        lldb_perf::GetResult<typename GaugeType::ValueType>(
            NULL, metric.GetAverage()));
  }

  void WriteAverageAndStandardDeviation(Results &results) {
    auto metric = GetMetric();
    auto dictionary =
        (Results::Dictionary *)results.GetDictionary()
            .Add(metric.GetName(), metric.GetDescription(),
                 lldb_perf::GetResult<typename GaugeType::ValueType>(
                     NULL, metric.GetAverage()))
            .get();
    if (dictionary) {
      dictionary->Add("stddev", NULL,
                      lldb_perf::GetResult<typename GaugeType::ValueType>(
                          NULL, metric.GetStandardDeviation()));
    }
  }

  void WriteStandardDeviation(Results &results) {
    auto metric = GetMetric();
    results.GetDictionary().Add(
        metric.GetName(), metric.GetDescription(),
        lldb_perf::GetResult<typename GaugeType::ValueType>(
            NULL, metric.GetStandardDeviation()));
  }

protected:
  GaugeType m_gauge;
  Callable m_callable;
  Metric<typename GaugeType::ValueType> m_metric;
};

template <typename Callable>
class TimeMeasurement : public Measurement<TimeGauge, Callable> {
public:
  TimeMeasurement() : Measurement<TimeGauge, Callable>() {}

  TimeMeasurement(Callable callable, const char *name = NULL,
                  const char *descr = NULL)
      : Measurement<TimeGauge, Callable>(callable, name, descr) {}

  template <typename Callable_Rhs>
  TimeMeasurement(const TimeMeasurement<Callable_Rhs> &rhs)
      : Measurement<TimeGauge, Callable>(rhs) {}

  template <typename GaugeType_Rhs, typename Callable_Rhs>
  TimeMeasurement(const Measurement<GaugeType_Rhs, Callable_Rhs> &rhs)
      : Measurement<GaugeType_Rhs, Callable_Rhs>(rhs) {}

  template <typename... Args> void operator()(Args... args) {
    Measurement<TimeGauge, Callable>::operator()(args...);
  }
};

template <typename Callable>
class MemoryMeasurement : public Measurement<MemoryGauge, Callable> {
public:
  MemoryMeasurement() : Measurement<MemoryGauge, Callable>() {}

  MemoryMeasurement(Callable callable, const char *name, const char *descr)
      : Measurement<MemoryGauge, Callable>(callable, name, descr) {}

  MemoryMeasurement(const char *name, const char *descr)
      : Measurement<MemoryGauge, Callable>(name, descr) {}

  template <typename Callable_Rhs>
  MemoryMeasurement(const MemoryMeasurement<Callable_Rhs> &rhs)
      : Measurement<MemoryGauge, Callable>(rhs) {}

  template <typename GaugeType_Rhs, typename Callable_Rhs>
  MemoryMeasurement(const Measurement<GaugeType_Rhs, Callable_Rhs> &rhs)
      : Measurement<GaugeType_Rhs, Callable_Rhs>(rhs) {}

  template <typename... Args> void operator()(Args... args) {
    Measurement<MemoryGauge, Callable>::operator()(args...);
  }
};
}

#endif /* defined(__PerfTestDriver__Measurement__) */
