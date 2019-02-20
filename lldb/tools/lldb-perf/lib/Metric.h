//===-- Metric.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __PerfTestDriver__Metric__
#define __PerfTestDriver__Metric__

#include <mach/task_info.h>
#include <string>
#include <vector>

namespace lldb_perf {

class MemoryStats;

template <class ValueType> class Metric {
public:
  enum class StandardDeviationMode { eSample, ePopulation };

  Metric();
  Metric(const char *, const char * = NULL);

  void Append(ValueType v);

  ValueType GetAverage() const;

  size_t GetCount() const;

  ValueType GetSum() const;

  ValueType GetStandardDeviation(
      StandardDeviationMode mode = StandardDeviationMode::ePopulation) const;

  const char *GetName() const {
    if (m_name.empty())
      return NULL;
    return m_name.c_str();
  }

  const char *GetDescription() const {
    if (m_description.empty())
      return NULL;
    return m_description.c_str();
  }

private:
  std::string m_name;
  std::string m_description;
  std::vector<ValueType> m_dataset;
};
}

#endif /* defined(__PerfTestDriver__Metric__) */
