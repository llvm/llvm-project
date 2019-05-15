//===-- MemoryGauge.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __PerfTestDriver__MemoryGauge__
#define __PerfTestDriver__MemoryGauge__

#include "Gauge.h"
#include "Results.h"

#include <mach/task_info.h>

namespace lldb_perf {

class MemoryStats {
public:
  MemoryStats(mach_vm_size_t virtual_size = 0, mach_vm_size_t resident_size = 0,
              mach_vm_size_t max_resident_size = 0);

  MemoryStats &operator+=(const MemoryStats &rhs);

  MemoryStats operator-(const MemoryStats &rhs);

  MemoryStats operator+(const MemoryStats &rhs);

  MemoryStats operator/(size_t rhs);

  MemoryStats operator*(const MemoryStats &rhs);

  mach_vm_size_t GetVirtualSize() const { return m_virtual_size; }

  mach_vm_size_t GetResidentSize() const { return m_resident_size; }

  mach_vm_size_t GetMaxResidentSize() const { return m_max_resident_size; }

  void SetVirtualSize(mach_vm_size_t vs) { m_virtual_size = vs; }

  void SetResidentSize(mach_vm_size_t rs) { m_resident_size = rs; }

  void SetMaxResidentSize(mach_vm_size_t mrs) { m_max_resident_size = mrs; }

  Results::ResultSP GetResult(const char *name, const char *description) const;

private:
  mach_vm_size_t m_virtual_size;
  mach_vm_size_t m_resident_size;
  mach_vm_size_t m_max_resident_size;
};

class MemoryGauge : public Gauge<MemoryStats> {
public:
  MemoryGauge();

  virtual ~MemoryGauge() {}

  void Start();

  ValueType Stop();

  virtual ValueType GetStartValue() const { return m_start; }

  virtual ValueType GetStopValue() const { return m_stop; }

  virtual ValueType GetDeltaValue() const;

private:
  enum class State { eNeverUsed, eCounting, eStopped };

  ValueType Now();

  State m_state;
  ValueType m_start;
  ValueType m_stop;
  ValueType m_delta;
};

template <>
Results::ResultSP GetResult(const char *description, MemoryStats value);

} // namespace lldb_perf

lldb_perf::MemoryStats sqrt(const lldb_perf::MemoryStats &arg);

#endif // #ifndef __PerfTestDriver__MemoryGauge__
