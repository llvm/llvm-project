//===-- MemoryGauge.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MemoryGauge.h"
#include "lldb/lldb-forward.h"
#include <assert.h>
#include <cmath>
#include <mach/mach.h>
#include <mach/mach_traps.h>
#include <mach/task.h>

using namespace lldb_perf;

MemoryStats::MemoryStats(mach_vm_size_t virtual_size,
                         mach_vm_size_t resident_size,
                         mach_vm_size_t max_resident_size)
    : m_virtual_size(virtual_size), m_resident_size(resident_size),
      m_max_resident_size(max_resident_size) {}

MemoryStats &MemoryStats::operator+=(const MemoryStats &rhs) {
  m_virtual_size += rhs.m_virtual_size;
  m_resident_size += rhs.m_resident_size;
  m_max_resident_size += rhs.m_max_resident_size;
  return *this;
}

MemoryStats MemoryStats::operator-(const MemoryStats &rhs) {
  return MemoryStats(m_virtual_size - rhs.m_virtual_size,
                     m_resident_size - rhs.m_resident_size,
                     m_max_resident_size - rhs.m_max_resident_size);
}

MemoryStats MemoryStats::operator+(const MemoryStats &rhs) {
  return MemoryStats(m_virtual_size + rhs.m_virtual_size,
                     m_resident_size + rhs.m_resident_size,
                     m_max_resident_size + rhs.m_max_resident_size);
}

MemoryStats MemoryStats::operator/(size_t n) {
  MemoryStats result(*this);
  result.m_virtual_size /= n;
  result.m_resident_size /= n;
  result.m_max_resident_size /= n;
  return result;
}

MemoryStats MemoryStats::operator*(const MemoryStats &rhs) {
  return MemoryStats(m_virtual_size * rhs.m_virtual_size,
                     m_resident_size * rhs.m_resident_size,
                     m_max_resident_size * rhs.m_max_resident_size);
}

Results::ResultSP MemoryStats::GetResult(const char *name,
                                         const char *description) const {
  std::unique_ptr<Results::Dictionary> dict_up(
      new Results::Dictionary(name, NULL));
  dict_up->AddUnsigned("resident", NULL, GetResidentSize());
  dict_up->AddUnsigned("max_resident", NULL, GetMaxResidentSize());
  return Results::ResultSP(dict_up.release());
}

MemoryGauge::ValueType MemoryGauge::Now() {
  task_t task = mach_task_self();
  mach_task_basic_info_data_t taskBasicInfo;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  auto task_info_ret = task_info(task, MACH_TASK_BASIC_INFO,
                                 (task_info_t)&taskBasicInfo, &count);
  if (task_info_ret == KERN_SUCCESS) {
    return MemoryStats(taskBasicInfo.virtual_size, taskBasicInfo.resident_size,
                       taskBasicInfo.resident_size_max);
  }
  return 0;
}

MemoryGauge::MemoryGauge()
    : m_state(MemoryGauge::State::eNeverUsed), m_start(), m_delta() {}

void MemoryGauge::Start() {
  m_state = MemoryGauge::State::eCounting;
  m_start = Now();
}

MemoryGauge::ValueType MemoryGauge::Stop() {
  m_stop = Now();
  assert(m_state == MemoryGauge::State::eCounting &&
         "cannot stop a non-started gauge");
  m_state = MemoryGauge::State::eStopped;
  m_delta = m_stop - m_start;
  return m_delta;
}

MemoryGauge::ValueType MemoryGauge::GetDeltaValue() const {
  assert(m_state == MemoryGauge::State::eStopped &&
         "gauge must be used before you can evaluate it");
  return m_delta;
}

template <>
Results::ResultSP lldb_perf::GetResult(const char *description,
                                       MemoryStats value) {
  return value.GetResult(NULL, description);
}

MemoryStats sqrt(const MemoryStats &arg) {
  long double virt_size = arg.GetVirtualSize();
  long double resident_size = arg.GetResidentSize();
  long double max_resident_size = arg.GetMaxResidentSize();

  virt_size = sqrtl(virt_size);
  resident_size = sqrtl(resident_size);
  max_resident_size = sqrtl(max_resident_size);

  return MemoryStats(virt_size, resident_size, max_resident_size);
}
