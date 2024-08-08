//===-- SummaryStatistics.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/SummaryStatistics.h"

using namespace lldb;
using namespace lldb_private;


StatsDuration& SummaryStatistics::GetDurationReference() {
  m_summary_count++;
  return m_total_time;
}

ConstString SummaryStatistics::GetName() const {
  return m_name;
}

uint64_t SummaryStatistics::GetSummaryCount() const {
  return m_summary_count;
}
