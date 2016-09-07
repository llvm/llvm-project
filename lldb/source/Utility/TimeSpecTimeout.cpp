//===--------------------- TimeSpecTimeout.cpp ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TimeSpecTimeout.h"

using namespace lldb_private;

const struct timespec *
TimeSpecTimeout::SetAbsoluteTimeoutMircoSeconds32(uint32_t timeout_usec) {
  if (timeout_usec == UINT32_MAX) {
    m_infinite = true;
  } else {
    m_infinite = false;
    TimeValue time_value(TimeValue::Now());
    time_value.OffsetWithMicroSeconds(timeout_usec);
    m_timespec = time_value.GetAsTimeSpec();
  }
  return GetTimeSpecPtr();
}

const struct timespec *
TimeSpecTimeout::SetRelativeTimeoutMircoSeconds32(uint32_t timeout_usec) {
  if (timeout_usec == UINT32_MAX) {
    m_infinite = true;
  } else {
    m_infinite = false;
    TimeValue time_value;
    time_value.OffsetWithMicroSeconds(timeout_usec);
    m_timespec = time_value.GetAsTimeSpec();
  }
  return GetTimeSpecPtr();
}
