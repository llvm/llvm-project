//===-- TimeValue.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/TimeValue.h"
#include "lldb/Host/Config.h"

// C Includes
#include <cstring>
#include <stddef.h>
#include <time.h>

#ifdef _MSC_VER
#include "lldb/Host/windows/windows.h"
#endif

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Stream.h"

using namespace lldb_private;

//----------------------------------------------------------------------
// TimeValue constructor
//----------------------------------------------------------------------
TimeValue::TimeValue() : m_nano_seconds(0) {}

//----------------------------------------------------------------------
// TimeValue copy constructor
//----------------------------------------------------------------------
TimeValue::TimeValue(const TimeValue &rhs)
    : m_nano_seconds(rhs.m_nano_seconds) {}

TimeValue::TimeValue(const struct timespec &ts)
    : m_nano_seconds((uint64_t)ts.tv_sec * NanoSecPerSec + ts.tv_nsec) {}

TimeValue::TimeValue(uint32_t seconds, uint32_t nanos)
    : m_nano_seconds((uint64_t)seconds * NanoSecPerSec + nanos) {}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
TimeValue::~TimeValue() {}

uint64_t TimeValue::GetAsNanoSecondsSinceJan1_1970() const {
  return m_nano_seconds;
}

uint64_t TimeValue::GetAsMicroSecondsSinceJan1_1970() const {
  return m_nano_seconds / NanoSecPerMicroSec;
}

uint64_t TimeValue::GetAsSecondsSinceJan1_1970() const {
  return m_nano_seconds / NanoSecPerSec;
}

struct timespec TimeValue::GetAsTimeSpec() const {
  struct timespec ts;
  ts.tv_sec = m_nano_seconds / NanoSecPerSec;
  ts.tv_nsec = m_nano_seconds % NanoSecPerSec;
  return ts;
}

void TimeValue::Clear() { m_nano_seconds = 0; }

bool TimeValue::IsValid() const { return m_nano_seconds != 0; }

void TimeValue::OffsetWithSeconds(uint64_t sec) {
  m_nano_seconds += sec * NanoSecPerSec;
}

void TimeValue::OffsetWithMicroSeconds(uint64_t usec) {
  m_nano_seconds += usec * NanoSecPerMicroSec;
}

void TimeValue::OffsetWithNanoSeconds(uint64_t nsec) { m_nano_seconds += nsec; }

TimeValue TimeValue::Now() {
  uint32_t seconds, nanoseconds;
#if _MSC_VER
  SYSTEMTIME st;
  GetSystemTime(&st);
  nanoseconds = st.wMilliseconds * 1000000;
  FILETIME ft;
  SystemTimeToFileTime(&st, &ft);

  seconds =
      ((((uint64_t)ft.dwHighDateTime) << 32 | ft.dwLowDateTime) / 10000000) -
      11644473600ULL;
#else
  struct timeval tv;
  gettimeofday(&tv, NULL);
  seconds = tv.tv_sec;
  nanoseconds = tv.tv_usec * NanoSecPerMicroSec;
#endif
  TimeValue now(seconds, nanoseconds);
  return now;
}

//----------------------------------------------------------------------
// TimeValue assignment operator
//----------------------------------------------------------------------
const TimeValue &TimeValue::operator=(const TimeValue &rhs) {
  m_nano_seconds = rhs.m_nano_seconds;
  return *this;
}

void TimeValue::Dump(Stream *s, uint32_t width) const {
  if (s == NULL)
    return;

#ifndef LLDB_DISABLE_POSIX
  char time_buf[32];
  time_t time = GetAsSecondsSinceJan1_1970();
  char *time_cstr = ::ctime_r(&time, time_buf);
  if (time_cstr) {
    char *newline = ::strpbrk(time_cstr, "\n\r");
    if (newline)
      *newline = '\0';
    if (width > 0)
      s->Printf("%-*s", width, time_cstr);
    else
      s->PutCString(time_cstr);
  } else if (width > 0)
    s->Printf("%-*s", width, "");
#endif
}

bool lldb_private::operator==(const TimeValue &lhs, const TimeValue &rhs) {
  return lhs.GetAsNanoSecondsSinceJan1_1970() ==
         rhs.GetAsNanoSecondsSinceJan1_1970();
}

bool lldb_private::operator!=(const TimeValue &lhs, const TimeValue &rhs) {
  return lhs.GetAsNanoSecondsSinceJan1_1970() !=
         rhs.GetAsNanoSecondsSinceJan1_1970();
}

bool lldb_private::operator<(const TimeValue &lhs, const TimeValue &rhs) {
  return lhs.GetAsNanoSecondsSinceJan1_1970() <
         rhs.GetAsNanoSecondsSinceJan1_1970();
}

bool lldb_private::operator<=(const TimeValue &lhs, const TimeValue &rhs) {
  return lhs.GetAsNanoSecondsSinceJan1_1970() <=
         rhs.GetAsNanoSecondsSinceJan1_1970();
}

bool lldb_private::operator>(const TimeValue &lhs, const TimeValue &rhs) {
  return lhs.GetAsNanoSecondsSinceJan1_1970() >
         rhs.GetAsNanoSecondsSinceJan1_1970();
}

bool lldb_private::operator>=(const TimeValue &lhs, const TimeValue &rhs) {
  return lhs.GetAsNanoSecondsSinceJan1_1970() >=
         rhs.GetAsNanoSecondsSinceJan1_1970();
}

uint64_t lldb_private::operator-(const TimeValue &lhs, const TimeValue &rhs) {
  return lhs.GetAsNanoSecondsSinceJan1_1970() -
         rhs.GetAsNanoSecondsSinceJan1_1970();
}
