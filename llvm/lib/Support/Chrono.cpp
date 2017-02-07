//===- Support/Chrono.cpp - Utilities for Timing Manipulation ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Chrono.h"
#include "llvm/Config/config.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

using namespace sys;

const char detail::unit<std::ratio<3600>>::value[] = "h";
const char detail::unit<std::ratio<60>>::value[] = "m";
const char detail::unit<std::ratio<1>>::value[] = "s";
const char detail::unit<std::milli>::value[] = "ms";
const char detail::unit<std::micro>::value[] = "us";
const char detail::unit<std::nano>::value[] = "ns";

static inline struct tm getStructTM(TimePoint<> TP) {
  struct tm Storage;
  std::time_t OurTime = toTimeT(TP);

#if defined(LLVM_ON_UNIX)
  struct tm *LT = ::localtime_r(&OurTime, &Storage);
  assert(LT);
  (void)LT;
#endif
#if defined(LLVM_ON_WIN32)
  int Error = ::localtime_s(&Storage, &OurTime);
  assert(!Error);
  (void)Error;
#endif

  return Storage;
}

raw_ostream &operator<<(raw_ostream &OS, TimePoint<> TP) {
  struct tm LT = getStructTM(TP);
  char Buffer[sizeof("YYYY-MM-DD HH:MM:SS")];
  strftime(Buffer, sizeof(Buffer), "%Y-%m-%d %H:%M:%S", &LT);
  return OS << Buffer << '.'
            << format("%.9lu",
                      long((TP.time_since_epoch() % std::chrono::seconds(1))
                               .count()));
}

} // namespace llvm
