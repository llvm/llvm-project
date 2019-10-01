//===-- PerfHelper.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Helpers for measuring perf events.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_EXEGESIS_PERFHELPER_H
#define LLVM_TOOLS_LLVM_EXEGESIS_PERFHELPER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/config.h"
#include <functional>
#include <memory>

struct perf_event_attr;

namespace llvm {
namespace exegesis {
namespace pfm {

// Returns true on error.
bool pfmInitialize();
void pfmTerminate();

// Retrieves the encoding for the event described by pfm_event_string.
// NOTE: pfm_initialize() must be called before creating PerfEvent objects.
class PerfEvent {
public:
  // http://perfmon2.sourceforge.net/manv4/libpfm.html
  // Events are expressed as strings. e.g. "INSTRUCTION_RETIRED"
  explicit PerfEvent(llvm::StringRef pfm_event_string);

  PerfEvent(const PerfEvent &) = delete;
  PerfEvent(PerfEvent &&other);
  ~PerfEvent();

  // The pfm_event_string passed at construction time.
  llvm::StringRef name() const;

  // Whether the event was successfully created.
  bool valid() const;

  // The encoded event to be passed to the Kernel.
  const perf_event_attr *attribute() const;

  // The fully qualified name for the event.
  // e.g. "snb_ep::INSTRUCTION_RETIRED:e=0:i=0:c=0:t=0:u=1:k=0:mg=0:mh=1"
  llvm::StringRef getPfmEventString() const;

private:
  const std::string EventString;
  std::string FullQualifiedEventString;
  perf_event_attr *Attr;
};

// Uses a valid PerfEvent to configure the Kernel so we can measure the
// underlying event.
struct Counter {
  // event: the PerfEvent to measure.
  explicit Counter(const PerfEvent &event);

  Counter(const Counter &) = delete;
  Counter(Counter &&other) = default;

  ~Counter();

  void start();         // Starts the measurement of the event.
  void stop();          // Stops the measurement of the event.
  int64_t read() const; // Return the current value of the counter.

private:
#ifdef HAVE_LIBPFM
  int FileDescriptor = -1;
#endif
};

// Helper to measure a list of PerfEvent for a particular function.
// callback is called for each successful measure (PerfEvent needs to be valid).
template <typename Function>
void Measure(
    llvm::ArrayRef<PerfEvent> Events,
    const std::function<void(const PerfEvent &Event, int64_t Value)> &Callback,
    Function Fn) {
  for (const auto &Event : Events) {
    if (!Event.valid())
      continue;
    Counter Cnt(Event);
    Cnt.start();
    Fn();
    Cnt.stop();
    Callback(Event, Cnt.read());
  }
}

} // namespace pfm
} // namespace exegesis
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_EXEGESIS_PERFHELPER_H
