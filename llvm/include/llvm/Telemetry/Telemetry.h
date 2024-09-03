//===- llvm/Telemetry/Telemetry.h - Telemetry -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides the basic framework for Telemetry
///
/// It comprises of three important structs/classes:
///
/// - Telemeter: The class responsible for collecting and forwarding
///              telemery data.
/// - TelemetryInfo: data courier
/// - TelemetryConfig: this stores configurations on Telemeter.
///
/// Refer to its documentation at llvm/docs/Telemetry.rst for more details.
//===---------------------------------------------------------------------===//

#ifndef LLVM_TELEMETRY_TELEMETRY_H
#define LLVM_TELEMETRY_TELEMETRY_H

#include <chrono>
#include <ctime>
#include <memory>
#include <optional>
#include <string>

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

namespace llvm {
namespace telemetry {

// Configuration for the Telemeter class.
// This struct can be extended as needed.
struct Config {
  // If true, telemetry will be enabled.
  bool EnableTelemetry;

  // Implementation-defined names of additional destinations to send
  // telemetry data (Could be stdout, stderr, or some local paths, etc).
  //
  // These strings will be interpreted by the vendor's code.
  // So the users must pick the  from their vendor's pre-defined
  // set of Destinations.
  std::vector<std::string> AdditionalDestinations;
};

// Defines a convenient type for timestamp of various events.
// This is used by the EventStats below.
using SteadyTimePoint = std::chrono::time_point<std::chrono::steady_clock>;

// Various time (and possibly memory) statistics of an event.
struct EventStats {
  // REQUIRED: Start time of an event
  SteadyTimePoint Start;
  // OPTIONAL: End time of an event - may be empty if not meaningful.
  std::optional<SteadyTimePoint> End;
  // TBD: could add some memory stats here too?

  TelemetryEventStats() = default;
  TelemetryEventStats(SteadyTimePoint Start) : Start(Start) {}
  TelemetryEventStats(SteadyTimePoint Start, SteadyTimePoint End)
      : Start(Start), End(End) {}
};

struct ExitDescription {
  int ExitCode;
  std::string Description;
};

// For isa, dyn_cast, etc operations on TelemetryInfo.
typedef unsigned KindType;
struct EntryKind {
  static const KindType Base = 0;
};

// TelemetryInfo is the data courier, used to forward data from
// the tool being monitored to the Telemery framework.
//
// This base class contains only the basic set of telemetry data.
// Downstream implementations can add more fields as needed.
struct TelemetryInfo {
  // This represents a unique-id, conventionally corresponding to
  // a tools' session - ie., every time the tool starts until it exits.
  //
  // Note: a tool could have mutliple sessions running at once, in which
  // case, these shall be multiple sets of TelemetryInfo with multiple unique
  // ids.
  //
  // Different usages can assign different types of IDs to this field.
  std::string SessionId;

  // Time/memory statistics of this event.
  TelemetryEventStats Stats;

  std::optional<ExitDescription> ExitDesc;

  // Counting number of entries.
  // (For each set of entries with the same SessionId, this value should
  // be unique for each entry)
  size_t Counter;

  TelemetryInfo() = default;
  ~TelemetryInfo() = default;

  virtual json::Object serializeToJson() const;

  // For isa, dyn_cast, etc, operations.
  virtual KindType getKind() const { return EntryKind::Base; }
  static bool classof(const TelemetryInfo *T) {
    return T->getKind() == EntryKind::Base;
  }
};

// This class presents a data sink to which the Telemetry framework
// sends data.
//
// Its implementation is transparent to the framework.
// It is up to the vendor to decide which pieces of data to forward
// and where to forward them.
class Destination {
public:
  virtual ~TelemetryDestination() = default;
  virtual Error emitEntry(const TelemetryInfo *Entry) = 0;
  virtual std::string name() const = 0;
};

// This class is the main interaction point between any LLVM tool
// and this framework.
// It is responsible for collecting telemetry data from the tool being
// monitored.
class Telemeter {
public:
  // Invoked upon tool startup
  virtual void logStartup(llvm::StringRef ToolPath, TelemetryInfo *Entry) = 0;

  // Invoked upon tool exit.
  virtual void logExit(llvm::StringRef ToolPath, TelemetryInfo *Entry) = 0;

  virtual void addDestination(TelemetryDestination *Destination) = 0;
};

} // namespace telemetry
} // namespace llvm

#endif // LLVM_TELEMETRY_TELEMETRY_H
