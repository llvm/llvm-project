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

/// Configuration for the Telemeter class.
/// This stores configurations from both users and vendors and is passed
/// to the Telemeter upon contruction. (Any changes to the config after
/// the Telemeter's construction will not have effect on it).
///
/// This struct can be extended as needed to add additional configuration
/// points specific to a vendor's implementation.
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

/// Defines a convenient type for timestamp of various events.
/// This is used by the EventStats below.
using SteadyTimePoint = std::chrono::time_point<std::chrono::steady_clock>;

/// Various time (and possibly memory) statistics of an event.
struct EventStats {
  // REQUIRED: Start time of an event
  SteadyTimePoint Start;
  // OPTIONAL: End time of an event - may be empty if not meaningful.
  std::optional<SteadyTimePoint> End;
  // TBD: could add some memory stats here too?

  EventStats() = default;
  EventStats(SteadyTimePoint Start) : Start(Start) {}
  EventStats(SteadyTimePoint Start, SteadyTimePoint End)
      : Start(Start), End(End) {}
};

/// Describes the exit signal of an event.
/// This is used by TelemetryInfo below.
struct ExitDescription {
  int ExitCode;
  std::string Description;
};

/// For isa, dyn_cast, etc operations on TelemetryInfo.
typedef unsigned KindType;
/// This struct is used by TelemetryInfo to support isa<>, dyn_cast<>
/// operations.
/// It is defined as a struct(rather than an enum) because it is
/// expectend to be extended by subclasses which may have
/// additional TelemetryInfo types defined to describe different events.
struct EntryKind {
  static const KindType Base = 0;
};

/// TelemetryInfo is the data courier, used to move instrumented data
/// from the tool being monitored to the Telemery framework.
///
/// This base class contains only the basic set of telemetry data.
/// Downstream implementations can add more fields as needed to describe
/// additional events.
///
/// For eg., The LLDB debugger can define a DebugCommandInfo subclass
/// which has additional fields about the debug-command being instrumented,
/// such as `CommandArguments` or `CommandName`.
struct TelemetryInfo {
  // This represents a unique-id, conventionally corresponding to
  // a tool's session - i.e., every time the tool starts until it exits.
  //
  // Note: a tool could have multiple sessions running at once, in which
  // case, these shall be multiple sets of TelemetryInfo with multiple unique
  // ids.
  //
  // Different usages can assign different types of IDs to this field.
  std::string SessionId;

  // Time/memory statistics of this event.
  EventStats Stats;

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
    if (T == nullptr)
      return false;
    return T->getKind() == EntryKind::Base;
  }
};

/// This class presents a data sink to which the Telemetry framework
/// sends data.
///
/// Its implementation is transparent to the framework.
/// It is up to the vendor to decide which pieces of data to forward
/// and where to forward them.
class Destination {
public:
  virtual ~Destination() = default;
  virtual Error emitEntry(const TelemetryInfo *Entry) = 0;
  virtual std::string name() const = 0;
};

/// This class is the main interaction point between any LLVM tool
/// and this framework.
/// It is responsible for collecting telemetry data from the tool being
/// monitored and transmitting the data elsewhere.
class Telemeter {
public:
  // Invoked upon tool startup
  virtual void logStartup(llvm::StringRef ToolPath, TelemetryInfo *Entry) = 0;

  // Invoked upon tool exit.
  virtual void logExit(llvm::StringRef ToolPath, TelemetryInfo *Entry) = 0;

  virtual void addDestination(Destination *Destination) = 0;
};

} // namespace telemetry
} // namespace llvm

#endif // LLVM_TELEMETRY_TELEMETRY_H
