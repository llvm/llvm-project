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
/// This framework is intended to be configurable and extensible:
///    - Any LLVM tool that wants to use Telemetry can extend/customize it.
///    - Toolchain vendors can also provide custom implementation/config of the
///      Telemetry library, which could either overrides or extends the given
///      tool's upstream implementation, to best fit their organization's usage
///      and security models.
///    - End users of such tool can also configure telemetry (as allowed
///      by their vendor).
///
/// Note: There are two important points to highlight about this package:
///
///  (0) There is (currently) no concrete implementation of Telemetry in
///      upstream LLVM. We only provide the abstract API here. Any tool
///      that wants telemetry will have to implement one.
///
///      The reason for this is because all the tools in llvm are
///      very different in what they care about (what/when/where to instrument)
///      Hence it might not be practical to a single implementation.
///      However, if in the future when we see any common pattern, we can
///      extract them into a shared place. That is TBD - contributions welcomed.
///
///  (1) No implementation of Telemetry in upstream LLVM shall directly store
///      any of the collected data due to privacy and security reasons:
///        + Different organizations have different opinions on which data
///          is sensitive and which is not.
///        + Data ownerships and data collection consents are hard to
///          accommodate from LLVM developers' point of view.
///          (Eg., the data collected by Telemetry framework is NOT neccessarily
///           owned by the user of a LLVM tool with Telemetry enabled, hence
///           their consent to data collection isn't meaningful. On the other
///           hand, we have no practical way to request consent from "real"
///           owners.
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

struct TelemetryConfig {
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

using SteadyTimePoint = std::chrono::time_point<std::chrono::steady_clock>;

struct TelemetryEventStats {
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
// the tool being monitored and the Telemery framework.
//
// This base class contains only the basic set of telemetry data.
// Downstream implementations can add more fields as needed.
struct TelemetryInfo {
  // A "session" corresponds to every time the tool starts.
  // All entries emitted for the same session will have
  // the same session_uuid
  std::string SessionUuid;

  TelemetryEventStats Stats;

  std::optional<ExitDescription> ExitDesc;

  // Counting number of entries.
  // (For each set of entries with the same session_uuid, this value should
  // be unique for each entry)
  size_t Counter;

  TelemetryInfo() = default;
  ~TelemetryInfo() = default;

  virtual json::Object serializeToJson() const;

  // For isa, dyn_cast, etc, operations.
  virtual KindType getEntryKind() const { return EntryKind::Base; }
  static bool classof(const TelemetryInfo *T) {
    return T->getEntryKind() == EntryKind::Base;
  }
};

// Where/how to send the telemetry entries.
class TelemetryDestination {
public:
  virtual ~TelemetryDestination() = default;
  virtual Error emitEntry(const TelemetryInfo *Entry) = 0;
  virtual std::string name() const = 0;
};

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
