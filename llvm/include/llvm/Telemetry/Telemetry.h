//===- llvm/Telemetry/Telemetry.h - Telemetry -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

using SteadyTimePoint = std::chrono::time_point<std::chrono::steady_clock>;

struct TelemetryConfig {
  // If true, telemetry will be enabled.
  bool EnableTelemetry;

  // Additional destinations to send the logged entries.
  // Could be stdout, stderr, or some local paths.
  // Note: these are destinations are __in addition to__ whatever the default
  // destination(s) are, as implemented by vendors.
  std::vector<std::string> AdditionalDestinations;
};

struct TelemetryEventStats {
  // REQUIRED: Start time of event
  SteadyTimePoint Start;
  // OPTIONAL: End time of event - may be empty if not meaningful.
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

// The base class contains the basic set of data.
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
  static bool classof(const TelemetryInfo* T) {
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
