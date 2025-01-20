//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides the basic framework for Telemetry.
/// Refer to its documentation at llvm/docs/Telemetry.rst for more details.
//===---------------------------------------------------------------------===//

#ifndef LLVM_TELEMETRY_TELEMETRY_H
#define LLVM_TELEMETRY_TELEMETRY_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace llvm {
namespace telemetry {

class Serializer {
public:
  virtual ~Serializer() = default;

  virtual Error init() = 0;
  virtual void write(StringRef KeyName, bool Value) = 0;
  virtual void write(StringRef KeyName, StringRef Value) = 0;
  virtual void write(StringRef KeyName, int Value) = 0;
  virtual void write(StringRef KeyName, long Value) = 0;
  virtual void write(StringRef KeyName, long long Value) = 0;
  virtual void write(StringRef KeyName, unsigned int Value) = 0;
  virtual void write(StringRef KeyName, unsigned long Value) = 0;
  virtual void write(StringRef KeyName, unsigned long long Value) = 0;
  virtual void beginObject(StringRef KeyName) = 0;
  virtual void endObject() = 0;
  virtual Error finalize() = 0;

  template <typename T, typename = typename T::mapped_type>
  void write(StringRef KeyName, const T &Map) {
    static_assert(std::is_convertible_v<typename T::key_type, StringRef>,
                  "KeyType must be convertible to string");
    beginObject(KeyName);
    for (const auto &KeyVal : Map)
      write(KeyVal.first, KeyVal.second);
    endObject();
  }
};

/// Configuration for the Manager class.
/// This stores configurations from both users and vendors and is passed
/// to the Manager upon construction. (Any changes to the config after
/// the Manager's construction will not have any effect on it).
///
/// This struct can be extended as needed to add additional configuration
/// points specific to a vendor's implementation.
struct Config {
  virtual ~Config() = default;

  // If true, telemetry will be enabled.
  const bool EnableTelemetry;
  Config(bool E) : EnableTelemetry(E) {}

  virtual std::optional<std::string> makeSessionId() { return std::nullopt; }
};

/// For isa, dyn_cast, etc operations on TelemetryInfo.
typedef unsigned KindType;
/// This struct is used by TelemetryInfo to support isa<>, dyn_cast<>
/// operations.
/// It is defined as a struct (rather than an enum) because it is
/// expected to be extended by subclasses which may have
/// additional TelemetryInfo types defined to describe different events.
struct EntryKind {
  static const KindType Base = 0;
};

/// TelemetryInfo is the data courier, used to move instrumented data
/// from the tool being monitored to the Telemetry framework.
///
/// This base class contains only the basic set of telemetry data.
/// Downstream implementations can define more subclasses with
/// additional fields to describe different events and concepts.
///
/// For example, The LLDB debugger can define a DebugCommandInfo subclass
/// which has additional fields about the debug-command being instrumented,
/// such as `CommandArguments` or `CommandName`.
struct TelemetryInfo {
  // This represents a unique-id, conventionally corresponding to
  // a tool's session - i.e., every time the tool starts until it exits.
  //
  // Note: a tool could have multiple sessions running at once, in which
  // case, these shall be multiple sets of TelemetryInfo with multiple unique
  // IDs.
  //
  // Different usages can assign different types of IDs to this field.
  std::string SessionId;

  TelemetryInfo() = default;
  virtual ~TelemetryInfo() = default;

  virtual void serialize(Serializer &serializer) const;

  // For isa, dyn_cast, etc, operations.
  virtual KindType getKind() const { return EntryKind::Base; }
  static bool classof(const TelemetryInfo *T) {
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
  virtual Error receiveEntry(const TelemetryInfo *Entry) = 0;
  virtual StringLiteral name() const = 0;
};

/// This class is the main interaction point between any LLVM tool
/// and this framework.
/// It is responsible for collecting telemetry data from the tool being
/// monitored and transmitting the data elsewhere.
class Manager {
public:
  virtual ~Manager() = default;

  // Optional callback for subclasses to perform additional tasks before
  // dispatching to Destinations.
  virtual Error preDispatch(TelemetryInfo *Entry) = 0;

  // Dispatch Telemetry data to the Destination(s).
  // The argument is non-const because the Manager may add or remove
  // data from the entry.
  virtual Error dispatch(TelemetryInfo *Entry);

  // Register a Destination.
  void addDestination(std::unique_ptr<Destination> Destination);

private:
  std::vector<std::unique_ptr<Destination>> Destinations;
};

} // namespace telemetry
} // namespace llvm

#endif // LLVM_TELEMETRY_TELEMETRY_H
