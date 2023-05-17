//===- BreakpointManager.h - Breakpoint Manager Support ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRACING_BREAKPOINTMANAGER_H
#define MLIR_TRACING_BREAKPOINTMANAGER_H

#include "mlir/IR/Action.h"
#include "llvm/ADT/MapVector.h"

namespace mlir {
namespace tracing {

/// This abstract class represents a breakpoint.
class Breakpoint {
public:
  virtual ~Breakpoint() = default;

  /// TypeID for the subclass, used for casting purpose.
  TypeID getTypeID() const { return typeID; }

  bool isEnabled() const { return enableStatus; }
  void enable() { enableStatus = true; }
  void disable() { enableStatus = false; }
  virtual void print(raw_ostream &os) const = 0;

protected:
  Breakpoint(TypeID typeID) : enableStatus(true), typeID(typeID) {}

private:
  /// The current state of the breakpoint. A breakpoint can be either enabled
  /// or disabled.
  bool enableStatus;
  TypeID typeID;
};

inline raw_ostream &operator<<(raw_ostream &os, const Breakpoint &breakpoint) {
  breakpoint.print(os);
  return os;
}

/// This class provides a CRTP wrapper around a base breakpoint class to define
/// a few necessary utility methods.
template <typename Derived>
class BreakpointBase : public Breakpoint {
public:
  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const Breakpoint *breakpoint) {
    return breakpoint->getTypeID() == TypeID::get<Derived>();
  }

protected:
  BreakpointBase() : Breakpoint(TypeID::get<Derived>()) {}
};

/// A breakpoint manager is responsible for managing a set of breakpoints and
/// matching them to a given action.
class BreakpointManager {
public:
  virtual ~BreakpointManager() = default;

  /// TypeID for the subclass, used for casting purpose.
  TypeID getTypeID() const { return typeID; }

  /// Try to match a Breakpoint to a given Action. If there is a match and
  /// the breakpoint is enabled, return the breakpoint. Otherwise, return
  /// nullptr.
  virtual Breakpoint *match(const Action &action) const = 0;

protected:
  BreakpointManager(TypeID typeID) : typeID(typeID) {}

  TypeID typeID;
};

/// CRTP base class for BreakpointManager implementations.
template <typename Derived>
class BreakpointManagerBase : public BreakpointManager {
public:
  BreakpointManagerBase() : BreakpointManager(TypeID::get<Derived>()) {}

  /// Provide classof to allow casting between breakpoint manager types.
  static bool classof(const BreakpointManager *breakpointManager) {
    return breakpointManager->getTypeID() == TypeID::get<Derived>();
  }
};

} // namespace tracing
} // namespace mlir

#endif // MLIR_TRACING_BREAKPOINTMANAGER_H
