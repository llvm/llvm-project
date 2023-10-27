//===- TagBreakpointManager.h - Simple breakpoint Support -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DEBUG_BREAKPOINTMANAGERS_TAGBREAKPOINTMANAGER_H
#define MLIR_DEBUG_BREAKPOINTMANAGERS_TAGBREAKPOINTMANAGER_H

#include "mlir/Debug/BreakpointManager.h"
#include "mlir/Debug/ExecutionContext.h"
#include "mlir/IR/Action.h"
#include "llvm/ADT/MapVector.h"

namespace mlir {
namespace tracing {

/// Simple breakpoint matching an action "tag".
class TagBreakpoint : public BreakpointBase<TagBreakpoint> {
public:
  TagBreakpoint(StringRef tag) : tag(tag) {}

  void print(raw_ostream &os) const override { os << "Tag: `" << tag << '`'; }

private:
  /// A tag to associate the TagBreakpoint with.
  std::string tag;

  /// Allow access to `tag`.
  friend class TagBreakpointManager;
};

/// This is a manager to store a collection of breakpoints that trigger
/// on tags.
class TagBreakpointManager
    : public BreakpointManagerBase<TagBreakpointManager> {
public:
  Breakpoint *match(const Action &action) const override {
    auto it = breakpoints.find(action.getTag());
    if (it != breakpoints.end() && it->second->isEnabled())
      return it->second.get();
    return {};
  }

  /// Add a breakpoint to the manager for the given tag and return it.
  /// If a breakpoint already exists for the given tag, return the existing
  /// instance.
  TagBreakpoint *addBreakpoint(StringRef tag) {
    auto result = breakpoints.insert({tag, nullptr});
    auto &it = result.first;
    if (result.second)
      it->second = std::make_unique<TagBreakpoint>(tag.str());
    return it->second.get();
  }

private:
  llvm::StringMap<std::unique_ptr<TagBreakpoint>> breakpoints;
};

} // namespace tracing
} // namespace mlir

#endif // MLIR_DEBUG_BREAKPOINTMANAGERS_TAGBREAKPOINTMANAGER_H
