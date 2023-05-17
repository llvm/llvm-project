//===- FileLineColLocBreakpointManager.h - TODO: add message ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRACING_BREAKPOINTMANAGERS_FILELINECOLLOCBREAKPOINTMANAGER_H
#define MLIR_TRACING_BREAKPOINTMANAGERS_FILELINECOLLOCBREAKPOINTMANAGER_H

#include "mlir/Debug/BreakpointManager.h"
#include "mlir/Debug/ExecutionContext.h"
#include "mlir/IR/Action.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include <memory>
#include <optional>

namespace mlir {
namespace tracing {

/// This breakpoing intends to match a FileLineColLocation, that is a tuple of
/// file name, line number, and column number. Using -1 for  the column and the
/// line number will match any column and line number respectively.
class FileLineColLocBreakpoint
    : public BreakpointBase<FileLineColLocBreakpoint> {
public:
  FileLineColLocBreakpoint(StringRef file, int64_t line, int64_t col)
      : line(line), col(col) {}

  void print(raw_ostream &os) const override {
    os << "Location: " << file << ':' << line << ':' << col;
  }

  /// Parse a string representation in the form of "<file>:<line>:<col>". Return
  /// a tuple with these three elements, the first one is a StringRef pointing
  /// into the original string.
  static FailureOr<std::tuple<StringRef, int64_t, int64_t>> parseFromString(
      StringRef str, llvm::function_ref<void(Twine)> diag = [](Twine) {});

private:
  /// A filename on which to break.
  StringRef file;

  /// A particular line on which to break, or -1 to break on any line.
  int64_t line;

  /// A particular column on which to break, or -1 to break on any column
  int64_t col;

  friend class FileLineColLocBreakpointManager;
};

/// This breakpoint manager is responsible for matching
/// FileLineColLocBreakpoint. It'll extract the location from the action context
/// looking for a FileLineColLocation, and match it against the registered
/// breakpoints.
class FileLineColLocBreakpointManager
    : public BreakpointManagerBase<FileLineColLocBreakpointManager> {
public:
  Breakpoint *match(const Action &action) const override {
    for (const IRUnit &unit : action.getContextIRUnits()) {
      if (auto *op = unit.dyn_cast<Operation *>()) {
        if (auto match = matchFromLocation(op->getLoc()))
          return *match;
        continue;
      }
      if (auto *block = unit.dyn_cast<Block *>()) {
        for (auto &op : block->getOperations()) {
          if (auto match = matchFromLocation(op.getLoc()))
            return *match;
        }
        continue;
      }
      if (Region *region = unit.dyn_cast<Region *>()) {
        if (auto match = matchFromLocation(region->getLoc()))
          return *match;
        continue;
      }
    }
    return {};
  }

  FileLineColLocBreakpoint *addBreakpoint(StringRef file, int64_t line,
                                          int64_t col = -1) {
    auto &breakpoint = breakpoints[std::make_tuple(file, line, col)];
    if (!breakpoint)
      breakpoint = std::make_unique<FileLineColLocBreakpoint>(file, line, col);
    return breakpoint.get();
  }

private:
  std::optional<Breakpoint *> matchFromLocation(Location initialLoc) const {
    std::optional<Breakpoint *> match = std::nullopt;
    initialLoc->walk([&](Location loc) {
      auto fileLoc = dyn_cast<FileLineColLoc>(loc);
      if (!fileLoc)
        return WalkResult::advance();
      StringRef file = fileLoc.getFilename();
      int64_t line = fileLoc.getLine();
      int64_t col = fileLoc.getColumn();
      auto lookup = breakpoints.find(std::make_tuple(file, line, col));
      if (lookup != breakpoints.end() && lookup->second->isEnabled()) {
        match = lookup->second.get();
        return WalkResult::interrupt();
      }
      // If not found, check with the -1 key if we have a breakpoint for any
      // col.
      lookup = breakpoints.find(std::make_tuple(file, line, -1));
      if (lookup != breakpoints.end() && lookup->second->isEnabled()) {
        match = lookup->second.get();
        return WalkResult::interrupt();
      }
      // If not found, check with the -1 key if we have a breakpoint for any
      // line.
      lookup = breakpoints.find(std::make_tuple(file, -1, -1));
      if (lookup != breakpoints.end() && lookup->second->isEnabled()) {
        match = lookup->second.get();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return match;
  }

  /// A map from a (filename, line, column) -> breakpoint.
  DenseMap<std::tuple<StringRef, int64_t, int64_t>,
           std::unique_ptr<FileLineColLocBreakpoint>>
      breakpoints;
};

} // namespace tracing
} // namespace mlir

#endif // MLIR_TRACING_BREAKPOINTMANAGERS_FILELINECOLLOCBREAKPOINTMANAGER_H
