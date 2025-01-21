//===- Linker.cpp - MLIR linker implementation -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Linker/Linker.h"

#include "llvm/ADT/StringSet.h"

using namespace mlir;

class ModuleLinker {
  IRMover &mover;
  OwningOpRef<Operation *> src;

  /// For symbol clashes, prefer those from src.
  unsigned flags;

  /// List of global value names that should be internalized.
  StringSet<> internalize;

  /// Function that will perform the actual internalization. The reason for a
  /// callback is that the linker cannot call internalizeModule without
  /// creating a circular dependency between IPO and the linker.
  std::function<void(Operation *, const StringSet<> &)> internalizeCallback;

public:
  ModuleLinker(IRMover &mover, OwningOpRef<Operation *> src, unsigned flags,
               std::function<void(Operation *, const StringSet<> &)>
                   internalizeCallback = {})
      : mover(mover), src(std::move(src)), flags(flags),
        internalizeCallback(std::move(internalizeCallback)) {}
  LogicalResult run();
};

LogicalResult ModuleLinker::run() { return failure(); }

Linker::Linker(Operation *composite) : mover(composite) {}

LogicalResult Linker::linkInModule(
    OwningOpRef<Operation *> src, unsigned flags,
    std::function<void(Operation *, const StringSet<> &)> internalizeCallback) {
  ModuleLinker modLinker(mover, std::move(src), flags,
                         std::move(internalizeCallback));
  return modLinker.run();
}
