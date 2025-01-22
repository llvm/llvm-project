//===- Linker.cpp - MLIR linker implementation -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Linker/Linker.h"

#include "mlir/Interfaces/LinkageInterfaces.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

enum class LinkFrom { Dst, Src, Both };

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
  InternalizeCallbackFn internalizeCallback;

  LinkableModuleOpInterface getSourceModule() {
    return cast<LinkableModuleOpInterface>(src.get());
  }

public:
  ModuleLinker(IRMover &mover, OwningOpRef<Operation *> src, unsigned flags,
               InternalizeCallbackFn internalizeCallback = {})
      : mover(mover), src(std::move(src)), flags(flags),
        internalizeCallback(std::move(internalizeCallback)) {}
  LogicalResult run();
};

LogicalResult ModuleLinker::run() {
  auto dst = mover.getComposite();

  return failure();
}

Linker::Linker(LinkableModuleOpInterface composite) : mover(composite) {}

LogicalResult Linker::linkInModule(OwningOpRef<Operation *> src, unsigned flags,
                                   InternalizeCallbackFn internalizeCallback) {
  ModuleLinker modLinker(mover, std::move(src), flags,
                         std::move(internalizeCallback));
  return modLinker.run();
}
