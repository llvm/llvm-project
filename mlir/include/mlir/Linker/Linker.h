//===- Linker.h - MLIR Module Linker ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LINKER_LINKER_H
#define MLIR_LINKER_LINKER_H

#include "mlir/Linker/IRMover.h"

#include "mlir/Interfaces/LinkageInterfaces.h"

namespace mlir {

using InternalizeCallbackFn =
    std::function<void(LinkableModuleOpInterface, const StringSet<> &)>;

/// This class provides the core functionality of linking in MLIR, it mirrors
/// functionality from `llvm/Linker/Linker.h` for MLIR. It keeps a pointer to
/// the merged module so far. It doesn't take ownership of the module since it
/// is assumed that the user of this class will want to do something with it
/// after the linking.
class Linker {
public:
  enum Flags {
    None = 0,
    OverrideFromSrc = (1 << 0),
    LinkOnlyNeeded = (1 << 1),
  };

  Linker(LinkableModuleOpInterface composite);

  MLIRContext *getContext() { return mover.getContext(); }

  LogicalResult linkInModule(OwningOpRef<Operation *> src,
                             unsigned flags = None,
                             InternalizeCallbackFn internalizeCallback = {});

private:
  IRMover mover;
};

} // namespace mlir

#endif
