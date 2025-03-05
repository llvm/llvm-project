//===- IRMover.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LINKER_IRMOVER_H
#define MLIR_LINKER_IRMOVER_H

#include "mlir/IR/BuiltinOps.h"

#include "mlir/Linker/LinkerInterface.h"

using llvm::Error;

namespace mlir::link {

class IRMover {
public:
  IRMover(mlir::ModuleOp composite) : composite(composite) {}

  ModuleOp getComposite() { return composite; }
  MLIRContext *getContext() { return composite->getContext(); }

  Error move(OwningOpRef<Operation *> src, ArrayRef<GlobalValue> valuesToLink);

private:
  mlir::ModuleOp composite;
};

} // namespace mlir::link

#endif
