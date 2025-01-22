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

#include "mlir/Interfaces/LinkageInterfaces.h"

namespace mlir {

class IRMover {
public:
  IRMover(Operation *composite);

  LinkableModuleOpInterface getComposite() { return composite; }
  MLIRContext *getContext() { return composite->getContext(); }

private:
  LinkableModuleOpInterface composite;
};

} // namespace mlir

#endif
