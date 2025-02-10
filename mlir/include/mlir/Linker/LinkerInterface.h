//===- LinkerInterface.h - MLIR Linker Interface ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header defines interfaces and utilities necessary for dialects
// to hook into mlir linker.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LINKER_LINKAGEDIALECTINTERFACE_H
#define MLIR_LINKER_LINKAGEDIALECTINTERFACE_H

#include "mlir/IR/DialectInterface.h"

#include "mlir/Interfaces/LinkageInterfaces.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// LinkageInterface
//===----------------------------------------------------------------------===//

class LinkerInterface : public DialectInterface::Base<LinkerInterface> {
public:
  LinkerInterface(Dialect *dialect) : Base(dialect) {}
};

} // namespace mlir

#endif // MLIR_LINKER_LINKAGEDIALECTINTERFACE_H
