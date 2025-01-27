//===- Comdat.h - MLIR Comdat -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LINKER_COMDAT_H
#define MLIR_LINKER_COMDAT_H

#include "llvm/ADT/StringMap.h"

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"

namespace mlir {
namespace link {

using ComdatSelectionKind = LLVM::comdat::Comdat;

using ComdatSymbolTable = llvm::StringMap<ComdatSelectionKind>;

} // namespace link
} // namespace mlir

#endif // MLIR_LINKER_COMDAT_H
