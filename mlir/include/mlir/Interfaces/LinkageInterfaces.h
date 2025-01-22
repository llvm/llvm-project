//===- LinkageInterfaces.h - Interfaces for Linkage -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of interfaces for ops that interact with linkage.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_LINKAGEINTERFACES_H_
#define MLIR_INTERFACES_LINKAGEINTERFACES_H_

#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/StringMap.h"

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"

namespace mlir {
namespace link {

using SelectionKind = LLVM::comdat::ComdatAttr;
using Linkage = LLVM::Linkage;

} // namespace link
} // namespace mlir

/// Include the generated interface declarations.
#include "mlir/Interfaces/LinkageInterfaces.h.inc"

#endif // MLIR_INTERFACES_LINKAGEINTERFACES_H_
