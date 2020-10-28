//===- PDLToPDLInterp.h - PDL to PDL Interpreter conversion -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a pass for PDL to PDL Interpreter dialect conversion.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_PDLTOPDLINTERP_PDLTOPDLINTERP_H
#define MLIR_CONVERSION_PDLTOPDLINTERP_PDLTOPDLINTERP_H

#include <memory>

namespace mlir {
class ModuleOp;
template <typename OpT>
class OperationPass;

/// Creates and returns a pass to convert PDL ops to PDL interpreter ops.
std::unique_ptr<OperationPass<ModuleOp>> createPDLToPDLInterpPass();

} // namespace mlir

#endif // MLIR_CONVERSION_PDLTOPDLINTERP_PDLTOPDLINTERP_H
