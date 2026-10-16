//===- InitAllDialects.h - MLIR Dialects Registration -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all dialects and
// passes to the system.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INITALLDIALECTS_H_
#define MLIR_INITALLDIALECTS_H_

namespace mlir {
class DialectRegistry;
class MLIRContext;

/// Add all the MLIR dialects to the provided registry.
void registerAllDialects(DialectRegistry &registry);

/// Append all the MLIR dialects to the registry contained in the given context.
void registerAllDialects(MLIRContext &context);

} // namespace mlir

#endif // MLIR_INITALLDIALECTS_H_
