//===----- All.h - MLIR To Cpp Translation Registration ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to register the all dialect translations to Cpp.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_CPP_DIALECT_ALL_H
#define MLIR_TARGET_CPP_DIALECT_ALL_H

#include "mlir/Target/Cpp/Dialect/Builtin/BuiltinToCppTranslation.h"
#include "mlir/Target/Cpp/Dialect/ControlFlow/ControlFlowToCppTranslation.h"
#include "mlir/Target/Cpp/Dialect/EmitC/EmitCToCppTranslation.h"
#include "mlir/Target/Cpp/Dialect/Func/FuncToCppTranslation.h"

namespace mlir {
class DialectRegistry;

/// Registers all dialects that can be translated to Cpp and the
/// corresponding translation interfaces.
static inline void registerAllToCppTranslations(DialectRegistry &registry) {
  registerBuiltinDialectCppTranslation(registry);
  registerControlFlowDialectCppTranslation(registry);
  registerEmitCDialectCppTranslation(registry);
  registerFuncDialectCppTranslation(registry);
}

} // namespace mlir

#endif // MLIR_TARGET_CPP_DIALECT_ALL_H
