//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARMSVE_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_ARMSVE_TRANSFORMS_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"

namespace mlir::arm_sve {

#define GEN_PASS_DECL
#include "mlir/Dialect/ArmSVE/Transforms/Passes.h.inc"

/// Pass to legalize the types of mask stores.
std::unique_ptr<Pass> createLegalizeVectorStoragePass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/ArmSVE/Transforms/Passes.h.inc"

} // namespace mlir::arm_sve

#endif // MLIR_DIALECT_ARMSVE_TRANSFORMS_PASSES_H
