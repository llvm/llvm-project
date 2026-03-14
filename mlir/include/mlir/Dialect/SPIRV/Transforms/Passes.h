//===- Passes.h - SPIR-V pass entry points ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_TRANSFORMS_PASSES_H_
#define MLIR_DIALECT_SPIRV_TRANSFORMS_PASSES_H_

#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

class ModuleOp;

namespace spirv {

class ModuleOp;

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "mlir/Dialect/SPIRV/Transforms/Passes.h.inc"

/// Creates an operation pass that unifies access of multiple aliased resources
/// into access of one single resource.
using GetTargetEnvFn = std::function<spirv::TargetEnvAttr(spirv::ModuleOp)>;
std::unique_ptr<OperationPass<spirv::ModuleOp>>
createUnifyAliasedResourcePass(GetTargetEnvFn getTargetEnv = nullptr);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/SPIRV/Transforms/Passes.h.inc"

} // namespace spirv
} // namespace mlir

#endif // MLIR_DIALECT_SPIRV_TRANSFORMS_PASSES_H_
