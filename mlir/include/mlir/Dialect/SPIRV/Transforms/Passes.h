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

#include "mlir/Pass/Pass.h"

namespace mlir {

class ModuleOp;

namespace spirv {

class ModuleOp;

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL_SPIRVCOMPOSITETYPELAYOUT
#define GEN_PASS_DECL_SPIRVCANONICALIZEGL
#define GEN_PASS_DECL_SPIRVLOWERABIATTRIBUTES
#define GEN_PASS_DECL_SPIRVREWRITEINSERTSPASS
#define GEN_PASS_DECL_SPIRVUNIFYALIASEDRESOURCEPASS
#define GEN_PASS_DECL_SPIRVUPDATEVCE
#include "mlir/Dialect/SPIRV/Transforms/Passes.h.inc"

/// Creates a pass to run canoncalization patterns that involve GL ops.
/// These patterns cannot be run in default canonicalization because GL ops
/// aren't always available. So they should be involed specifically when needed.
std::unique_ptr<OperationPass<>> createCanonicalizeGLPass();

/// Creates a module pass that converts composite types used by objects in the
/// StorageBuffer, PhysicalStorageBuffer, Uniform, and PushConstant storage
/// classes with layout information.
/// Right now this pass only supports Vulkan layout rules.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createDecorateSPIRVCompositeTypeLayoutPass();

/// Creates an operation pass that deduces and attaches the minimal version/
/// capabilities/extensions requirements for spv.module ops.
/// For each spv.module op, this pass requires a `spv.target_env` attribute on
/// it or an enclosing module-like op to drive the deduction. The reason is
/// that an op can be enabled by multiple extensions/capabilities. So we need
/// to know which one to pick. `spv.target_env` gives the hard limit as for
/// what the target environment can support; this pass deduces what are
/// actually needed for a specific spv.module op.
std::unique_ptr<OperationPass<spirv::ModuleOp>>
createUpdateVersionCapabilityExtensionPass();

/// Creates an operation pass that lowers the ABI attributes specified during
/// SPIR-V Lowering. Specifically,
/// 1. Creates the global variables for arguments of entry point function using
///    the specification in the `spv.interface_var_abi` attribute for each
///    argument.
/// 2. Inserts the EntryPointOp and the ExecutionModeOp for entry point
///    functions using the specification in the `spv.entry_point_abi` attribute.
std::unique_ptr<OperationPass<spirv::ModuleOp>> createLowerABIAttributesPass();

/// Creates an operation pass that rewrites sequential chains of
/// spv.CompositeInsert into spv.CompositeConstruct.
std::unique_ptr<OperationPass<spirv::ModuleOp>> createRewriteInsertsPass();

/// Creates an operation pass that unifies access of multiple aliased resources
/// into access of one single resource.
std::unique_ptr<OperationPass<spirv::ModuleOp>>
createUnifyAliasedResourcePass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/SPIRV/Transforms/Passes.h.inc"

} // namespace spirv
} // namespace mlir

#endif // MLIR_DIALECT_SPIRV_TRANSFORMS_PASSES_H_
