//===- ModuleCombiner.h - MLIR SPIR-V Module Combiner -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the entry point to the SPIR-V module combiner library.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_LINKING_MODULECOMBINER_H_
#define MLIR_DIALECT_SPIRV_LINKING_MODULECOMBINER_H_

#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class OpBuilder;

namespace spirv {
class ModuleOp;

/// The listener function to receive symbol renaming events.
///
/// `originalModule` is the input spirv::ModuleOp that contains the renamed
/// symbol. `oldSymbol` and `newSymbol` are the original and renamed symbol.
/// Note that it's the responsibility of the caller to properly retain the
/// storage underlying the passed StringRefs if the listener callback outlives
/// this function call.
using SymbolRenameListener = function_ref<void(
    spirv::ModuleOp originalModule, StringRef oldSymbol, StringRef newSymbol)>;

/// Combines a list of SPIR-V `inputModules` into one. Returns the combined
/// module on success; returns a null module otherwise.
//
/// \param inputModules the list of modules to combine. They won't be modified.
/// \param combinedMdouleBuilder an OpBuilder for building the combined module.
/// \param symbRenameListener a listener that gets called everytime a symbol in
///                           one of the input modules is renamed.
///
/// To combine multiple SPIR-V modules, we move all the module-level ops
/// from all the input modules into one big combined module. To that end, the
/// combination process proceeds in 2 phases:
///
/// 1. resolve conflicts between pairs of ops from different modules,
/// 2. deduplicate equivalent ops/sub-ops in the merged module.
///
/// For the conflict resolution phase, the following rules are employed to
/// resolve such conflicts:
///
/// - If 2 spirv.func's have the same symbol name, then rename one of the
///   functions.
/// - If an spirv.func and another op have the same symbol name, then rename the
///   other symbol.
/// - If none of the 2 conflicting ops are spirv.func, then rename either.
///
/// For deduplication, the following 3 cases are taken into consideration:
///
/// - If 2 spirv.GlobalVariable's have either the same descriptor set + binding
///   or the same build_in attribute value, then replace one of them using the
///   other.
/// - If 2 spirv.SpecConstant's have the same spec_id attribute value, then
///   replace one of them using the other.
/// - Deduplicating functions are not supported right now.
///
/// In all cases, the references to the updated symbol (whether renamed or
/// deduplicated) are also updated to reflect the change.
OwningOpRef<spirv::ModuleOp> combine(ArrayRef<spirv::ModuleOp> inputModules,
                                     OpBuilder &combinedModuleBuilder,
                                     SymbolRenameListener symRenameListener);
} // namespace spirv
} // namespace mlir

#endif // MLIR_DIALECT_SPIRV_LINKING_MODULECOMBINER_H_
