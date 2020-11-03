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

#ifndef MLIR_DIALECT_SPIRV_MODULECOMBINER_H_
#define MLIR_DIALECT_SPIRV_MODULECOMBINER_H_

#include "mlir/Dialect/SPIRV/SPIRVModule.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class OpBuilder;

namespace spirv {
class ModuleOp;

/// To combine a number of MLIR SPIR-V modules, we move all the module-level ops
/// from all the input modules into one big combined module. To that end, the
/// combination process proceeds in 2 phases:
///
///   (1) resolve conflicts between pairs of ops from different modules
///   (2) deduplicate equivalent ops/sub-ops in the merged module. (TODO)
///
/// For the conflict resolution phase, the following rules are employed to
/// resolve such conflicts:
///
///   - If 2 spv.func's have the same symbol name, then rename one of the
///   functions.
///   - If an spv.func and another op have the same symbol name, then rename the
///   other symbol.
///   - If none of the 2 conflicting ops are spv.func, then rename either.
///
/// In all cases, the references to the updated symbol are also updated to
/// reflect the change.
///
/// \param modules the list of modules to combine. Input modules are not
/// modified.
/// \param combinedMdouleBuilder an OpBuilder to be used for
/// building up the combined module.
/// \param symbRenameListener a listener that gets called everytime a symbol in
///                           one of the input modules is renamed. The arguments
///                           passed to the listener are: the input
///                           spirv::ModuleOp that contains the renamed symbol,
///                           a StringRef to the old symbol name, and a
///                           StringRef to the new symbol name. Note that it is
///                           the responsibility of the caller to properly
///                           retain the storage underlying the passed
///                           StringRefs if the listener callback outlives this
///                           function call.
///
/// \return the combined module.
OwningSPIRVModuleRef
combine(llvm::MutableArrayRef<ModuleOp> modules,
        OpBuilder &combinedModuleBuilder,
        llvm::function_ref<void(ModuleOp, StringRef, StringRef)>
            symbRenameListener);
} // namespace spirv
} // namespace mlir

#endif // MLIR_DIALECT_SPIRV_MODULECOMBINER_H_
