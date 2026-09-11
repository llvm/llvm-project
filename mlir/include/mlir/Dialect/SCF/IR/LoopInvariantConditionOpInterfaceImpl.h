//===- LoopInvariantConditionOpInterfaceImpl.h - Impl. for scf.if ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the registration function for scf::IfOp's external
// model implementation of LoopInvariantConditionOpInterface.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SCF_IR_LOOPINVARIANTCONDITIONOPINTERFACEIMPL_H
#define MLIR_DIALECT_SCF_IR_LOOPINVARIANTCONDITIONOPINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace scf {
/// Registers scf::IfOp's implementation of LoopInvariantConditionOpInterface
/// with the given registry.
/// Added for -loop-invariant-code-motion; must be called before that pass runs.
void registerLoopInvariantConditionOpInterfaceExternalModels(
    DialectRegistry &registry);
} // namespace scf
} // namespace mlir

#endif // MLIR_DIALECT_SCF_IR_LOOPINVARIANTCONDITIONOPINTERFACEIMPL_H
