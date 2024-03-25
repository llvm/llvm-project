//===- ControlFlowToSCF.h - ControlFlow to SCF -------------*- C++ ------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Define conversions from the ControlFlow dialect to the SCF dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_CONTROLFLOWTOSCF_CONTROLFLOWTOSCF_H
#define MLIR_CONVERSION_CONTROLFLOWTOSCF_CONTROLFLOWTOSCF_H

#include <memory>

#include "mlir/Transforms/CFGToSCF.h"

namespace mlir {
class Pass;

/// Implementation of `CFGToSCFInterface` used to lift Control Flow Dialect
/// operations to SCF Dialect operations.
class ControlFlowToSCFTransformation : public CFGToSCFInterface {
public:
  /// Creates an `scf.if` op if `controlFlowCondOp` is a `cf.cond_br` op or
  /// an `scf.index_switch` if `controlFlowCondOp` is a `cf.switch`.
  /// Returns failure otherwise.
  FailureOr<Operation *> createStructuredBranchRegionOp(
      OpBuilder &builder, Operation *controlFlowCondOp, TypeRange resultTypes,
      MutableArrayRef<Region> regions) override;

  /// Creates an `scf.yield` op returning the given results.
  LogicalResult createStructuredBranchRegionTerminatorOp(
      Location loc, OpBuilder &builder, Operation *branchRegionOp,
      Operation *replacedControlFlowOp, ValueRange results) override;

  /// Creates an `scf.while` op. The loop body is made the before-region of the
  /// while op and terminated with an `scf.condition` op. The after-region does
  /// nothing but forward the iteration variables.
  FailureOr<Operation *>
  createStructuredDoWhileLoopOp(OpBuilder &builder, Operation *replacedOp,
                                ValueRange loopVariablesInit, Value condition,
                                ValueRange loopVariablesNextIter,
                                Region &&loopBody) override;

  /// Creates an `arith.constant` with an i32 attribute of the given value.
  Value getCFGSwitchValue(Location loc, OpBuilder &builder,
                          unsigned value) override;

  /// Creates a `cf.switch` op with the given cases and flag.
  void createCFGSwitchOp(Location loc, OpBuilder &builder, Value flag,
                         ArrayRef<unsigned> caseValues,
                         BlockRange caseDestinations,
                         ArrayRef<ValueRange> caseArguments, Block *defaultDest,
                         ValueRange defaultArgs) override;

  /// Creates a `ub.poison` op of the given type.
  Value getUndefValue(Location loc, OpBuilder &builder, Type type) override;

  /// Creates a `func.return` op with poison for each of the return values of
  /// the function. It is guaranteed to be directly within the function body.
  /// TODO: This can be made independent of the `func` dialect once the UB
  ///       dialect has a `ub.unreachable` op.
  FailureOr<Operation *> createUnreachableTerminator(Location loc,
                                                     OpBuilder &builder,
                                                     Region &region) override;
};

#define GEN_PASS_DECL_LIFTCONTROLFLOWTOSCFPASS
#include "mlir/Conversion/Passes.h.inc"

} // namespace mlir

#endif // MLIR_CONVERSION_CONTROLFLOWTOSCF_CONTROLFLOWTOSCF_H
