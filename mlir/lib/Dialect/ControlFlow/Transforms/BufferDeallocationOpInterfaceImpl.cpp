//===- BufferDeallocationOpInterfaceImpl.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ControlFlow/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/BufferDeallocationOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::bufferization;

static bool isMemref(Value v) { return isa<BaseMemRefType>(v.getType()); }

namespace {
/// While CondBranchOp also implement the BranchOpInterface, we add a
/// special-case implementation here because the BranchOpInterface does not
/// offer all of the functionallity we need to insert dealloc oeprations in an
/// efficient way. More precisely, there is no way to extract the branch
/// condition without casting to CondBranchOp specifically. It is still
/// possible to implement deallocation for cases where we don't know to which
/// successor the terminator branches before the actual branch happens by
/// inserting auxiliary blocks and putting the dealloc op there, however, this
/// can lead to less efficient code.
/// This function inserts two dealloc operations (one for each successor) and
/// adjusts the dealloc conditions according to the branch condition, then the
/// ownerships of the retained MemRefs are updated by combining the result
/// values of the two dealloc operations.
///
/// Example:
/// ```
/// ^bb1:
///   <more ops...>
///   cf.cond_br cond, ^bb2(<forward-to-bb2>), ^bb3(<forward-to-bb2>)
/// ```
/// becomes
/// ```
/// // let (m, c) = getMemrefsAndConditionsToDeallocate(bb1)
/// // let r0 = getMemrefsToRetain(bb1, bb2, <forward-to-bb2>)
/// // let r1 = getMemrefsToRetain(bb1, bb3, <forward-to-bb3>)
/// ^bb1:
///   <more ops...>
///   let thenCond = map(c, (c) -> arith.andi cond, c)
///   let elseCond = map(c, (c) -> arith.andi (arith.xori cond, true), c)
///   o0 = bufferization.dealloc m if thenCond retain r0
///   o1 = bufferization.dealloc m if elseCond retain r1
///   // replace ownership(r0) with o0 element-wise
///   // replace ownership(r1) with o1 element-wise
///   // let ownership0 := (r) -> o in o0 corresponding to r
///   // let ownership1 := (r) -> o in o1 corresponding to r
///   // let cmn := intersection(r0, r1)
///   foreach (a, b) in zip(map(cmn, ownership0), map(cmn, ownership1)):
///     forall r in r0: replace ownership0(r) with arith.select cond, a, b)
///     forall r in r1: replace ownership1(r) with arith.select cond, a, b)
///   cf.cond_br cond, ^bb2(<forward-to-bb2>, o0), ^bb3(<forward-to-bb3>, o1)
/// ```
struct CondBranchOpInterface
    : public BufferDeallocationOpInterface::ExternalModel<CondBranchOpInterface,
                                                          cf::CondBranchOp> {
  FailureOr<Operation *> process(Operation *op, DeallocationState &state,
                                 const DeallocationOptions &options) const {
    OpBuilder builder(op);
    auto condBr = cast<cf::CondBranchOp>(op);

    // The list of memrefs to deallocate in this block is independent of which
    // branch is taken.
    SmallVector<Value> memrefs, conditions;
    if (failed(state.getMemrefsAndConditionsToDeallocate(
            builder, condBr.getLoc(), condBr->getBlock(), memrefs, conditions)))
      return failure();

    // Helper lambda to factor out common logic for inserting the dealloc
    // operations for each successor.
    auto insertDeallocForBranch =
        [&](Block *target, MutableOperandRange destOperands,
            const std::function<Value(Value)> &conditionModifier,
            DenseMap<Value, Value> &mapping) -> DeallocOp {
      SmallVector<Value> toRetain;
      state.getMemrefsToRetain(condBr->getBlock(), target,
                               destOperands.getAsOperandRange(), toRetain);
      SmallVector<Value> adaptedConditions(
          llvm::map_range(conditions, conditionModifier));
      auto deallocOp = bufferization::DeallocOp::create(
          builder, condBr.getLoc(), memrefs, adaptedConditions, toRetain);
      state.resetOwnerships(deallocOp.getRetained(), condBr->getBlock());
      for (auto [retained, ownership] : llvm::zip(
               deallocOp.getRetained(), deallocOp.getUpdatedConditions())) {
        state.updateOwnership(retained, ownership, condBr->getBlock());
        mapping[retained] = ownership;
      }
      SmallVector<Value> replacements, ownerships;
      for (OpOperand &operand : destOperands) {
        replacements.push_back(operand.get());
        if (isMemref(operand.get())) {
          assert(mapping.contains(operand.get()) &&
                 "Should be contained at this point");
          ownerships.push_back(mapping[operand.get()]);
        }
      }
      replacements.append(ownerships);
      destOperands.assign(replacements);
      return deallocOp;
    };

    // Call the helper lambda and make sure the dealloc conditions are properly
    // modified to reflect the branch condition as well.
    DenseMap<Value, Value> thenMapping, elseMapping;
    DeallocOp thenTakenDeallocOp = insertDeallocForBranch(
        condBr.getTrueDest(), condBr.getTrueDestOperandsMutable(),
        [&](Value cond) {
          return arith::AndIOp::create(builder, condBr.getLoc(), cond,
                                       condBr.getCondition());
        },
        thenMapping);
    DeallocOp elseTakenDeallocOp = insertDeallocForBranch(
        condBr.getFalseDest(), condBr.getFalseDestOperandsMutable(),
        [&](Value cond) {
          Value trueVal = arith::ConstantOp::create(builder, condBr.getLoc(),
                                                    builder.getBoolAttr(true));
          Value negation = arith::XOrIOp::create(
              builder, condBr.getLoc(), trueVal, condBr.getCondition());
          return arith::AndIOp::create(builder, condBr.getLoc(), cond,
                                       negation);
        },
        elseMapping);

    // We specifically need to update the ownerships of values that are retained
    // in both dealloc operations again to get a combined 'Unique' ownership
    // instead of an 'Unknown' ownership.
    SmallPtrSet<Value, 16> thenValues(llvm::from_range,
                                      thenTakenDeallocOp.getRetained());
    SetVector<Value> commonValues;
    for (Value val : elseTakenDeallocOp.getRetained()) {
      if (thenValues.contains(val))
        commonValues.insert(val);
    }

    for (Value retained : commonValues) {
      state.resetOwnerships(retained, condBr->getBlock());
      Value combinedOwnership = arith::SelectOp::create(
          builder, condBr.getLoc(), condBr.getCondition(),
          thenMapping[retained], elseMapping[retained]);
      state.updateOwnership(retained, combinedOwnership, condBr->getBlock());
    }

    return condBr.getOperation();
  }
};

} // namespace

void mlir::cf::registerBufferDeallocationOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, ControlFlowDialect *dialect) {
    CondBranchOp::attachInterface<CondBranchOpInterface>(*ctx);
  });
}
