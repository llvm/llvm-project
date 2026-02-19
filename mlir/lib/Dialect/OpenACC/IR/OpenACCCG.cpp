//===- OpenACCCG.cpp - OpenACC codegen ops, attributes, and types ---------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation for OpenACC codegen operations, attributes, and types.
// These correspond to the definitions in OpenACCCG*.td tablegen files
// and are kept in a separate file because they do not represent direct mappings
// of OpenACC language constructs; they are intermediate representations used
// when decomposing and lowering primary `acc` dialect operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/Region.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace acc;

namespace {

/// Generic helper for single-region OpenACC ops that execute their body once
/// and then return to the parent operation with their results (if any).
static void
getSingleRegionOpSuccessorRegions(Operation *op, Region &region,
                                  RegionBranchPoint point,
                                  SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.push_back(RegionSuccessor(&region));
    return;
  }
  regions.push_back(RegionSuccessor::parent());
}

static ValueRange getSingleRegionSuccessorInputs(Operation *op,
                                                 RegionSuccessor successor) {
  return successor.isParent() ? ValueRange(op->getResults()) : ValueRange();
}

/// Remove empty acc.kernel_environment operations. If the operation has wait
/// operands, create a acc.wait operation to preserve synchronization.
struct RemoveEmptyKernelEnvironment
    : public OpRewritePattern<acc::KernelEnvironmentOp> {
  using OpRewritePattern<acc::KernelEnvironmentOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(acc::KernelEnvironmentOp op,
                                PatternRewriter &rewriter) const override {
    assert(op->getNumRegions() == 1 && "expected op to have one region");

    Block &block = op.getRegion().front();
    if (!block.empty())
      return failure();

    // Conservatively disable canonicalization of empty acc.kernel_environment
    // operations if the wait operands in the kernel_environment cannot be fully
    // represented by acc.wait operation.

    // Disable canonicalization if device type is not the default
    if (auto deviceTypeAttr = op.getWaitOperandsDeviceTypeAttr()) {
      for (auto attr : deviceTypeAttr) {
        if (auto dtAttr = mlir::dyn_cast<acc::DeviceTypeAttr>(attr)) {
          if (dtAttr.getValue() != mlir::acc::DeviceType::None)
            return failure();
        }
      }
    }

    // Disable canonicalization if any wait segment has a devnum
    if (auto hasDevnumAttr = op.getHasWaitDevnumAttr()) {
      for (auto attr : hasDevnumAttr) {
        if (auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attr)) {
          if (boolAttr.getValue())
            return failure();
        }
      }
    }

    // Disable canonicalization if there are multiple wait segments
    if (auto segmentsAttr = op.getWaitOperandsSegmentsAttr()) {
      if (segmentsAttr.size() > 1)
        return failure();
    }

    // Remove empty kernel environment.
    // Preserve synchronization by creating acc.wait operation if needed.
    if (!op.getWaitOperands().empty() || op.getWaitOnlyAttr())
      rewriter.replaceOpWithNewOp<acc::WaitOp>(op, op.getWaitOperands(),
                                               /*asyncOperand=*/Value(),
                                               /*waitDevnum=*/Value(),
                                               /*async=*/nullptr,
                                               /*ifCond=*/Value());
    else
      rewriter.eraseOp(op);

    return success();
  }
};

template <typename EffectTy>
static void addOperandEffect(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    const MutableOperandRange &operand) {
  for (unsigned i = 0, e = operand.size(); i < e; ++i)
    effects.emplace_back(EffectTy::get(), &operand[i]);
}

template <typename EffectTy>
static void addResultEffect(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    Value result) {
  effects.emplace_back(EffectTy::get(), mlir::cast<mlir::OpResult>(result));
}

} // namespace

//===----------------------------------------------------------------------===//
// KernelEnvironmentOp
//===----------------------------------------------------------------------===//

void KernelEnvironmentOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  getSingleRegionOpSuccessorRegions(getOperation(), getRegion(), point,
                                    regions);
}

ValueRange KernelEnvironmentOp::getSuccessorInputs(RegionSuccessor successor) {
  return getSingleRegionSuccessorInputs(getOperation(), successor);
}

void KernelEnvironmentOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<RemoveEmptyKernelEnvironment>(context);
}

//===----------------------------------------------------------------------===//
// FirstprivateMapInitialOp
//===----------------------------------------------------------------------===//

LogicalResult FirstprivateMapInitialOp::verify() {
  if (getDataClause() != acc::DataClause::acc_firstprivate)
    return emitError("data clause associated with firstprivate operation must "
                     "match its intent");
  if (!getVar())
    return emitError("must have var operand");
  if (!mlir::isa<mlir::acc::PointerLikeType>(getVar().getType()) &&
      !mlir::isa<mlir::acc::MappableType>(getVar().getType()))
    return emitError("var must be mappable or pointer-like");
  if (mlir::isa<mlir::acc::PointerLikeType>(getVar().getType()) &&
      getVarType() == getVar().getType())
    return emitError("varType must capture the element type of var");
  if (getModifiers() != acc::DataClauseModifier::none)
    return emitError("no data clause modifiers are allowed");
  return success();
}

void FirstprivateMapInitialOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       acc::CurrentDeviceIdResource::get());
  addOperandEffect<MemoryEffects::Read>(effects, getVarMutable());
  addResultEffect<MemoryEffects::Write>(effects, getAccVar());
}

//===----------------------------------------------------------------------===//
// ReductionCombineOp
//===----------------------------------------------------------------------===//

void ReductionCombineOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMemrefMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getDestMemrefMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestMemrefMutable(),
                       SideEffects::DefaultResource::get());
}
