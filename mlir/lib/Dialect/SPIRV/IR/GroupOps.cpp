//===- GroupOps.cpp - MLIR SPIR-V Group Ops  ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the group operations in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"

#include "SPIRVOpUtils.h"
#include "SPIRVParsingUtils.h"

using namespace mlir::spirv::AttrNames;

namespace mlir::spirv {

template <typename OpTy>
static LogicalResult verifyGroupNonUniformArithmeticOp(Operation *groupOp) {
  GroupOperation operation =
      groupOp
          ->getAttrOfType<GroupOperationAttr>(
              OpTy::getGroupOperationAttrName(groupOp->getName()))
          .getValue();
  if (operation == GroupOperation::ClusteredReduce &&
      groupOp->getNumOperands() == 1)
    return groupOp->emitOpError("cluster size operand must be provided for "
                                "'ClusteredReduce' group operation");
  if (groupOp->getNumOperands() > 1) {
    Operation *sizeOp = groupOp->getOperand(1).getDefiningOp();
    int32_t clusterSize = 0;

    // TODO: support specialization constant here.
    if (failed(extractValueFromConstOp(sizeOp, clusterSize)))
      return groupOp->emitOpError(
          "cluster size operand must come from a constant op");

    if (!llvm::isPowerOf2_32(clusterSize))
      return groupOp->emitOpError(
          "cluster size operand must be a power of two");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.GroupBroadcast
//===----------------------------------------------------------------------===//

LogicalResult GroupBroadcastOp::verify() {
  if (auto localIdTy = dyn_cast<VectorType>(getLocalid().getType()))
    if (localIdTy.getNumElements() != 2 && localIdTy.getNumElements() != 3)
      return emitOpError("localid is a vector and can be with only "
                         " 2 or 3 components, actual number is ")
             << localIdTy.getNumElements();

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformBroadcast
//===----------------------------------------------------------------------===//

LogicalResult GroupNonUniformBroadcastOp::verify() {
  // SPIR-V spec: "Before version 1.5, Id must come from a
  // constant instruction.
  auto targetEnv = spirv::getDefaultTargetEnv(getContext());
  if (auto spirvModule = (*this)->getParentOfType<spirv::ModuleOp>())
    targetEnv = spirv::lookupTargetEnvOrDefault(spirvModule);

  if (targetEnv.getVersion() < spirv::Version::V_1_5) {
    auto *idOp = getId().getDefiningOp();
    if (!idOp || !isa<spirv::ConstantOp,           // for normal constant
                      spirv::ReferenceOfOp>(idOp)) // for spec constant
      return emitOpError("id must be the result of a constant op");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformShuffle*
//===----------------------------------------------------------------------===//

template <typename OpTy>
static LogicalResult verifyGroupNonUniformShuffleOp(OpTy op) {
  if (op.getOperands().back().getType().isSignedInteger())
    return op.emitOpError("second operand must be a singless/unsigned integer");

  return success();
}

LogicalResult GroupNonUniformShuffleOp::verify() {
  return verifyGroupNonUniformShuffleOp(*this);
}
LogicalResult GroupNonUniformShuffleDownOp::verify() {
  return verifyGroupNonUniformShuffleOp(*this);
}
LogicalResult GroupNonUniformShuffleUpOp::verify() {
  return verifyGroupNonUniformShuffleOp(*this);
}
LogicalResult GroupNonUniformShuffleXorOp::verify() {
  return verifyGroupNonUniformShuffleOp(*this);
}

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformFAddOp
//===----------------------------------------------------------------------===//

LogicalResult GroupNonUniformFAddOp::verify() {
  return verifyGroupNonUniformArithmeticOp<GroupNonUniformFAddOp>(*this);
}

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformFMaxOp
//===----------------------------------------------------------------------===//

LogicalResult GroupNonUniformFMaxOp::verify() {
  return verifyGroupNonUniformArithmeticOp<GroupNonUniformFMaxOp>(*this);
}

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformFMinOp
//===----------------------------------------------------------------------===//

LogicalResult GroupNonUniformFMinOp::verify() {
  return verifyGroupNonUniformArithmeticOp<GroupNonUniformFMinOp>(*this);
}

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformFMulOp
//===----------------------------------------------------------------------===//

LogicalResult GroupNonUniformFMulOp::verify() {
  return verifyGroupNonUniformArithmeticOp<GroupNonUniformFMulOp>(*this);
}

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformIAddOp
//===----------------------------------------------------------------------===//

LogicalResult GroupNonUniformIAddOp::verify() {
  return verifyGroupNonUniformArithmeticOp<GroupNonUniformIAddOp>(*this);
}

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformIMulOp
//===----------------------------------------------------------------------===//

LogicalResult GroupNonUniformIMulOp::verify() {
  return verifyGroupNonUniformArithmeticOp<GroupNonUniformIMulOp>(*this);
}

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformSMaxOp
//===----------------------------------------------------------------------===//

LogicalResult GroupNonUniformSMaxOp::verify() {
  return verifyGroupNonUniformArithmeticOp<GroupNonUniformSMaxOp>(*this);
}

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformSMinOp
//===----------------------------------------------------------------------===//

LogicalResult GroupNonUniformSMinOp::verify() {
  return verifyGroupNonUniformArithmeticOp<GroupNonUniformSMinOp>(*this);
}

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformUMaxOp
//===----------------------------------------------------------------------===//

LogicalResult GroupNonUniformUMaxOp::verify() {
  return verifyGroupNonUniformArithmeticOp<GroupNonUniformUMaxOp>(*this);
}

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformUMinOp
//===----------------------------------------------------------------------===//

LogicalResult GroupNonUniformUMinOp::verify() {
  return verifyGroupNonUniformArithmeticOp<GroupNonUniformUMinOp>(*this);
}

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformBitwiseAnd
//===----------------------------------------------------------------------===//

LogicalResult GroupNonUniformBitwiseAndOp::verify() {
  return verifyGroupNonUniformArithmeticOp<GroupNonUniformBitwiseAndOp>(*this);
}

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformBitwiseOr
//===----------------------------------------------------------------------===//

LogicalResult GroupNonUniformBitwiseOrOp::verify() {
  return verifyGroupNonUniformArithmeticOp<GroupNonUniformBitwiseOrOp>(*this);
}

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformBitwiseXor
//===----------------------------------------------------------------------===//

LogicalResult GroupNonUniformBitwiseXorOp::verify() {
  return verifyGroupNonUniformArithmeticOp<GroupNonUniformBitwiseXorOp>(*this);
}

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformLogicalAnd
//===----------------------------------------------------------------------===//

LogicalResult GroupNonUniformLogicalAndOp::verify() {
  return verifyGroupNonUniformArithmeticOp<GroupNonUniformLogicalAndOp>(*this);
}

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformLogicalOr
//===----------------------------------------------------------------------===//

LogicalResult GroupNonUniformLogicalOrOp::verify() {
  return verifyGroupNonUniformArithmeticOp<GroupNonUniformLogicalOrOp>(*this);
}

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformLogicalXor
//===----------------------------------------------------------------------===//

LogicalResult GroupNonUniformLogicalXorOp::verify() {
  return verifyGroupNonUniformArithmeticOp<GroupNonUniformLogicalXorOp>(*this);
}

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformRotateKHR
//===----------------------------------------------------------------------===//

LogicalResult GroupNonUniformRotateKHROp::verify() {
  if (Value clusterSizeVal = getClusterSize()) {
    mlir::Operation *defOp = clusterSizeVal.getDefiningOp();
    int32_t clusterSize = 0;

    if (failed(extractValueFromConstOp(defOp, clusterSize)))
      return emitOpError("cluster size operand must come from a constant op");

    if (!llvm::isPowerOf2_32(clusterSize))
      return emitOpError("cluster size operand must be a power of two");
  }

  return success();
}

} // namespace mlir::spirv
