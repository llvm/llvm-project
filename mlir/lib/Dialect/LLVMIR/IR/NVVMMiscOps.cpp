//===- NVVMMiscOps.cpp - NVVM dialect miscellaneous ops -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ops of op partition `Misc` of the NVVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/NVVMMiscOps.h"

#include "IR/NVVMOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"

#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::NVVM;

//===----------------------------------------------------------------------===//
// Verifier methods
//===----------------------------------------------------------------------===//

LogicalResult PMEventOp::verify() {
  auto eventId = getEventId();
  auto maskedEventId = getMaskedEventId();
  if (!maskedEventId && !eventId) {
    return emitOpError() << "either `id` or `mask` must be set";
  }

  if (maskedEventId && eventId) {
    return emitOpError() << "`id` and `mask` cannot be set at the same time";
  }

  if (eventId) {
    if (eventId < 0 || eventId > 15) {
      return emitOpError() << "`id` must be between 0 and 15";
    }
  }

  return llvm::success();
}

LogicalResult NVVM::SetMaxRegisterOp::verify() {
  if (getRegCount() % 8)
    return emitOpError("new register size must be multiple of 8");
  if (getRegCount() < 24 || getRegCount() > 256)
    return emitOpError("new register size must be in between 24 to 256");
  return success();
}

LogicalResult NVVM::ClusterLaunchControlQueryCancelOp::verify() {
  switch (getQueryType()) {
  case NVVM::ClusterLaunchControlQueryType::IS_CANCELED:
    if (!getType().isInteger(1))
      return emitOpError("is_canceled query type returns an i1");
    break;
  case NVVM::ClusterLaunchControlQueryType::GET_FIRST_CTA_ID_X:
  case NVVM::ClusterLaunchControlQueryType::GET_FIRST_CTA_ID_Y:
  case NVVM::ClusterLaunchControlQueryType::GET_FIRST_CTA_ID_Z:
    if (!getType().isInteger(32)) {
      return emitOpError("get_first_cta_id_x, get_first_cta_id_y, "
                         "get_first_cta_id_z query types return an i32");
    }
    break;
  }
  return success();
}

LogicalResult ClusterLaunchControlQueryCancelOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    ClusterLaunchControlQueryCancelOp::Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  unsigned width =
      adaptor.getQueryType() == NVVM::ClusterLaunchControlQueryType::IS_CANCELED
          ? 1
          : 32;
  inferredReturnTypes.push_back(IntegerType::get(context, width));
  return success();
}

//===----------------------------------------------------------------------===//
// getIntrinsicID/getIntrinsicIDAndArgs methods
//===----------------------------------------------------------------------===//

mlir::NVVM::IDArgPair
PMEventOp::getIntrinsicIDAndArgs(Operation &op, LLVM::ModuleTranslation &mt,
                                 llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::PMEventOp>(op);
  llvm::Type *i16Ty = llvm::Type::getInt16Ty(mt.getLLVMContext());

  // With event-id, mask is generated as (1 << event-id)
  llvm::Value *maskVal;
  if (auto eventAttr = thisOp.getEventIdAttr()) {
    uint16_t mask = static_cast<uint16_t>(1u << eventAttr.getInt());
    maskVal = llvm::ConstantInt::get(i16Ty, mask);
  } else {
    maskVal =
        llvm::ConstantInt::get(i16Ty, thisOp.getMaskedEventIdAttr().getValue());
  }

  return {llvm::Intrinsic::nvvm_pm_event_mask, {maskVal}};
}

bool NVVM::InlinePtxOp::getAsmValues(
    RewriterBase &rewriter,
    llvm::SmallVectorImpl<std::pair<mlir::Value, mlir::NVVM::PTXRegisterMod>>
        &asmValues) {
  for (auto arg : getReadWriteArgs())
    asmValues.push_back({arg, mlir::NVVM::PTXRegisterMod::ReadWrite});
  for (auto arg : getResults())
    asmValues.push_back({arg, mlir::NVVM::PTXRegisterMod::Write});
  for (auto arg : getReadOnlyArgs())
    asmValues.push_back({arg, mlir::NVVM::PTXRegisterMod::Read});
  if (getPredicate())
    asmValues.push_back({getPredicate(), mlir::NVVM::PTXRegisterMod::Read});
  return false; // No manual mapping needed
}

NVVM::IDArgPair ClusterLaunchControlTryCancelOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto curOp = cast<NVVM::ClusterLaunchControlTryCancelOp>(op);
  llvm::SmallVector<llvm::Value *> args;
  args.push_back(mt.lookupValue(curOp.getSmemAddress()));
  args.push_back(mt.lookupValue(curOp.getMbarrier()));

  llvm::Intrinsic::ID intrinsicID =
      curOp.getMulticast()
          ? llvm::Intrinsic::
                nvvm_clusterlaunchcontrol_try_cancel_async_multicast_shared
          : llvm::Intrinsic::nvvm_clusterlaunchcontrol_try_cancel_async_shared;

  return {intrinsicID, args};
}

NVVM::IDArgPair ClusterLaunchControlQueryCancelOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto curOp = cast<NVVM::ClusterLaunchControlQueryCancelOp>(op);
  llvm::SmallVector<llvm::Value *> args;
  args.push_back(mt.lookupValue(curOp.getTryCancelResponse()));

  llvm::Intrinsic::ID intrinsicID;

  switch (curOp.getQueryType()) {
  case NVVM::ClusterLaunchControlQueryType::IS_CANCELED:
    intrinsicID =
        llvm::Intrinsic::nvvm_clusterlaunchcontrol_query_cancel_is_canceled;
    break;
  case NVVM::ClusterLaunchControlQueryType::GET_FIRST_CTA_ID_X:
    intrinsicID = llvm::Intrinsic::
        nvvm_clusterlaunchcontrol_query_cancel_get_first_ctaid_x;
    break;
  case NVVM::ClusterLaunchControlQueryType::GET_FIRST_CTA_ID_Y:
    intrinsicID = llvm::Intrinsic::
        nvvm_clusterlaunchcontrol_query_cancel_get_first_ctaid_y;
    break;
  case NVVM::ClusterLaunchControlQueryType::GET_FIRST_CTA_ID_Z:
    intrinsicID = llvm::Intrinsic::
        nvvm_clusterlaunchcontrol_query_cancel_get_first_ctaid_z;
    break;
  }
  return {intrinsicID, args};
}

//===----------------------------------------------------------------------===//
// sparse.compress and sparse.decompress Ops
//===----------------------------------------------------------------------===//

// Returns the number of 32-bit registers in a metadata register bundle, which
// is either an i32 or a vector of i32.
static unsigned getNumSPRegisters(Type type) {
  if (auto vecTy = dyn_cast<VectorType>(type))
    return vecTy.getNumElements();
  return 1;
}

// The repeat factor of spcompress and spdecompress is a power of two up to 64.
static bool isValidSPRepFactor(unsigned repFactor) {
  return llvm::isPowerOf2_32(repFactor) && repFactor <= 64;
}

LogicalResult SparseCompressOp::verify() {
  auto dataTy = cast<VectorType>(getData().getType());
  auto compressedTy = cast<VectorType>(getCompressedData().getType());
  if (dataTy.getElementType() != compressedTy.getElementType())
    return emitOpError(
        "expects data and compressed data to have the same element type");

  // data occupies 2 * repFactor 32-bit registers.
  unsigned elemSize = dataTy.getElementTypeBitWidth();
  unsigned dataBits = dataTy.getNumElements() * elemSize;
  unsigned repFactor = dataBits / 64;
  if (dataBits % 64 != 0 || !isValidSPRepFactor(repFactor))
    return emitOpError("expects data to occupy 2 * repFactor 32-bit "
                       "registers, where repFactor is a power of two up to 64");

  // spcompress only supports the 2:4 sparsity factor.
  if (2 * compressedTy.getNumElements() != dataTy.getNumElements())
    return emitOpError(
        "expects compressed data to have half as many elements as data");

  unsigned numMetadataRegs =
      llvm::divideCeil(repFactor * getIdxSize(), elemSize);
  if (getNumSPRegisters(getMetadata().getType()) != numMetadataRegs)
    return emitOpError("expects metadata to occupy ")
           << numMetadataRegs << " 32-bit register(s)";

  return success();
}

LogicalResult SparseDecompressOp::verify() {
  auto dataTy = cast<VectorType>(getData().getType());
  auto compressedTy = cast<VectorType>(getCompressedData().getType());
  if (dataTy.getElementType() != compressedTy.getElementType())
    return emitOpError(
        "expects data and compressed data to have the same element type");

  // data holds repFactor groups of numTgt elements, and compressed data holds
  // numSrc elements for each of those groups.
  unsigned idxSize = getIdxSize();
  unsigned numTgt = getNumTgt();
  unsigned numData = dataTy.getNumElements();
  unsigned numCompressed = compressedTy.getNumElements();
  unsigned repFactor = numData / numTgt;
  if (numData % numTgt != 0 || !isValidSPRepFactor(repFactor))
    return emitOpError("expects data to have repFactor * num_tgt elements, "
                       "where repFactor is a power of two up to 64");

  unsigned numSrc = numCompressed / repFactor;
  if (numCompressed % repFactor != 0 ||
      !llvm::is_contained({1u, 2u, 4u}, numSrc) || numSrc >= numTgt)
    return emitOpError("unsupported sparsity factor: compressed data has ")
           << numCompressed << " elements for " << repFactor << " group(s) of "
           << numTgt << " elements";

  unsigned elemSize = dataTy.getElementTypeBitWidth();
  unsigned dataBits = numData * elemSize;
  if (dataBits < 32 || dataBits > 4096)
    return emitOpError("expects data to be between 32 and 4096 bits");

  if (numSrc * elemSize > 32)
    return emitOpError("expects at most 32 bits of compressed data per group");

  if (idxSize == 2 && numTgt > 4)
    return emitOpError("expects num_tgt to be at most 4 if idx_size is 2");

  unsigned numMetadataRegs = llvm::divideCeil(numSrc * idxSize * repFactor, 32);
  if (getNumSPRegisters(getMetadata().getType()) != numMetadataRegs)
    return emitOpError("expects metadata to occupy ")
           << numMetadataRegs << " 32-bit register(s)";

  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMMiscOps.cpp.inc"
