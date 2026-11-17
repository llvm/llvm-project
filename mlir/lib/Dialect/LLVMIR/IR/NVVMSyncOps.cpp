//===- NVVMSyncOps.cpp - NVVM dialect synchronization ops -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ops of op partition `Sync` of the NVVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/NVVMSyncOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::NVVM;

//===----------------------------------------------------------------------===//
// Verifier methods
//===----------------------------------------------------------------------===//

LogicalResult NVVM::FenceProxyOp::verify() {
  if (getKind() == NVVM::ProxyKind::async_shared && !getSpace().has_value()) {
    return emitOpError() << "async_shared fence requires space attribute";
  }
  if (getKind() != NVVM::ProxyKind::async_shared && getSpace().has_value()) {
    return emitOpError() << "only async_shared fence can have space attribute";
  }
  return success();
}

LogicalResult NVVM::FenceProxyAcquireOp::verify() {
  if (getFromProxy() != NVVM::ProxyKind::GENERIC)
    return emitOpError("uni-directional proxies only support generic for "
                       "from_proxy attribute");

  if (getToProxy() != NVVM::ProxyKind::TENSORMAP)
    return emitOpError("uni-directional proxies only support tensormap "
                       "for to_proxy attribute");
  return success();
}

LogicalResult NVVM::FenceProxyReleaseOp::verify() {
  if (getFromProxy() != NVVM::ProxyKind::GENERIC)
    return emitOpError("uni-directional proxies only support generic for "
                       "from_proxy attribute");

  if (getToProxy() != NVVM::ProxyKind::TENSORMAP)
    return emitOpError("uni-directional proxies only support tensormap "
                       "for to_proxy attribute");
  return success();
}

LogicalResult NVVM::FenceProxySyncRestrictOp::verify() {
  if (getFromProxy() != NVVM::ProxyKind::GENERIC)
    return emitOpError("only generic is support for from_proxy attribute");

  if (getToProxy() != NVVM::ProxyKind::async)
    return emitOpError("only async is supported for to_proxy attribute");
  return success();
}

LogicalResult NVVM::MatchSyncOp::verify() {
  if (getKind() == NVVM::MatchSyncKind::all) {
    auto type = llvm::dyn_cast<LLVM::LLVMStructType>(getType());
    if (!type || type.getBody().size() != 2 ||
        !type.getBody()[0].isInteger(32) || !type.getBody()[1].isInteger(1)) {
      return emitOpError("match.sync 'all' returns a two element struct with "
                         "first element as i32 and second element as i1");
    }
  } else {
    if (!getType().isInteger(32)) {
      return emitOpError("match.sync 'any' returns an i32");
    }
  }
  return success();
}

LogicalResult MatchSyncOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    MatchSyncOp::Adaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {
  if (adaptor.getKind() == NVVM::MatchSyncKind::all)
    inferredReturnTypes.push_back(LLVM::LLVMStructType::getLiteral(
        context,
        {IntegerType::get(context, 32), IntegerType::get(context, 1)}));
  else
    inferredReturnTypes.push_back(IntegerType::get(context, 32));
  return success();
}

LogicalResult NVVM::VoteSyncOp::verify() {
  if (getKind() == NVVM::VoteSyncKind::ballot) {
    if (!getType().isInteger(32)) {
      return emitOpError("vote.sync 'ballot' returns an i32");
    }
  } else {
    if (!getType().isInteger(1)) {
      return emitOpError("vote.sync 'any', 'all' and 'uni' returns an i1");
    }
  }
  return success();
}

LogicalResult VoteSyncOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    VoteSyncOp::Adaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {
  unsigned width = adaptor.getKind() == NVVM::VoteSyncKind::ballot ? 32 : 1;
  inferredReturnTypes.push_back(IntegerType::get(context, width));
  return success();
}

LogicalResult NVVM::ReduxOp::verify() {
  mlir::Type reduxType = getType();

  if (!reduxType.isF32()) {
    if (getAbs())
      return emitOpError("abs attribute is supported only for f32 type");
    if (getNan())
      return emitOpError("nan attribute is supported only for f32 type");
  }

  NVVM::ReductionKind kind = getKind();
  switch (kind) {
  case NVVM::ReductionKind::ADD:
  case NVVM::ReductionKind::AND:
  case NVVM::ReductionKind::OR:
  case NVVM::ReductionKind::XOR:
  case NVVM::ReductionKind::MAX:
  case NVVM::ReductionKind::MIN:
  case NVVM::ReductionKind::UMAX:
  case NVVM::ReductionKind::UMIN:
    if (!reduxType.isInteger(32))
      return emitOpError("'")
             << kind << "' reduction kind unsupported with " << reduxType
             << " type. Only supported type is 'i32'.";
    break;
  case NVVM::ReductionKind::FMIN:
  case NVVM::ReductionKind::FMAX:
    if (!reduxType.isF32())
      return emitOpError("'")
             << kind << "' reduction kind unsupported with " << reduxType
             << " type. Only supported type is 'f32'.";
    break;
  }

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

/// Maps the (aligned, hasCount) pair to the `@llvm.nvvm.barrier.cta.sync.*`
/// intrinsic ID.
static llvm::Intrinsic::ID getBarrierSyncIntrinsic(bool aligned,
                                                   bool hasCount) {
  if (hasCount) {
    return aligned ? llvm::Intrinsic::nvvm_barrier_cta_sync_aligned_count
                   : llvm::Intrinsic::nvvm_barrier_cta_sync_count;
  }
  return aligned ? llvm::Intrinsic::nvvm_barrier_cta_sync_aligned_all
                 : llvm::Intrinsic::nvvm_barrier_cta_sync_all;
}

/// Maps the (aligned, kind) pair to the `@llvm.nvvm.barrier.cta.red.*`
/// intrinsic ID.
static llvm::Intrinsic::ID
getBarrierReductionIntrinsic(bool aligned, NVVM::BarrierReduction kind) {
  switch (kind) {
  case NVVM::BarrierReduction::AND:
    return aligned ? llvm::Intrinsic::nvvm_barrier_cta_red_and_aligned_all
                   : llvm::Intrinsic::nvvm_barrier_cta_red_and_all;
  case NVVM::BarrierReduction::OR:
    return aligned ? llvm::Intrinsic::nvvm_barrier_cta_red_or_aligned_all
                   : llvm::Intrinsic::nvvm_barrier_cta_red_or_all;
  case NVVM::BarrierReduction::POPC:
    return aligned ? llvm::Intrinsic::nvvm_barrier_cta_red_popc_aligned_all
                   : llvm::Intrinsic::nvvm_barrier_cta_red_popc_all;
  }
  llvm_unreachable("unknown BarrierReduction kind");
}

mlir::NVVM::IDArgPair NVVM::BarrierOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::BarrierOp>(op);
  llvm::Value *barrierId = thisOp.getBarrierId()
                               ? mt.lookupValue(thisOp.getBarrierId())
                               : builder.getInt32(0);
  bool hasCount = static_cast<bool>(thisOp.getNumberOfThreads());
  llvm::Intrinsic::ID id =
      getBarrierSyncIntrinsic(thisOp.getAligned(), hasCount);
  llvm::SmallVector<llvm::Value *> args = {barrierId};
  if (hasCount)
    args.push_back(mt.lookupValue(thisOp.getNumberOfThreads()));
  return {id, std::move(args)};
}

mlir::NVVM::IDArgPair NVVM::BarrierArriveOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::BarrierArriveOp>(op);
  llvm::Value *barrierId = thisOp.getBarrierId()
                               ? mt.lookupValue(thisOp.getBarrierId())
                               : builder.getInt32(0);
  llvm::Value *numThreads = mt.lookupValue(thisOp.getNumberOfThreads());
  llvm::Intrinsic::ID id =
      thisOp.getAligned()
          ? llvm::Intrinsic::nvvm_barrier_cta_arrive_aligned_count
          : llvm::Intrinsic::nvvm_barrier_cta_arrive_count;
  return {id, {barrierId, numThreads}};
}

mlir::NVVM::IDArgPair NVVM::BarrierReductionOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::BarrierReductionOp>(op);
  llvm::Intrinsic::ID id = getBarrierReductionIntrinsic(
      thisOp.getAligned(), thisOp.getReductionOp());
  llvm::Value *barrierId = thisOp.getBarrierId()
                               ? mt.lookupValue(thisOp.getBarrierId())
                               : builder.getInt32(0);
  llvm::SmallVector<llvm::Value *> args = {
      barrierId,
      builder.CreateICmpNE(mt.lookupValue(thisOp.getReductionPredicate()),
                           builder.getInt32(0))};
  return {id, std::move(args)};
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

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMSyncOps.cpp.inc"
