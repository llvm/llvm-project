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

#include "IR/NVVMOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"

#include <optional>
#include <string>
#include <utility>

using namespace mlir;
using namespace mlir::NVVM;

//===----------------------------------------------------------------------===//
// Helper/Utility methods
//===----------------------------------------------------------------------===//

static bool isPtrInGenericSpace(mlir::Value ptr) {
  return isPtrInAddrSpace(ptr, NVVMMemorySpace::Generic);
}

static bool isPtrInSharedClusterSpace(mlir::Value ptr) {
  return isPtrInAddrSpace(ptr, NVVMMemorySpace::SharedCluster);
}

static llvm::Value *castPtrToAddrSpace(llvm::IRBuilderBase &builder,
                                       llvm::Value *ptr,
                                       NVVMMemorySpace targetAS) {
  unsigned AS = static_cast<unsigned>(targetAS);
  return builder.CreateAddrSpaceCast(
      ptr, llvm::PointerType::get(builder.getContext(), AS));
}

//===----------------------------------------------------------------------===//
// Verifier methods
//===----------------------------------------------------------------------===//

static LogicalResult verifyMBarrierArriveLikeOp(Operation *op, Value addr,
                                                NVVM::MemScopeKind scope,
                                                Value retVal = nullptr,
                                                Value multicastMask = nullptr) {
  if (scope != NVVM::MemScopeKind::CTA && scope != NVVM::MemScopeKind::CLUSTER)
    return op->emitError("mbarrier scope must be either CTA or Cluster");

  bool isSharedCluster = isPtrInSharedClusterSpace(addr);
  bool hasRetValue = static_cast<bool>(retVal);
  if (isSharedCluster && hasRetValue)
    return op->emitError(
        "mbarrier in shared_cluster space cannot return any value");

  if (multicastMask && !isSharedCluster)
    return op->emitError("multicast requires shared::cluster address space");

  return success();
}

LogicalResult MBarrierArriveOp::verify() {
  return verifyMBarrierArriveLikeOp(getOperation(), getAddr(), getScope(),
                                    getRes(), getMulticastMask());
}

LogicalResult MBarrierArriveDropOp::verify() {
  return verifyMBarrierArriveLikeOp(getOperation(), getAddr(), getScope(),
                                    getRes(), getMulticastMask());
}

LogicalResult MBarrierArriveExpectTxOp::verify() {
  // The inline-ptx version of this Op does not support all features.
  // With predicate, this Op lowers to inline-ptx. So, verify and
  // error-out if there are unsupported features.
  if (getPredicate()) {
    if (getScope() != NVVM::MemScopeKind::CTA)
      return emitError("mbarrier scope must be CTA when using predicate");

    if (isPtrInSharedClusterSpace(getAddr()))
      return emitError("mbarrier in shared_cluster space is not supported when "
                       "using predicate");

    if (getRes())
      return emitError("return-value is not supported when using predicate");

    if (getRelaxed() == true)
      return emitError("mbarrier with relaxed semantics is not supported when "
                       "using predicate");
  }
  return verifyMBarrierArriveLikeOp(getOperation(), getAddr(), getScope(),
                                    getRes(), getMulticastMask());
}

LogicalResult MBarrierArriveDropExpectTxOp::verify() {
  return verifyMBarrierArriveLikeOp(getOperation(), getAddr(), getScope(),
                                    getRes(), getMulticastMask());
}

//===----------------------------------------------------------------------===//
// inferReturnTypes for mbarrier arrive-like ops
//===----------------------------------------------------------------------===//

/// Only shared_cluster (ptr<7>) produces zero results; all other address
/// spaces (including generic) return i64.
static LogicalResult
inferMBarrierArriveResultTypes(MLIRContext *context, Value addr,
                               SmallVectorImpl<Type> &inferredReturnTypes) {
  if (!isPtrInSharedClusterSpace(addr))
    inferredReturnTypes.push_back(IntegerType::get(context, 64));
  return success();
}

LogicalResult
MBarrierArriveOp::inferReturnTypes(MLIRContext *context,
                                   std::optional<Location> location,
                                   MBarrierArriveOp::Adaptor adaptor,
                                   SmallVectorImpl<Type> &inferredReturnTypes) {
  return inferMBarrierArriveResultTypes(context, adaptor.getAddr(),
                                        inferredReturnTypes);
}

LogicalResult MBarrierArriveDropOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    MBarrierArriveDropOp::Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  return inferMBarrierArriveResultTypes(context, adaptor.getAddr(),
                                        inferredReturnTypes);
}

LogicalResult MBarrierArriveExpectTxOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    MBarrierArriveExpectTxOp::Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // Predicate forces no return value (inline PTX path).
  // Note: predicate + shared_cluster is rejected by the verifier separately.
  if (adaptor.getPredicate())
    return success();
  return inferMBarrierArriveResultTypes(context, adaptor.getAddr(),
                                        inferredReturnTypes);
}

LogicalResult MBarrierArriveDropExpectTxOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    MBarrierArriveDropExpectTxOp::Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  return inferMBarrierArriveResultTypes(context, adaptor.getAddr(),
                                        inferredReturnTypes);
}

/// For ops with optional results, allow the user to omit the result even when
/// inference would produce one. This preserves backward compatibility: the
/// result can be silently discarded (e.g., for fire-and-forget arrive ops).
static bool isCompatibleReturnTypesOptionalResult(TypeRange inferred,
                                                  TypeRange actual) {
  if (actual.empty())
    return true;
  return inferred == actual;
}

bool MBarrierArriveOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  return isCompatibleReturnTypesOptionalResult(l, r);
}

bool MBarrierArriveDropOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  return isCompatibleReturnTypesOptionalResult(l, r);
}

bool MBarrierArriveExpectTxOp::isCompatibleReturnTypes(TypeRange l,
                                                       TypeRange r) {
  return isCompatibleReturnTypesOptionalResult(l, r);
}

bool MBarrierArriveDropExpectTxOp::isCompatibleReturnTypes(TypeRange l,
                                                           TypeRange r) {
  return isCompatibleReturnTypesOptionalResult(l, r);
}

LogicalResult MBarrierExpectTxOp::verify() {
  return verifyMBarrierArriveLikeOp(getOperation(), getAddr(), getScope(),
                                    /*retVal=*/Value{}, getMulticastMask());
}

LogicalResult MBarrierCompleteTxOp::verify() {
  return verifyMBarrierArriveLikeOp(getOperation(), getAddr(), getScope(),
                                    /*retVal=*/Value{}, getMulticastMask());
}

LogicalResult MBarrierTestWaitOp::verify() {
  return verifyMBarrierArriveLikeOp(getOperation(), getAddr(), getScope());
}

LogicalResult MBarrierTryWaitOp::verify() {
  return verifyMBarrierArriveLikeOp(getOperation(), getAddr(), getScope());
}

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

//===----------------------------------------------------------------------===//
// getPtx methods
//===----------------------------------------------------------------------===//

std::string NVVM::MBarrierInitOp::getPtx() {
  std::string space = isPtrInSharedCTASpace(getAddr()) ? ".shared" : "";
  // Layout v0 is the default, so it is emitted as a plain mbarrier.init.
  std::string layout =
      getLayout() == 1 ? std::string(".layout::v1") : std::string();

  return llvm::formatv("mbarrier.init{0}{1}.b64 [%0], %1;", layout, space)
      .str();
}

std::string NVVM::MBarrierArriveExpectTxOp::getPtx() {
  bool isShared = isPtrInSharedCTASpace(getAddr());
  return isShared
             ? std::string("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;")
             : std::string("mbarrier.arrive.expect_tx.b64 _, [%0], %1;");
}

std::string NVVM::MBarrierTryWaitParityOp::getPtx() {
  bool isShared = isPtrInSharedCTASpace(getAddr());
  llvm::StringRef space = isShared ? ".shared" : "";

  return llvm::formatv("{\n\t"
                       ".reg .pred       P1; \n\t"
                       "LAB_WAIT: \n\t"
                       "mbarrier.try_wait.parity{0}.b64 P1, [%0], %1, %2; \n\t"
                       "@P1 bra.uni DONE; \n\t"
                       "bra.uni     LAB_WAIT; \n\t"
                       "DONE: \n\t"
                       "}",
                       space);
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

bool MBarrierInitOp::getAsmValues(
    RewriterBase &rewriter,
    llvm::SmallVectorImpl<std::pair<mlir::Value, mlir::NVVM::PTXRegisterMod>>
        &asmValues) {
  // Add all the operands but not the attrs to the asmValues list.
  // The layout attr is already baked into the PTX string by getPtx(), so
  // passing it along here too would shift the operand numbering.
  for (auto val : getOperands())
    asmValues.push_back({val, mlir::NVVM::PTXRegisterMod::Read});

  return false;
}

mlir::NVVM::IDArgPair MBarrierInitOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::MBarrierInitOp>(op);

  // The intrinsic is overloaded on the mbarrier pointer, so the address space
  // selects the generic or shared::cta form on its own.
  return {llvm::Intrinsic::nvvm_mbarrier_init,
          {mt.lookupValue(thisOp.getAddr()), mt.lookupValue(thisOp.getCount()),
           builder.getInt32(thisOp.getLayout())}};
}

mlir::NVVM::IDArgPair MBarrierInvalOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::MBarrierInvalOp>(op);
  bool isShared = isPtrInSharedCTASpace(thisOp.getAddr());
  llvm::Intrinsic::ID id = isShared
                               ? llvm::Intrinsic::nvvm_mbarrier_inval_shared
                               : llvm::Intrinsic::nvvm_mbarrier_inval;

  return {id, {mt.lookupValue(thisOp.getAddr())}};
}

mlir::NVVM::IDArgPair MBarrierCheckLayoutOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::MBarrierCheckLayoutOp>(op);

  return {
      llvm::Intrinsic::nvvm_mbarrier_check_layout,
      {mt.lookupValue(thisOp.getAddr()), builder.getInt32(thisOp.getLayout())}};
}

static void appendMBarrierMulticastArgs(Value multicastMask,
                                        LLVM::ModuleTranslation &mt,
                                        llvm::IRBuilderBase &builder,
                                        SmallVectorImpl<llvm::Value *> &args) {
  bool hasMulticastMask = static_cast<bool>(multicastMask);
  args.push_back(hasMulticastMask ? mt.lookupValue(multicastMask)
                                  : builder.getInt32(0));
  args.push_back(builder.getInt1(hasMulticastMask));
}

mlir::NVVM::IDArgPair MBarrierExpectTxOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::MBarrierExpectTxOp>(op);

  bool isClusterSpace = isPtrInSharedClusterSpace(thisOp.getAddr());
  bool isClusterScope = thisOp.getScope() == NVVM::MemScopeKind::CLUSTER;
  // bit-0: Space
  // bit-1: Scope
  size_t index = ((isClusterScope ? 1 : 0) << 1) | (isClusterSpace ? 1 : 0);

  static constexpr llvm::Intrinsic::ID IDs[] = {
      llvm::Intrinsic::nvvm_mbarrier_expect_tx_scope_cta_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_expect_tx_scope_cta_space_cluster,
      llvm::Intrinsic::nvvm_mbarrier_expect_tx_scope_cluster_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_expect_tx_scope_cluster_space_cluster};

  // Fill the Intrinsic Args
  llvm::SmallVector<llvm::Value *> args;
  args.push_back(mt.lookupValue(thisOp.getAddr()));
  args.push_back(mt.lookupValue(thisOp.getTxcount()));

  if (isClusterSpace)
    appendMBarrierMulticastArgs(thisOp.getMulticastMask(), mt, builder, args);

  return {IDs[index], std::move(args)};
}

mlir::NVVM::IDArgPair MBarrierCompleteTxOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::MBarrierCompleteTxOp>(op);

  bool isClusterSpace = isPtrInSharedClusterSpace(thisOp.getAddr());
  bool isClusterScope = thisOp.getScope() == NVVM::MemScopeKind::CLUSTER;
  // bit-0: Space
  // bit-1: Scope
  size_t index = ((isClusterScope ? 1 : 0) << 1) | (isClusterSpace ? 1 : 0);

  static constexpr llvm::Intrinsic::ID IDs[] = {
      llvm::Intrinsic::nvvm_mbarrier_complete_tx_scope_cta_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_complete_tx_scope_cta_space_cluster,
      llvm::Intrinsic::nvvm_mbarrier_complete_tx_scope_cluster_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_complete_tx_scope_cluster_space_cluster};

  // Fill the Intrinsic Args
  llvm::SmallVector<llvm::Value *> args;
  args.push_back(mt.lookupValue(thisOp.getAddr()));
  args.push_back(mt.lookupValue(thisOp.getTxcount()));

  if (isClusterSpace)
    appendMBarrierMulticastArgs(thisOp.getMulticastMask(), mt, builder, args);

  return {IDs[index], std::move(args)};
}

mlir::NVVM::IDArgPair MBarrierArriveOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::MBarrierArriveOp>(op);

  bool isClusterSpace = isPtrInSharedClusterSpace(thisOp.getAddr());
  bool isClusterScope = thisOp.getScope() == NVVM::MemScopeKind::CLUSTER;
  // bit-0: Space
  // bit-1: Scope
  size_t index = ((isClusterScope ? 1 : 0) << 1) | (isClusterSpace ? 1 : 0);

  static constexpr llvm::Intrinsic::ID IDs[] = {
      llvm::Intrinsic::nvvm_mbarrier_arrive_scope_cta_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_arrive_scope_cta_space_cluster,
      llvm::Intrinsic::nvvm_mbarrier_arrive_scope_cluster_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_arrive_scope_cluster_space_cluster};
  static constexpr llvm::Intrinsic::ID relaxedIDs[] = {
      llvm::Intrinsic::nvvm_mbarrier_arrive_relaxed_scope_cta_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_arrive_relaxed_scope_cta_space_cluster,
      llvm::Intrinsic::nvvm_mbarrier_arrive_relaxed_scope_cluster_space_cta,
      llvm::Intrinsic::
          nvvm_mbarrier_arrive_relaxed_scope_cluster_space_cluster};
  auto id = thisOp.getRelaxed() ? relaxedIDs[index] : IDs[index];

  // Tidy-up the Intrinsic Args
  bool needCast = isPtrInGenericSpace(thisOp.getAddr());
  llvm::Value *mbar = mt.lookupValue(thisOp.getAddr());
  if (needCast)
    mbar = castPtrToAddrSpace(builder, mbar, NVVMMemorySpace::Shared);

  // We have the most basic mbarrier.arrive supported on sm_80.
  // It supports: Space=cta, scope=cta, No relaxed, No explicit count.
  // So, only for this combination use the legacy intrinsic.
  bool hasCount = static_cast<bool>(thisOp.getCount());
  if (!hasCount &&
      (id == llvm::Intrinsic::nvvm_mbarrier_arrive_scope_cta_space_cta))
    return {llvm::Intrinsic::nvvm_mbarrier_arrive_shared, {mbar}};

  // When count is not explicitly specified, the default is 1.
  llvm::LLVMContext &ctx = mt.getLLVMContext();
  llvm::Value *count =
      hasCount ? mt.lookupValue(thisOp.getCount())
               : llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 1);

  llvm::SmallVector<llvm::Value *> args = {mbar, count};
  if (isClusterSpace)
    appendMBarrierMulticastArgs(thisOp.getMulticastMask(), mt, builder, args);
  return {id, std::move(args)};
}

mlir::NVVM::IDArgPair MBarrierArriveDropOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::MBarrierArriveDropOp>(op);

  bool isClusterSpace = isPtrInSharedClusterSpace(thisOp.getAddr());
  bool isClusterScope = thisOp.getScope() == NVVM::MemScopeKind::CLUSTER;
  // bit-0: Space
  // bit-1: Scope
  size_t index = ((isClusterScope ? 1 : 0) << 1) | (isClusterSpace ? 1 : 0);

  static constexpr llvm::Intrinsic::ID IDs[] = {
      llvm::Intrinsic::nvvm_mbarrier_arrive_drop_scope_cta_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_arrive_drop_scope_cta_space_cluster,
      llvm::Intrinsic::nvvm_mbarrier_arrive_drop_scope_cluster_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_arrive_drop_scope_cluster_space_cluster};
  static constexpr llvm::Intrinsic::ID relaxedIDs[] = {
      llvm::Intrinsic::nvvm_mbarrier_arrive_drop_relaxed_scope_cta_space_cta,
      llvm::Intrinsic::
          nvvm_mbarrier_arrive_drop_relaxed_scope_cta_space_cluster,
      llvm::Intrinsic::
          nvvm_mbarrier_arrive_drop_relaxed_scope_cluster_space_cta,
      llvm::Intrinsic::
          nvvm_mbarrier_arrive_drop_relaxed_scope_cluster_space_cluster};
  auto id = thisOp.getRelaxed() ? relaxedIDs[index] : IDs[index];

  // Tidy-up the Intrinsic Args
  bool needCast = isPtrInGenericSpace(thisOp.getAddr());
  llvm::Value *mbar = mt.lookupValue(thisOp.getAddr());
  if (needCast)
    mbar = castPtrToAddrSpace(builder, mbar, NVVMMemorySpace::Shared);

  // We have the most basic mbarrier.arrive_drop supported on sm_80.
  // It supports: Space=cta, scope=cta, No relaxed, No explicit count.
  // So, only for this combination use the legacy intrinsic.
  bool hasCount = static_cast<bool>(thisOp.getCount());
  if (!hasCount &&
      (id == llvm::Intrinsic::nvvm_mbarrier_arrive_drop_scope_cta_space_cta))
    return {llvm::Intrinsic::nvvm_mbarrier_arrive_drop_shared, {mbar}};

  // When count is not explicitly specified, the default is 1.
  llvm::LLVMContext &ctx = mt.getLLVMContext();
  llvm::Value *count =
      hasCount ? mt.lookupValue(thisOp.getCount())
               : llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 1);

  llvm::SmallVector<llvm::Value *> args = {mbar, count};
  if (isClusterSpace)
    appendMBarrierMulticastArgs(thisOp.getMulticastMask(), mt, builder, args);
  return {id, std::move(args)};
}

bool MBarrierArriveExpectTxOp::getAsmValues(
    RewriterBase &rewriter,
    llvm::SmallVectorImpl<std::pair<mlir::Value, mlir::NVVM::PTXRegisterMod>>
        &asmValues) {
  // Add all the operands but not the attrs to the asmValues list.
  // The attrs here are used to generate the right variants for
  // intrinsics-lowering. So, we ignore them while generating inline-PTX.
  for (auto val : getOperands())
    asmValues.push_back({val, mlir::NVVM::PTXRegisterMod::Read});

  return false;
}

mlir::NVVM::IDArgPair MBarrierArriveExpectTxOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::MBarrierArriveExpectTxOp>(op);

  bool isClusterSpace = isPtrInSharedClusterSpace(thisOp.getAddr());
  bool isClusterScope = thisOp.getScope() == NVVM::MemScopeKind::CLUSTER;
  // bit-0: Space
  // bit-1: Scope
  size_t index = ((isClusterScope ? 1 : 0) << 1) | (isClusterSpace ? 1 : 0);

  // clang-format off
  static constexpr llvm::Intrinsic::ID IDs[] = {
      llvm::Intrinsic::nvvm_mbarrier_arrive_expect_tx_scope_cta_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_arrive_expect_tx_scope_cta_space_cluster,
      llvm::Intrinsic::nvvm_mbarrier_arrive_expect_tx_scope_cluster_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_arrive_expect_tx_scope_cluster_space_cluster};
  static constexpr llvm::Intrinsic::ID relaxedIDs[] = {
      llvm::Intrinsic::nvvm_mbarrier_arrive_expect_tx_relaxed_scope_cta_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_arrive_expect_tx_relaxed_scope_cta_space_cluster,
      llvm::Intrinsic::nvvm_mbarrier_arrive_expect_tx_relaxed_scope_cluster_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_arrive_expect_tx_relaxed_scope_cluster_space_cluster};
  // clang-format on
  auto id = thisOp.getRelaxed() ? relaxedIDs[index] : IDs[index];

  // Tidy-up the Intrinsic Args
  llvm::Value *txcount = mt.lookupValue(thisOp.getTxcount());
  llvm::Value *mbar = mt.lookupValue(thisOp.getAddr());
  bool needCast = isPtrInGenericSpace(thisOp.getAddr());
  if (needCast)
    mbar = castPtrToAddrSpace(builder, mbar, NVVMMemorySpace::Shared);

  llvm::SmallVector<llvm::Value *> args = {mbar, txcount};
  if (isClusterSpace)
    appendMBarrierMulticastArgs(thisOp.getMulticastMask(), mt, builder, args);
  return {id, std::move(args)};
}

mlir::NVVM::IDArgPair MBarrierArriveDropExpectTxOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::MBarrierArriveDropExpectTxOp>(op);

  bool isClusterSpace = isPtrInSharedClusterSpace(thisOp.getAddr());
  bool isClusterScope = thisOp.getScope() == NVVM::MemScopeKind::CLUSTER;
  // bit-0: Space
  // bit-1: Scope
  size_t index = ((isClusterScope ? 1 : 0) << 1) | (isClusterSpace ? 1 : 0);

  // clang-format off
  static constexpr llvm::Intrinsic::ID IDs[] = {
      llvm::Intrinsic::nvvm_mbarrier_arrive_drop_expect_tx_scope_cta_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_arrive_drop_expect_tx_scope_cta_space_cluster,
      llvm::Intrinsic::nvvm_mbarrier_arrive_drop_expect_tx_scope_cluster_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_arrive_drop_expect_tx_scope_cluster_space_cluster};
  static constexpr llvm::Intrinsic::ID relaxedIDs[] = {
      llvm::Intrinsic::nvvm_mbarrier_arrive_drop_expect_tx_relaxed_scope_cta_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_arrive_drop_expect_tx_relaxed_scope_cta_space_cluster,
      llvm::Intrinsic::nvvm_mbarrier_arrive_drop_expect_tx_relaxed_scope_cluster_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_arrive_drop_expect_tx_relaxed_scope_cluster_space_cluster};
  // clang-format on
  auto id = thisOp.getRelaxed() ? relaxedIDs[index] : IDs[index];

  // Tidy-up the Intrinsic Args
  llvm::Value *txcount = mt.lookupValue(thisOp.getTxcount());
  llvm::Value *mbar = mt.lookupValue(thisOp.getAddr());
  bool needCast = isPtrInGenericSpace(thisOp.getAddr());
  if (needCast)
    mbar = castPtrToAddrSpace(builder, mbar, NVVMMemorySpace::Shared);

  llvm::SmallVector<llvm::Value *> args = {mbar, txcount};
  if (isClusterSpace)
    appendMBarrierMulticastArgs(thisOp.getMulticastMask(), mt, builder, args);
  return {id, std::move(args)};
}

mlir::NVVM::IDArgPair MBarrierArriveNocompleteOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::MBarrierArriveNocompleteOp>(op);
  bool isShared = isPtrInSharedCTASpace(thisOp.getAddr());
  llvm::Intrinsic::ID id =
      isShared ? llvm::Intrinsic::nvvm_mbarrier_arrive_noComplete_shared
               : llvm::Intrinsic::nvvm_mbarrier_arrive_noComplete;
  // Fill the Intrinsic Args
  llvm::SmallVector<llvm::Value *> args;
  args.push_back(mt.lookupValue(thisOp.getAddr()));
  args.push_back(mt.lookupValue(thisOp.getCount()));

  return {id, std::move(args)};
}

mlir::NVVM::IDArgPair MBarrierArriveDropNocompleteOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::MBarrierArriveDropNocompleteOp>(op);
  bool isShared = isPtrInSharedCTASpace(thisOp.getAddr());
  llvm::Intrinsic::ID id =
      isShared ? llvm::Intrinsic::nvvm_mbarrier_arrive_drop_noComplete_shared
               : llvm::Intrinsic::nvvm_mbarrier_arrive_drop_noComplete;
  // Fill the Intrinsic Args
  llvm::SmallVector<llvm::Value *> args;
  args.push_back(mt.lookupValue(thisOp.getAddr()));
  args.push_back(mt.lookupValue(thisOp.getCount()));

  return {id, std::move(args)};
}

mlir::NVVM::IDArgPair MBarrierTestWaitOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::MBarrierTestWaitOp>(op);
  bool isPhaseParity = thisOp.getStateOrPhase().getType().isInteger(32);
  bool isClusterScope = thisOp.getScope() == NVVM::MemScopeKind::CLUSTER;
  // bit-0: isPhaseParity
  // bit-1: Scope
  size_t index = ((isClusterScope ? 1 : 0) << 1) | (isPhaseParity ? 1 : 0);

  // clang-format off
  static constexpr llvm::Intrinsic::ID IDs[] = {
      llvm::Intrinsic::nvvm_mbarrier_test_wait_scope_cta_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_test_wait_parity_scope_cta_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_test_wait_scope_cluster_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_test_wait_parity_scope_cluster_space_cta};
  static constexpr llvm::Intrinsic::ID relaxedIDs[] = {
      llvm::Intrinsic::nvvm_mbarrier_test_wait_relaxed_scope_cta_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_test_wait_parity_relaxed_scope_cta_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_test_wait_relaxed_scope_cluster_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_test_wait_parity_relaxed_scope_cluster_space_cta};
  // clang-format on
  auto id = thisOp.getRelaxed() ? relaxedIDs[index] : IDs[index];

  // Tidy-up the Intrinsic Args
  llvm::Value *mbar = mt.lookupValue(thisOp.getAddr());
  llvm::Value *input = mt.lookupValue(thisOp.getStateOrPhase());
  bool needCast = isPtrInGenericSpace(thisOp.getAddr());
  if (needCast)
    mbar = castPtrToAddrSpace(builder, mbar, NVVMMemorySpace::Shared);

  return {id, {mbar, input}};
}

mlir::NVVM::IDArgPair MBarrierTryWaitOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::MBarrierTryWaitOp>(op);
  bool isPhaseParity = thisOp.getStateOrPhase().getType().isInteger(32);
  bool isClusterScope = thisOp.getScope() == NVVM::MemScopeKind::CLUSTER;
  bool hasTicks = static_cast<bool>(thisOp.getTicks());
  // bit-0: isPhaseParity
  // bit-1: Scope
  // bit-2: hasTicks
  size_t index = ((hasTicks ? 1 : 0) << 2) | ((isClusterScope ? 1 : 0) << 1) |
                 (isPhaseParity ? 1 : 0);

  // clang-format off
  static constexpr llvm::Intrinsic::ID IDs[] = {
      llvm::Intrinsic::nvvm_mbarrier_try_wait_scope_cta_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_try_wait_parity_scope_cta_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_try_wait_scope_cluster_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_try_wait_parity_scope_cluster_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_try_wait_tl_scope_cta_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_try_wait_parity_tl_scope_cta_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_try_wait_tl_scope_cluster_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_try_wait_parity_tl_scope_cluster_space_cta};
  static constexpr llvm::Intrinsic::ID relaxedIDs[] = {
      llvm::Intrinsic::nvvm_mbarrier_try_wait_relaxed_scope_cta_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_try_wait_parity_relaxed_scope_cta_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_try_wait_relaxed_scope_cluster_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_try_wait_parity_relaxed_scope_cluster_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_try_wait_tl_relaxed_scope_cta_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_try_wait_parity_tl_relaxed_scope_cta_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_try_wait_tl_relaxed_scope_cluster_space_cta,
      llvm::Intrinsic::nvvm_mbarrier_try_wait_parity_tl_relaxed_scope_cluster_space_cta};
  // clang-format on
  auto id = thisOp.getRelaxed() ? relaxedIDs[index] : IDs[index];

  // Tidy-up the mbarrier pointer
  llvm::Value *mbar = mt.lookupValue(thisOp.getAddr());
  bool needCast = isPtrInGenericSpace(thisOp.getAddr());
  if (needCast)
    mbar = castPtrToAddrSpace(builder, mbar, NVVMMemorySpace::Shared);

  // Fill the Intrinsic Args
  llvm::SmallVector<llvm::Value *> args;
  args.push_back(mbar);
  args.push_back(mt.lookupValue(thisOp.getStateOrPhase()));
  if (hasTicks)
    args.push_back(mt.lookupValue(thisOp.getTicks()));

  return {id, std::move(args)};
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMSyncOps.cpp.inc"
