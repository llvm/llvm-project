//===- NVVMMemoryOps.cpp - NVVM dialect memory ops ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ops of op partition `Memory` of the NVVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/NVVMMemoryOps.h"

#include "IR/NVVMOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/NVPTXAddrSpace.h"

#include <array>
#include <cassert>
#include <optional>
#include <string>
#include <utility>

using namespace mlir;
using namespace mlir::NVVM;

//===----------------------------------------------------------------------===//
// Helper/Utility methods
//===----------------------------------------------------------------------===//

static ParseResult parseCTAGroup(OpAsmParser &parser,
                                 NVVM::CTAGroupKindAttr &groupAttr) {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return failure();
  std::optional<NVVM::CTAGroupKind> group =
      NVVM::symbolizeCTAGroupKind(keyword);
  if (!group)
    return parser.emitError(parser.getNameLoc()) << " expected CTA group";
  groupAttr = NVVM::CTAGroupKindAttr::get(parser.getContext(), *group);
  return success();
}

static void printCTAGroup(OpAsmPrinter &printer, Operation *,
                          NVVM::CTAGroupKindAttr groupAttr) {
  printer << NVVM::stringifyCTAGroupKind(groupAttr.getValue());
}

//===----------------------------------------------------------------------===//
// Verifier methods
//===----------------------------------------------------------------------===//

// This verifier is shared among the following Ops:
// CpAsyncBulkTensorSharedCTAToGlobalOp (TMA Store)
// CpAsyncBulkTensorReduceOp (TMA Store-Reduce)
static LogicalResult cpAsyncBulkTensorCommonVerifier(size_t tensorDims,
                                                     bool isIm2Col,
                                                     size_t numIm2ColOffsets,
                                                     Location loc) {
  if (tensorDims < 1 || tensorDims > 5)
    return emitError(loc, "expects coordinates between 1 to 5 dimension");

  // For Im2Col mode, there are two constraints:
  if (isIm2Col) {
    // 1. Tensor must always be at least 3-d.
    if (tensorDims < 3)
      return emitError(
          loc,
          "to use im2col mode, the tensor has to be at least 3-dimensional");
    // 2. When there are Im2ColOffsets, they must be (Dims - 2) in number.
    if (numIm2ColOffsets && (tensorDims != (numIm2ColOffsets + 2)))
      return emitError(
          loc, "im2col offsets must be 2 less than number of coordinates");
  }
  return success();
}

LogicalResult CpAsyncBulkTensorOverrideAddrCommonVerifier(
    OperandRange coordinates, OperandRange tensorSize, OperandRange lowerStride,
    Value upperStride, bool isTile, Location loc) {
  LogicalResult res = success();
  if (!tensorSize.empty() && coordinates.size() != tensorSize.size()) {
    res =
        emitError(loc, "Expected coordinates size to be equal to tensor size");
  }

  if (!lowerStride.empty() && tensorSize.empty()) {
    res = emitError(
        loc,
        "Expected tensor_size to be present when lower_stride is provided");
  } else if (!lowerStride.empty() &&
             lowerStride.size() != tensorSize.size() - 1) {
    res = emitError(
        loc,
        "Expected lower_stride size to be equal to one less than tensor size");
  }

  if (!lowerStride.empty() != static_cast<bool>(upperStride)) {
    res = emitError(loc,
                    "Expected lower_stride and upper_stride to be either both "
                    "present or both absent");
  }

  bool isDimStride = tensorSize.size() > 0;
  if (!isTile && isDimStride) {
    res = emitError(
        loc, "Only tile mode supports override address with dim and stride");
  }

  return res;
}

LogicalResult CpAsyncBulkTensorSharedCTAToGlobalOp::verify() {
  TMAStoreMode mode = getMode();
  // We lower through inline-ptx when getPredicate() is true.
  // a) Only TILE mode is supported
  // b) Cache-hint is not supported
  if (getPredicate()) {
    if (mode != TMAStoreMode::TILE)
      return emitError("Inline-ptx lowering supported only for Tile mode.");
    if (getL2CacheHint())
      return emitError("Inline-ptx lowering unsupported with L2 cache-hint.");
  }

  size_t dims = getCoordinates().size();
  switch (mode) {
  case TMAStoreMode::TILE:
    return cpAsyncBulkTensorCommonVerifier(dims, false, 0, getLoc());
  case TMAStoreMode::IM2COL:
  case TMAStoreMode::IM2COL_W:
    return cpAsyncBulkTensorCommonVerifier(dims, true, 0, getLoc());
  case TMAStoreMode::TILE_SCATTER4:
    if (dims != 5)
      return emitError("Scatter4 mode expects 5 coordinates");
  }
  return success();
}

LogicalResult CpAsyncBulkTensorSharedCTAToGlobalOverrideAddrOp::verify() {
  TMAStoreMode mode = getMode();
  bool isIm2Col =
      mode == TMAStoreMode::IM2COL || mode == TMAStoreMode::IM2COL_W;
  bool isTile = mode == TMAStoreMode::TILE;

  LogicalResult commonRes = cpAsyncBulkTensorCommonVerifier(
      getCoordinates().size(), isIm2Col, 0, getLoc());

  LogicalResult overrideAddrRes = CpAsyncBulkTensorOverrideAddrCommonVerifier(
      getCoordinates(), getTensorSize(), getLowerStride(), getUpperStride(),
      isTile, getLoc());

  if (mode == TMAStoreMode::TILE_SCATTER4 && getCoordinates().size() != 5)
    overrideAddrRes = emitError("Mode tile scatter4 expects 5 coordinates");

  return failed(commonRes) || failed(overrideAddrRes) ? failure() : success();
}

LogicalResult CpAsyncOp::verify() {
  if (getModifier() != LoadCacheModifierKind::CG &&
      getModifier() != LoadCacheModifierKind::CA)
    return emitError("Only CG and CA cache modifiers are supported.");
  if (getSize() != 4 && getSize() != 8 && getSize() != 16)
    return emitError("expected byte size to be either 4, 8 or 16.");
  if (getModifier() == LoadCacheModifierKind::CG && getSize() != 16)
    return emitError("CG cache modifier is only support for 16 bytes copy.");
  return success();
}

// This verify params can be shared across TMA Load and Prefetch Ops.
static LogicalResult verifyTMALoadParams(size_t tensorDims, size_t numIm2colOff,
                                         TMALoadMode mode, Location loc) {
  if (tensorDims < 1 || tensorDims > 5)
    return emitError(loc, "expects coordinates between 1 to 5 dimension");

  auto checkTMALoadParams = [&](TMALoadMode mode, bool isIm2col,
                                size_t expectedIm2colOff) -> LogicalResult {
    if (isIm2col && (tensorDims < 3))
      return emitError(loc)
             << "to use " << mode
             << " mode, the tensor has to be at least 3-dimensional";

    if (numIm2colOff != expectedIm2colOff)
      return emitError(loc) << " im2col offsets expected " << expectedIm2colOff
                            << " (provided " << numIm2colOff << ")";

    return success();
  };

  switch (mode) {
  case TMALoadMode::TILE:
    return checkTMALoadParams(mode, false, 0);
  case TMALoadMode::IM2COL:
    return checkTMALoadParams(mode, true, tensorDims - 2);
  case TMALoadMode::IM2COL_W:
  case TMALoadMode::IM2COL_W_128:
    return checkTMALoadParams(mode, true, 2);
  case TMALoadMode::TILE_GATHER4:
    return (tensorDims == 5)
               ? checkTMALoadParams(mode, false, 0)
               : emitError(loc, "Gather4 mode expects 5 coordinates");
  }
  return success();
}

LogicalResult CpAsyncBulkTensorPrefetchOp::verify() {
  return verifyTMALoadParams(getCoordinates().size(), getIm2colOffsets().size(),
                             getMode(), getLoc());
}

LogicalResult CpAsyncBulkTensorGlobalToSharedClusterOp::verify() {
  TMALoadMode mode = getMode();
  bool isCTAOnly = getIsCTAOnly();
  if (getPredicate()) { // Inline-asm based lowering
    if (isCTAOnly)
      return emitError("Predicate is supported only for shared::cluster mode.");
    if (mode != TMALoadMode::TILE && mode != TMALoadMode::IM2COL)
      return emitError(
          "Predicate is supported only for Tile and Im2col modes.");
  } else { // Intrinsics-based lowering
    NVVMMemorySpace expectedAS =
        isCTAOnly ? NVVMMemorySpace::Shared : NVVMMemorySpace::SharedCluster;
    unsigned AS = llvm::cast<LLVM::LLVMPointerType>(getDstMem().getType())
                      .getAddressSpace();
    if (AS != expectedAS)
      return emitError()
             << (isCTAOnly
                     ? "Shared::cta destination requires address-space 3."
                     : "Shared::cluster destination requires address-space 7.");
    // Checks specific to shared::cta mode
    if (isCTAOnly) {
      if (getMulticastMask())
        return emitError("Multicast is not supported with shared::cta mode.");
      if (getGroup())
        return emitError("CTAGroup is not supported with shared::cta mode.");
    }
  }

  return verifyTMALoadParams(getCoordinates().size(), getIm2colOffsets().size(),
                             getMode(), getLoc());
}

LogicalResult CpAsyncBulkTensorReduceOp::verify() {
  TMAStoreMode mode = getMode();
  size_t dims = getCoordinates().size();
  switch (mode) {
  case TMAStoreMode::TILE:
    return cpAsyncBulkTensorCommonVerifier(dims, false, 0, getLoc());
  case TMAStoreMode::IM2COL:
  case TMAStoreMode::IM2COL_W:
    return cpAsyncBulkTensorCommonVerifier(dims, true, 0, getLoc());
  case TMAStoreMode::TILE_SCATTER4:
    return emitError("Scatter mode unsupported for CpAsyncBulkTensorReduceOp");
  }
  return success();
}

LogicalResult CpAsyncBulkTensorReduceOverrideAddrOp::verify() {
  bool isIm2Col =
      getMode() == TMAStoreMode::IM2COL || getMode() == TMAStoreMode::IM2COL_W;
  bool isTile = getMode() == TMAStoreMode::TILE;

  LogicalResult commonRes = cpAsyncBulkTensorCommonVerifier(
      getCoordinates().size(), isIm2Col, 0, getLoc());

  LogicalResult overrideAddrRes = CpAsyncBulkTensorOverrideAddrCommonVerifier(
      getCoordinates(), getTensorSize(), getLowerStride(), getUpperStride(),
      isTile, getLoc());

  if (getMode() == TMAStoreMode::TILE_SCATTER4)
    overrideAddrRes = emitError(
        "Scatter mode unsupported for CpAsyncBulkTensorReduceOverrideAddrOp");

  return failed(commonRes) || failed(overrideAddrRes) ? failure() : success();
}

LogicalResult CpAsyncBulkGlobalToSharedClusterOp::verify() {
  bool isSharedCTA = isPtrInSharedCTASpace(getDstMem());
  if (isSharedCTA && getMulticastMask())
    return emitError("Multicast is not supported with shared::cta mode.");

  return success();
}

LogicalResult BulkStoreOp::verify() {
  if (getInitVal() != 0)
    return emitOpError("only 0 is supported for initVal, got ") << getInitVal();
  return success();
}

LogicalResult AsyncStoreGlobalOp::verify() {
  NVVM::MemScopeKind scope = getScope();
  bool isMmio = getMmio();
  bool isMultimem = getMultimem();

  if (scope != MemScopeKind::SYS && scope != MemScopeKind::GPU)
    return emitOpError("scope must be either SYS or GPU");

  if (isMmio && scope != MemScopeKind::SYS)
    return emitOpError("mmio is only supported for SYS scope");

  if (isMmio && isMultimem)
    return emitOpError("multimem is not supported with mmio");

  return success();
}

LogicalResult NVVM::PrefetchOp::verify() {
  using MemSpace = NVVM::NVVMMemorySpace;
  using CacheLevel = NVVM::PrefetchCacheLevel;

  unsigned addressSpace =
      llvm::cast<LLVM::LLVMPointerType>(getAddr().getType()).getAddressSpace();
  std::optional<NVVM::CacheEvictionPriority> evictPriority = getEvictPriority();
  std::optional<NVVM::PrefetchCacheLevel> cacheLevel = getCacheLevel();

  if (getTensormap() && cacheLevel)
    return emitOpError("cannot specify both tensormap and cache level");

  if (getTensormap()) {
    if (addressSpace != MemSpace::Generic &&
        addressSpace != MemSpace::Constant) {
      return emitOpError(
          "prefetch tensormap requires a generic or constant pointer");
    }

    if (evictPriority) {
      return emitOpError(
          "prefetch tensormap does not support eviction priority");
    }

    if (getInParamSpace() && addressSpace != MemSpace::Generic) {
      return emitOpError(
          "in_param_space can only be specified for a generic pointer");
    }

  } else if (cacheLevel) {
    if (addressSpace != MemSpace::Generic && addressSpace != MemSpace::Global &&
        addressSpace != MemSpace::Local) {
      return emitOpError("prefetch to cache level requires a generic, global, "
                         "or local pointer");
    }

    if (getUniform()) {
      if (*cacheLevel != CacheLevel::L1) {
        return emitOpError(
            "unsupported cache level, the only supported uniform "
            "cache level is L1");
      }

      if (addressSpace != MemSpace::Generic) {
        return emitOpError(
            "prefetch to uniform cache requires a generic pointer");
      }
    }

    if (evictPriority) {
      if (*cacheLevel != CacheLevel::L2)
        return emitOpError(
            "cache eviction priority supported only for cache level L2");

      if (addressSpace != MemSpace::Global)
        return emitOpError("cache eviction priority requires a global pointer");

      if (*evictPriority != NVVM::CacheEvictionPriority::EvictNormal &&
          *evictPriority != NVVM::CacheEvictionPriority::EvictLast)
        return emitOpError(
            "unsupported cache eviction priority, only evict_last and "
            "evict_normal are supported");
    }

    if (getPredicate())
      return emitOpError("predicate supported only on prefetch tensormap");

  } else {
    return emitOpError(
        "requires specification of either cache level or tensormap");
  }

  return success();
}

LogicalResult NVVM::TensormapReplaceOp::verify() {
  auto ord = getOrd();
  Value newVal = getNewValue();
  auto newValAttr = getNewValueAttr();
  auto fieldName = stringifyEnum(getField());

  if (ord && !llvm::is_contained({NVVM::TensormapField::BOX_DIM,
                                  NVVM::TensormapField::GLOBAL_DIM,
                                  NVVM::TensormapField::GLOBAL_STRIDE,
                                  NVVM::TensormapField::ELEMENT_STRIDE},
                                 getField()))
    return emitOpError("ordinal is not supported for ")
           << fieldName << " field";

  auto invalidNewVal = [&](llvm::Twine type) -> std::string {
    return llvm::Twine("new_value must be specified and must be an " + type +
                       " for " + llvm::Twine(fieldName) + " field")
        .str();
  };

  auto invalidNewValAttr = [&]() -> std::string {
    return (llvm::Twine(
                "new_value_attr must be specified and must be a valid ") +
            llvm::Twine(fieldName) + " attribute for " + fieldName + " field")
        .str();
  };

  switch (getField()) {
  case NVVM::TensormapField::GLOBAL_ADDRESS:
    if (!(newVal && newVal.getType().isInteger(64)))
      return emitOpError(invalidNewVal("i64"));
    break;
  case NVVM::TensormapField::RANK:
    if (!(newVal && newVal.getType().isInteger(32)))
      return emitOpError(invalidNewVal("i32"));
    break;
  case NVVM::TensormapField::GLOBAL_STRIDE:
    if (!ord)
      return emitOpError("ordinal is required for global_stride field");
    if (!(newVal && newVal.getType().isInteger(64)))
      return emitOpError(invalidNewVal("i64"));
    break;
  case NVVM::TensormapField::BOX_DIM:
  case NVVM::TensormapField::GLOBAL_DIM:
  case NVVM::TensormapField::ELEMENT_STRIDE:
    if (!ord)
      return emitOpError("ordinal is required for ")
             << stringifyEnum(getField()) << " field";
    if (!(newVal && newVal.getType().isInteger(32)))
      return emitOpError(invalidNewVal("i32"));
    break;
  case NVVM::TensormapField::ELEMTYPE:
    if (!(newValAttr && llvm::isa<TensormapElemtypeAttr>(*newValAttr)))
      return emitOpError(invalidNewValAttr());
    break;
  case NVVM::TensormapField::INTERLEAVE_LAYOUT:
    if (!(newValAttr && llvm::isa<TensormapInterleaveLayoutAttr>(*newValAttr)))
      return emitOpError(invalidNewValAttr());
    break;
  case NVVM::TensormapField::SWIZZLE_MODE:
    if (!(newValAttr && llvm::isa<TensormapSwizzleModeAttr>(*newValAttr)))
      return emitOpError(invalidNewValAttr());
    break;
  case NVVM::TensormapField::SWIZZLE_ATOMICITY:
    if (!(newValAttr && llvm::isa<TensormapSwizzleAtomicityAttr>(*newValAttr)))
      return emitOpError(invalidNewValAttr());
    break;
  case NVVM::TensormapField::FILL_MODE:
    if (!(newValAttr && llvm::isa<TensormapFillModeAttr>(*newValAttr)))
      return emitOpError(invalidNewValAttr());
    break;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// getIntrinsicID/getIntrinsicIDAndArgs methods
//===----------------------------------------------------------------------===//

mlir::NVVM::IDArgPair AsyncStoreGlobalOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  using IDArgPair = mlir::NVVM::IDArgPair;
  auto thisOp = cast<NVVM::AsyncStoreGlobalOp>(op);
  mlir::NVVM::MemScopeKind scope = thisOp.getScope();
  bool isMmio = thisOp.getMmio();

  llvm::Value *addr = mt.lookupValue(thisOp.getAddr());
  llvm::Value *value = mt.lookupValue(thisOp.getValue());
  llvm::Value *isMultimem = builder.getInt1(thisOp.getMultimem());

  if (scope == MemScopeKind::SYS) {
    return isMmio ? IDArgPair(llvm::Intrinsic::nvvm_st_async_mmio_sys,
                              {addr, value})
                  : IDArgPair(llvm::Intrinsic::nvvm_st_async_sys,
                              {addr, value, isMultimem});
  } else if (scope == MemScopeKind::GPU) {
    return IDArgPair(llvm::Intrinsic::nvvm_st_async_gpu,
                     {addr, value, isMultimem});
  }
  llvm_unreachable("unsupported scope for AsyncStoreGlobalOp");
}

mlir::NVVM::IDArgPair CpAsyncMBarrierArriveOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::CpAsyncMBarrierArriveOp>(op);
  bool isShared = isPtrInSharedCTASpace(thisOp.getAddr());

  llvm::Intrinsic::ID id;
  if (thisOp.getNoinc()) {
    id = isShared ? llvm::Intrinsic::nvvm_cp_async_mbarrier_arrive_noinc_shared
                  : llvm::Intrinsic::nvvm_cp_async_mbarrier_arrive_noinc;
  } else {
    id = isShared ? llvm::Intrinsic::nvvm_cp_async_mbarrier_arrive_shared
                  : llvm::Intrinsic::nvvm_cp_async_mbarrier_arrive;
  }

  return {id, {mt.lookupValue(thisOp.getAddr())}};
}

#define CP_ASYNC_ID_IMPL(mod, size, suffix)                                    \
  llvm::Intrinsic::nvvm_cp_async_##mod##_shared_global_##size##suffix

#define GET_CP_ASYNC_ID(mod, size, has_cpsize)                                 \
  has_cpsize ? CP_ASYNC_ID_IMPL(mod, size, _s) : CP_ASYNC_ID_IMPL(mod, size, )

llvm::Intrinsic::ID
CpAsyncOp::getIntrinsicIDAndArgs(Operation &op, LLVM::ModuleTranslation &mt,
                                 llvm::SmallVector<llvm::Value *> &args) {
  llvm::Intrinsic::ID id;

  auto cpAsyncOp = cast<NVVM::CpAsyncOp>(op);
  bool hasCpSize = static_cast<bool>(cpAsyncOp.getCpSize());
  switch (cpAsyncOp.getSize()) {
  case 4:
    id = GET_CP_ASYNC_ID(ca, 4, hasCpSize);
    break;
  case 8:
    id = GET_CP_ASYNC_ID(ca, 8, hasCpSize);
    break;
  case 16:
    id = (cpAsyncOp.getModifier() == NVVM::LoadCacheModifierKind::CG)
             ? GET_CP_ASYNC_ID(cg, 16, hasCpSize)
             : GET_CP_ASYNC_ID(ca, 16, hasCpSize);
    break;
  default:
    llvm_unreachable("Invalid copy size in CpAsyncOp.");
  }

  // Fill the Intrinsic Args
  args.push_back(mt.lookupValue(cpAsyncOp.getDst()));
  args.push_back(mt.lookupValue(cpAsyncOp.getSrc()));
  if (hasCpSize)
    args.push_back(mt.lookupValue(cpAsyncOp.getCpSize()));

  return id;
}

mlir::NVVM::IDArgPair CpAsyncBulkPrefetchOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::CpAsyncBulkPrefetchOp>(op);
  llvm::SmallVector<llvm::Value *> args;
  llvm::Intrinsic::ID id = llvm::Intrinsic::nvvm_cp_async_bulk_prefetch_L2;

  // Fill the Intrinsic Args
  args.push_back(mt.lookupValue(thisOp.getSrcMem()));
  args.push_back(mt.lookupValue(thisOp.getSize()));

  mlir::Value cacheHint = thisOp.getL2CacheHint();
  const bool hasCacheHint = static_cast<bool>(cacheHint);
  llvm::Value *i64Unused =
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(mt.getLLVMContext()), 0);
  args.push_back(hasCacheHint ? mt.lookupValue(cacheHint) : i64Unused);
  args.push_back(builder.getInt1(hasCacheHint));

  return {id, std::move(args)};
}

mlir::NVVM::IDArgPair CpAsyncBulkGlobalToSharedClusterOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::CpAsyncBulkGlobalToSharedClusterOp>(op);
  llvm::SmallVector<llvm::Value *> args;

  // Fill the Intrinsic Args: dst, mbar, src, size.
  args.push_back(mt.lookupValue(thisOp.getDstMem()));
  args.push_back(mt.lookupValue(thisOp.getMbar()));
  args.push_back(mt.lookupValue(thisOp.getSrcMem()));
  args.push_back(mt.lookupValue(thisOp.getSize()));

  // Multicast mask for shared::cluster only, if available.
  mlir::Value multicastMask = thisOp.getMulticastMask();
  const bool hasMulticastMask = static_cast<bool>(multicastMask);
  const bool isSharedCTA = isPtrInSharedCTASpace(thisOp.getDstMem());
  if (!isSharedCTA) {
    llvm::Value *i16Unused = llvm::ConstantInt::get(builder.getInt16Ty(), 0);
    args.push_back(hasMulticastMask ? mt.lookupValue(multicastMask)
                                    : i16Unused);
  } else {
    args.push_back(builder.getInt32(0)); // ignore_bytes_left
    args.push_back(builder.getInt32(0)); // ignore_bytes_right
  }

  // Cache hint, if available.
  mlir::Value cacheHint = thisOp.getL2CacheHint();
  const bool hasCacheHint = static_cast<bool>(cacheHint);
  llvm::Value *i64Unused = llvm::ConstantInt::get(builder.getInt64Ty(), 0);
  args.push_back(hasCacheHint ? mt.lookupValue(cacheHint) : i64Unused);

  // Flag arguments for multicast/ignore_oob and cachehint.
  if (isSharedCTA) {
    args.push_back(builder.getInt1(hasCacheHint)); // flag_ch
    args.push_back(builder.getInt1(false));        // flag_oob
  } else {
    args.push_back(builder.getInt1(hasMulticastMask)); // flag_mc
    args.push_back(builder.getInt1(hasCacheHint));     // flag_ch
  }

  // flag_valid_pattern = disabled.
  args.push_back(builder.getInt32(0));

  llvm::Intrinsic::ID id =
      isSharedCTA
          ? llvm::Intrinsic::nvvm_cp_async_bulk_global_to_shared_cta
          : llvm::Intrinsic::nvvm_cp_async_bulk_global_to_shared_cluster;

  return {id, std::move(args)};
}

mlir::NVVM::IDArgPair CpAsyncBulkSharedCTAToGlobalOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::CpAsyncBulkSharedCTAToGlobalOp>(op);
  llvm::SmallVector<llvm::Value *> args;
  llvm::Intrinsic::ID id =
      llvm::Intrinsic::nvvm_cp_async_bulk_shared_cta_to_global;

  // Fill the Intrinsic Args
  args.push_back(mt.lookupValue(thisOp.getDstMem()));
  args.push_back(mt.lookupValue(thisOp.getSrcMem()));
  args.push_back(mt.lookupValue(thisOp.getSize()));

  mlir::Value cacheHint = thisOp.getL2CacheHint();
  const bool hasCacheHint = static_cast<bool>(cacheHint);
  llvm::Value *i64Unused =
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(mt.getLLVMContext()), 0);
  args.push_back(hasCacheHint ? mt.lookupValue(cacheHint) : i64Unused);
  args.push_back(builder.getInt1(hasCacheHint));

  // Choose the bytemask variant
  if (mlir::Value byteMask = thisOp.getByteMask()) {
    args.push_back(mt.lookupValue(byteMask));
    id = llvm::Intrinsic::nvvm_cp_async_bulk_shared_cta_to_global_bytemask;
  }

  return {id, std::move(args)};
}

bool CpAsyncBulkTensorGlobalToSharedClusterOp::getAsmValues(
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

mlir::NVVM::IDArgPair
CpAsyncBulkTensorGlobalToSharedClusterOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::CpAsyncBulkTensorGlobalToSharedClusterOp>(op);
  const bool isCTAOnly = thisOp.getIsCTAOnly();
  llvm::SmallVector<llvm::Value *> args;

  // Fill the Intrinsic Args
  args.push_back(mt.lookupValue(thisOp.getDstMem()));
  args.push_back(mt.lookupValue(thisOp.getMbar()));
  args.push_back(mt.lookupValue(thisOp.getTmaDescriptor()));

  // Coordinates and im2col-offsets
  for (mlir::Value v : thisOp.getCoordinates())
    args.push_back(mt.lookupValue(v));
  for (mlir::Value v : thisOp.getIm2colOffsets())
    args.push_back(mt.lookupValue(v));

  // MulticastMask, if available
  mlir::Value mcMask = thisOp.getMulticastMask();
  const bool hasMC = static_cast<bool>(mcMask);
  llvm::Value *i16Zero =
      llvm::ConstantInt::get(llvm::Type::getInt16Ty(mt.getLLVMContext()), 0);

  // CacheHint, if available
  mlir::Value cacheHint = thisOp.getL2CacheHint();
  const bool hasCacheHint = static_cast<bool>(cacheHint);
  llvm::Value *i64Zero =
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(mt.getLLVMContext()), 0);

  // Flag argument CTAGroup
  // CTA_1/2 is mapped to values 1 and 2 for the intrinsics.
  // Hence, the +1 to getGroup().
  const int32_t val =
      thisOp.getGroup() ? (static_cast<int32_t>(*thisOp.getGroup()) + 1) : 0;
  llvm::Value *cg =
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(mt.getLLVMContext()), val);

  // flag_valid_pattern = disabled
  llvm::Value *flagValidPattern = builder.getInt32(0);

  if (!isCTAOnly) {
    // For shared::cluster, all the arguments that we build are applicable.
    args.push_back(hasMC ? mt.lookupValue(mcMask) : i16Zero);
    args.push_back(hasCacheHint ? mt.lookupValue(cacheHint) : i64Zero);
    args.push_back(builder.getInt1(hasMC));
    args.push_back(builder.getInt1(hasCacheHint));
    args.push_back(cg);
    args.push_back(flagValidPattern);
  } else {
    // For shared::cta, only cache-hint is applicable.
    args.push_back(hasCacheHint ? mt.lookupValue(cacheHint) : i64Zero);
    args.push_back(builder.getInt1(hasCacheHint));
    args.push_back(flagValidPattern);
  }

  constexpr size_t numDims = 5;  // 1D to 5D
  constexpr size_t numModes = 5; // Tile, Im2col, w, w_128, gather4
  using rowTy = std::array<llvm::Intrinsic::ID, numDims + 1>;
  using TableTy = std::array<rowTy, numModes>;
  static constexpr TableTy IDTable{
      {{notIntrinsic, llvm::Intrinsic::nvvm_cp_async_bulk_tensor_g2s_tile_1d,
        llvm::Intrinsic::nvvm_cp_async_bulk_tensor_g2s_tile_2d,
        llvm::Intrinsic::nvvm_cp_async_bulk_tensor_g2s_tile_3d,
        llvm::Intrinsic::nvvm_cp_async_bulk_tensor_g2s_tile_4d,
        llvm::Intrinsic::nvvm_cp_async_bulk_tensor_g2s_tile_5d},
       {notIntrinsic, notIntrinsic, notIntrinsic,
        llvm::Intrinsic::nvvm_cp_async_bulk_tensor_g2s_im2col_3d,
        llvm::Intrinsic::nvvm_cp_async_bulk_tensor_g2s_im2col_4d,
        llvm::Intrinsic::nvvm_cp_async_bulk_tensor_g2s_im2col_5d},
       {notIntrinsic, notIntrinsic, notIntrinsic,
        llvm::Intrinsic::nvvm_cp_async_bulk_tensor_g2s_im2col_w_3d,
        llvm::Intrinsic::nvvm_cp_async_bulk_tensor_g2s_im2col_w_4d,
        llvm::Intrinsic::nvvm_cp_async_bulk_tensor_g2s_im2col_w_5d},
       {notIntrinsic, notIntrinsic, notIntrinsic,
        llvm::Intrinsic::nvvm_cp_async_bulk_tensor_g2s_im2col_w_128_3d,
        llvm::Intrinsic::nvvm_cp_async_bulk_tensor_g2s_im2col_w_128_4d,
        llvm::Intrinsic::nvvm_cp_async_bulk_tensor_g2s_im2col_w_128_5d},
       {notIntrinsic, notIntrinsic, notIntrinsic, notIntrinsic, notIntrinsic,
        llvm::Intrinsic::nvvm_cp_async_bulk_tensor_g2s_tile_gather4_2d}}};

  static constexpr TableTy IDTableCTA{
      {{notIntrinsic,
        llvm::Intrinsic::nvvm_cp_async_bulk_tensor_g2s_cta_tile_1d,
        llvm::Intrinsic::nvvm_cp_async_bulk_tensor_g2s_cta_tile_2d,
        llvm::Intrinsic::nvvm_cp_async_bulk_tensor_g2s_cta_tile_3d,
        llvm::Intrinsic::nvvm_cp_async_bulk_tensor_g2s_cta_tile_4d,
        llvm::Intrinsic::nvvm_cp_async_bulk_tensor_g2s_cta_tile_5d},
       {notIntrinsic, notIntrinsic, notIntrinsic,
        llvm::Intrinsic::nvvm_cp_async_bulk_tensor_g2s_cta_im2col_3d,
        llvm::Intrinsic::nvvm_cp_async_bulk_tensor_g2s_cta_im2col_4d,
        llvm::Intrinsic::nvvm_cp_async_bulk_tensor_g2s_cta_im2col_5d},
       {notIntrinsic, notIntrinsic, notIntrinsic,
        llvm::Intrinsic::nvvm_cp_async_bulk_tensor_g2s_cta_im2col_w_3d,
        llvm::Intrinsic::nvvm_cp_async_bulk_tensor_g2s_cta_im2col_w_4d,
        llvm::Intrinsic::nvvm_cp_async_bulk_tensor_g2s_cta_im2col_w_5d},
       {notIntrinsic, notIntrinsic, notIntrinsic,
        llvm::Intrinsic::nvvm_cp_async_bulk_tensor_g2s_cta_im2col_w_128_3d,
        llvm::Intrinsic::nvvm_cp_async_bulk_tensor_g2s_cta_im2col_w_128_4d,
        llvm::Intrinsic::nvvm_cp_async_bulk_tensor_g2s_cta_im2col_w_128_5d},
       {notIntrinsic, notIntrinsic, notIntrinsic, notIntrinsic, notIntrinsic,
        llvm::Intrinsic::nvvm_cp_async_bulk_tensor_g2s_cta_tile_gather4_2d}}};

  static_assert(
      (getMaxEnumValForTMALoadMode() == std::size(IDTable) - 1) &&
          (getMaxEnumValForTMALoadMode() == std::size(IDTableCTA) - 1),
      "TMALoadModes must match number of rows in IDTable and IDTableCTA");
  size_t mode = static_cast<size_t>(thisOp.getMode());
  size_t dim = thisOp.getCoordinates().size();
  auto id = isCTAOnly ? IDTableCTA[mode][dim] : IDTable[mode][dim];
  assert(id != notIntrinsic &&
         "Invalid intrinsic for CpAsyncBulkTensorGlobalToSharedClusterOp.");

  return {id, std::move(args)};
}

mlir::NVVM::IDArgPair CpAsyncBulkTensorPrefetchOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::CpAsyncBulkTensorPrefetchOp>(op);
  llvm::SmallVector<llvm::Value *> args;

  // Fill the Intrinsic Args
  args.push_back(mt.lookupValue(thisOp.getTmaDescriptor()));

  for (auto v : thisOp.getCoordinates())
    args.push_back(mt.lookupValue(v));
  for (auto v : thisOp.getIm2colOffsets())
    args.push_back(mt.lookupValue(v));

  mlir::Value cacheHint = thisOp.getL2CacheHint();
  const bool hasCacheHint = static_cast<bool>(cacheHint);
  llvm::Value *i64Unused =
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(mt.getLLVMContext()), 0);
  args.push_back(hasCacheHint ? mt.lookupValue(cacheHint) : i64Unused);
  args.push_back(builder.getInt1(hasCacheHint));

  const unsigned NI = llvm::Intrinsic::not_intrinsic;
  static constexpr llvm::Intrinsic::ID IDTable[][6] = {
      {NI, llvm::Intrinsic::nvvm_cp_async_bulk_tensor_prefetch_tile_1d,
       llvm::Intrinsic::nvvm_cp_async_bulk_tensor_prefetch_tile_2d,
       llvm::Intrinsic::nvvm_cp_async_bulk_tensor_prefetch_tile_3d,
       llvm::Intrinsic::nvvm_cp_async_bulk_tensor_prefetch_tile_4d,
       llvm::Intrinsic::nvvm_cp_async_bulk_tensor_prefetch_tile_5d},
      {NI, NI, NI,
       llvm::Intrinsic::nvvm_cp_async_bulk_tensor_prefetch_im2col_3d,
       llvm::Intrinsic::nvvm_cp_async_bulk_tensor_prefetch_im2col_4d,
       llvm::Intrinsic::nvvm_cp_async_bulk_tensor_prefetch_im2col_5d},
      {NI, NI, NI,
       llvm::Intrinsic::nvvm_cp_async_bulk_tensor_prefetch_im2col_w_3d,
       llvm::Intrinsic::nvvm_cp_async_bulk_tensor_prefetch_im2col_w_4d,
       llvm::Intrinsic::nvvm_cp_async_bulk_tensor_prefetch_im2col_w_5d},
      {NI, NI, NI,
       llvm::Intrinsic::nvvm_cp_async_bulk_tensor_prefetch_im2col_w_128_3d,
       llvm::Intrinsic::nvvm_cp_async_bulk_tensor_prefetch_im2col_w_128_4d,
       llvm::Intrinsic::nvvm_cp_async_bulk_tensor_prefetch_im2col_w_128_5d},
      {NI, NI, NI, NI, NI,
       llvm::Intrinsic::nvvm_cp_async_bulk_tensor_prefetch_tile_gather4_2d}};

  static_assert(getMaxEnumValForTMALoadMode() == std::size(IDTable) - 1,
                "TMALoadModes must match number of rows in IDTable");
  size_t mode = static_cast<size_t>(thisOp.getMode());
  size_t dim = thisOp.getCoordinates().size();
  llvm::Intrinsic::ID id = IDTable[mode][dim];
  if (id == llvm::Intrinsic::not_intrinsic)
    llvm_unreachable("Invalid intrinsic for CpAsyncBulkTensorPrefetchOp.");

  return {id, std::move(args)};
}

mlir::NVVM::IDArgPair
CpAsyncBulkTensorSharedCTAToGlobalOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::CpAsyncBulkTensorSharedCTAToGlobalOp>(op);
  llvm::SmallVector<llvm::Value *> args;

  // Fill the Intrinsic Args
  args.push_back(mt.lookupValue(thisOp.getSrcMem()));
  args.push_back(mt.lookupValue(thisOp.getTmaDescriptor()));

  for (auto v : thisOp.getCoordinates())
    args.push_back(mt.lookupValue(v));

  mlir::Value cacheHint = thisOp.getL2CacheHint();
  const bool hasCacheHint = static_cast<bool>(cacheHint);
  llvm::Value *i64Unused =
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(mt.getLLVMContext()), 0);
  args.push_back(hasCacheHint ? mt.lookupValue(cacheHint) : i64Unused);
  args.push_back(builder.getInt1(hasCacheHint));

  using namespace llvm::Intrinsic;
  const unsigned NI = not_intrinsic;
  static constexpr ID IDTable[][6] = {
      {NI, nvvm_cp_async_bulk_tensor_s2g_tile_1d,
       nvvm_cp_async_bulk_tensor_s2g_tile_2d,
       nvvm_cp_async_bulk_tensor_s2g_tile_3d,
       nvvm_cp_async_bulk_tensor_s2g_tile_4d,
       nvvm_cp_async_bulk_tensor_s2g_tile_5d},
      {NI, NI, NI, nvvm_cp_async_bulk_tensor_s2g_im2col_3d,
       nvvm_cp_async_bulk_tensor_s2g_im2col_4d,
       nvvm_cp_async_bulk_tensor_s2g_im2col_5d},
      {NI, NI, NI, NI, NI, nvvm_cp_async_bulk_tensor_s2g_tile_scatter4_2d},
      {NI, NI, NI, nvvm_cp_async_bulk_tensor_s2g_im2col_w_3d,
       nvvm_cp_async_bulk_tensor_s2g_im2col_w_4d,
       nvvm_cp_async_bulk_tensor_s2g_im2col_w_5d}};

  static_assert(getMaxEnumValForTMAStoreMode() == std::size(IDTable) - 1,
                "TMAStoreModes must match number of rows in IDTable");
  size_t mode = static_cast<size_t>(thisOp.getMode());
  size_t dim = thisOp.getCoordinates().size();
  ID id = IDTable[mode][dim];
  if (id == llvm::Intrinsic::not_intrinsic)
    llvm_unreachable(
        "Invalid intrinsic for CpAsyncBulkTensorSharedCTAToGlobalOp.");

  return {id, std::move(args)};
}

NVVM::IDArgPair
CpAsyncBulkTensorSharedCTAToGlobalOverrideAddrOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp =
      cast<NVVM::CpAsyncBulkTensorSharedCTAToGlobalOverrideAddrOp>(op);

  llvm::SmallVector<llvm::Value *> args;
  args.push_back(mt.lookupValue(thisOp.getSrcMem()));
  args.push_back(mt.lookupValue(thisOp.getTmaDescriptor()));
  args.push_back(mt.lookupValue(thisOp.getOverrideAddr()));
  for (Value v : thisOp.getTensorSize())
    args.push_back(mt.lookupValue(v));
  for (Value v : thisOp.getLowerStride())
    args.push_back(mt.lookupValue(v));
  if (thisOp.getUpperStride())
    args.push_back(mt.lookupValue(thisOp.getUpperStride()));
  for (Value v : thisOp.getCoordinates())
    args.push_back(mt.lookupValue(v));

  mlir::Value cacheHint = thisOp.getL2CacheHint();
  const bool hasCacheHint = static_cast<bool>(cacheHint);
  args.push_back(hasCacheHint ? mt.lookupValue(cacheHint)
                              : builder.getInt64(0));
  args.push_back(builder.getInt1(hasCacheHint));

  using namespace llvm::Intrinsic;
  const unsigned NI = not_intrinsic;
  // clang-format off
  // override_addr variants, indexed [mode][dim].
  static constexpr ID IDTable[][6] = {
      {NI, nvvm_cp_async_bulk_tensor_s2g_tile_override_addr_1d,
       nvvm_cp_async_bulk_tensor_s2g_tile_override_addr_2d,
       nvvm_cp_async_bulk_tensor_s2g_tile_override_addr_3d,
       nvvm_cp_async_bulk_tensor_s2g_tile_override_addr_4d,
       nvvm_cp_async_bulk_tensor_s2g_tile_override_addr_5d},
      {NI, NI, NI, nvvm_cp_async_bulk_tensor_s2g_im2col_override_addr_3d,
       nvvm_cp_async_bulk_tensor_s2g_im2col_override_addr_4d,
       nvvm_cp_async_bulk_tensor_s2g_im2col_override_addr_5d},
      {NI, NI, NI, NI, NI,
       nvvm_cp_async_bulk_tensor_s2g_tile_scatter4_override_addr_2d},
      {NI, NI, NI, nvvm_cp_async_bulk_tensor_s2g_im2col_w_override_addr_3d,
       nvvm_cp_async_bulk_tensor_s2g_im2col_w_override_addr_4d,
       nvvm_cp_async_bulk_tensor_s2g_im2col_w_override_addr_5d}};

  // Tile-only override_addr_dim (1D) / override_addr_dim_stride (2D-5D)
  // variants, indexed [dim].
  static constexpr ID dimStrideIDTable[] = {
      NI, nvvm_cp_async_bulk_tensor_s2g_tile_override_addr_dim_1d,
      nvvm_cp_async_bulk_tensor_s2g_tile_override_addr_dim_stride_2d,
      nvvm_cp_async_bulk_tensor_s2g_tile_override_addr_dim_stride_3d,
      nvvm_cp_async_bulk_tensor_s2g_tile_override_addr_dim_stride_4d,
      nvvm_cp_async_bulk_tensor_s2g_tile_override_addr_dim_stride_5d};
  // clang-format on

  size_t mode = static_cast<size_t>(thisOp.getMode());
  size_t dim = thisOp.getCoordinates().size();
  bool isDimStride = !thisOp.getTensorSize().empty();

  assert(mode < std::size(IDTable) &&
         "Invalid mode for CpAsyncBulkTensorSharedCTAToGlobalOverrideAddrOp");
  assert(dim < std::size(IDTable[mode]) && dim < std::size(dimStrideIDTable) &&
         "Invalid dim for CpAsyncBulkTensorSharedCTAToGlobalOverrideAddrOp");

  ID intrinsicID = isDimStride ? dimStrideIDTable[dim] : IDTable[mode][dim];
  assert(
      intrinsicID != NI &&
      "Invalid intrinsic for CpAsyncBulkTensorSharedCTAToGlobalOverrideAddrOp");
  return {intrinsicID, std::move(args)};
}

NVVM::IDArgPair CpAsyncBulkTensorReduceOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::CpAsyncBulkTensorReduceOp>(op);

  llvm::SmallVector<llvm::Value *> args;
  args.push_back(mt.lookupValue(thisOp.getSrcMem()));
  args.push_back(mt.lookupValue(thisOp.getTmaDescriptor()));
  for (Value v : thisOp.getCoordinates())
    args.push_back(mt.lookupValue(v));

  mlir::Value cacheHint = thisOp.getL2CacheHint();
  const bool hasCacheHint = static_cast<bool>(cacheHint);
  args.push_back(hasCacheHint ? mt.lookupValue(cacheHint)
                              : builder.getInt64(0));
  args.push_back(builder.getInt32(static_cast<uint32_t>(thisOp.getRedKind())));
  args.push_back(builder.getInt1(hasCacheHint));

  using namespace llvm::Intrinsic;
  const unsigned NI = not_intrinsic;
  static constexpr ID IDTable[][6] = {
      {NI, nvvm_cp_async_bulk_tensor_reduce_tile_1d,
       nvvm_cp_async_bulk_tensor_reduce_tile_2d,
       nvvm_cp_async_bulk_tensor_reduce_tile_3d,
       nvvm_cp_async_bulk_tensor_reduce_tile_4d,
       nvvm_cp_async_bulk_tensor_reduce_tile_5d},
      {NI, NI, NI, nvvm_cp_async_bulk_tensor_reduce_im2col_3d,
       nvvm_cp_async_bulk_tensor_reduce_im2col_4d,
       nvvm_cp_async_bulk_tensor_reduce_im2col_5d},
      {NI, NI, NI, NI, NI, NI}, // scatter4 not supported for reduce
      {NI, NI, NI, nvvm_cp_async_bulk_tensor_reduce_im2col_w_3d,
       nvvm_cp_async_bulk_tensor_reduce_im2col_w_4d,
       nvvm_cp_async_bulk_tensor_reduce_im2col_w_5d}};

  size_t mode = static_cast<size_t>(thisOp.getMode());
  size_t dim = thisOp.getCoordinates().size();
  assert(mode < std::size(IDTable) &&
         "Invalid mode for CpAsyncBulkTensorReduceOp");
  assert(dim < std::size(IDTable[mode]) &&
         "Invalid dim for CpAsyncBulkTensorReduceOp");

  ID intrinsicID = IDTable[mode][dim];
  assert(intrinsicID != NI &&
         "Invalid intrinsic for CpAsyncBulkTensorReduceOp");
  return {intrinsicID, std::move(args)};
}

NVVM::IDArgPair CpAsyncBulkTensorReduceOverrideAddrOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::CpAsyncBulkTensorReduceOverrideAddrOp>(op);

  llvm::SmallVector<llvm::Value *> args;
  args.push_back(mt.lookupValue(thisOp.getSrcMem()));
  args.push_back(mt.lookupValue(thisOp.getTmaDescriptor()));
  args.push_back(mt.lookupValue(thisOp.getOverrideAddr()));

  for (Value v : thisOp.getTensorSize())
    args.push_back(mt.lookupValue(v));
  for (Value v : thisOp.getLowerStride())
    args.push_back(mt.lookupValue(v));
  if (thisOp.getUpperStride())
    args.push_back(mt.lookupValue(thisOp.getUpperStride()));
  for (Value v : thisOp.getCoordinates())
    args.push_back(mt.lookupValue(v));

  mlir::Value cacheHint = thisOp.getL2CacheHint();
  const bool hasCacheHint = static_cast<bool>(cacheHint);
  args.push_back(hasCacheHint ? mt.lookupValue(cacheHint)
                              : builder.getInt64(0));
  args.push_back(builder.getInt32(static_cast<uint32_t>(thisOp.getRedKind())));
  args.push_back(builder.getInt1(hasCacheHint));

  using namespace llvm::Intrinsic;
  const unsigned NI = not_intrinsic;
  // clang-format off
// override_addr variants, indexed [mode][dim].
static constexpr ID IDTable[][6] = {
    {NI, nvvm_cp_async_bulk_tensor_reduce_tile_override_addr_1d,
     nvvm_cp_async_bulk_tensor_reduce_tile_override_addr_2d,
     nvvm_cp_async_bulk_tensor_reduce_tile_override_addr_3d,
     nvvm_cp_async_bulk_tensor_reduce_tile_override_addr_4d,
     nvvm_cp_async_bulk_tensor_reduce_tile_override_addr_5d},
    {NI, NI, NI, nvvm_cp_async_bulk_tensor_reduce_im2col_override_addr_3d,
     nvvm_cp_async_bulk_tensor_reduce_im2col_override_addr_4d,
     nvvm_cp_async_bulk_tensor_reduce_im2col_override_addr_5d},
    {NI, NI, NI, NI, NI, NI}, // scatter4 not supported for reduce
    {NI, NI, NI, nvvm_cp_async_bulk_tensor_reduce_im2col_w_override_addr_3d,
     nvvm_cp_async_bulk_tensor_reduce_im2col_w_override_addr_4d,
     nvvm_cp_async_bulk_tensor_reduce_im2col_w_override_addr_5d}};

// Tile-only override_addr_dim (1D) / override_addr_dim_stride (2D-5D)
// variants, indexed [dim].
static constexpr ID dimStrideIDTable[] = {
    NI, nvvm_cp_async_bulk_tensor_reduce_tile_override_addr_dim_1d,
    nvvm_cp_async_bulk_tensor_reduce_tile_override_addr_dim_stride_2d,
    nvvm_cp_async_bulk_tensor_reduce_tile_override_addr_dim_stride_3d,
    nvvm_cp_async_bulk_tensor_reduce_tile_override_addr_dim_stride_4d,
    nvvm_cp_async_bulk_tensor_reduce_tile_override_addr_dim_stride_5d};
  // clang-format on

  size_t mode = static_cast<size_t>(thisOp.getMode());
  size_t dim = thisOp.getCoordinates().size();
  bool isDimStride = !thisOp.getTensorSize().empty();

  assert(mode < std::size(IDTable) &&
         "Invalid mode for CpAsyncBulkTensorReduceOverrideAddrOp");
  assert(dim < std::size(IDTable[mode]) && dim < std::size(dimStrideIDTable) &&
         "Invalid dim for CpAsyncBulkTensorReduceOverrideAddrOp");

  ID intrinsicID = isDimStride ? dimStrideIDTable[dim] : IDTable[mode][dim];
  assert(intrinsicID != NI &&
         "Invalid intrinsic for CpAsyncBulkTensorReduceOverrideAddrOp");
  return {intrinsicID, std::move(args)};
}

static llvm::Value *getParamCastedAddr(llvm::Value *addr,
                                       llvm::IRBuilderBase &builder) {
  return builder.CreateAddrSpaceCast(
      addr, builder.getPtrTy(llvm::NVPTXAS::ADDRESS_SPACE_ENTRY_PARAM));
}

NVVM::IDArgPair
PrefetchOp::getIntrinsicIDAndArgs(NVVM::PrefetchOp &op,
                                  LLVM::ModuleTranslation &mt,
                                  llvm::IRBuilderBase &builder) {
  using MemSpace = NVVM::NVVMMemorySpace;
  using CacheLevel = NVVM::PrefetchCacheLevel;

  std::optional<NVVM::PrefetchCacheLevel> cacheLevel = op.getCacheLevel();
  std::optional<NVVM::CacheEvictionPriority> evictPriority =
      op.getEvictPriority();
  unsigned addressSpace =
      llvm::cast<LLVM::LLVMPointerType>(op.getAddr().getType())
          .getAddressSpace();

  llvm::SmallVector<llvm::Value *> args;
  llvm::Value *addr = mt.lookupValue(op.getAddr());
  args.push_back(op.getInParamSpace() ? getParamCastedAddr(addr, builder)
                                      : addr);

  if (op.getTensormap())
    return {llvm::Intrinsic::nvvm_prefetch_tensormap, args};

  assert(cacheLevel && "expected cache level for non-tensormap prefetch");

  if (op.getUniform() && *cacheLevel == CacheLevel::L1)
    return {llvm::Intrinsic::nvvm_prefetchu_L1, args};

  if (evictPriority && *cacheLevel == CacheLevel::L2) {
    switch (*evictPriority) {
    case NVVM::CacheEvictionPriority::EvictLast:
      return {llvm::Intrinsic::nvvm_prefetch_global_L2_evict_last, args};
    case NVVM::CacheEvictionPriority::EvictNormal:
      return {llvm::Intrinsic::nvvm_prefetch_global_L2_evict_normal, args};
    default:
      llvm_unreachable("Invalid cache eviction priority");
    }
  }

  switch (static_cast<MemSpace>(addressSpace)) {
  case MemSpace::Generic:
    return *cacheLevel == CacheLevel::L1
               ? NVVM::IDArgPair({llvm::Intrinsic::nvvm_prefetch_L1, args})
               : NVVM::IDArgPair({llvm::Intrinsic::nvvm_prefetch_L2, args});
  case MemSpace::Global:
    return *cacheLevel == CacheLevel::L1
               ? NVVM::IDArgPair(
                     {llvm::Intrinsic::nvvm_prefetch_global_L1, args})
               : NVVM::IDArgPair(
                     {llvm::Intrinsic::nvvm_prefetch_global_L2, args});
  case MemSpace::Local:
    return *cacheLevel == CacheLevel::L1
               ? NVVM::IDArgPair(
                     {llvm::Intrinsic::nvvm_prefetch_local_L1, args})
               : NVVM::IDArgPair(
                     {llvm::Intrinsic::nvvm_prefetch_local_L2, args});
  default:
    llvm_unreachable("Invalid pointer address space");
  }
}

mlir::NVVM::IDArgPair TensormapReplaceOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::TensormapReplaceOp>(op);

  llvm::SmallVector<llvm::Value *> args;
  args.push_back(mt.lookupValue(thisOp.getAddr()));
  if (thisOp.getOrd())
    args.push_back(builder.getInt32(thisOp.getOrd().value()));
  if (thisOp.getNewValue())
    args.push_back(mt.lookupValue(thisOp.getNewValue()));
  if (auto attr = thisOp.getNewValueAttr()) {
    auto val =
        llvm::TypeSwitch<mlir::Attribute, unsigned>(*attr)
            .Case<TensormapElemtypeAttr, TensormapInterleaveLayoutAttr,
                  TensormapSwizzleModeAttr, TensormapSwizzleAtomicityAttr,
                  TensormapFillModeAttr>([](auto attr) {
              return static_cast<unsigned>(attr.getValue());
            })
            .Default([](auto attr) {
              llvm_unreachable("Invalid attribute type");
              return 0;
            });
    args.push_back(builder.getInt32(val));
  }

  static constexpr llvm::Intrinsic::ID IDs[] = {
      llvm::Intrinsic::nvvm_tensormap_replace_global_address,
      llvm::Intrinsic::nvvm_tensormap_replace_rank,
      llvm::Intrinsic::nvvm_tensormap_replace_box_dim,
      llvm::Intrinsic::nvvm_tensormap_replace_global_dim,
      llvm::Intrinsic::nvvm_tensormap_replace_global_stride,
      llvm::Intrinsic::nvvm_tensormap_replace_element_stride,
      llvm::Intrinsic::nvvm_tensormap_replace_elemtype,
      llvm::Intrinsic::nvvm_tensormap_replace_interleave_layout,
      llvm::Intrinsic::nvvm_tensormap_replace_swizzle_mode,
      llvm::Intrinsic::nvvm_tensormap_replace_swizzle_atomicity,
      llvm::Intrinsic::nvvm_tensormap_replace_fill_mode,
  };

  unsigned fieldIndex = static_cast<unsigned>(thisOp.getField());

  return {IDs[fieldIndex], args};
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMMemoryOps.cpp.inc"
