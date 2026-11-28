//===- NVVMDataMovementOps.cpp - NVVM dialect data movement ops -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ops of op partition `DataMovement` of the NVVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/NVVMDataMovementOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/NVPTXAddrSpace.h"

#include <cassert>
#include <optional>

using namespace mlir;
using namespace mlir::NVVM;

//===----------------------------------------------------------------------===//
// Verifier methods
//===----------------------------------------------------------------------===//

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

LogicalResult ShflOp::verify() {
  auto returnStructType = llvm::dyn_cast<LLVM::LLVMStructType>(getType());

  auto verifyTypeError = [&](Twine desc, Type expectedType,
                             Type actualType) -> LogicalResult {
    return emitOpError("expected " + desc + " to be of type ")
           << expectedType << " but got " << actualType << " instead";
  };

  if (returnStructType) {
    if (!getReturnValueAndIsValid())
      return emitOpError("\"return_value_and_is_valid\" attribute must be "
                         "specified when the return type is a struct type");

    if (returnStructType.getBody().size() != 2)
      return emitOpError("expected return type to be a two-element struct");

    llvm::ArrayRef<Type> returnStruct = returnStructType.getBody();
    auto resultType = returnStruct[0];
    if (resultType != getVal().getType())
      return verifyTypeError("first element in the returned struct",
                             getVal().getType(), resultType);

    auto predicateType = returnStruct[1];
    if (!predicateType.isInteger(1))
      return verifyTypeError("second element in the returned struct",
                             mlir::IntegerType::get(getContext(), 1),
                             predicateType);
  } else {
    if (getReturnValueAndIsValid())
      return emitOpError("expected return type to be a two-element struct");

    if (getType() != getVal().getType())
      return verifyTypeError("return type", getVal().getType(), getType());
  }
  return success();
}

LogicalResult
ShflOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                         ShflOp::Adaptor adaptor,
                         SmallVectorImpl<Type> &inferredReturnTypes) {
  Type valType = adaptor.getVal().getType();
  if (adaptor.getReturnValueAndIsValid())
    inferredReturnTypes.push_back(LLVM::LLVMStructType::getLiteral(
        context, {valType, IntegerType::get(context, 1)}));
  else
    inferredReturnTypes.push_back(valType);
  return success();
}

LogicalResult PermuteOp::verify() {
  using Mode = NVVM::PermuteMode;
  bool hasHi = static_cast<bool>(getHi());

  switch (getMode()) {
  case Mode::DEFAULT:
  case Mode::F4E:
  case Mode::B4E:
    if (!hasHi)
      return emitError("mode '") << getMode() << "' requires 'hi' operand.";
    break;
  case Mode::RC8:
  case Mode::ECL:
  case Mode::ECR:
  case Mode::RC16:
    if (hasHi)
      return emitError("mode '")
             << getMode() << "' does not accept 'hi' operand.";
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

mlir::NVVM::IDArgPair
PermuteOp::getIntrinsicIDAndArgs(Operation &op, LLVM::ModuleTranslation &mt,
                                 llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::PermuteOp>(op);
  NVVM::PermuteMode mode = thisOp.getMode();

  static constexpr llvm::Intrinsic::ID IDs[] = {
      llvm::Intrinsic::nvvm_prmt,     llvm::Intrinsic::nvvm_prmt_f4e,
      llvm::Intrinsic::nvvm_prmt_b4e, llvm::Intrinsic::nvvm_prmt_rc8,
      llvm::Intrinsic::nvvm_prmt_ecl, llvm::Intrinsic::nvvm_prmt_ecr,
      llvm::Intrinsic::nvvm_prmt_rc16};

  unsigned modeIndex = static_cast<unsigned>(mode);
  llvm::SmallVector<llvm::Value *> args;
  args.push_back(mt.lookupValue(thisOp.getLo()));

  // Only first 3 modes (Default, f4e, b4e) need the hi operand.
  if (modeIndex < 3)
    args.push_back(mt.lookupValue(thisOp.getHi()));

  args.push_back(mt.lookupValue(thisOp.getSelector()));

  return {IDs[modeIndex], args};
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
#include "mlir/Dialect/LLVMIR/NVVMDataMovementOps.cpp.inc"
