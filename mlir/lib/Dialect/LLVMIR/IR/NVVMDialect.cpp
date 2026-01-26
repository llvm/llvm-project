//===- NVVMDialect.cpp - NVVM IR Ops and Dialect registration -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the types and operation details for the NVVM IR dialect in
// MLIR, and the LLVM IR dialect.  It also registers the dialect.
//
// The NVVM dialect only contains GPU specific additions on top of the general
// LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/NVVMIntrinsicUtils.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/NVPTXAddrSpace.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <optional>
#include <string>

using namespace mlir;
using namespace NVVM;

#include "mlir/Dialect/LLVMIR/NVVMOpsDialect.cpp.inc"
#include "mlir/Dialect/LLVMIR/NVVMOpsEnums.cpp.inc"

static constexpr unsigned notIntrinsic = llvm::Intrinsic::not_intrinsic;

//===----------------------------------------------------------------------===//
// Helper/Utility methods
//===----------------------------------------------------------------------===//

static bool isPtrInAddrSpace(mlir::Value ptr, NVVMMemorySpace targetAS) {
  auto ptrTy = llvm::cast<LLVM::LLVMPointerType>(ptr.getType());
  return ptrTy.getAddressSpace() == static_cast<unsigned>(targetAS);
}

static bool isPtrInGenericSpace(mlir::Value ptr) {
  return isPtrInAddrSpace(ptr, NVVMMemorySpace::Generic);
}

static bool isPtrInSharedCTASpace(mlir::Value ptr) {
  return isPtrInAddrSpace(ptr, NVVMMemorySpace::Shared);
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

// Helper method to convert CtaGroupKind in NVVM Dialect to CtaGroupKind in LLVM
static llvm::nvvm::CTAGroupKind
getNVVMCtaGroupKind(NVVM::CTAGroupKind ctaGroup) {
  switch (ctaGroup) {
  case NVVM::CTAGroupKind::CTA_1:
    return llvm::nvvm::CTAGroupKind::CG_1;
  case NVVM::CTAGroupKind::CTA_2:
    return llvm::nvvm::CTAGroupKind::CG_2;
  }
  llvm_unreachable("unsupported cta_group value");
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
    return cpAsyncBulkTensorCommonVerifier(dims, true, 0, getLoc());
  case TMAStoreMode::TILE_SCATTER4:
    if (dims != 5)
      return emitError("Scatter4 mode expects 5 coordinates");
  }
  return success();
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
    return cpAsyncBulkTensorCommonVerifier(dims, true, 0, getLoc());
  case TMAStoreMode::TILE_SCATTER4:
    return emitError("Scatter mode unsupported for CpAsyncBulkTensorReduceOp");
  }
  return success();
}

LogicalResult CpAsyncBulkGlobalToSharedClusterOp::verify() {
  bool isSharedCTA = isPtrInSharedCTASpace(getDstMem());
  if (isSharedCTA && getMulticastMask())
    return emitError("Multicast is not supported with shared::cta mode.");

  return success();
}

static LogicalResult verifyMBarrierArriveLikeOp(Operation *op, Value addr,
                                                NVVM::MemScopeKind scope,
                                                Value retVal = nullptr) {
  if (scope != NVVM::MemScopeKind::CTA && scope != NVVM::MemScopeKind::CLUSTER)
    return op->emitError("mbarrier scope must be either CTA or Cluster");

  bool isSharedCluster = isPtrInSharedClusterSpace(addr);
  bool hasRetValue = static_cast<bool>(retVal);
  if (isSharedCluster && hasRetValue)
    return op->emitError(
        "mbarrier in shared_cluster space cannot return any value");

  return success();
}

LogicalResult MBarrierArriveOp::verify() {
  return verifyMBarrierArriveLikeOp(getOperation(), getAddr(), getScope(),
                                    getRes());
}

LogicalResult MBarrierArriveDropOp::verify() {
  return verifyMBarrierArriveLikeOp(getOperation(), getAddr(), getScope(),
                                    getRes());
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
                                    getRes());
}

LogicalResult MBarrierArriveDropExpectTxOp::verify() {
  return verifyMBarrierArriveLikeOp(getOperation(), getAddr(), getScope(),
                                    getRes());
}

LogicalResult MBarrierExpectTxOp::verify() {
  return verifyMBarrierArriveLikeOp(getOperation(), getAddr(), getScope());
}

LogicalResult MBarrierCompleteTxOp::verify() {
  return verifyMBarrierArriveLikeOp(getOperation(), getAddr(), getScope());
}

LogicalResult MBarrierTestWaitOp::verify() {
  return verifyMBarrierArriveLikeOp(getOperation(), getAddr(), getScope());
}

LogicalResult MBarrierTryWaitOp::verify() {
  return verifyMBarrierArriveLikeOp(getOperation(), getAddr(), getScope());
}

LogicalResult ConvertFloatToTF32Op::verify() {
  using RndMode = NVVM::FPRoundingMode;
  switch (getRnd()) {
  case RndMode::RNA:
    if (getRelu())
      return emitError("Relu not supported with rna rounding mode.");
    break;
  case RndMode::RN:
  case RndMode::RZ:
    break;
  default:
    return emitError(
        "Only {rn,rz,rna} rounding modes supported for ConvertFloatToTF32Op.");
  }
  return success();
}

LogicalResult ConvertF32x2ToF6x2Op::verify() {
  mlir::MLIRContext *ctx = getContext();

  if (!llvm::isa<mlir::Float6E2M3FNType, mlir::Float6E3M2FNType>(getDstTy())) {
    return emitOpError("Only ")
           << mlir::Float6E2M3FNType::get(ctx) << " and "
           << mlir::Float6E3M2FNType::get(ctx)
           << " types are supported for conversions from f32x2 to f6x2.";
  }
  return success();
}

LogicalResult ConvertF32x2ToF8x2Op::verify() {
  using RndMode = NVVM::FPRoundingMode;
  using SatMode = NVVM::SaturationMode;

  bool isRoundingModeRN = getRnd() == RndMode::RN;
  bool isRoundingModeRZ = getRnd() == RndMode::RZ;
  bool isRoundingModeRP = getRnd() == RndMode::RP;
  bool isSatFinite = getSat() == SatMode::SATFINITE;

  bool hasRelu = getRelu();

  mlir::MLIRContext *ctx = getContext();

  return llvm::TypeSwitch<mlir::Type, LogicalResult>(getDstTy())
      .Case<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(
          [&](mlir::Type) -> LogicalResult {
            if (!isRoundingModeRN) {
              return emitOpError("Only RN rounding mode is supported for "
                                 "conversions from f32x2 to ")
                     << mlir::Float8E4M3FNType::get(ctx) << " and "
                     << mlir::Float8E5M2Type::get(ctx) << " types";
            }
            if (!isSatFinite) {
              return emitOpError("Only SATFINITE saturation mode is supported "
                                 "for conversions "
                                 "from f32x2 to ")
                     << mlir::Float8E4M3FNType::get(ctx) << " and "
                     << mlir::Float8E5M2Type::get(ctx) << " types";
            }
            return success();
          })
      .Case<mlir::Float8E8M0FNUType>([&](mlir::Type) -> LogicalResult {
        if (!(isRoundingModeRZ || isRoundingModeRP)) {
          return emitOpError("Only RZ and RP rounding modes are supported for "
                             "conversions from f32x2 to ")
                 << mlir::Float8E8M0FNUType::get(ctx) << " type";
        }
        if (hasRelu) {
          return emitOpError("relu not supported for conversions to ")
                 << mlir::Float8E8M0FNUType::get(ctx) << " type";
        }
        return success();
      })
      .Default([&](mlir::Type) {
        return emitOpError("Only ")
               << mlir::Float8E4M3FNType::get(ctx) << ", "
               << mlir::Float8E5M2Type::get(ctx) << ", and "
               << mlir::Float8E8M0FNUType::get(ctx)
               << " types are "
                  "supported for conversions from f32x2 to f8x2";
      });
}

LogicalResult ConvertF16x2ToF8x2Op::verify() {
  mlir::MLIRContext *ctx = getContext();

  if (!llvm::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(getDstTy())) {
    return emitOpError("Only ")
           << mlir::Float8E4M3FNType::get(ctx) << " and "
           << mlir::Float8E5M2Type::get(ctx)
           << " types are supported for conversions from f16x2 to f8x2.";
  }
  return success();
}

LogicalResult ConvertBF16x2ToF8x2Op::verify() {
  using RndMode = NVVM::FPRoundingMode;

  if (!llvm::isa<mlir::Float8E8M0FNUType>(getDstTy()))
    return emitOpError("Only ") << mlir::Float8E8M0FNUType::get(getContext())
                                << " type is supported for conversions from "
                                   "bf16x2 to f8x2.";

  auto rnd = getRnd();
  if (rnd != RndMode::RZ && rnd != RndMode::RP)
    return emitOpError("Only RZ and RP rounding modes are supported for "
                       "conversions from bf16x2 to f8x2.");

  return success();
}

LogicalResult ConvertF32x2ToF4x2Op::verify() {
  mlir::MLIRContext *ctx = getContext();

  if (!llvm::isa<mlir::Float4E2M1FNType>(getDstTy()))
    return emitOpError("Only ")
           << mlir::Float4E2M1FNType::get(ctx)
           << " type is supported for conversions from f32x2 to f4x2.";

  return success();
}

LogicalResult ConvertF8x2ToF16x2Op::verify() {
  mlir::MLIRContext *ctx = getContext();

  if (!llvm::isa<Float8E4M3FNType, Float8E5M2Type>(getSrcType()))
    return emitOpError("Only ")
           << mlir::Float8E4M3FNType::get(ctx) << " and "
           << mlir::Float8E5M2Type::get(ctx)
           << " types are supported for conversions from f8x2 to f16x2.";

  return success();
}

LogicalResult ConvertF8x2ToBF16x2Op::verify() {
  mlir::MLIRContext *ctx = getContext();
  if (!llvm::isa<Float8E8M0FNUType>(getSrcType()))
    return emitOpError("Only ")
           << mlir::Float8E8M0FNUType::get(ctx)
           << " type is supported for conversions from f8x2 to bf16x2.";

  return success();
}

LogicalResult ConvertF6x2ToF16x2Op::verify() {
  mlir::MLIRContext *ctx = getContext();

  if (!llvm::isa<Float6E2M3FNType, Float6E3M2FNType>(getSrcType()))
    return emitOpError("Only ")
           << mlir::Float6E2M3FNType::get(ctx) << " and "
           << mlir::Float6E3M2FNType::get(ctx)
           << " types are supported for conversions from f6x2 to f16x2.";

  return success();
}

LogicalResult ConvertF4x2ToF16x2Op::verify() {
  mlir::MLIRContext *ctx = getContext();

  if (!llvm::isa<Float4E2M1FNType>(getSrcType()))
    return emitOpError("Only ")
           << mlir::Float4E2M1FNType::get(ctx)
           << " type is supported for conversions from f4x2 to f16x2.";

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
// Stochastic Rounding Conversion Ops
//===----------------------------------------------------------------------===//

static LogicalResult verifyConvertF32x2ToFP16x2Op(Twine dstType,
                                                  FPRoundingMode rnd,
                                                  bool hasRandomBits,
                                                  Operation *op) {
  static constexpr FPRoundingMode validRndModes[] = {
      FPRoundingMode::RN, FPRoundingMode::RZ, FPRoundingMode::RS};

  if (!llvm::is_contained(validRndModes, rnd)) {
    return op->emitOpError(
               "Only RN, RZ, and RS rounding modes are supported for "
               "conversions from f32x2 to ")
           << dstType << ".";
  }

  if (rnd == FPRoundingMode::RS) {
    if (!hasRandomBits) {
      return op->emitOpError("random_bits is required for RS rounding mode.");
    }
  } else {
    if (hasRandomBits) {
      return op->emitOpError(
          "random_bits not supported for RN and RZ rounding modes.");
    }
  }

  return success();
}

LogicalResult ConvertF32x2ToF16x2Op::verify() {
  return verifyConvertF32x2ToFP16x2Op("f16x2", getRnd(),
                                      getRandomBits() ? true : false, *this);
}

LogicalResult ConvertF32x2ToBF16x2Op::verify() {
  return verifyConvertF32x2ToFP16x2Op("bf16x2", getRnd(),
                                      getRandomBits() ? true : false, *this);
}

LogicalResult ConvertF32x4ToF8x4Op::verify() {
  mlir::MLIRContext *ctx = getContext();

  if (!llvm::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(getDstTy()))
    return emitOpError("Only ")
           << mlir::Float8E4M3FNType::get(ctx) << " and "
           << mlir::Float8E5M2Type::get(ctx)
           << " types are supported for conversions from f32x4 to f8x4.";

  return success();
}

LogicalResult ConvertF32x4ToF6x4Op::verify() {
  mlir::MLIRContext *ctx = getContext();

  if (!llvm::isa<mlir::Float6E2M3FNType, mlir::Float6E3M2FNType>(getDstTy()))
    return emitOpError("Only ")
           << mlir::Float6E2M3FNType::get(ctx) << " and "
           << mlir::Float6E3M2FNType::get(ctx)
           << " types are supported for conversions from f32x4 to f6x4.";

  return success();
}

LogicalResult ConvertF32x4ToF4x4Op::verify() {
  mlir::MLIRContext *ctx = getContext();

  if (!llvm::isa<mlir::Float4E2M1FNType>(getDstTy()))
    return emitOpError("Only ") << mlir::Float4E2M1FNType::get(ctx)
                                << " type is supported for conversions from "
                                   "f32x4 to f4x4.";

  return success();
}

LogicalResult BulkStoreOp::verify() {
  if (getInitVal() != 0)
    return emitOpError("only 0 is supported for initVal, got ") << getInitVal();
  return success();
}

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

// Given the element type of an operand and whether or not it is an accumulator,
// this function returns the PTX type (`NVVM::MMATypes`) that corresponds to the
// operand's element type.
std::optional<mlir::NVVM::MMATypes>
MmaOp::inferOperandMMAType(Type operandElType, bool isAccumulator) {
  auto half2Type =
      VectorType::get(2, Float16Type::get(operandElType.getContext()));
  if (operandElType.isF64())
    return NVVM::MMATypes::f64;
  if (operandElType.isF16() || operandElType == half2Type)
    return NVVM::MMATypes::f16;
  if (operandElType.isF32() && isAccumulator)
    return NVVM::MMATypes::f32;
  if (operandElType.isF32() && !isAccumulator)
    return NVVM::MMATypes::tf32;
  if (llvm::isa<IntegerType>(operandElType)) {
    if (isAccumulator)
      return NVVM::MMATypes::s32;
    return std::nullopt;
  }

  if (auto structType = llvm::dyn_cast<LLVM::LLVMStructType>(operandElType)) {
    if (structType.getBody().empty())
      return std::nullopt;
    return inferOperandMMAType(structType.getBody()[0], isAccumulator);
  }

  return std::nullopt;
}

static bool isInt4PtxType(MMATypes type) {
  return (type == MMATypes::u4 || type == MMATypes::s4);
}

static bool isInt8PtxType(MMATypes type) {
  return (type == MMATypes::u8 || type == MMATypes::s8);
}

static bool isIntegerPtxType(MMATypes type) {
  return isInt4PtxType(type) || isInt8PtxType(type) || type == MMATypes::b1 ||
         type == MMATypes::s32;
}

MMATypes MmaOp::accumPtxType() {
  std::optional<mlir::NVVM::MMATypes> val = inferOperandMMAType(
      getODSOperands(2).getTypes().front(), /*isAccumulator=*/true);
  assert(val.has_value() && "accumulator PTX type should always be inferrable");
  return val.value();
}

MMATypes MmaOp::resultPtxType() {
  std::optional<mlir::NVVM::MMATypes> val =
      inferOperandMMAType(getResult().getType(), /*isAccumulator=*/true);
  assert(val.has_value() && "result PTX type should always be inferrable");
  return val.value();
}

void MmaOp::print(OpAsmPrinter &p) {
  SmallVector<Type, 4> regTypes;
  struct MMAOperandFragment {
    StringRef operandName;
    StringRef ptxTypeAttr;
    SmallVector<Value, 4> regs;
    explicit MMAOperandFragment(StringRef name, StringRef ptxTypeName)
        : operandName(name), ptxTypeAttr(ptxTypeName) {}
  };

  std::array<MMAOperandFragment, 3> frags{
      MMAOperandFragment("A", getMultiplicandAPtxTypeAttrName()),
      MMAOperandFragment("B", getMultiplicandBPtxTypeAttrName()),
      MMAOperandFragment("C", "")};
  SmallVector<StringRef, 4> ignoreAttrNames{
      mlir::NVVM::MmaOp::getOperandSegmentSizeAttr()};

  for (unsigned fragIdx = 0; fragIdx < frags.size(); fragIdx++) {
    auto &frag = frags[fragIdx];
    auto varOperandSpec = getODSOperandIndexAndLength(fragIdx);
    for (auto operandIdx = varOperandSpec.first;
         operandIdx < varOperandSpec.first + varOperandSpec.second;
         operandIdx++) {
      frag.regs.push_back(this->getOperand(operandIdx));
      if (operandIdx == 0) {
        regTypes.push_back(this->getOperand(operandIdx).getType());
      }
    }
    std::optional<MMATypes> inferredType = MmaOp::inferOperandMMAType(
        regTypes.back(), /*isAccumulator=*/fragIdx >= 2);
    if (inferredType)
      ignoreAttrNames.push_back(frag.ptxTypeAttr);
  }

  auto printMmaOperand = [&](const MMAOperandFragment &frag) -> void {
    p << " " << frag.operandName;
    p << "[";
    p.printOperands(frag.regs);
    p << "] ";
  };

  for (const auto &frag : frags) {
    printMmaOperand(frag);
  }

  p.printOptionalAttrDict(this->getOperation()->getAttrs(), ignoreAttrNames);

  // Print the types of the operands and result.
  p << " : "
    << "(";
  llvm::interleaveComma(SmallVector<Type, 3>{frags[0].regs[0].getType(),
                                             frags[1].regs[0].getType(),
                                             frags[2].regs[0].getType()},
                        p);
  p << ")";
  p.printArrowTypeList(TypeRange{this->getRes().getType()});
}

void MmaOp::build(OpBuilder &builder, OperationState &result, Type resultType,
                  ValueRange operandA, ValueRange operandB, ValueRange operandC,
                  ArrayRef<int64_t> shape, std::optional<MMAB1Op> b1Op,
                  std::optional<MMAIntOverflow> intOverflow,
                  std::optional<std::array<MMATypes, 2>> multiplicandPtxTypes,
                  std::optional<std::array<MMALayout, 2>> multiplicandLayouts) {

  assert(shape.size() == 3 && "expected shape to have size 3 (m, n, k)");
  MLIRContext *ctx = builder.getContext();
  result.addAttribute(
      "shape", builder.getAttr<MMAShapeAttr>(shape[0], shape[1], shape[2]));

  result.addOperands(operandA);
  result.addOperands(operandB);
  result.addOperands(operandC);

  if (multiplicandPtxTypes) {
    result.addAttribute("multiplicandAPtxType",
                        MMATypesAttr::get(ctx, (*multiplicandPtxTypes)[0]));
    result.addAttribute("multiplicandBPtxType",
                        MMATypesAttr::get(ctx, (*multiplicandPtxTypes)[1]));
  } else {
    if (auto res = inferOperandMMAType(operandA[0].getType(), false))
      result.addAttribute("multiplicandAPtxType", MMATypesAttr::get(ctx, *res));
    if (auto res = inferOperandMMAType(operandB[0].getType(), false))
      result.addAttribute("multiplicandBPtxType", MMATypesAttr::get(ctx, *res));
  }

  if (multiplicandLayouts) {
    result.addAttribute("layoutA",
                        MMALayoutAttr::get(ctx, (*multiplicandLayouts)[0]));
    result.addAttribute("layoutB",
                        MMALayoutAttr::get(ctx, (*multiplicandLayouts)[1]));
  } else {
    result.addAttribute("layoutA", MMALayoutAttr::get(ctx, MMALayout::row));
    result.addAttribute("layoutB", MMALayoutAttr::get(ctx, MMALayout::col));
  }

  if (intOverflow.has_value())
    result.addAttribute("intOverflowBehavior",
                        MMAIntOverflowAttr::get(ctx, *intOverflow));
  if (b1Op.has_value())
    result.addAttribute("b1Op", MMAB1OpAttr::get(ctx, *b1Op));

  result.addTypes(resultType);
  result.addAttribute(
      MmaOp::getOperandSegmentSizeAttr(),
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(operandA.size()),
                                    static_cast<int32_t>(operandB.size()),
                                    static_cast<int32_t>(operandC.size())}));
}

// <operation> :=
//   A `[` $operandA `]` B `[` $operandB `]` C `[` $operandC `]`
//   attr-dict : (type($operandA[0]), type($operandB[0]), type($operandC[0]))
//     `->` type($res)
ParseResult MmaOp::parse(OpAsmParser &parser, OperationState &result) {
  struct MMAOperandFragment {
    std::optional<MMATypes> elemtype;
    SmallVector<OpAsmParser::UnresolvedOperand, 4> regs;
    SmallVector<Type> regTypes;
  };

  Builder &builder = parser.getBuilder();
  std::array<MMAOperandFragment, 4> frags;

  NamedAttrList namedAttributes;

  // A helper to parse the operand segments.
  auto parseMmaOperand = [&](StringRef operandName,
                             MMAOperandFragment &frag) -> LogicalResult {
    if (parser.parseKeyword(operandName).failed())
      return failure();
    if (parser
            .parseOperandList(frag.regs, OpAsmParser::Delimiter::OptionalSquare)
            .failed())
      return failure();
    return success();
  };

  // Parse the operand segments.
  if (parseMmaOperand("A", frags[0]).failed())
    return failure();
  if (parseMmaOperand("B", frags[1]).failed())
    return failure();
  if (parseMmaOperand("C", frags[2]).failed())
    return failure();

  if (parser.parseOptionalAttrDict(namedAttributes).failed())
    return failure();

  // Parse the type specification and resolve operands.
  SmallVector<Type, 3> operandTypes;
  if (failed(parser.parseColon()))
    return failure();
  if (failed(parser.parseLParen()))
    return failure();
  if (failed(parser.parseTypeList(operandTypes)))
    return failure();
  if (failed(parser.parseRParen()))
    if (operandTypes.size() != 3)
      return parser.emitError(
          parser.getNameLoc(),
          "expected one type for each operand segment but got " +
              Twine(operandTypes.size()) + " types");
  for (const auto &iter : llvm::enumerate(operandTypes)) {
    auto &frag = frags[iter.index()];
    frag.regTypes.resize(frag.regs.size(), iter.value());
    if (failed(parser.resolveOperands(frag.regs, frag.regTypes,
                                      parser.getNameLoc(), result.operands)))
      return failure();
    frag.elemtype = inferOperandMMAType(frag.regTypes[0],
                                        /*isAccumulator*/ iter.index() < 2);
  }

  Type resultType;
  if (parser.parseArrow() || parser.parseType(resultType))
    return failure();
  frags[3].elemtype = inferOperandMMAType(resultType, /*isAccumulator*/ true);

  std::array<StringRef, 2> names{"multiplicandAPtxType",
                                 "multiplicandBPtxType"};
  for (unsigned idx = 0; idx < names.size(); idx++) {
    const auto &frag = frags[idx];
    std::optional<NamedAttribute> attr = namedAttributes.getNamed(names[idx]);
    if (!frag.elemtype.has_value() && !attr.has_value()) {
      return parser.emitError(
          parser.getNameLoc(),
          "attribute " + names[idx] +
              " is not provided explicitly and cannot be inferred");
    }
    if (!attr.has_value())
      result.addAttribute(
          names[idx], MMATypesAttr::get(parser.getContext(), *frag.elemtype));
  }

  result.addTypes(resultType);
  if (!namedAttributes.empty())
    result.addAttributes(namedAttributes);
  result.addAttribute(MmaOp::getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr({
                          static_cast<int32_t>(frags[0].regs.size()),
                          static_cast<int32_t>(frags[1].regs.size()),
                          static_cast<int32_t>(frags[2].regs.size()),
                      }));
  return success();
}

LogicalResult MmaOp::verify() {
  MLIRContext *context = getContext();
  auto f16Ty = Float16Type::get(context);
  auto i32Ty = IntegerType::get(context, 32);
  auto f16x2Ty = VectorType::get(2, f16Ty);
  auto f32Ty = Float32Type::get(context);
  auto f16x2x4StructTy = LLVM::LLVMStructType::getLiteral(
      context, {f16x2Ty, f16x2Ty, f16x2Ty, f16x2Ty});

  auto s32x4StructTy =
      LLVM::LLVMStructType::getLiteral(context, {i32Ty, i32Ty, i32Ty, i32Ty});
  auto f32x8StructTy =
      LLVM::LLVMStructType::getLiteral(context, SmallVector<Type>(8, f32Ty));
  auto f16x2x2StructTy =
      LLVM::LLVMStructType::getLiteral(context, {f16x2Ty, f16x2Ty});
  auto f32x4StructTy =
      LLVM::LLVMStructType::getLiteral(context, {f32Ty, f32Ty, f32Ty, f32Ty});
  auto s32x2StructTy =
      LLVM::LLVMStructType::getLiteral(context, {i32Ty, i32Ty});

  std::array<int64_t, 3> mmaShape{getShapeAttr().getM(), getShapeAttr().getN(),
                                  getShapeAttr().getK()};

  // These variables define the set of allowed data types for matrices A, B, C,
  // and result.
  using AllowedShapes = SmallVector<std::array<int64_t, 3>, 2>;
  using AllowedTypes = SmallVector<SmallVector<Type, 4>, 2>;
  AllowedShapes allowedShapes;
  AllowedTypes expectedA;
  AllowedTypes expectedB;
  AllowedTypes expectedC;
  SmallVector<Type> expectedResult;

  // When M = 16, we just need to calculate the number of 8xk tiles, where
  // k is a factor that depends on the data type.
  if (mmaShape[0] == 16) {
    int64_t kFactor;
    Type multiplicandFragType;
    switch (*getMultiplicandAPtxType()) {
    case MMATypes::tf32:
      kFactor = 4;
      multiplicandFragType = i32Ty;
      expectedResult.push_back(LLVM::LLVMStructType::getLiteral(
          context, {f32Ty, f32Ty, f32Ty, f32Ty}));
      break;
    case MMATypes::bf16:
      kFactor = 8;
      multiplicandFragType = i32Ty;
      expectedResult.push_back(LLVM::LLVMStructType::getLiteral(
          context, {f32Ty, f32Ty, f32Ty, f32Ty}));
      break;
    case MMATypes::f16:
      kFactor = 8;
      multiplicandFragType = f16x2Ty;
      expectedResult.push_back(f16x2x2StructTy);
      expectedResult.push_back(f32x4StructTy);
      break;
    case MMATypes::s4:
    case MMATypes::u4:
      kFactor = 32;
      break;
    case MMATypes::b1:
      kFactor = 128;
      break;
    case MMATypes::s8:
    case MMATypes::u8:
      kFactor = 16;
      break;
    default:
      return emitError("invalid shape or multiplicand type: ")
             << getMultiplicandAPtxType().value();
    }

    if (isIntegerPtxType(getMultiplicandAPtxType().value())) {
      expectedResult.push_back(s32x4StructTy);
      expectedC.emplace_back(4, i32Ty);
      multiplicandFragType = i32Ty;
    } else {
      expectedC.emplace_back(2, f16x2Ty);
      expectedC.emplace_back(4, f32Ty);
    }

    int64_t unitA = (mmaShape[0] / 8) * (mmaShape[2] / kFactor);
    int64_t unitB = (mmaShape[1] / 8) * (mmaShape[2] / kFactor);
    expectedA.emplace_back(unitA, multiplicandFragType);
    expectedB.emplace_back(unitB, multiplicandFragType);
    allowedShapes.push_back({16, 8, kFactor});
    allowedShapes.push_back({16, 8, kFactor * 2});

    if (resultPtxType() != accumPtxType())
      return emitOpError("ctype does not match dtype");
  }

  // In the M=8 case, there is only 1 possible case per data type.
  if (mmaShape[0] == 8) {
    if (*getMultiplicandAPtxType() == MMATypes::f16) {
      expectedA.emplace_back(2, f16x2Ty);
      expectedB.emplace_back(2, f16x2Ty);
      expectedResult.push_back(f16x2x4StructTy);
      expectedResult.push_back(f32x8StructTy);
      expectedC.emplace_back(4, f16x2Ty);
      expectedC.emplace_back(8, f32Ty);
      allowedShapes.push_back({8, 8, 4});
    }
    if (*getMultiplicandAPtxType() == MMATypes::f64) {
      Type f64Ty = Float64Type::get(context);
      expectedA.emplace_back(1, f64Ty);
      expectedB.emplace_back(1, f64Ty);
      expectedC.emplace_back(2, f64Ty);
      expectedResult.emplace_back(LLVM::LLVMStructType::getLiteral(
          context, SmallVector<Type>(2, f64Ty)));
      allowedShapes.push_back({8, 8, 4});
    }
    if (isIntegerPtxType(getMultiplicandAPtxType().value())) {
      expectedA.push_back({i32Ty});
      expectedB.push_back({i32Ty});
      expectedC.push_back({i32Ty, i32Ty});
      expectedResult.push_back(s32x2StructTy);
      if (isInt4PtxType(getMultiplicandAPtxType().value()))
        allowedShapes.push_back({8, 8, 32});
      if (isInt8PtxType(getMultiplicandAPtxType().value()))
        allowedShapes.push_back({8, 8, 16});
      if (getMultiplicandAPtxType().value() == MMATypes::b1)
        allowedShapes.push_back({8, 8, 128});
    }
  }

  std::string errorMessage;
  llvm::raw_string_ostream errorStream(errorMessage);

  // Check that we matched an existing shape/dtype combination.
  if (expectedA.empty() || expectedB.empty() || expectedC.empty() ||
      !llvm::is_contained(allowedShapes, mmaShape)) {
    errorStream << "unimplemented variant for MMA shape <";
    llvm::interleaveComma(mmaShape, errorStream);
    errorStream << ">";
    return emitOpError(errorMessage);
  }

  // Verify the operand types for segments of A, B, and C operands.
  std::array<StringRef, 3> operandNames{"A", "B", "C"};
  for (const auto &iter : llvm::enumerate(
           SmallVector<AllowedTypes, 3>{expectedA, expectedB, expectedC})) {
    auto spec = this->getODSOperandIndexAndLength(iter.index());
    SmallVector<Type, 4> operandTySeg(operand_type_begin() + spec.first,
                                      operand_type_begin() + spec.first +
                                          spec.second);
    bool match = llvm::is_contained(iter.value(), operandTySeg);

    if (!match) {
      errorStream << "Could not match types for the "
                  << operandNames[iter.index()]
                  << " operands; expected one of ";
      for (const auto &x : iter.value()) {
        errorStream << x.size() << "x" << x[0] << " ";
      }
      errorStream << "but got ";
      llvm::interleaveComma(operandTySeg, errorStream);
      return emitOpError(errorMessage);
    }
  }

  // Check the result type
  if (!llvm::any_of(expectedResult, [&](Type expectedResultType) {
        return expectedResultType == getResult().getType();
      })) {
    errorStream
        << "Could not match allowed types for the result; expected one of ";
    llvm::interleaveComma(expectedResult, errorStream);
    errorStream << " but got " << getResult().getType();
    return emitOpError(errorMessage);
  }

  // Ensure that binary MMA variants have a b1 MMA operation defined.
  if (getMultiplicandAPtxType() == MMATypes::b1 && !getB1Op()) {
    return emitOpError("op requires " + getB1OpAttrName().strref() +
                       " attribute");
  }

  // Ensure int4/int8 MMA variants specify the accum overflow behavior
  // attribute.
  if (isInt4PtxType(*getMultiplicandAPtxType()) ||
      isInt8PtxType(*getMultiplicandAPtxType())) {
    if (!getIntOverflowBehavior())
      return emitOpError("op requires " +
                         getIntOverflowBehaviorAttrName().strref() +
                         " attribute");
  }

  // Validate layout combinations. According to the operation description, most
  // MMA operations require layoutA=row and layoutB=col. Only m8n8k4 with f16
  // can use other layout combinations.
  bool isM8N8K4_F16 =
      (mmaShape[0] == 8 && mmaShape[1] == 8 && mmaShape[2] == 4 &&
       getMultiplicandAPtxType() == MMATypes::f16);

  if (!isM8N8K4_F16) {
    // For all other shapes/types, layoutA must be row and layoutB must be col
    if (getLayoutA() != MMALayout::row || getLayoutB() != MMALayout::col) {
      return emitOpError("requires layoutA = #nvvm.mma_layout<row> and "
                         "layoutB = #nvvm.mma_layout<col> for shape <")
             << mmaShape[0] << ", " << mmaShape[1] << ", " << mmaShape[2]
             << "> with element types " << *getMultiplicandAPtxType() << " and "
             << *getMultiplicandBPtxType()
             << ". Only m8n8k4 with f16 supports other layouts.";
    }
  }

  return success();
}

MMATypes MmaSpOp::accumPtxType() {
  std::optional<mlir::NVVM::MMATypes> val = MmaOp::inferOperandMMAType(
      getODSOperands(2).getTypes().front(), /*isAccumulator=*/true);
  assert(val.has_value() && "accumulator PTX type should always be inferrable");
  return val.value();
}

MMATypes MmaSpOp::resultPtxType() {
  std::optional<mlir::NVVM::MMATypes> val =
      MmaOp::inferOperandMMAType(getResult().getType(), /*isAccumulator=*/true);
  assert(val.has_value() && "result PTX type should always be inferrable");
  return val.value();
}

mlir::NVVM::IDArgPair
MmaSpOp::getIntrinsicIDAndArgs(Operation &op, LLVM::ModuleTranslation &mt,
                               llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::MmaSpOp>(op);

  // Get operands
  llvm::SmallVector<llvm::Value *> args;
  for (mlir::Value v : thisOp.getOperands())
    args.push_back(mt.lookupValue(v));

  // Get intrinsic ID using the existing getIntrinsicID method
  auto intId = MmaSpOp::getIntrinsicID(
      thisOp.getShape().getM(), thisOp.getShape().getN(),
      thisOp.getShape().getK(), thisOp.getIntOverflowBehavior(),
      thisOp.getOrderedMetadata(), thisOp.getKind(),
      *thisOp.getMultiplicandAPtxType(), *thisOp.getMultiplicandBPtxType(),
      thisOp.accumPtxType(), thisOp.resultPtxType());

  return {intId, args};
}

void MmaSpOp::print(OpAsmPrinter &p) {
  SmallVector<Type, 4> regTypes;
  struct MMAOperandFragment {
    StringRef operandName;
    StringRef ptxTypeAttr;
    SmallVector<Value, 4> regs;
    explicit MMAOperandFragment(StringRef name, StringRef ptxTypeName)
        : operandName(name), ptxTypeAttr(ptxTypeName) {}
  };

  std::array<MMAOperandFragment, 5> frags{
      MMAOperandFragment("A", getMultiplicandAPtxTypeAttrName()),
      MMAOperandFragment("B", getMultiplicandBPtxTypeAttrName()),
      MMAOperandFragment("C", ""), MMAOperandFragment("sparseMetadata", ""),
      MMAOperandFragment("selector", "")};
  SmallVector<StringRef, 4> ignoreAttrNames{
      mlir::NVVM::MmaSpOp::getOperandSegmentSizeAttr()};

  // Handle variadic operands A, B, C
  for (unsigned fragIdx = 0; fragIdx < 3; fragIdx++) {
    auto &frag = frags[fragIdx];
    auto varOperandSpec = getODSOperandIndexAndLength(fragIdx);
    for (auto operandIdx = varOperandSpec.first;
         operandIdx < varOperandSpec.first + varOperandSpec.second;
         operandIdx++) {
      frag.regs.push_back(this->getOperand(operandIdx));
      if (operandIdx == varOperandSpec.first) {
        regTypes.push_back(this->getOperand(operandIdx).getType());
      }
    }
    std::optional<MMATypes> inferredType = MmaOp::inferOperandMMAType(
        regTypes.back(), /*isAccumulator=*/fragIdx >= 2);
    if (inferredType)
      ignoreAttrNames.push_back(frag.ptxTypeAttr);
  }

  // Handle sparse metadata and selector (single operands)
  frags[3].regs.push_back(getSparseMetadata());
  frags[4].regs.push_back(getSparsitySelector());

  auto printMmaSpOperand = [&](const MMAOperandFragment &frag) -> void {
    p << " " << frag.operandName;
    p << "[";
    p.printOperands(frag.regs);
    p << "]";
  };

  for (const auto &frag : frags)
    printMmaSpOperand(frag);

  p.printOptionalAttrDict((*this)->getAttrs(), ignoreAttrNames);
  p << " : ";
  p << "(";
  for (int i = 0; i < 3; ++i) {
    p << regTypes[i];
    if (i < 2)
      p << ", ";
  }
  p << ") -> " << getResult().getType();
}

void MmaSpOp::build(
    OpBuilder &builder, OperationState &result, Type resultType,
    ValueRange operandA, ValueRange operandB, ValueRange operandC,
    Value sparseMetadata, Value sparsitySelector, ArrayRef<int64_t> shape,
    std::optional<MMAIntOverflow> intOverflow,
    std::optional<std::array<MMATypes, 2>> multiplicandPtxTypes) {

  assert(shape.size() == 3 && "expected shape to have size 3 (m, n, k)");
  MLIRContext *ctx = builder.getContext();
  result.addAttribute(
      "shape", builder.getAttr<MMAShapeAttr>(shape[0], shape[1], shape[2]));

  result.addOperands(operandA);
  result.addOperands(operandB);
  result.addOperands(operandC);
  result.addOperands(sparseMetadata);
  result.addOperands(sparsitySelector);

  if (multiplicandPtxTypes) {
    result.addAttribute("multiplicandAPtxType",
                        MMATypesAttr::get(ctx, (*multiplicandPtxTypes)[0]));
    result.addAttribute("multiplicandBPtxType",
                        MMATypesAttr::get(ctx, (*multiplicandPtxTypes)[1]));
  } else {
    if (auto res = MmaOp::inferOperandMMAType(operandA[0].getType(), false))
      result.addAttribute("multiplicandAPtxType", MMATypesAttr::get(ctx, *res));
    if (auto res = MmaOp::inferOperandMMAType(operandB[0].getType(), false))
      result.addAttribute("multiplicandBPtxType", MMATypesAttr::get(ctx, *res));
  }

  if (intOverflow.has_value())
    result.addAttribute("intOverflowBehavior",
                        MMAIntOverflowAttr::get(ctx, *intOverflow));

  result.addTypes(resultType);
  result.addAttribute(
      MmaSpOp::getOperandSegmentSizeAttr(),
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(operandA.size()),
                                    static_cast<int32_t>(operandB.size()),
                                    static_cast<int32_t>(operandC.size()), 1,
                                    1})); // sparseMetadata and sparsitySelector
}

ParseResult MmaSpOp::parse(OpAsmParser &parser, OperationState &result) {
  struct MMAOperandFragment {
    std::optional<MMATypes> elemtype;
    SmallVector<OpAsmParser::UnresolvedOperand, 4> regs;
    SmallVector<Type> regTypes;
  };

  Builder &builder = parser.getBuilder();
  std::array<MMAOperandFragment, 6> frags; // A, B, C, sparseMetadata, selector

  NamedAttrList namedAttributes;

  // A helper to parse the operand segments.
  auto parseMmaSpOperand = [&](StringRef operandName,
                               MMAOperandFragment &frag) -> LogicalResult {
    if (parser.parseKeyword(operandName).failed())
      return failure();
    if (parser
            .parseOperandList(frag.regs, OpAsmParser::Delimiter::OptionalSquare)
            .failed())
      return failure();
    return success();
  };

  // Parse the operand segments.
  if (parseMmaSpOperand("A", frags[0]).failed())
    return failure();
  if (parseMmaSpOperand("B", frags[1]).failed())
    return failure();
  if (parseMmaSpOperand("C", frags[2]).failed())
    return failure();
  if (parseMmaSpOperand("sparseMetadata", frags[3]).failed())
    return failure();
  if (parseMmaSpOperand("selector", frags[4]).failed())
    return failure();

  if (parser.parseOptionalAttrDict(namedAttributes).failed())
    return failure();

  // Parse the type specification and resolve operands.
  SmallVector<Type, 3> operandTypes;
  if (failed(parser.parseColon()))
    return failure();
  if (failed(parser.parseLParen()))
    return failure();
  if (failed(parser.parseTypeList(operandTypes)))
    return failure();
  if (failed(parser.parseRParen()))
    return failure();
  if (operandTypes.size() != 3)
    return parser.emitError(
        parser.getNameLoc(),
        "expected one type for each operand segment but got " +
            Twine(operandTypes.size()) + " types");
  for (const auto &iter : llvm::enumerate(operandTypes)) {
    auto &frag = frags[iter.index()];
    frag.regTypes.resize(frag.regs.size(), iter.value());
    if (failed(parser.resolveOperands(frag.regs, frag.regTypes,
                                      parser.getNameLoc(), result.operands)))
      return failure();
    frag.elemtype =
        MmaOp::inferOperandMMAType(frag.regTypes[0],
                                   /*isAccumulator*/ iter.index() >= 2);
  }

  Type resultType;
  if (parser.parseArrow() || parser.parseType(resultType))
    return failure();
  frags[5].elemtype =
      MmaOp::inferOperandMMAType(resultType, /*isAccumulator*/ true);

  // Resolve sparse metadata and selector (assume i32 type)
  Type i32Type = builder.getIntegerType(32);
  if (parser
          .resolveOperands(frags[3].regs, i32Type, parser.getCurrentLocation(),
                           result.operands)
          .failed())
    return failure();
  if (parser
          .resolveOperands(frags[4].regs, i32Type, parser.getCurrentLocation(),
                           result.operands)
          .failed())
    return failure();

  std::array<StringRef, 2> names{"multiplicandAPtxType",
                                 "multiplicandBPtxType"};
  for (unsigned idx = 0; idx < names.size(); idx++) {
    const auto &frag = frags[idx];
    std::optional<NamedAttribute> attr = namedAttributes.getNamed(names[idx]);
    if (!frag.elemtype.has_value() && !attr.has_value()) {
      return parser.emitError(
          parser.getNameLoc(),
          "attribute " + names[idx] +
              " is not provided explicitly and cannot be inferred");
    }
    if (!attr.has_value())
      result.addAttribute(
          names[idx], MMATypesAttr::get(parser.getContext(), *frag.elemtype));
  }

  result.addTypes(resultType);
  if (!namedAttributes.empty())
    result.addAttributes(namedAttributes);
  result.addAttribute(MmaSpOp::getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr({
                          static_cast<int32_t>(frags[0].regs.size()),
                          static_cast<int32_t>(frags[1].regs.size()),
                          static_cast<int32_t>(frags[2].regs.size()),
                          1, // sparseMetadata
                          1  // sparsitySelector
                      }));
  return success();
}

LogicalResult MmaSpOp::verify() {
  MLIRContext *context = getContext();
  auto f16Ty = Float16Type::get(context);
  auto i32Ty = IntegerType::get(context, 32);
  auto f16x2Ty = VectorType::get(2, f16Ty);
  auto f32Ty = Float32Type::get(context);
  auto f16x2x4StructTy = LLVM::LLVMStructType::getLiteral(
      context, {f16x2Ty, f16x2Ty, f16x2Ty, f16x2Ty});

  auto s32x4StructTy =
      LLVM::LLVMStructType::getLiteral(context, {i32Ty, i32Ty, i32Ty, i32Ty});
  auto f32x8StructTy =
      LLVM::LLVMStructType::getLiteral(context, SmallVector<Type>(8, f32Ty));
  auto f16x2x2StructTy =
      LLVM::LLVMStructType::getLiteral(context, {f16x2Ty, f16x2Ty});
  auto f32x4StructTy =
      LLVM::LLVMStructType::getLiteral(context, {f32Ty, f32Ty, f32Ty, f32Ty});
  auto s32x2StructTy =
      LLVM::LLVMStructType::getLiteral(context, {i32Ty, i32Ty});

  std::array<int64_t, 3> mmaShape{getShapeAttr().getM(), getShapeAttr().getN(),
                                  getShapeAttr().getK()};

  // These variables define the set of allowed data types for matrices A, B, C,
  // and result.
  using AllowedShapes = SmallVector<std::array<int64_t, 3>, 2>;
  using AllowedTypes = SmallVector<SmallVector<Type, 4>, 2>;
  AllowedShapes allowedShapes;
  AllowedTypes expectedA;
  AllowedTypes expectedB;
  AllowedTypes expectedC;
  SmallVector<Type> expectedResult;

  // When M = 16, we just need to calculate the number of 8xk tiles, where
  // k is a factor that depends on the data type.
  if (mmaShape[0] == 16) {
    int64_t kFactor;
    Type multiplicandFragType;
    switch (*getMultiplicandAPtxType()) {
    case MMATypes::tf32:
      kFactor = 4;
      multiplicandFragType = i32Ty;
      expectedResult.push_back(LLVM::LLVMStructType::getLiteral(
          context, {f32Ty, f32Ty, f32Ty, f32Ty}));
      // Sparse MMA supports m16n8k8 and m16n8k16 for tf32
      allowedShapes.push_back({16, 8, 8});
      allowedShapes.push_back({16, 8, 16});
      break;
    case MMATypes::bf16:
      kFactor = 8;
      multiplicandFragType = i32Ty;
      expectedResult.push_back(LLVM::LLVMStructType::getLiteral(
          context, {f32Ty, f32Ty, f32Ty, f32Ty}));
      // Sparse MMA supports m16n8k16 and m16n8k32 for bf16
      allowedShapes.push_back({16, 8, 16});
      allowedShapes.push_back({16, 8, 32});
      break;
    case MMATypes::f16:
      kFactor = 8;
      multiplicandFragType = f16x2Ty;
      expectedResult.push_back(f16x2x2StructTy);
      expectedResult.push_back(f32x4StructTy);
      // Sparse MMA supports m16n8k16 and m16n8k32 for f16
      allowedShapes.push_back({16, 8, 16});
      allowedShapes.push_back({16, 8, 32});
      break;
    case MMATypes::s4:
    case MMATypes::u4:
      kFactor = 32;
      // Sparse MMA supports m16n8k64 and m16n8k128 for s4/u4
      allowedShapes.push_back({16, 8, 64});
      allowedShapes.push_back({16, 8, 128});
      break;
    case MMATypes::s8:
    case MMATypes::u8:
      kFactor = 16;
      // Sparse MMA supports m16n8k32 and m16n8k64 for s8/u8
      allowedShapes.push_back({16, 8, 32});
      allowedShapes.push_back({16, 8, 64});
      break;
    case MMATypes::e4m3:
    case MMATypes::e5m2:
    case MMATypes::e3m2:
    case MMATypes::e2m3:
    case MMATypes::e2m1:
      kFactor = 32;
      multiplicandFragType = i32Ty;
      expectedResult.push_back(f16x2x2StructTy);
      expectedResult.push_back(f32x4StructTy);
      // Sparse MMA supports m16n8k64 for FP8 types
      allowedShapes.push_back({16, 8, 64});
      break;
    default:
      return emitError("invalid shape or multiplicand type: ")
             << getMultiplicandAPtxType().value();
    }

    if (isIntegerPtxType(getMultiplicandAPtxType().value())) {
      expectedResult.push_back(s32x4StructTy);
      expectedC.emplace_back(4, i32Ty);
      multiplicandFragType = i32Ty;
    } else if (*getMultiplicandAPtxType() >= MMATypes::e4m3 &&
               *getMultiplicandAPtxType() <= MMATypes::e2m1) {
      // FP8 types
      expectedC.emplace_back(2, f16x2Ty);
      expectedC.emplace_back(4, f32Ty);
    } else {
      expectedC.emplace_back(2, f16x2Ty);
      expectedC.emplace_back(4, f32Ty);
    }

    // For sparse MMA, A operand is compressed (2:4 sparsity means half the
    // elements)
    int64_t unitA = (mmaShape[0] / 8) * (mmaShape[2] / kFactor) / 2;
    int64_t unitB = (mmaShape[1] / 8) * (mmaShape[2] / kFactor);
    expectedA.emplace_back(unitA, multiplicandFragType);
    expectedB.emplace_back(unitB, multiplicandFragType);

    if (resultPtxType() != accumPtxType())
      return emitOpError("ctype does not match dtype");
  }

  // In the M=8 case, there is only 1 possible case per data type.
  if (mmaShape[0] == 8) {
    if (*getMultiplicandAPtxType() == MMATypes::f16) {
      expectedA.emplace_back(2, f16x2Ty);
      expectedB.emplace_back(2, f16x2Ty);
      expectedResult.push_back(f16x2x4StructTy);
      expectedResult.push_back(f32x8StructTy);
      expectedC.emplace_back(4, f16x2Ty);
      expectedC.emplace_back(8, f32Ty);
      allowedShapes.push_back({8, 8, 4});
    }
    if (*getMultiplicandAPtxType() == MMATypes::f64) {
      Type f64Ty = Float64Type::get(context);
      expectedA.emplace_back(1, f64Ty);
      expectedB.emplace_back(1, f64Ty);
      expectedC.emplace_back(2, f64Ty);
      expectedResult.emplace_back(LLVM::LLVMStructType::getLiteral(
          context, SmallVector<Type>(2, f64Ty)));
      allowedShapes.push_back({8, 8, 4});
    }
    if (isIntegerPtxType(getMultiplicandAPtxType().value())) {
      expectedA.push_back({i32Ty});
      expectedB.push_back({i32Ty});
      expectedC.push_back({i32Ty, i32Ty});
      expectedResult.push_back(s32x2StructTy);
      if (isInt4PtxType(getMultiplicandAPtxType().value()))
        allowedShapes.push_back({8, 8, 32});
      if (isInt8PtxType(getMultiplicandAPtxType().value()))
        allowedShapes.push_back({8, 8, 16});
    }
  }

  std::string errorMessage;
  llvm::raw_string_ostream errorStream(errorMessage);

  // Check that we matched an existing shape/dtype combination.
  if (expectedA.empty() || expectedB.empty() || expectedC.empty() ||
      !llvm::is_contained(allowedShapes, mmaShape)) {
    errorStream << "unimplemented variant for MMA shape <";
    llvm::interleaveComma(mmaShape, errorStream);
    errorStream << ">";
    return emitOpError(errorMessage);
  }

  // Verify the operand types for segments of A, B, and C operands.
  std::array<StringRef, 3> operandNames{"A", "B", "C"};
  for (const auto &iter : llvm::enumerate(
           SmallVector<AllowedTypes, 3>{expectedA, expectedB, expectedC})) {
    auto spec = this->getODSOperandIndexAndLength(iter.index());
    SmallVector<Type, 4> operandTySeg(operand_type_begin() + spec.first,
                                      operand_type_begin() + spec.first +
                                          spec.second);
    bool match = llvm::is_contained(iter.value(), operandTySeg);

    if (!match) {
      errorStream << "Could not match types for the "
                  << operandNames[iter.index()]
                  << " operands; expected one of ";
      for (const auto &x : iter.value()) {
        errorStream << x.size() << "x" << x[0] << " ";
      }
      errorStream << "but got ";
      llvm::interleaveComma(operandTySeg, errorStream);
      return emitOpError(errorMessage);
    }
  }

  // Check the result type
  if (!llvm::any_of(expectedResult, [&](Type expectedResultType) {
        return expectedResultType == getResult().getType();
      })) {
    errorStream
        << "Could not match allowed types for the result; expected one of ";
    llvm::interleaveComma(expectedResult, errorStream);
    errorStream << " but got " << getResult().getType();
    return emitOpError(errorMessage);
  }

  // Ensure int4/int8 MMA variants specify the accum overflow behavior
  // attribute.
  if (isInt4PtxType(*getMultiplicandAPtxType()) ||
      isInt8PtxType(*getMultiplicandAPtxType())) {
    if (!getIntOverflowBehavior())
      return emitOpError("op requires " +
                         getIntOverflowBehaviorAttrName().strref() +
                         " attribute");
  }

  // Validate sparse metadata type (should be i32)
  if (!getSparseMetadata().getType().isInteger(32)) {
    return emitOpError() << "sparse metadata must be i32 type";
  }

  // Validate sparsity selector type (should be i32)
  if (!getSparsitySelector().getType().isInteger(32)) {
    return emitOpError() << "sparsity selector must be i32 type";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MMA Block Scale Operations - Shared Helpers
//===----------------------------------------------------------------------===//

namespace {
// Shared structure for MMA operand fragments (A, B, C)
struct MMAOperandFragment {
  StringRef operandName;
  StringRef ptxTypeAttr;
  SmallVector<Value, 4> regs;
  explicit MMAOperandFragment(StringRef name, StringRef ptxTypeName)
      : operandName(name), ptxTypeAttr(ptxTypeName) {}
};
} // namespace

// Helper to print operand list in the format: name[operands]
static void printOperandList(OpAsmPrinter &p, StringRef name,
                             ArrayRef<Value> operands) {
  p << " " << name << "[";
  p.printOperands(operands);
  p << "]";
}

// Helper to parse operand list in the format: name[operands]
static LogicalResult
parseMmaOperand(OpAsmParser &parser, StringRef operandName,
                SmallVectorImpl<OpAsmParser::UnresolvedOperand> &regs) {
  if (parser.parseKeyword(operandName).failed())
    return failure();
  if (parser.parseOperandList(regs, OpAsmParser::Delimiter::OptionalSquare)
          .failed())
    return failure();
  return success();
}

// Helper to process operand fragments and determine which attributes can be
// inferred
template <typename Op>
static void
processOperandFragments(Op &op, std::array<MMAOperandFragment, 3> &frags,
                        SmallVectorImpl<Type> &regTypes,
                        SmallVectorImpl<StringRef> &ignoreAttrNames) {
  for (unsigned fragIdx = 0; fragIdx < frags.size(); fragIdx++) {
    auto &frag = frags[fragIdx];
    auto varOperandSpec = op.getODSOperandIndexAndLength(fragIdx);
    for (auto operandIdx = varOperandSpec.first;
         operandIdx < varOperandSpec.first + varOperandSpec.second;
         operandIdx++) {
      frag.regs.push_back(op.getOperand(operandIdx));
      if (fragIdx == 0 && operandIdx == varOperandSpec.first) {
        regTypes.push_back(op.getOperand(operandIdx).getType());
      }
    }
    if (fragIdx < 2) {
      regTypes.push_back(frag.regs[0].getType());
    }
    std::optional<MMATypes> inferredType =
        MmaOp::inferOperandMMAType(regTypes.back(),
                                   /*isAccumulator=*/fragIdx >= 2);
    if (inferredType)
      ignoreAttrNames.push_back(frag.ptxTypeAttr);
  }
}

// Helper to parse type signature: (A_type, B_type, C_type)
static LogicalResult
parseMmaTypeSignature(OpAsmParser &parser,
                      SmallVectorImpl<Type> &operandTypes) {
  if (parser.parseColon().failed() || parser.parseLParen().failed())
    return failure();

  auto typeParser = [&]() {
    Type ty;
    if (parser.parseType(ty).failed())
      return failure();
    operandTypes.push_back(ty);
    return success();
  };
  if (parser.parseCommaSeparatedList(typeParser))
    return failure();

  if (operandTypes.size() != 3)
    return parser.emitError(parser.getCurrentLocation(),
                            "expected exactly 3 types");

  return parser.parseRParen();
}

// Helper to infer and set multiplicand PTX type attributes
static void
inferAndSetMultiplicandTypes(MLIRContext *ctx, NamedAttrList &attrs,
                             const SmallVectorImpl<Type> &operandTypes) {
  if (!attrs.get("multiplicandAPtxType")) {
    if (auto inferredType =
            MmaOp::inferOperandMMAType(operandTypes[0], false)) {
      attrs.set("multiplicandAPtxType", MMATypesAttr::get(ctx, *inferredType));
    }
  }
  if (!attrs.get("multiplicandBPtxType")) {
    if (auto inferredType =
            MmaOp::inferOperandMMAType(operandTypes[1], false)) {
      attrs.set("multiplicandBPtxType", MMATypesAttr::get(ctx, *inferredType));
    }
  }
}

// Helper to add common block scale properties
template <typename OpType>
static void addBlockScaleProperties(OpBuilder &builder, OperationState &result,
                                    ArrayRef<int64_t> shape,
                                    ScaleVecSize scaleVecSize,
                                    BlockScaleFormat blockScaleFormat,
                                    MMABlockScaleKind kind) {
  MLIRContext *ctx = builder.getContext();
  auto &properties = result.getOrAddProperties<typename OpType::Properties>();
  properties.setShape(
      builder.getAttr<MMAShapeAttr>(shape[0], shape[1], shape[2]));
  properties.setScaleVecSize(ScaleVecSizeAttr::get(ctx, scaleVecSize));
  properties.setBlockScaleFormat(
      BlockScaleFormatAttr::get(ctx, blockScaleFormat));
  properties.setKind(MMABlockScaleKindAttr::get(ctx, kind));
}

// Helper to infer and add multiplicand PTX types to builder
static void addInferredMultiplicandTypes(
    MLIRContext *ctx, OperationState &result, ValueRange operandA,
    ValueRange operandB,
    std::optional<std::array<MMATypes, 2>> multiplicandPtxTypes) {
  if (multiplicandPtxTypes) {
    result.addAttribute("multiplicandAPtxType",
                        MMATypesAttr::get(ctx, (*multiplicandPtxTypes)[0]));
    result.addAttribute("multiplicandBPtxType",
                        MMATypesAttr::get(ctx, (*multiplicandPtxTypes)[1]));
  } else {
    if (auto res = MmaOp::inferOperandMMAType(operandA[0].getType(), false))
      result.addAttribute("multiplicandAPtxType", MMATypesAttr::get(ctx, *res));
    if (auto res = MmaOp::inferOperandMMAType(operandB[0].getType(), false))
      result.addAttribute("multiplicandBPtxType", MMATypesAttr::get(ctx, *res));
  }
}

// Template helper for common accumPtxType/resultPtxType implementation
template <typename OpTy>
static MMATypes inferPtxTypeFromResult(OpTy op) {
  return *MmaOp::inferOperandMMAType(
      cast<LLVM::LLVMStructType>(op.getRes().getType()).getBody()[0],
      /*isAccumulator=*/true);
}

//===----------------------------------------------------------------------===//
// MmaBlockScaleOp
//===----------------------------------------------------------------------===//

void MmaBlockScaleOp::print(OpAsmPrinter &p) {
  SmallVector<Type, 4> regTypes;
  std::array<MMAOperandFragment, 3> frags{
      MMAOperandFragment("A", getMultiplicandAPtxTypeAttrName()),
      MMAOperandFragment("B", getMultiplicandBPtxTypeAttrName()),
      MMAOperandFragment("C", "")};
  SmallVector<StringRef, 4> ignoreAttrNames{
      mlir::NVVM::MmaBlockScaleOp::getOperandSegmentSizeAttr()};

  processOperandFragments(*this, frags, regTypes, ignoreAttrNames);

  // Print A, B, C operands
  for (const auto &frag : frags)
    printOperandList(p, frag.operandName, frag.regs);

  // Print scale operands
  printOperandList(p, "scaleA",
                   {getScaleAData(), getByteIdA(), getThreadIdA()});
  printOperandList(p, "scaleB",
                   {getScaleBData(), getByteIdB(), getThreadIdB()});

  p.printOptionalAttrDict(this->getOperation()->getAttrs(), ignoreAttrNames);

  // Print type signature
  p << " : (";
  llvm::interleaveComma(SmallVector<Type, 3>{frags[0].regs[0].getType(),
                                             frags[1].regs[0].getType(),
                                             frags[2].regs[0].getType()},
                        p);
  p << ")";
  p.printArrowTypeList(TypeRange{this->getRes().getType()});
}

ParseResult MmaBlockScaleOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  struct LocalOperandFragment {
    std::optional<MMATypes> elemtype;
    SmallVector<OpAsmParser::UnresolvedOperand, 4> regs;
  };

  Builder &builder = parser.getBuilder();
  std::array<LocalOperandFragment, 3> frags;
  NamedAttrList namedAttributes;

  // Parse A[...] B[...] C[...]
  if (parseMmaOperand(parser, "A", frags[0].regs).failed() ||
      parseMmaOperand(parser, "B", frags[1].regs).failed() ||
      parseMmaOperand(parser, "C", frags[2].regs).failed())
    return failure();

  // Parse scale operands: scaleA[...] scaleB[...]
  SmallVector<OpAsmParser::UnresolvedOperand, 3> scaleAOperands, scaleBOperands;
  if (parseMmaOperand(parser, "scaleA", scaleAOperands).failed() ||
      parseMmaOperand(parser, "scaleB", scaleBOperands).failed())
    return failure();

  if (parser.parseOptionalAttrDict(namedAttributes).failed())
    return failure();

  // Parse type signature
  SmallVector<Type, 3> operandTypes;
  if (parseMmaTypeSignature(parser, operandTypes).failed())
    return failure();

  // Parse result type
  SmallVector<Type, 1> resultTypes;
  if (parser.parseArrowTypeList(resultTypes).failed())
    return failure();

  // Infer element types and resolve operands
  for (const auto &[idx, frag] : llvm::enumerate(frags)) {
    frag.elemtype = MmaOp::inferOperandMMAType(operandTypes[idx],
                                               /*isAccumulator=*/idx >= 2);
    if (parser
            .resolveOperands(frag.regs, operandTypes[idx], parser.getNameLoc(),
                             result.operands)
            .failed())
      return failure();
  }

  // Resolve scale operands
  SmallVector<Type, 3> scaleTypes = {builder.getI32Type(), builder.getI16Type(),
                                     builder.getI16Type()};
  if (parser
          .resolveOperands(scaleAOperands, scaleTypes, parser.getNameLoc(),
                           result.operands)
          .failed() ||
      parser
          .resolveOperands(scaleBOperands, scaleTypes, parser.getNameLoc(),
                           result.operands)
          .failed())
    return failure();

  // Add attributes
  result.addAttributes(namedAttributes);
  inferAndSetMultiplicandTypes(parser.getContext(), result.attributes,
                               operandTypes);

  result.addTypes(resultTypes);
  result.addAttribute(MmaBlockScaleOp::getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr({
                          static_cast<int32_t>(frags[0].regs.size()),
                          static_cast<int32_t>(frags[1].regs.size()),
                          static_cast<int32_t>(frags[2].regs.size()),
                          1, // scaleAData
                          1, // byteIdA
                          1, // threadIdA
                          1, // scaleBData
                          1, // byteIdB
                          1  // threadIdB
                      }));
  return success();
}

void MmaBlockScaleOp::build(
    OpBuilder &builder, OperationState &result, Type resultType,
    ValueRange operandA, ValueRange operandB, ValueRange operandC,
    Value scaleAData, Value byteIdA, Value threadIdA, Value scaleBData,
    Value byteIdB, Value threadIdB, ArrayRef<int64_t> shape,
    std::optional<std::array<MMATypes, 2>> multiplicandPtxTypes,
    ScaleVecSize scaleVecSize, BlockScaleFormat blockScaleFormat,
    MMABlockScaleKind kind) {
  assert(shape.size() == 3 && "expected shape to have size 3 (m, n, k)");

  addBlockScaleProperties<MmaBlockScaleOp>(builder, result, shape, scaleVecSize,
                                           blockScaleFormat, kind);

  result.addOperands(operandA);
  result.addOperands(operandB);
  result.addOperands(operandC);
  result.addOperands(
      {scaleAData, byteIdA, threadIdA, scaleBData, byteIdB, threadIdB});

  addInferredMultiplicandTypes(builder.getContext(), result, operandA, operandB,
                               multiplicandPtxTypes);

  result.addTypes(resultType);
  result.addAttribute(MmaBlockScaleOp::getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr({
                          static_cast<int32_t>(operandA.size()),
                          static_cast<int32_t>(operandB.size()),
                          static_cast<int32_t>(operandC.size()),
                          1, // scaleAData
                          1, // byteIdA
                          1, // threadIdA
                          1, // scaleBData
                          1, // byteIdB
                          1  // threadIdB
                      }));
}

NVVM::IDArgPair MmaBlockScaleOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto curOp = cast<NVVM::MmaBlockScaleOp>(op);

  SmallVector<llvm::Value *> args;
  // Add A, B, C operands
  for (Value operand : curOp.getOperandA())
    args.push_back(mt.lookupValue(operand));
  for (Value operand : curOp.getOperandB())
    args.push_back(mt.lookupValue(operand));
  for (Value operand : curOp.getOperandC())
    args.push_back(mt.lookupValue(operand));

  // Add scale operands
  args.push_back(mt.lookupValue(curOp.getScaleAData()));
  args.push_back(mt.lookupValue(curOp.getByteIdA()));
  args.push_back(mt.lookupValue(curOp.getThreadIdA()));
  args.push_back(mt.lookupValue(curOp.getScaleBData()));
  args.push_back(mt.lookupValue(curOp.getByteIdB()));
  args.push_back(mt.lookupValue(curOp.getThreadIdB()));

  unsigned intId = MmaBlockScaleOp::getIntrinsicID(
      curOp.getShape().getM(), curOp.getShape().getN(), curOp.getShape().getK(),
      *curOp.getMultiplicandAPtxType(), *curOp.getMultiplicandBPtxType(),
      inferPtxTypeFromResult(curOp), curOp.getScaleVecSize(),
      curOp.getBlockScaleFormat(), curOp.getKind());

  return {intId, args};
}

LogicalResult MmaBlockScaleOp::verify() {
  LogicalResult result = success();
  int m = getShape().getM();
  int n = getShape().getN();
  int k = getShape().getK();

  if (m == 16 && n == 8 && k == 64) {
    if (getMultiplicandAPtxType() != NVVM::MMATypes::e2m1 ||
        getMultiplicandBPtxType() != NVVM::MMATypes::e2m1)
      result = emitOpError(
          "unsupported MMATypes attribute for mma.m16n8k64.(mxf4nvf4|mxf4)");
    if (getKind() == NVVM::MMABlockScaleKind::MXF4) {
      if (getScaleVecSize() != NVVM::ScaleVecSize::X2)
        result = emitOpError(
            "unsupported ScaleVecSize attribute for mma.m16n8k64.mxf4");
      if (getBlockScaleFormat() != NVVM::BlockScaleFormat::UE8M0)
        result = emitOpError(
            "unsupported BlockScaleFormat attribute for mma.m16n8k64.mxf4");
    } else if (getKind() == NVVM::MMABlockScaleKind::MXF4NVF4) {
      if (!((getScaleVecSize() == NVVM::ScaleVecSize::X2 &&
             getBlockScaleFormat() == NVVM::BlockScaleFormat::UE8M0) ||
            (getScaleVecSize() == NVVM::ScaleVecSize::X4 &&
             getBlockScaleFormat() == NVVM::BlockScaleFormat::UE4M3)))
        result = emitOpError("unsupported ScaleVecSize and BlockScaleFormat "
                             "attributes for mma.m16n8k64.mxf4nvf4");
    } else {
      result = emitOpError("unsupported Kind attribute for mma.m16n8k64");
    }
  } else if (m == 16 && n == 8 && k == 32) {
    if (!(getKind() == NVVM::MMABlockScaleKind::MXF8F6F4 &&
          getScaleVecSize() == NVVM::ScaleVecSize::X1 &&
          getBlockScaleFormat() == NVVM::BlockScaleFormat::UE8M0))
      result =
          emitOpError("unsupported Kind, ScaleVecSize and BlockScaleFormat "
                      "attributes for mma.m16n8k32");
  } else {
    result = emitOpError("unsupported Geom for mma with block scaling");
  }
  return result;
}

//===----------------------------------------------------------------------===//
// MmaSpBlockScaleOp
//===----------------------------------------------------------------------===//

void MmaSpBlockScaleOp::print(OpAsmPrinter &p) {
  SmallVector<Type, 4> regTypes;
  std::array<MMAOperandFragment, 3> frags{
      MMAOperandFragment("A", getMultiplicandAPtxTypeAttrName()),
      MMAOperandFragment("B", getMultiplicandBPtxTypeAttrName()),
      MMAOperandFragment("C", "")};
  SmallVector<StringRef, 4> ignoreAttrNames{
      mlir::NVVM::MmaSpBlockScaleOp::getOperandSegmentSizeAttr()};

  processOperandFragments(*this, frags, regTypes, ignoreAttrNames);

  // Print A, B, C operands
  for (const auto &frag : frags)
    printOperandList(p, frag.operandName, frag.regs);

  // Print sparse-specific operands
  printOperandList(p, "sparseMetadata", {getSparseMetadata()});
  printOperandList(p, "selector", {getSparsitySelector()});

  // Print scale operands
  printOperandList(p, "scaleA",
                   {getScaleAData(), getByteIdA(), getThreadIdA()});
  printOperandList(p, "scaleB",
                   {getScaleBData(), getByteIdB(), getThreadIdB()});

  p.printOptionalAttrDict(this->getOperation()->getAttrs(), ignoreAttrNames);

  // Print type signature
  p << " : (";
  llvm::interleaveComma(SmallVector<Type, 3>{frags[0].regs[0].getType(),
                                             frags[1].regs[0].getType(),
                                             frags[2].regs[0].getType()},
                        p);
  p << ")";
  p.printArrowTypeList(TypeRange{this->getRes().getType()});
}

ParseResult MmaSpBlockScaleOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  struct LocalOperandFragment {
    std::optional<MMATypes> elemtype;
    SmallVector<OpAsmParser::UnresolvedOperand, 4> regs;
  };

  Builder &builder = parser.getBuilder();
  std::array<LocalOperandFragment, 3> frags;
  NamedAttrList namedAttributes;

  // Parse A[...] B[...] C[...]
  if (parseMmaOperand(parser, "A", frags[0].regs).failed() ||
      parseMmaOperand(parser, "B", frags[1].regs).failed() ||
      parseMmaOperand(parser, "C", frags[2].regs).failed())
    return failure();

  // Parse sparse-specific operands
  SmallVector<OpAsmParser::UnresolvedOperand, 1> metadataOperands,
      selectorOperands;
  if (parseMmaOperand(parser, "sparseMetadata", metadataOperands).failed() ||
      parseMmaOperand(parser, "selector", selectorOperands).failed())
    return failure();

  // Parse scale operands
  SmallVector<OpAsmParser::UnresolvedOperand, 3> scaleAOperands, scaleBOperands;
  if (parseMmaOperand(parser, "scaleA", scaleAOperands).failed() ||
      parseMmaOperand(parser, "scaleB", scaleBOperands).failed())
    return failure();

  if (parser.parseOptionalAttrDict(namedAttributes).failed())
    return failure();

  // Parse type signature
  SmallVector<Type, 3> operandTypes;
  if (parseMmaTypeSignature(parser, operandTypes).failed())
    return failure();

  // Parse result type
  SmallVector<Type, 1> resultTypes;
  if (parser.parseArrowTypeList(resultTypes).failed())
    return failure();

  // Infer element types and resolve operands
  for (const auto &[idx, frag] : llvm::enumerate(frags)) {
    frag.elemtype = MmaOp::inferOperandMMAType(operandTypes[idx],
                                               /*isAccumulator=*/idx >= 2);
    if (parser
            .resolveOperands(frag.regs, operandTypes[idx], parser.getNameLoc(),
                             result.operands)
            .failed())
      return failure();
  }

  // Resolve sparse metadata and selector
  Type i32Type = builder.getI32Type();
  if (parser
          .resolveOperands(metadataOperands, i32Type, parser.getNameLoc(),
                           result.operands)
          .failed() ||
      parser
          .resolveOperands(selectorOperands, i32Type, parser.getNameLoc(),
                           result.operands)
          .failed())
    return failure();

  // Resolve scale operands
  SmallVector<Type, 3> scaleTypes = {i32Type, builder.getI16Type(),
                                     builder.getI16Type()};
  if (parser
          .resolveOperands(scaleAOperands, scaleTypes, parser.getNameLoc(),
                           result.operands)
          .failed() ||
      parser
          .resolveOperands(scaleBOperands, scaleTypes, parser.getNameLoc(),
                           result.operands)
          .failed())
    return failure();

  // Add attributes
  result.addAttributes(namedAttributes);
  inferAndSetMultiplicandTypes(parser.getContext(), result.attributes,
                               operandTypes);

  // orderedMetadata is mandatory
  if (!result.attributes.get("orderedMetadata"))
    result.addAttribute("orderedMetadata", builder.getUnitAttr());

  result.addTypes(resultTypes);
  result.addAttribute(MmaSpBlockScaleOp::getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr({
                          static_cast<int32_t>(frags[0].regs.size()),
                          static_cast<int32_t>(frags[1].regs.size()),
                          static_cast<int32_t>(frags[2].regs.size()),
                          1, // sparseMetadata
                          1, // sparsitySelector
                          1, // scaleAData
                          1, // byteIdA
                          1, // threadIdA
                          1, // scaleBData
                          1, // byteIdB
                          1  // threadIdB
                      }));
  return success();
}

void MmaSpBlockScaleOp::build(
    OpBuilder &builder, OperationState &result, Type resultType,
    ValueRange operandA, ValueRange operandB, ValueRange operandC,
    Value sparseMetadata, Value sparsitySelector, Value scaleAData,
    Value byteIdA, Value threadIdA, Value scaleBData, Value byteIdB,
    Value threadIdB, ArrayRef<int64_t> shape,
    std::optional<std::array<MMATypes, 2>> multiplicandPtxTypes,
    ScaleVecSize scaleVecSize, BlockScaleFormat blockScaleFormat,
    MMABlockScaleKind kind) {
  assert(shape.size() == 3 && "expected shape to have size 3 (m, n, k)");

  addBlockScaleProperties<MmaSpBlockScaleOp>(
      builder, result, shape, scaleVecSize, blockScaleFormat, kind);
  result.addAttribute("orderedMetadata", builder.getUnitAttr());

  result.addOperands(operandA);
  result.addOperands(operandB);
  result.addOperands(operandC);
  result.addOperands({sparseMetadata, sparsitySelector, scaleAData, byteIdA,
                      threadIdA, scaleBData, byteIdB, threadIdB});

  addInferredMultiplicandTypes(builder.getContext(), result, operandA, operandB,
                               multiplicandPtxTypes);

  result.addTypes(resultType);
  result.addAttribute(MmaSpBlockScaleOp::getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr({
                          static_cast<int32_t>(operandA.size()),
                          static_cast<int32_t>(operandB.size()),
                          static_cast<int32_t>(operandC.size()),
                          1, // sparseMetadata
                          1, // sparsitySelector
                          1, // scaleAData
                          1, // byteIdA
                          1, // threadIdA
                          1, // scaleBData
                          1, // byteIdB
                          1  // threadIdB
                      }));
}

NVVM::IDArgPair MmaSpBlockScaleOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto curOp = cast<NVVM::MmaSpBlockScaleOp>(op);

  SmallVector<llvm::Value *> args;
  // Add A, B, C operands
  for (Value operand : curOp.getOperandA())
    args.push_back(mt.lookupValue(operand));
  for (Value operand : curOp.getOperandB())
    args.push_back(mt.lookupValue(operand));
  for (Value operand : curOp.getOperandC())
    args.push_back(mt.lookupValue(operand));

  // Add sparse metadata and selector
  args.push_back(mt.lookupValue(curOp.getSparseMetadata()));
  args.push_back(mt.lookupValue(curOp.getSparsitySelector()));

  // Add scale operands
  args.push_back(mt.lookupValue(curOp.getScaleAData()));
  args.push_back(mt.lookupValue(curOp.getByteIdA()));
  args.push_back(mt.lookupValue(curOp.getThreadIdA()));
  args.push_back(mt.lookupValue(curOp.getScaleBData()));
  args.push_back(mt.lookupValue(curOp.getByteIdB()));
  args.push_back(mt.lookupValue(curOp.getThreadIdB()));

  unsigned intId = MmaSpBlockScaleOp::getIntrinsicID(
      curOp.getShape().getM(), curOp.getShape().getN(), curOp.getShape().getK(),
      *curOp.getMultiplicandAPtxType(), *curOp.getMultiplicandBPtxType(),
      inferPtxTypeFromResult(curOp), curOp.getScaleVecSize(),
      curOp.getBlockScaleFormat(), curOp.getKind());

  return {intId, args};
}

LogicalResult MmaSpBlockScaleOp::verify() {
  // Check that orderedMetadata is present
  if (!getOrderedMetadata()) {
    return emitOpError("'orderedMetadata' attribute is mandatory");
  }

  LogicalResult result = success();
  int m = getShape().getM();
  int n = getShape().getN();
  int k = getShape().getK();

  if (m == 16 && n == 8 && k == 128) {
    if (getMultiplicandAPtxType() != NVVM::MMATypes::e2m1 ||
        getMultiplicandBPtxType() != NVVM::MMATypes::e2m1)
      result = emitOpError(
          "unsupported MMATypes attribute for mma.m16n8k128.(mxf4nvf4|mxf4)");
    if (getKind() == NVVM::MMABlockScaleKind::MXF4) {
      if (getScaleVecSize() != NVVM::ScaleVecSize::X2)
        result = emitOpError(
            "unsupported ScaleVecSize attribute for mma.m16n8k128.mxf4");
      if (getBlockScaleFormat() != NVVM::BlockScaleFormat::UE8M0)
        result = emitOpError(
            "unsupported BlockScaleFormat attribute for mma.m16n8k128.mxf4");
    } else if (getKind() == NVVM::MMABlockScaleKind::MXF4NVF4) {
      if (!((getScaleVecSize() == NVVM::ScaleVecSize::X2 &&
             getBlockScaleFormat() == NVVM::BlockScaleFormat::UE8M0) ||
            (getScaleVecSize() == NVVM::ScaleVecSize::X4 &&
             getBlockScaleFormat() == NVVM::BlockScaleFormat::UE4M3)))
        result = emitOpError("unsupported ScaleVecSize and BlockScaleFormat "
                             "attributes for mma.m16n8k128.mxf4nvf4");
    } else {
      result = emitOpError("unsupported Kind attribute for mma.m16n8k128");
    }
  } else if (m == 16 && n == 8 && k == 64) {
    if (!(getKind() == NVVM::MMABlockScaleKind::MXF8F6F4 &&
          getScaleVecSize() == NVVM::ScaleVecSize::X1 &&
          getBlockScaleFormat() == NVVM::BlockScaleFormat::UE8M0))
      result =
          emitOpError("unsupported Kind, ScaleVecSize and BlockScaleFormat "
                      "attributes for mma.m16n8k64");
  } else {
    result = emitOpError("unsupported Geom for sparse mma with block scaling");
  }
  return result;
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

std::pair<mlir::Type, unsigned> NVVM::inferMMAType(NVVM::MMATypes type,
                                                   NVVM::MMAFrag frag, int nRow,
                                                   int nCol,
                                                   MLIRContext *context) {
  unsigned numberElements = 0;
  Type elementType;
  OpBuilder builder(context);
  Type f16x2 = VectorType::get(2, builder.getF16Type());
  if (type == NVVM::MMATypes::f16) {
    elementType = f16x2;
    if (frag == NVVM::MMAFrag::a || frag == NVVM::MMAFrag::b)
      numberElements = 8;
    else
      numberElements = 4;
  } else if (type == NVVM::MMATypes::f32) {
    elementType = builder.getF32Type();
    numberElements = 8;
  } else if (type == NVVM::MMATypes::f64) {
    elementType = builder.getF64Type();
    if (frag == NVVM::MMAFrag::a || frag == NVVM::MMAFrag::b)
      numberElements = 1;
    else
      numberElements = 2;
  } else if (type == NVVM::MMATypes::tf32) {
    elementType = builder.getI32Type();
    numberElements = 4;
  } else if (type == NVVM::MMATypes::s8 || type == NVVM::MMATypes::u8) {
    elementType = builder.getI32Type();
    int parallelSize = 0;
    if (frag == NVVM::MMAFrag::a)
      parallelSize = nRow;
    if (frag == NVVM::MMAFrag::b)
      parallelSize = nCol;

    // m == 16 && n == 16 && k == 16
    if (parallelSize == 16)
      numberElements = 2;
    // m == 8 && n == 32 && k == 16 or m == 32 && n == 8 && k == 16
    else if (parallelSize == 8)
      numberElements = 1;
    else if (parallelSize == 32)
      numberElements = 4;
  } else if (type == NVVM::MMATypes::s32) {
    elementType = builder.getI32Type();
    numberElements = 8;
  }
  assert(numberElements != 0 && elementType != nullptr);
  return std::make_pair(elementType, numberElements);
}

static std::pair<mlir::Type, unsigned>
inferMMATypeFromMNK(NVVM::MMATypes type, NVVM::MMAFrag frag, int m, int n,
                    int k, MLIRContext *context) {
  int nRow, nCol;
  if (frag == NVVM::MMAFrag::a) {
    nRow = m;
    nCol = k;
  } else if (frag == NVVM::MMAFrag::b) {
    nRow = k;
    nCol = n;
  } else {
    nRow = m;
    nCol = n;
  }
  assert(nRow && nCol);
  return inferMMAType(type, frag, nRow, nCol, context);
}

LogicalResult NVVM::WMMALoadOp::verify() {
  unsigned addressSpace =
      llvm::cast<LLVM::LLVMPointerType>(getPtr().getType()).getAddressSpace();
  if (addressSpace != 0 && addressSpace != NVVMMemorySpace::Global &&
      addressSpace != NVVMMemorySpace::Shared)
    return emitOpError("expected source pointer in memory "
                       "space 0, 1, 3");

  if (NVVM::WMMALoadOp::getIntrinsicID(getM(), getN(), getK(), getLayout(),
                                       getEltype(), getFrag()) == 0)
    return emitOpError() << "invalid attribute combination";
  std::pair<Type, unsigned> typeInfo = inferMMATypeFromMNK(
      getEltype(), getFrag(), getM(), getN(), getK(), getContext());
  // Special case for f64 fragments
  Type f64Ty = Float64Type::get(getContext());
  if (typeInfo.first == f64Ty && typeInfo.second == 1) {
    if (getType() != f64Ty)
      return emitOpError("expected destination type to be f64");
    return success();
  }
  // Everything else is a struct
  Type dstType = LLVM::LLVMStructType::getLiteral(
      getContext(), SmallVector<Type, 8>(typeInfo.second, typeInfo.first));
  if (getType() != dstType)
    return emitOpError("expected destination type is a structure of ")
           << typeInfo.second << " elements of type " << typeInfo.first;
  return success();
}

LogicalResult NVVM::WMMAStoreOp::verify() {
  unsigned addressSpace =
      llvm::cast<LLVM::LLVMPointerType>(getPtr().getType()).getAddressSpace();
  if (addressSpace != 0 && addressSpace != NVVMMemorySpace::Global &&
      addressSpace != NVVMMemorySpace::Shared)
    return emitOpError("expected operands to be a source pointer in memory "
                       "space 0, 1, 3");

  if (NVVM::WMMAStoreOp::getIntrinsicID(getM(), getN(), getK(), getLayout(),
                                        getEltype()) == 0)
    return emitOpError() << "invalid attribute combination";
  std::pair<Type, unsigned> typeInfo = inferMMATypeFromMNK(
      getEltype(), NVVM::MMAFrag::c, getM(), getN(), getK(), getContext());
  if (getArgs().size() != typeInfo.second)
    return emitOpError() << "expected " << typeInfo.second << " data operands";
  if (llvm::any_of(getArgs(), [&typeInfo](Value operands) {
        return operands.getType() != typeInfo.first;
      }))
    return emitOpError() << "expected data operands of type " << typeInfo.first;
  return success();
}

LogicalResult NVVM::WMMAMmaOp::verify() {
  if (NVVM::WMMAMmaOp::getIntrinsicID(getM(), getN(), getK(), getLayoutA(),
                                      getLayoutB(), getEltypeA(),
                                      getEltypeB()) == 0)
    return emitOpError() << "invalid attribute combination";
  std::pair<Type, unsigned> typeInfoA = inferMMATypeFromMNK(
      getEltypeA(), NVVM::MMAFrag::a, getM(), getN(), getK(), getContext());
  std::pair<Type, unsigned> typeInfoB = inferMMATypeFromMNK(
      getEltypeA(), NVVM::MMAFrag::b, getM(), getN(), getK(), getContext());
  std::pair<Type, unsigned> typeInfoC = inferMMATypeFromMNK(
      getEltypeB(), NVVM::MMAFrag::c, getM(), getN(), getK(), getContext());
  SmallVector<Type, 32> arguments;
  arguments.append(typeInfoA.second, typeInfoA.first);
  arguments.append(typeInfoB.second, typeInfoB.first);
  arguments.append(typeInfoC.second, typeInfoC.first);
  unsigned numArgs = arguments.size();
  if (getArgs().size() != numArgs)
    return emitOpError() << "expected " << numArgs << " arguments";
  for (unsigned i = 0; i < numArgs; i++) {
    if (getArgs()[i].getType() != arguments[i])
      return emitOpError() << "expected argument " << i << " to be of type "
                           << arguments[i];
  }
  Type dstType = LLVM::LLVMStructType::getLiteral(
      getContext(), SmallVector<Type, 8>(typeInfoC.second, typeInfoC.first));
  if (getType() != dstType)
    return emitOpError("expected destination type is a structure of ")
           << typeInfoC.second << " elements of type " << typeInfoC.first;
  return success();
}

LogicalResult NVVM::LdMatrixOp::verify() {
  uint32_t num = getNum(), m = getShape().getM(), n = getShape().getN();
  if (m == 8 && n == 8) {
    if (num != 1 && num != 2 && num != 4) {
      return emitOpError("expected num attribute to be 1, 2 or 4 for 8x8 "
                         "matrix");
    }
    if (getEltType() != LdStMatrixEltType::B16) {
      return emitOpError("expected element type to be b16 for 8x8 matrix");
    }
  } else if (m == 8 && n == 16) {
    if (num != 1 && num != 2 && num != 4) {
      return emitOpError("expected num attribute to be 1, 2 or 4 for 8x16 "
                         "matrix");
    }
    if (getLayout() != MMALayout::row) {
      return emitOpError("expected layout to be row for 8x16 matrix");
    }
    if (getEltType() != LdStMatrixEltType::B8X16_B4X16_P64 &&
        getEltType() != LdStMatrixEltType::B8X16_B6X16_P32) {
      return emitOpError("expected element type to be b8x16.b4x16_p64 or "
                         "b8x16.b6x16_p32 for 8x16 matrix");
    }
  } else if (m == 16 && n == 16) {
    if (num != 1 && num != 2) {
      return emitOpError("expected num attribute to be 1 or 2 for 16x16 "
                         "matrix");
    }
    if (getLayout() != MMALayout::col) {
      return emitOpError("expected layout to be col for 16x16 matrix");
    }
    if (getEltType() != LdStMatrixEltType::B8 &&
        getEltType() != LdStMatrixEltType::B8X16_B4X16_P64 &&
        getEltType() != LdStMatrixEltType::B8X16_B6X16_P32) {
      return emitOpError("expected element type to be b8, b8x16.b4x16_p64 or "
                         "b8x16.b6x16_p32 for 16x16 matrix");
    }
  } else {
    return emitOpError("expected shape to be 8x8, 8x16 or 16x16");
  }

  Type i32 = IntegerType::get(getContext(), 32);
  uint32_t numElements = (m == 16 && n == 16 ? num * 2 : num);
  if (numElements == 1 && getType() != i32)
    return emitOpError("expected destination type is i32");
  if (numElements == 2 || numElements == 4) {
    Type dstType = LLVM::LLVMStructType::getLiteral(
        getContext(), SmallVector<Type>(numElements, i32));
    if (getType() != dstType)
      return emitOpError("expected destination type is a structure of ")
             << numElements << " elements of type i32";
  }

  return success();
}

LogicalResult NVVM::StMatrixOp::verify() {
  int numMatrix = getSources().size();
  if (numMatrix != 1 && numMatrix != 2 && numMatrix != 4)
    return emitOpError("expected num attribute to be 1, 2 or 4");

  int m = getShape().getM(), n = getShape().getN();
  if (m == 8 && n == 8) {
    if (getEltType() != NVVM::LdStMatrixEltType::B16) {
      return emitOpError("expected element type to be B16 for 8x8 matrix");
    }
  } else if (m == 16 && n == 8) {
    if (getEltType() != NVVM::LdStMatrixEltType::B8) {
      return emitOpError("expected element type to be B8 for 16x8 matrix");
    }
    if (getLayout() != NVVM::MMALayout::col) {
      return emitOpError("expected layout to be col for 16x8 matrix");
    }
  } else {
    return emitOpError("expected shape to be 8x8 or 16x8");
  }

  return success();
}

static FailureOr<int> getAllowedSizeK(NVVM::WGMMATypes typeA) {
  if (typeA == NVVM::WGMMATypes::tf32)
    return 8;
  if (typeA == NVVM::WGMMATypes::f16 || typeA == NVVM::WGMMATypes::bf16)
    return 16;
  if (typeA == NVVM::WGMMATypes::s8 || typeA == NVVM::WGMMATypes::u8)
    return 32;
  if (typeA == NVVM::WGMMATypes::e4m3 || typeA == NVVM::WGMMATypes::e5m2)
    return 32;
  if (typeA == NVVM::WGMMATypes::b1)
    return 256;
  return failure();
}

static LogicalResult isAllowedWGMMADataType(NVVM::WGMMATypes typeD,
                                            NVVM::WGMMATypes typeA,
                                            NVVM::WGMMATypes typeB) {
  switch (typeA) {
  case NVVM::WGMMATypes::f16:
    if ((typeD == NVVM::WGMMATypes::f32 || typeD == NVVM::WGMMATypes::f16) &&
        typeB == NVVM::WGMMATypes::f16)
      return success();
    break;
  case NVVM::WGMMATypes::tf32:
    if (typeD == NVVM::WGMMATypes::f32 && typeB == NVVM::WGMMATypes::tf32)
      return success();
    break;
  case NVVM::WGMMATypes::u8:
  case NVVM::WGMMATypes::s8:
    if (typeD == NVVM::WGMMATypes::s32 &&
        (typeB == NVVM::WGMMATypes::u8 || typeB == NVVM::WGMMATypes::s8))
      return success();
    break;
  case NVVM::WGMMATypes::b1:
    if (typeD == NVVM::WGMMATypes::s32 && typeB == NVVM::WGMMATypes::b1)
      return success();
    break;
  case NVVM::WGMMATypes::bf16:
    if ((typeD == NVVM::WGMMATypes::f32 || typeD == NVVM::WGMMATypes::f16) &&
        typeB == NVVM::WGMMATypes::bf16)
      return success();
    break;
  case NVVM::WGMMATypes::e4m3:
  case NVVM::WGMMATypes::e5m2:
    if ((typeD == NVVM::WGMMATypes::f32 || typeD == NVVM::WGMMATypes::f16) &&
        (typeB == NVVM::WGMMATypes::e5m2 || typeB == NVVM::WGMMATypes::e4m3))
      return success();
    break;
  case WGMMATypes::f32:
  case WGMMATypes::s32:
    llvm_unreachable("unsupported input types");
    break;
  }
  return failure();
}

static LogicalResult isAllowedSizeN(int sizeN, NVVM::WGMMATypes typeA) {
  SmallVector<int> allowedN = {8,   16,  24,  32,  40,  48,  56,  64,
                               72,  80,  88,  96,  104, 112, 120, 128,
                               136, 144, 152, 160, 168, 176, 184, 192,
                               200, 208, 216, 224, 232, 240, 248, 256};
  SmallVector<int> allowedNshort = {8,   16,  24,  32,  48,  64,
                                    80,  96,  112, 128, 144, 160,
                                    176, 192, 208, 224, 240, 256};
  switch (typeA) {
  case WGMMATypes::f16:
  case WGMMATypes::tf32:
  case WGMMATypes::bf16:
  case WGMMATypes::e4m3:
  case WGMMATypes::e5m2:
    if (llvm::is_contained(allowedN, sizeN))
      return success();
    break;
  case WGMMATypes::u8:
  case WGMMATypes::s8:
  case WGMMATypes::b1:
    if (llvm::is_contained(allowedNshort, sizeN))
      return success();
    break;
  case WGMMATypes::f32:
  case WGMMATypes::s32:
    llvm_unreachable("unsupported input types");
    break;
  }
  return failure();
}

LogicalResult NVVM::WgmmaMmaAsyncOp::verify() {
  Value outValue = getResults();
  auto stype = dyn_cast<LLVM::LLVMStructType>(outValue.getType());
  if (!stype)
    return emitOpError() << "expected results to be struct";
  int outputSize = stype.getBody().size();
  WGMMATypes typeD = getTypeD();
  WGMMATypes typeA = getTypeA();
  WGMMATypes typeB = getTypeB();

  for (Type t : stype.getBody()) {
    if (t != stype.getBody().front())
      return emitOpError()
             << "all elements in struct must be same type but there is " << t;
  }

  if (typeD != WGMMATypes::f32 && typeD != WGMMATypes::f16 &&
      typeD != WGMMATypes::s32) {
    return emitOpError() << "does not support the given output type " << typeD;
  }
  if (typeD == WGMMATypes::s32 &&
      (getScaleA() == WGMMAScaleIn::neg || getScaleB() == WGMMAScaleIn::neg)) {
    return emitOpError() << "has s32 output, scaleA and scaleB cannot be neg";
  }

  if (failed(isAllowedWGMMADataType(typeD, typeA, typeB))) {
    return emitOpError() << typeD << " += " << typeA << " * " << typeB
                         << ", it is not supported.";
  }

  // Check M
  if (getShape().getM() != 64)
    return emitOpError() << "shape 'm' must be 64";

  // Check K
  FailureOr<int> allowedK = getAllowedSizeK(typeA);
  if (failed(allowedK) || allowedK.value() != getShape().getK())
    return emitOpError() << "shape 'k' must be " << allowedK.value()
                         << " for input type " << typeA;

  // Check N
  if (failed(isAllowedSizeN(getShape().getN(), typeA))) {
    return emitOpError() << "has input type " << typeA << " n is set to "
                         << getShape().getN() << ", it is not supported.";
  }

  // Check transpose (only available for f16/bf16)
  // Matrices A should be stored in row-major and B in column-major.
  // Only f16/bf16 matrices can be stored in either column-major or row-major
  // by setting the transpose value(imm-trans-a,imm-trans-b) in PTX code.
  if ((typeA != WGMMATypes::f16 && typeA != WGMMATypes::bf16) &&
      (getLayoutA() == mlir::NVVM::MMALayout::col ||
       getLayoutB() == mlir::NVVM::MMALayout::row)) {
    return emitOpError()
           << "given layouts layout_a = " << getLayoutA()
           << " and layout_b = " << getLayoutB() << " for input types " << typeA
           << " and " << typeB
           << " requires transpose. However, this is only supported for: "
           << MMATypes::f16 << " and " << MMATypes::bf16;
  }

  // Check result registers
  int expectedOutput = 0;
  if (typeD == WGMMATypes::f32 || typeD == WGMMATypes::s32)
    expectedOutput = getShape().getN() / 2;
  if (typeD == WGMMATypes::f16)
    expectedOutput = getShape().getN() / 4;
  if (outputSize != expectedOutput) {
    return emitOpError() << "results " << expectedOutput
                         << ", however output struct has " << outputSize
                         << " elements";
  }
  // Check satfinite (only available for s32 accumulator)
  if (typeD != WGMMATypes::s32 &&
      getSatfinite().value_or(NVVM::MMAIntOverflow::wrapped) ==
          NVVM::MMAIntOverflow::satfinite) {
    return emitOpError()
           << " `satfinite` can be only used with s32 accumulator, however "
              "the current accumulator is "
           << typeD;
  }

  return success();
}

std::string NVVM::WgmmaMmaAsyncOp::getPtx() {

  int m = getShape().getM(), n = getShape().getN(), k = getShape().getK();
  bool isF16 = getTypeA() == WGMMATypes::f16 || getTypeA() == WGMMATypes::bf16;

  StringRef outputTypeName = stringifyWGMMATypes(getTypeD());

  int expectedOutputRegisters = 0;
  if (getTypeD() == WGMMATypes::f16)
    expectedOutputRegisters = getShape().getN() / 4;
  else
    expectedOutputRegisters = getShape().getN() / 2;

  std::string ptx;
  llvm::raw_string_ostream ss(ptx);

  ss << "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, $"
     << ((expectedOutputRegisters * 2) + 2)
     << ", 0;\n"
        "wgmma.mma_async.sync.aligned.m"
     << m << "n" << n << "k" << k << "." << outputTypeName << "." << getTypeA()
     << "." << getTypeB();
  if (getSatfinite().value_or(NVVM::MMAIntOverflow::wrapped) ==
      NVVM::MMAIntOverflow::satfinite)
    ss << ".satfinite";
  ss << " {";
  int regCnt = 0;
  for (; regCnt < expectedOutputRegisters; ++regCnt) {
    ss << "$" << regCnt;
    if (regCnt != expectedOutputRegisters - 1)
      ss << ", ";
  }

  ss << "},";
  // Need to map read/write registers correctly.
  regCnt = (regCnt * 2);
  ss << " $" << (regCnt) << ","
     << " $" << (regCnt + 1) << ","
     << " p";
  if (getTypeD() != WGMMATypes::s32) {
    ss << ", $" << (regCnt + 3) << ",  $" << (regCnt + 4);
  }
  // Don't add transpose parameters unless needed.
  if (isF16) {
    ss << ", $" << (regCnt + 5) << ",  $" << (regCnt + 6);
  }
  ss << ";\n"
     << "}\n";
  return ptx;
}

bool NVVM::WgmmaMmaAsyncOp::getAsmValues(
    RewriterBase &rewriter,
    llvm::SmallVectorImpl<std::pair<mlir::Value, mlir::NVVM::PTXRegisterMod>>
        &asmValues) {
  bool isF16 = getTypeA() == WGMMATypes::f16 || getTypeA() == WGMMATypes::bf16;
  if (getResults())
    asmValues.push_back({getResults(), mlir::NVVM::PTXRegisterMod::Write});
  if (getInouts())
    asmValues.push_back({getInouts(), mlir::NVVM::PTXRegisterMod::ReadWrite});
  asmValues.push_back({getDescriptorA(), mlir::NVVM::PTXRegisterMod::Read});
  asmValues.push_back({getDescriptorB(), mlir::NVVM::PTXRegisterMod::Read});
  asmValues.push_back({makeConstantI32(rewriter, static_cast<int>(getScaleD())),
                       mlir::NVVM::PTXRegisterMod::Read});
  if (getTypeD() != WGMMATypes::s32) {
    asmValues.push_back(
        {makeConstantI32(rewriter,
                         getScaleA() == NVVM::WGMMAScaleIn::neg ? -1 : 1),
         mlir::NVVM::PTXRegisterMod::Read});
    asmValues.push_back(
        {makeConstantI32(rewriter,
                         getScaleB() == NVVM::WGMMAScaleIn::neg ? -1 : 1),
         mlir::NVVM::PTXRegisterMod::Read});
  }
  if (isF16) {
    asmValues.push_back(
        {makeConstantI32(rewriter, static_cast<int>(getLayoutA())),
         mlir::NVVM::PTXRegisterMod::Read});
    asmValues.push_back(
        {makeConstantI32(rewriter, 1 - static_cast<int>(getLayoutB())),
         mlir::NVVM::PTXRegisterMod::Read});
  }
  return true; // Has manual mapping
}

LogicalResult NVVM::FenceSyncRestrictOp::verify() {
  if (getOrder() != NVVM::MemOrderKind::ACQUIRE &&
      getOrder() != NVVM::MemOrderKind::RELEASE)
    return emitOpError("only acquire and release semantics are supported");
  return success();
}

LogicalResult NVVM::FenceProxyOp::verify() {
  if (getKind() == NVVM::ProxyKind::TENSORMAP)
    return emitOpError() << "tensormap proxy is not a supported proxy kind";
  if (getKind() == NVVM::ProxyKind::GENERIC)
    return emitOpError() << "generic proxy not a supported proxy kind";
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
  if (getOrder() != NVVM::MemOrderKind::ACQUIRE &&
      getOrder() != NVVM::MemOrderKind::RELEASE)
    return emitOpError("only acquire and release semantics are supported");

  if (getFromProxy() != NVVM::ProxyKind::GENERIC)
    return emitOpError("only generic is support for from_proxy attribute");

  if (getToProxy() != NVVM::ProxyKind::async)
    return emitOpError("only async is supported for to_proxy attribute");
  return success();
}

LogicalResult NVVM::SetMaxRegisterOp::verify() {
  if (getRegCount() % 8)
    return emitOpError("new register size must be multiple of 8");
  if (getRegCount() < 24 || getRegCount() > 256)
    return emitOpError("new register size must be in between 24 to 256");
  return success();
}

LogicalResult NVVM::BarrierOp::verify() {
  if (getNumberOfThreads() && !getBarrierId())
    return emitOpError(
        "barrier id is missing, it should be set between 0 to 15");

  if (getBarrierId() && (getReductionOp() || getReductionPredicate()))
    return emitOpError("reduction are only available when id is 0");

  if ((getReductionOp() && !getReductionPredicate()) ||
      (!getReductionOp() && getReductionPredicate()))
    return emitOpError("reduction predicate and reduction operation must be "
                       "specified together");

  return success();
}

LogicalResult NVVM::Tcgen05CpOp::verify() {
  auto mc = getMulticast();

  using SH = Tcgen05CpShape;
  using MC = Tcgen05CpMulticast;
  switch (getShape()) {
  case SH::SHAPE_128x256b:
  case SH::SHAPE_128x128b:
  case SH::SHAPE_4x256b:
    if (mc != MC::NONE)
      return emitError("Invalid multicast type for tcgen05.cp Op");
    break;
  case SH::SHAPE_64x128b:
    if (mc != MC::WARPX2_01_23 && mc != MC::WARPX2_02_13)
      return emitError("Shape 64x128b requires multicast warpx2_01_23 or "
                       "warpx2_02_13 for tcgen05.cp Op");
    break;
  case SH::SHAPE_32x128b:
    if (mc != MC::WARPX4)
      return emitError(
          "Shape 32x128b requires multicast warpx4 for tcgen05.cp Op");
    break;
  }
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

LogicalResult NVVM::ReduxOp::verify() {
  mlir::Type reduxType = getType();

  if (!reduxType.isF32()) {
    if (getAbs())
      return emitOpError("abs attribute is supported only for f32 type");
    if (getNan())
      return emitOpError("nan attribute is supported only for f32 type");
  }

  NVVM::ReduxKind kind = getKind();
  switch (kind) {
  case NVVM::ReduxKind::ADD:
  case NVVM::ReduxKind::AND:
  case NVVM::ReduxKind::OR:
  case NVVM::ReduxKind::XOR:
  case NVVM::ReduxKind::MAX:
  case NVVM::ReduxKind::MIN:
  case NVVM::ReduxKind::UMAX:
  case NVVM::ReduxKind::UMIN:
    if (!reduxType.isInteger(32))
      return emitOpError("'")
             << kind << "' redux kind unsupported with " << reduxType
             << " type. Only supported type is 'i32'.";
    break;
  case NVVM::ReduxKind::FMIN:
  case NVVM::ReduxKind::FMAX:
    if (!reduxType.isF32())
      return emitOpError("'")
             << kind << "' redux kind unsupported with " << reduxType
             << " type. Only supported type is 'f32'.";
    break;
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

/// Packs the given `field` into the `result`.
/// The `result` is 64-bits and each `field` can be 32-bits or narrower.
static llvm::Value *
packValInto64Bits(llvm::IRBuilderBase &builder,
                  llvm::Value *result, // the `result` (unset bits are zero)
                  llvm::Value *field,  // `field` to pack into `result`
                  unsigned sizeInBits, // Size of `field` in bits
                  unsigned start) {    // Starting bit within `result`
  field = builder.CreateZExtOrBitCast(field, builder.getInt32Ty());

  unsigned mask = (sizeInBits < 32 ? ((1u << sizeInBits) - 1) : 0xffffffffu);
  if (mask != 0xffffffffu)
    field = builder.CreateAnd(field, builder.getInt32(mask));

  field = builder.CreateZExtOrBitCast(field, builder.getInt64Ty());
  field = builder.CreateShl(field, start);

  return builder.CreateOr(result, field);
}

void Tcgen05MmaSmemDescOp::createSmemDescriptor(Operation &op,
                                                LLVM::ModuleTranslation &mt,
                                                llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::Tcgen05MmaSmemDescOp>(op);
  llvm::Value *smemDesc = builder.getInt64(0);

  smemDesc = packValInto64Bits(builder, smemDesc,
                               mt.lookupValue(thisOp.getStartAddr()), 14, 0);
  smemDesc = packValInto64Bits(
      builder, smemDesc, mt.lookupValue(thisOp.getLeadingDimOffset()), 14, 16);
  smemDesc = packValInto64Bits(
      builder, smemDesc, mt.lookupValue(thisOp.getStrideDimOffset()), 14, 32);

  smemDesc = packValInto64Bits(builder, smemDesc, builder.getInt32(1), 3, 46);
  smemDesc = packValInto64Bits(builder, smemDesc,
                               mt.lookupValue(thisOp.getBaseOffset()), 3, 49);
  smemDesc = packValInto64Bits(
      builder, smemDesc, mt.lookupValue(thisOp.getLeadingDimMode()), 1, 52);
  smemDesc = packValInto64Bits(builder, smemDesc,
                               mt.lookupValue(thisOp.getSwizzleMode()), 3, 61);

  mt.mapValue(thisOp.getRes()) = smemDesc;
}

//===----------------------------------------------------------------------===//
// getPtx methods
//===----------------------------------------------------------------------===//

std::string NVVM::MBarrierInitOp::getPtx() {
  bool isShared = isPtrInSharedCTASpace(getAddr());
  return isShared ? std::string("mbarrier.init.shared.b64 [%0], %1;")
                  : std::string("mbarrier.init.b64 [%0], %1;");
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

mlir::NVVM::IDArgPair NVVM::BarrierOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::BarrierOp>(op);
  llvm::Value *barrierId = thisOp.getBarrierId()
                               ? mt.lookupValue(thisOp.getBarrierId())
                               : builder.getInt32(0);
  llvm::Intrinsic::ID id;
  llvm::SmallVector<llvm::Value *> args = {barrierId};
  if (thisOp.getNumberOfThreads()) {
    id = llvm::Intrinsic::nvvm_barrier_cta_sync_aligned_count;
    args.push_back(mt.lookupValue(thisOp.getNumberOfThreads()));
  } else if (thisOp.getReductionOp()) {
    switch (*thisOp.getReductionOp()) {
    case NVVM::BarrierReduction::AND:
      id = llvm::Intrinsic::nvvm_barrier_cta_red_and_aligned_all;
      break;
    case NVVM::BarrierReduction::OR:
      id = llvm::Intrinsic::nvvm_barrier_cta_red_or_aligned_all;
      break;
    case NVVM::BarrierReduction::POPC:
      id = llvm::Intrinsic::nvvm_barrier_cta_red_popc_aligned_all;
      break;
    }
    args.push_back(builder.CreateICmpNE(
        mt.lookupValue(thisOp.getReductionPredicate()), builder.getInt32(0)));
  } else {
    id = llvm::Intrinsic::nvvm_barrier_cta_sync_aligned_all;
  }

  return {id, std::move(args)};
}

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

mlir::NVVM::IDArgPair MBarrierInitOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::MBarrierInitOp>(op);
  bool isShared = isPtrInSharedCTASpace(thisOp.getAddr());
  llvm::Intrinsic::ID id = isShared ? llvm::Intrinsic::nvvm_mbarrier_init_shared
                                    : llvm::Intrinsic::nvvm_mbarrier_init;

  // Fill the Intrinsic Args
  llvm::SmallVector<llvm::Value *> args;
  args.push_back(mt.lookupValue(thisOp.getAddr()));
  args.push_back(mt.lookupValue(thisOp.getCount()));

  return {id, std::move(args)};
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
  return {id, {mbar, count}};
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

  // When count is not explicitly specified, the default is 1.
  llvm::LLVMContext &ctx = mt.getLLVMContext();
  bool hasCount = static_cast<bool>(thisOp.getCount());
  llvm::Value *count =
      hasCount ? mt.lookupValue(thisOp.getCount())
               : llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 1);

  return {id, {mbar, count}};
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

  return {id, {mbar, txcount}};
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

  return {id, {mbar, txcount}};
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
  }

  // Cache hint, if available.
  mlir::Value cacheHint = thisOp.getL2CacheHint();
  const bool hasCacheHint = static_cast<bool>(cacheHint);
  llvm::Value *i64Unused = llvm::ConstantInt::get(builder.getInt64Ty(), 0);
  args.push_back(hasCacheHint ? mt.lookupValue(cacheHint) : i64Unused);

  // Flag arguments for multicast and cachehint.
  if (!isSharedCTA)
    args.push_back(builder.getInt1(hasMulticastMask));
  args.push_back(builder.getInt1(hasCacheHint));

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

  if (!isCTAOnly) {
    // For shared::cluster, all the arguments that we build are applicable.
    args.push_back(hasMC ? mt.lookupValue(mcMask) : i16Zero);
    args.push_back(hasCacheHint ? mt.lookupValue(cacheHint) : i64Zero);
    args.push_back(builder.getInt1(hasMC));
    args.push_back(builder.getInt1(hasCacheHint));
    args.push_back(cg);
  } else {
    // For shared::cta, only cache-hint is applicable.
    args.push_back(hasCacheHint ? mt.lookupValue(cacheHint) : i64Zero);
    args.push_back(builder.getInt1(hasCacheHint));
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

  const unsigned NI = llvm::Intrinsic::not_intrinsic;
  static constexpr llvm::Intrinsic::ID IDTable[][6] = {
      {NI, llvm::Intrinsic::nvvm_cp_async_bulk_tensor_s2g_tile_1d,
       llvm::Intrinsic::nvvm_cp_async_bulk_tensor_s2g_tile_2d,
       llvm::Intrinsic::nvvm_cp_async_bulk_tensor_s2g_tile_3d,
       llvm::Intrinsic::nvvm_cp_async_bulk_tensor_s2g_tile_4d,
       llvm::Intrinsic::nvvm_cp_async_bulk_tensor_s2g_tile_5d},
      {NI, NI, NI, llvm::Intrinsic::nvvm_cp_async_bulk_tensor_s2g_im2col_3d,
       llvm::Intrinsic::nvvm_cp_async_bulk_tensor_s2g_im2col_4d,
       llvm::Intrinsic::nvvm_cp_async_bulk_tensor_s2g_im2col_5d},
      {NI, NI, NI, NI, NI,
       llvm::Intrinsic::nvvm_cp_async_bulk_tensor_s2g_tile_scatter4_2d}};

  static_assert(getMaxEnumValForTMAStoreMode() == std::size(IDTable) - 1,
                "TMAStoreModes must match number of rows in IDTable");
  size_t mode = static_cast<size_t>(thisOp.getMode());
  size_t dim = thisOp.getCoordinates().size();
  llvm::Intrinsic::ID id = IDTable[mode][dim];
  if (id == llvm::Intrinsic::not_intrinsic)
    llvm_unreachable(
        "Invalid intrinsic for CpAsyncBulkTensorSharedCTAToGlobalOp.");

  return {id, std::move(args)};
}

NVVM::IDArgPair CpAsyncBulkTensorReduceOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::CpAsyncBulkTensorReduceOp>(op);
  llvm::LLVMContext &ctx = mt.getLLVMContext();

  llvm::SmallVector<llvm::Value *> args;

  // Arguments to the intrinsic:
  // shared_mem_ptr, tmaDesc, tensorDims
  // cache_hint(if applicable) and flag(boolean)
  args.push_back(mt.lookupValue(thisOp.getSrcMem()));
  args.push_back(mt.lookupValue(thisOp.getTmaDescriptor()));

  for (Value v : thisOp.getCoordinates())
    args.push_back(mt.lookupValue(v));

  mlir::Value cacheHint = thisOp.getL2CacheHint();
  const bool hasCacheHint = static_cast<bool>(cacheHint);
  llvm::Value *i64ZeroValue =
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx), 0);
  args.push_back(hasCacheHint ? mt.lookupValue(cacheHint) : i64ZeroValue);
  args.push_back(builder.getInt1(hasCacheHint));

  const llvm::Intrinsic::ID notIntrinsic = llvm::Intrinsic::not_intrinsic;

  constexpr unsigned numRedKinds = 8; // ADD, MIN, MAX, INC, DEC, AND, OR, XOR
  constexpr unsigned numLayouts = 2;  // TILE, IM2COL
  constexpr unsigned maxDim = 5;      // 1D to 5D
  using row = std::array<llvm::Intrinsic::ID, maxDim + 1>;
  using layoutTable = std::array<row, numLayouts>;
  using fullTable = std::array<layoutTable, numRedKinds>;
  static constexpr fullTable IDTable{
      {// RedTy::ADD
       {{{{notIntrinsic,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_add_tile_1d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_add_tile_2d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_add_tile_3d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_add_tile_4d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_add_tile_5d}},
         {{notIntrinsic, notIntrinsic, notIntrinsic,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_add_im2col_3d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_add_im2col_4d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_add_im2col_5d}}}},
       // RedTy::MIN
       {{{{notIntrinsic,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_min_tile_1d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_min_tile_2d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_min_tile_3d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_min_tile_4d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_min_tile_5d}},
         {{notIntrinsic, notIntrinsic, notIntrinsic,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_min_im2col_3d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_min_im2col_4d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_min_im2col_5d}}}},
       // RedTy::MAX
       {{{{notIntrinsic,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_max_tile_1d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_max_tile_2d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_max_tile_3d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_max_tile_4d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_max_tile_5d}},
         {{notIntrinsic, notIntrinsic, notIntrinsic,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_max_im2col_3d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_max_im2col_4d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_max_im2col_5d}}}},
       // RedTy::INC
       {{{{notIntrinsic,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_inc_tile_1d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_inc_tile_2d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_inc_tile_3d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_inc_tile_4d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_inc_tile_5d}},
         {{notIntrinsic, notIntrinsic, notIntrinsic,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_inc_im2col_3d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_inc_im2col_4d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_inc_im2col_5d}}}},
       // RedTy::DEC
       {{{{notIntrinsic,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_dec_tile_1d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_dec_tile_2d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_dec_tile_3d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_dec_tile_4d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_dec_tile_5d}},
         {{notIntrinsic, notIntrinsic, notIntrinsic,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_dec_im2col_3d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_dec_im2col_4d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_dec_im2col_5d}}}},
       // RedTy::AND
       {{{{notIntrinsic,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_and_tile_1d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_and_tile_2d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_and_tile_3d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_and_tile_4d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_and_tile_5d}},
         {{notIntrinsic, notIntrinsic, notIntrinsic,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_and_im2col_3d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_and_im2col_4d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_and_im2col_5d}}}},
       // RedTy::OR
       {{{{notIntrinsic,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_or_tile_1d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_or_tile_2d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_or_tile_3d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_or_tile_4d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_or_tile_5d}},
         {{notIntrinsic, notIntrinsic, notIntrinsic,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_or_im2col_3d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_or_im2col_4d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_or_im2col_5d}}}},
       // RedTy::XOR
       {{{{notIntrinsic,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_xor_tile_1d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_xor_tile_2d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_xor_tile_3d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_xor_tile_4d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_xor_tile_5d}},
         {{notIntrinsic, notIntrinsic, notIntrinsic,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_xor_im2col_3d,
           llvm::Intrinsic::nvvm_cp_async_bulk_tensor_reduce_xor_im2col_4d,
           llvm::Intrinsic::
               nvvm_cp_async_bulk_tensor_reduce_xor_im2col_5d}}}}}};

  static_assert(getMaxEnumValForTMAReduxKind() == std::size(IDTable) - 1,
                "TMAReduxKinds must match number of rows in IDTable");

  size_t redKind = static_cast<size_t>(thisOp.getRedKind());
  size_t mode = static_cast<size_t>(thisOp.getMode());
  size_t dim = thisOp.getCoordinates().size();

  assert(redKind < IDTable.size() &&
         "Invalid redKind for CpAsyncBulkTensorReduceOp");
  assert(mode < IDTable[redKind].size() &&
         "Invalid mode for CpAsyncBulkTensorReduceOp");
  assert(dim < IDTable[redKind][mode].size() &&
         "Invalid dim for CpAsyncBulkTensorReduceOp");

  llvm::Intrinsic::ID intrinsicID = IDTable[redKind][mode][dim];

  assert(intrinsicID != notIntrinsic &&
         "Invalid intrinsic for CpAsyncBulkTensorReduceOp.");

  return {intrinsicID, std::move(args)};
}

#define _none

#define CVT_F2TF32_ID_IMPL(rnd, relu, sf)                                      \
  hasRelu ? llvm::Intrinsic::nvvm_f2tf32_##rnd##relu##sf                       \
          : llvm::Intrinsic::nvvm_f2tf32_##rnd##sf

#define GET_CVT_F2TF32_ID(rnd, relu, sf)                                       \
  hasSatFinite ? CVT_F2TF32_ID_IMPL(rnd, relu, sf)                             \
               : CVT_F2TF32_ID_IMPL(rnd, relu, )

llvm::Intrinsic::ID
ConvertFloatToTF32Op::getIntrinsicID(NVVM::FPRoundingMode rnd,
                                     NVVM::SaturationMode sat, bool hasRelu) {
  using RndMode = NVVM::FPRoundingMode;
  bool hasSatFinite = (sat == NVVM::SaturationMode::SATFINITE);
  switch (rnd) {
  case RndMode::RN:
    return GET_CVT_F2TF32_ID(rn, _relu, _satfinite);
  case RndMode::RZ:
    return GET_CVT_F2TF32_ID(rz, _relu, _satfinite);
  case RndMode::RNA:
    return GET_CVT_F2TF32_ID(rna, _none, _satfinite);
  default:
    llvm_unreachable("Invalid RoundingMode for CvtFloatToTF32Op");
  }
}

NVVM::IDArgPair
ConvertF32x2ToF4x2Op::getIntrinsicIDAndArgs(NVVM::ConvertF32x2ToF4x2Op op,
                                            LLVM::ModuleTranslation &mt,
                                            llvm::IRBuilderBase &builder) {
  llvm::SmallVector<llvm::Value *> args;
  args.push_back(mt.lookupValue(op.getA()));
  args.push_back(mt.lookupValue(op.getB()));

  bool hasRelu = op.getRelu();

  llvm::Intrinsic::ID intId =
      hasRelu ? llvm::Intrinsic::nvvm_ff_to_e2m1x2_rn_relu_satfinite
              : llvm::Intrinsic::nvvm_ff_to_e2m1x2_rn_satfinite;

  return {intId, std::move(args)};
}

#define GET_F32x2_TO_F6x2_ID(type, has_relu)                                   \
  has_relu ? llvm::Intrinsic::nvvm_ff_to_##type##_rn_relu_satfinite            \
           : llvm::Intrinsic::nvvm_ff_to_##type##_rn_satfinite

llvm::Intrinsic::ID ConvertF32x2ToF6x2Op::getIntrinsicID(mlir::Type dstTy,
                                                         bool hasRelu) {
  return llvm::TypeSwitch<mlir::Type, llvm::Intrinsic::ID>(dstTy)
      .Case<mlir::Float6E2M3FNType>([&](mlir::Float6E2M3FNType) {
        return GET_F32x2_TO_F6x2_ID(e2m3x2, hasRelu);
      })
      .Case<mlir::Float6E3M2FNType>([&](mlir::Float6E3M2FNType) {
        return GET_F32x2_TO_F6x2_ID(e3m2x2, hasRelu);
      })
      .Default([](mlir::Type) {
        llvm_unreachable("Invalid conversion in ConvertF32x2ToF6x2Op");
        return llvm::Intrinsic::not_intrinsic;
      });
}

#define GET_F32x2_TO_F8X2_US_ID(rnd, has_satf)                                 \
  has_satf ? llvm::Intrinsic::nvvm_ff_to_ue8m0x2_##rnd##_satfinite             \
           : llvm::Intrinsic::nvvm_ff_to_ue8m0x2_##rnd

#define GET_F32x2_TO_F8X2_S_ID(type, has_relu)                                 \
  has_relu ? llvm::Intrinsic::nvvm_ff_to_##type##_rn_relu                      \
           : llvm::Intrinsic::nvvm_ff_to_##type##_rn

llvm::Intrinsic::ID
ConvertF32x2ToF8x2Op::getIntrinsicID(mlir::Type dstTy, NVVM::FPRoundingMode rnd,
                                     NVVM::SaturationMode sat, bool hasRelu) {
  bool hasSatFinite = (sat == NVVM::SaturationMode::SATFINITE);
  bool hasRoundingModeRZ = (rnd == NVVM::FPRoundingMode::RZ);
  bool hasRoundingModeRP = (rnd == NVVM::FPRoundingMode::RP);

  return llvm::TypeSwitch<mlir::Type, llvm::Intrinsic::ID>(dstTy)
      .Case<mlir::Float8E4M3FNType>([&](mlir::Float8E4M3FNType) {
        return GET_F32x2_TO_F8X2_S_ID(e4m3x2, hasRelu);
      })
      .Case<mlir::Float8E5M2Type>([&](mlir::Float8E5M2Type) {
        return GET_F32x2_TO_F8X2_S_ID(e5m2x2, hasRelu);
      })
      .Case<mlir::Float8E8M0FNUType>([&](mlir::Float8E8M0FNUType) {
        if (hasRoundingModeRZ)
          return GET_F32x2_TO_F8X2_US_ID(rz, hasSatFinite);
        else if (hasRoundingModeRP)
          return GET_F32x2_TO_F8X2_US_ID(rp, hasSatFinite);

        llvm_unreachable("Invalid conversion in ConvertF32x2ToF8x2Op");
      })
      .Default([](mlir::Type) {
        llvm_unreachable("Invalid conversion in ConvertF32x2ToF8x2Op");
        return llvm::Intrinsic::not_intrinsic;
      });
}

#define GET_F16x2_TO_F8X2_ID(type, has_relu)                                   \
  has_relu ? llvm::Intrinsic::nvvm_f16x2_to_##type##_rn_relu                   \
           : llvm::Intrinsic::nvvm_f16x2_to_##type##_rn

llvm::Intrinsic::ID ConvertF16x2ToF8x2Op::getIntrinsicID(mlir::Type dstTy,
                                                         bool hasRelu) {
  return llvm::TypeSwitch<mlir::Type, llvm::Intrinsic::ID>(dstTy)
      .Case<mlir::Float8E4M3FNType>([&](mlir::Float8E4M3FNType) {
        return GET_F16x2_TO_F8X2_ID(e4m3x2, hasRelu);
      })
      .Case<mlir::Float8E5M2Type>([&](mlir::Float8E5M2Type) {
        return GET_F16x2_TO_F8X2_ID(e5m2x2, hasRelu);
      })
      .Default([](mlir::Type) {
        llvm_unreachable("Invalid conversion in ConvertF16x2ToF8x2Op");
        return llvm::Intrinsic::not_intrinsic;
      });
}

#define GET_BF16X2_TO_F8X2_ID(rnd, has_satf)                                   \
  has_satf ? llvm::Intrinsic::nvvm_bf16x2_to_ue8m0x2_##rnd##_satfinite         \
           : llvm::Intrinsic::nvvm_bf16x2_to_ue8m0x2_##rnd

llvm::Intrinsic::ID
ConvertBF16x2ToF8x2Op::getIntrinsicID(NVVM::FPRoundingMode rnd,
                                      NVVM::SaturationMode sat) {
  bool hasSatFinite = (sat == NVVM::SaturationMode::SATFINITE);
  switch (rnd) {
  case NVVM::FPRoundingMode::RZ:
    return GET_BF16X2_TO_F8X2_ID(rz, hasSatFinite);
  case NVVM::FPRoundingMode::RP:
    return GET_BF16X2_TO_F8X2_ID(rp, hasSatFinite);
  default:
    llvm_unreachable("Invalid rounding mode for CvtBF16x2ToF8x2Op");
  }
}

NVVM::IDArgPair ConvertF8x2ToF16x2Op::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto curOp = cast<NVVM::ConvertF8x2ToF16x2Op>(op);

  bool hasRelu = curOp.getRelu();

  llvm::Intrinsic::ID intId =
      llvm::TypeSwitch<mlir::Type, llvm::Intrinsic::ID>(curOp.getSrcType())
          .Case<Float8E4M3FNType>([&](Float8E4M3FNType type) {
            return hasRelu ? llvm::Intrinsic::nvvm_e4m3x2_to_f16x2_rn_relu
                           : llvm::Intrinsic::nvvm_e4m3x2_to_f16x2_rn;
          })
          .Case<Float8E5M2Type>([&](Float8E5M2Type type) {
            return hasRelu ? llvm::Intrinsic::nvvm_e5m2x2_to_f16x2_rn_relu
                           : llvm::Intrinsic::nvvm_e5m2x2_to_f16x2_rn;
          })
          .Default([](mlir::Type type) {
            llvm_unreachable("Invalid type for ConvertF8x2ToF16x2Op");
            return llvm::Intrinsic::not_intrinsic;
          });

  llvm::Value *packedI16 =
      builder.CreateBitCast(mt.lookupValue(curOp.getSrc()),
                            llvm::Type::getInt16Ty(builder.getContext()));

  return {intId, {packedI16}};
}

NVVM::IDArgPair ConvertF8x2ToBF16x2Op::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto curOp = cast<NVVM::ConvertF8x2ToBF16x2Op>(op);

  llvm::Intrinsic::ID intId = llvm::Intrinsic::nvvm_ue8m0x2_to_bf16x2;
  llvm::Value *packedI16 =
      builder.CreateBitCast(mt.lookupValue(curOp.getSrc()),
                            llvm::Type::getInt16Ty(builder.getContext()));

  return {intId, {packedI16}};
}

NVVM::IDArgPair ConvertF6x2ToF16x2Op::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto curOp = cast<NVVM::ConvertF6x2ToF16x2Op>(op);

  bool hasRelu = curOp.getRelu();

  llvm::Intrinsic::ID intId =
      llvm::TypeSwitch<mlir::Type, llvm::Intrinsic::ID>(curOp.getSrcType())
          .Case<Float6E2M3FNType>([&](Float6E2M3FNType type) {
            return hasRelu ? llvm::Intrinsic::nvvm_e2m3x2_to_f16x2_rn_relu
                           : llvm::Intrinsic::nvvm_e2m3x2_to_f16x2_rn;
          })
          .Case<Float6E3M2FNType>([&](Float6E3M2FNType type) {
            return hasRelu ? llvm::Intrinsic::nvvm_e3m2x2_to_f16x2_rn_relu
                           : llvm::Intrinsic::nvvm_e3m2x2_to_f16x2_rn;
          })
          .Default([](mlir::Type type) {
            llvm_unreachable("Invalid type for ConvertF6x2ToF16x2Op");
            return llvm::Intrinsic::not_intrinsic;
          });

  llvm::Value *packedI16 =
      builder.CreateBitCast(mt.lookupValue(curOp.getSrc()),
                            llvm::Type::getInt16Ty(builder.getContext()));

  return {intId, {packedI16}};
}

NVVM::IDArgPair ConvertF4x2ToF16x2Op::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto curOp = cast<NVVM::ConvertF4x2ToF16x2Op>(op);

  bool hasRelu = curOp.getRelu();

  llvm::Intrinsic::ID intId =
      llvm::TypeSwitch<mlir::Type, llvm::Intrinsic::ID>(curOp.getSrcType())
          .Case<Float4E2M1FNType>([&](Float4E2M1FNType type) {
            return hasRelu ? llvm::Intrinsic::nvvm_e2m1x2_to_f16x2_rn_relu
                           : llvm::Intrinsic::nvvm_e2m1x2_to_f16x2_rn;
          })
          .Default([](mlir::Type type) {
            llvm_unreachable("Invalid type for ConvertF4x2ToF16x2Op");
            return llvm::Intrinsic::not_intrinsic;
          });

  llvm::Value *extendedI16 =
      builder.CreateZExt(mt.lookupValue(curOp.getSrc()),
                         llvm::Type::getInt16Ty(builder.getContext()));

  return {intId, {extendedI16}};
}

llvm::Intrinsic::ID
Tcgen05AllocOp::getIntrinsicIDAndArgs(Operation &op,
                                      LLVM::ModuleTranslation &mt,
                                      llvm::SmallVector<llvm::Value *> &args) {
  auto curOp = cast<NVVM::Tcgen05AllocOp>(op);
  unsigned as = llvm::cast<LLVM::LLVMPointerType>(curOp.getAddr().getType())
                    .getAddressSpace();
  bool isShared = as == NVVMMemorySpace::Shared;
  bool is2CTAMode = curOp.getGroup() == CTAGroupKind::CTA_2;

  llvm::Intrinsic::ID id;
  if (isShared) {
    id = is2CTAMode ? llvm::Intrinsic::nvvm_tcgen05_alloc_shared_cg2
                    : llvm::Intrinsic::nvvm_tcgen05_alloc_shared_cg1;
  } else {
    id = is2CTAMode ? llvm::Intrinsic::nvvm_tcgen05_alloc_cg2
                    : llvm::Intrinsic::nvvm_tcgen05_alloc_cg1;
  }

  // Fill the Intrinsic Args
  args.push_back(mt.lookupValue(curOp.getAddr()));
  args.push_back(mt.lookupValue(curOp.getNCols()));

  return id;
}

llvm::Intrinsic::ID Tcgen05DeallocOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt,
    llvm::SmallVector<llvm::Value *> &args) {
  auto curOp = cast<NVVM::Tcgen05DeallocOp>(op);
  auto id = (curOp.getGroup() == CTAGroupKind::CTA_1)
                ? llvm::Intrinsic::nvvm_tcgen05_dealloc_cg1
                : llvm::Intrinsic::nvvm_tcgen05_dealloc_cg2;

  // Fill the Intrinsic Args
  args.push_back(mt.lookupValue(curOp.getTaddr()));
  args.push_back(mt.lookupValue(curOp.getNCols()));

  return id;
}

#define TCGEN05_COMMIT_IMPL(cg, is_shared, mc)                                 \
  is_shared ? llvm::Intrinsic::nvvm_tcgen05_commit##mc##_shared##_##cg         \
            : llvm::Intrinsic::nvvm_tcgen05_commit##mc##_##cg

#define GET_TCGEN05_COMMIT_ID(cta_group, is_shared, has_mc)                    \
  has_mc ? TCGEN05_COMMIT_IMPL(cta_group, is_shared, _mc)                      \
         : TCGEN05_COMMIT_IMPL(cta_group, is_shared, )

llvm::Intrinsic::ID
Tcgen05CommitOp::getIntrinsicIDAndArgs(Operation &op,
                                       LLVM::ModuleTranslation &mt,
                                       llvm::SmallVector<llvm::Value *> &args) {
  auto curOp = cast<NVVM::Tcgen05CommitOp>(op);
  unsigned as = llvm::cast<LLVM::LLVMPointerType>(curOp.getAddr().getType())
                    .getAddressSpace();
  bool isShared = as == NVVMMemorySpace::Shared;
  bool hasMulticast = static_cast<bool>(curOp.getMulticastMask());
  bool is2CTAMode = curOp.getGroup() == CTAGroupKind::CTA_2;

  llvm::Intrinsic::ID id =
      is2CTAMode ? GET_TCGEN05_COMMIT_ID(cg2, isShared, hasMulticast)
                 : GET_TCGEN05_COMMIT_ID(cg1, isShared, hasMulticast);

  // Fill the Intrinsic Args
  args.push_back(mt.lookupValue(curOp.getAddr()));
  if (hasMulticast)
    args.push_back(mt.lookupValue(curOp.getMulticastMask()));

  return id;
}

#define TCGEN05_CP_IMPL(shape_mc, src_fmt, cg)                                 \
  llvm::Intrinsic::nvvm_tcgen05_cp##shape_mc##src_fmt##cg

#define TCGEN05_CP_2CTA(shape_mc, src_fmt, is_2cta)                            \
  is_2cta ? TCGEN05_CP_IMPL(shape_mc, src_fmt, _cg2)                           \
          : TCGEN05_CP_IMPL(shape_mc, src_fmt, _cg1)

#define GET_TCGEN05_CP_ID(shape_mc, src_fmt, is_2cta)                          \
  [&]() -> auto {                                                              \
    if ((src_fmt) == Tcgen05CpSrcFormat::B6x16_P32)                            \
      return TCGEN05_CP_2CTA(shape_mc, _b6x16_p32, is_2cta);                   \
    if ((src_fmt) == Tcgen05CpSrcFormat::B4x16_P64)                            \
      return TCGEN05_CP_2CTA(shape_mc, _b4x16_p64, is_2cta);                   \
    return TCGEN05_CP_2CTA(shape_mc, , is_2cta);                               \
  }()

NVVM::IDArgPair
ConvertF32x2ToF16x2Op::getIntrinsicIDAndArgs(NVVM::ConvertF32x2ToF16x2Op &op,
                                             LLVM::ModuleTranslation &mt,
                                             llvm::IRBuilderBase &builder) {
  static constexpr llvm::Intrinsic::ID rndRNIds[] = {
      llvm::Intrinsic::nvvm_ff2f16x2_rn,
      llvm::Intrinsic::nvvm_ff2f16x2_rn_relu,
      llvm::Intrinsic::nvvm_ff2f16x2_rn_satfinite,
      llvm::Intrinsic::nvvm_ff2f16x2_rn_relu_satfinite,
  };
  static constexpr llvm::Intrinsic::ID rndRZIds[] = {
      llvm::Intrinsic::nvvm_ff2f16x2_rz,
      llvm::Intrinsic::nvvm_ff2f16x2_rz_relu,
      llvm::Intrinsic::nvvm_ff2f16x2_rz_satfinite,
      llvm::Intrinsic::nvvm_ff2f16x2_rz_relu_satfinite,
  };
  static constexpr llvm::Intrinsic::ID rndRSIds[] = {
      llvm::Intrinsic::nvvm_ff2f16x2_rs,
      llvm::Intrinsic::nvvm_ff2f16x2_rs_relu,
      llvm::Intrinsic::nvvm_ff2f16x2_rs_satfinite,
      llvm::Intrinsic::nvvm_ff2f16x2_rs_relu_satfinite,
  };

  unsigned hasRelu = op.getRelu() ? 1 : 0;
  unsigned hasSatFinite =
      (op.getSat() == NVVM::SaturationMode::SATFINITE) ? 1 : 0;
  // idx: bit-0 - relu
  //      bit-1 - satfinite
  unsigned idx = (hasSatFinite << 1) | hasRelu;

  llvm::SmallVector<llvm::Value *> args;
  args.push_back(mt.lookupValue(op.getSrcHi()));
  args.push_back(mt.lookupValue(op.getSrcLo()));
  if (op.getRandomBits())
    args.push_back(mt.lookupValue(op.getRandomBits()));

  switch (op.getRnd()) {
  case FPRoundingMode::RN:
    return {rndRNIds[idx], std::move(args)};
  case FPRoundingMode::RZ:
    return {rndRZIds[idx], std::move(args)};
  case FPRoundingMode::RS:
    return {rndRSIds[idx], std::move(args)};
  default:
    llvm_unreachable("Invalid rounding mode for ConvertF32x2ToF16x2Op");
  }
}

NVVM::IDArgPair
ConvertF32x2ToBF16x2Op::getIntrinsicIDAndArgs(NVVM::ConvertF32x2ToBF16x2Op &op,
                                              LLVM::ModuleTranslation &mt,
                                              llvm::IRBuilderBase &builder) {
  static constexpr llvm::Intrinsic::ID rndRNIds[] = {
      llvm::Intrinsic::nvvm_ff2bf16x2_rn,
      llvm::Intrinsic::nvvm_ff2bf16x2_rn_relu,
      llvm::Intrinsic::nvvm_ff2bf16x2_rn_satfinite,
      llvm::Intrinsic::nvvm_ff2bf16x2_rn_relu_satfinite,
  };
  static constexpr llvm::Intrinsic::ID rndRZIds[] = {
      llvm::Intrinsic::nvvm_ff2bf16x2_rz,
      llvm::Intrinsic::nvvm_ff2bf16x2_rz_relu,
      llvm::Intrinsic::nvvm_ff2bf16x2_rz_satfinite,
      llvm::Intrinsic::nvvm_ff2bf16x2_rz_relu_satfinite,
  };
  static constexpr llvm::Intrinsic::ID rndRSIds[] = {
      llvm::Intrinsic::nvvm_ff2bf16x2_rs,
      llvm::Intrinsic::nvvm_ff2bf16x2_rs_relu,
      llvm::Intrinsic::nvvm_ff2bf16x2_rs_satfinite,
      llvm::Intrinsic::nvvm_ff2bf16x2_rs_relu_satfinite,
  };

  unsigned hasRelu = op.getRelu() ? 1 : 0;
  unsigned hasSatFinite =
      (op.getSat() == NVVM::SaturationMode::SATFINITE) ? 1 : 0;
  // idx: bit-0 - relu
  //      bit-1 - satfinite
  unsigned idx = (hasSatFinite << 1) | hasRelu;

  llvm::SmallVector<llvm::Value *> args;
  args.push_back(mt.lookupValue(op.getSrcHi()));
  args.push_back(mt.lookupValue(op.getSrcLo()));
  if (op.getRandomBits())
    args.push_back(mt.lookupValue(op.getRandomBits()));

  switch (op.getRnd()) {
  case FPRoundingMode::RN:
    return {rndRNIds[idx], std::move(args)};
  case FPRoundingMode::RZ:
    return {rndRZIds[idx], std::move(args)};
  case FPRoundingMode::RS:
    return {rndRSIds[idx], std::move(args)};
  default:
    llvm_unreachable("Invalid rounding mode for ConvertF32x2ToBF16x2Op");
  }
}

llvm::Intrinsic::ID ConvertF32x4ToF8x4Op::getIntrinsicID() {
  mlir::Type dstTy = getDstTy();
  bool hasRelu = getRelu();

  return llvm::TypeSwitch<mlir::Type, llvm::Intrinsic::ID>(dstTy)
      .Case<mlir::Float8E4M3FNType>([&](mlir::Float8E4M3FNType) {
        return hasRelu ? llvm::Intrinsic::nvvm_f32x4_to_e4m3x4_rs_relu_satfinite
                       : llvm::Intrinsic::nvvm_f32x4_to_e4m3x4_rs_satfinite;
      })
      .Case<mlir::Float8E5M2Type>([&](mlir::Float8E5M2Type) {
        return hasRelu ? llvm::Intrinsic::nvvm_f32x4_to_e5m2x4_rs_relu_satfinite
                       : llvm::Intrinsic::nvvm_f32x4_to_e5m2x4_rs_satfinite;
      })
      .Default([](mlir::Type) {
        llvm_unreachable("Invalid F8 type in ConvertF32x4ToF8x4Op");
        return llvm::Intrinsic::not_intrinsic;
      });
}

llvm::Intrinsic::ID ConvertF32x4ToF6x4Op::getIntrinsicID() {
  mlir::Type dstTy = getDstTy();
  bool hasRelu = getRelu();

  return llvm::TypeSwitch<mlir::Type, llvm::Intrinsic::ID>(dstTy)
      .Case<mlir::Float6E2M3FNType>([&](mlir::Float6E2M3FNType) {
        return hasRelu ? llvm::Intrinsic::nvvm_f32x4_to_e2m3x4_rs_relu_satfinite
                       : llvm::Intrinsic::nvvm_f32x4_to_e2m3x4_rs_satfinite;
      })
      .Case<mlir::Float6E3M2FNType>([&](mlir::Float6E3M2FNType) {
        return hasRelu ? llvm::Intrinsic::nvvm_f32x4_to_e3m2x4_rs_relu_satfinite
                       : llvm::Intrinsic::nvvm_f32x4_to_e3m2x4_rs_satfinite;
      })
      .Default([](mlir::Type) {
        llvm_unreachable("Invalid F6 type in ConvertF32x4ToF6x4Op");
        return llvm::Intrinsic::not_intrinsic;
      });
}

llvm::Intrinsic::ID ConvertF32x4ToF4x4Op::getIntrinsicID() {
  mlir::Type dstTy = getDstTy();
  bool hasRelu = getRelu();

  return llvm::TypeSwitch<mlir::Type, llvm::Intrinsic::ID>(dstTy)
      .Case<mlir::Float4E2M1FNType>([&](mlir::Float4E2M1FNType) {
        return hasRelu ? llvm::Intrinsic::nvvm_f32x4_to_e2m1x4_rs_relu_satfinite
                       : llvm::Intrinsic::nvvm_f32x4_to_e2m1x4_rs_satfinite;
      })
      .Default([](mlir::Type) {
        llvm_unreachable("Invalid F4 type in ConvertF32x4ToF4x4Op");
        return llvm::Intrinsic::not_intrinsic;
      });
}

llvm::Intrinsic::ID Tcgen05CpOp::getIntrinsicID(Operation &op) {
  auto curOp = cast<NVVM::Tcgen05CpOp>(op);
  bool is2CTA = curOp.getGroup() == CTAGroupKind::CTA_2;
  auto srcFmt = curOp.getSrcFormat();
  auto mc = curOp.getMulticast();

  switch (curOp.getShape()) {
  case Tcgen05CpShape::SHAPE_128x256b:
    return GET_TCGEN05_CP_ID(_128x256b, srcFmt, is2CTA);
  case Tcgen05CpShape::SHAPE_128x128b:
    return GET_TCGEN05_CP_ID(_128x128b, srcFmt, is2CTA);
  case Tcgen05CpShape::SHAPE_4x256b:
    return GET_TCGEN05_CP_ID(_4x256b, srcFmt, is2CTA);
  case Tcgen05CpShape::SHAPE_32x128b:
    return GET_TCGEN05_CP_ID(_32x128b_warpx4, srcFmt, is2CTA);
  case Tcgen05CpShape::SHAPE_64x128b:
    return (mc == Tcgen05CpMulticast::WARPX2_01_23)
               ? GET_TCGEN05_CP_ID(_64x128b_warpx2_01_23, srcFmt, is2CTA)
               : GET_TCGEN05_CP_ID(_64x128b_warpx2_02_13, srcFmt, is2CTA);
  }
  llvm_unreachable("Invalid shape in tcgen05 cp Op");
}

// Returns the valid vector length for a given shape and vector length, the
// function models the table mentioned in the tcgen05.{ld, st} Op description
static unsigned isValidVectorLength(NVVM::Tcgen05LdStShape shape,
                                    unsigned vecLen) {
  if (shape == NVVM::Tcgen05LdStShape::SHAPE_16X128B)
    return vecLen >= 2;
  if (shape == NVVM::Tcgen05LdStShape::SHAPE_16X256B)
    return vecLen >= 4;
  return true;
}

LogicalResult Tcgen05LdOp::verify() {
  LogicalResult result = success();
  if (getShape() == NVVM::Tcgen05LdStShape::SHAPE_16X32BX2 && !getOffset())
    result = emitError("shape 16x32bx2 requires offset argument");

  if (getShape() != NVVM::Tcgen05LdStShape::SHAPE_16X32BX2 && getOffset())
    result = emitError("offset argument is only supported for shape 16x32bx2");

  auto resTy = getRes().getType();
  unsigned resLen = isa<VectorType>(resTy)
                        ? llvm::cast<VectorType>(resTy).getNumElements()
                        : 1;
  if (!isValidVectorLength(getShape(), resLen))
    result = emitError(llvm::formatv("invalid result type length {0} for shape "
                                     "{1} in tcgen05.ld Op",
                                     resLen, stringifyEnum(getShape())));

  return result;
}

LogicalResult Tcgen05StOp::verify() {
  LogicalResult result = success();
  if (getShape() == NVVM::Tcgen05LdStShape::SHAPE_16X32BX2 && !getOffset())
    result = emitError("shape 16x32bx2 requires offset argument");

  auto valTy = getVal().getType();
  unsigned valLen = isa<VectorType>(valTy)
                        ? llvm::cast<VectorType>(valTy).getNumElements()
                        : 1;
  if (!isValidVectorLength(getShape(), valLen))
    result = emitError(llvm::formatv("invalid input length {0} for shape "
                                     "{1} in tcgen05.st Op",
                                     valLen, stringifyEnum(getShape())));

  return result;
}

/// Infer the result ranges for the NVVM SpecialRangeableRegisterOp that might
/// have ConstantRangeAttr.
static void nvvmInferResultRanges(Operation *op, Value result,
                                  ArrayRef<::mlir::ConstantIntRanges> argRanges,
                                  SetIntRangeFn setResultRanges) {
  if (auto rangeAttr = op->getAttrOfType<LLVM::ConstantRangeAttr>("range")) {
    setResultRanges(result, {rangeAttr.getLower(), rangeAttr.getUpper(),
                             rangeAttr.getLower(), rangeAttr.getUpper()});
  } else {
    setResultRanges(result, IntegerValueRange::getMaxRange(result).getValue());
  }
}

/// Verify the range attribute satisfies LLVM ConstantRange constructor
/// requirements for NVVM SpecialRangeableRegisterOp.
static LogicalResult
verifyConstantRangeAttr(Operation *op,
                        std::optional<LLVM::ConstantRangeAttr> rangeAttr) {
  if (!rangeAttr)
    return success();

  const llvm::APInt &lower = rangeAttr->getLower();
  const llvm::APInt &upper = rangeAttr->getUpper();

  // Check LLVM ConstantRange constructor condition
  if (lower == upper && !lower.isMaxValue() && !lower.isMinValue()) {
    unsigned bitWidth = lower.getBitWidth();
    llvm::APInt minVal = llvm::APInt::getMinValue(bitWidth);
    llvm::APInt maxVal = llvm::APInt::getMaxValue(bitWidth);
    return op->emitOpError(
               "invalid range attribute: Lower == Upper, but they aren't min (")
           << llvm::toString(minVal, 10, false) << ") or max ("
           << llvm::toString(maxVal, 10, false)
           << ") value! This is an invalid constant range.";
  }

  return success();
}

static llvm::Value *getAsPackedI32(llvm::Value *arg,
                                   llvm::IRBuilderBase &builder) {
  return builder.CreateBitCast(arg,
                               llvm::Type::getInt32Ty(builder.getContext()));
}

NVVM::IDArgPair DotAccumulate4WayOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto curOp = cast<NVVM::DotAccumulate4WayOp>(op);

  llvm::SmallVector<llvm::Value *> args;
  args.push_back(getAsPackedI32(mt.lookupValue(curOp.getA()), builder));
  args.push_back(getAsPackedI32(mt.lookupValue(curOp.getB()), builder));
  args.push_back(mt.lookupValue(curOp.getC()));

  bool isASigned = curOp.getAType() == NVVM::DotAccumulateType::SIGNED;
  bool isBSigned = curOp.getBType() == NVVM::DotAccumulateType::SIGNED;
  unsigned type = (isASigned << 1) | isBSigned;
  const llvm::Intrinsic::ID ids[] = {
      llvm::Intrinsic::nvvm_idp4a_u_u,
      llvm::Intrinsic::nvvm_idp4a_u_s,
      llvm::Intrinsic::nvvm_idp4a_s_u,
      llvm::Intrinsic::nvvm_idp4a_s_s,
  };
  return {ids[type], args};
}

NVVM::IDArgPair DotAccumulate2WayOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto curOp = cast<NVVM::DotAccumulate2WayOp>(op);

  llvm::SmallVector<llvm::Value *> args;
  args.push_back(getAsPackedI32(mt.lookupValue(curOp.getA()), builder));
  args.push_back(getAsPackedI32(mt.lookupValue(curOp.getB()), builder));
  args.push_back(builder.getInt1(curOp.getBHi()));
  args.push_back(mt.lookupValue(curOp.getC()));

  bool isASigned = curOp.getAType() == NVVM::DotAccumulateType::SIGNED;
  bool isBSigned = curOp.getBType() == NVVM::DotAccumulateType::SIGNED;
  unsigned type = (isASigned << 1) | isBSigned;
  const llvm::Intrinsic::ID ids[] = {
      llvm::Intrinsic::nvvm_idp2a_u_u,
      llvm::Intrinsic::nvvm_idp2a_u_s,
      llvm::Intrinsic::nvvm_idp2a_s_u,
      llvm::Intrinsic::nvvm_idp2a_s_s,
  };
  return {ids[type], args};
}

static llvm::Value *getParamCastedAddr(llvm::Value *addr,
                                       llvm::IRBuilderBase &builder) {
  return builder.CreateAddrSpaceCast(
      addr,
      llvm::PointerType::get(builder.getContext(),
                             llvm::NVPTXAS::AddressSpace::ADDRESS_SPACE_PARAM));
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

//===----------------------------------------------------------------------===//
// NVVM tcgen05.mma functions
//===----------------------------------------------------------------------===//

mlir::NVVM::IDArgPair
Tcgen05MMAOp::getIntrinsicIDAndArgs(Operation &op, LLVM::ModuleTranslation &mt,
                                    llvm::IRBuilderBase &builder) {

  auto thisOp = cast<NVVM::Tcgen05MMAOp>(op);
  llvm::SmallVector<llvm::Value *> args;

  args.push_back(mt.lookupValue(thisOp.getMatrixD()));

  llvm::Value *A = mt.lookupValue(thisOp.getMatrixA());
  const bool isATensor = isa<llvm::PointerType>(A->getType());
  args.push_back(A);

  args.push_back(mt.lookupValue(thisOp.getMatrixB()));
  args.push_back(mt.lookupValue(thisOp.getIdesc()));
  args.push_back(mt.lookupValue(thisOp.getEnableInputD()));

  using EnableAShiftArray = std::array<llvm::Intrinsic::ID, 2>;
  using CtaGroupArray = std::array<EnableAShiftArray, 2>;
  using IsATensorArray = std::array<CtaGroupArray, 2>;
  using HasScaleInputDArray = std::array<IsATensorArray, 2>;
  using HasDisableOutputLaneArray = std::array<HasScaleInputDArray, 2>;

  // [hasDisableOutputLane][hasScaleInputD][isATensor][CtaGroup][EnableAShift]
  static constexpr HasDisableOutputLaneArray tcgen05MMAIDs = {
      {  // without diable output lane
       {{// without scale input D
         {{
             // shared
             {{// cg1
               {llvm::Intrinsic::nvvm_tcgen05_mma_shared, notIntrinsic},
               // cg2
               {llvm::Intrinsic::nvvm_tcgen05_mma_shared, notIntrinsic}}},
             {{// tensor
               {
                   // cg1
                   llvm::Intrinsic::nvvm_tcgen05_mma_tensor,
                   llvm::Intrinsic::nvvm_tcgen05_mma_tensor_ashift,
               },
               {
                   // cg2
                   llvm::Intrinsic::nvvm_tcgen05_mma_tensor,
                   llvm::Intrinsic::nvvm_tcgen05_mma_tensor_ashift,
               }}},
         }},
         // with scale input D
         {{  // shared
           {{// cg1
             {llvm::Intrinsic::nvvm_tcgen05_mma_shared_scale_d, notIntrinsic},
             // cg2
             {llvm::Intrinsic::nvvm_tcgen05_mma_shared_scale_d, notIntrinsic}}},
           {{// tensor
             {
                 // cg1
                 llvm::Intrinsic::nvvm_tcgen05_mma_tensor_scale_d,
                 llvm::Intrinsic::nvvm_tcgen05_mma_tensor_scale_d_ashift,
             },
             {
                 // cg2
                 llvm::Intrinsic::nvvm_tcgen05_mma_tensor_scale_d,
                 llvm::Intrinsic::nvvm_tcgen05_mma_tensor_scale_d_ashift,
             }}}}}}},
       // with disable output lane
       {{    // without scale input D
         {{  // shared
           {{// cg1
             {llvm::Intrinsic::nvvm_tcgen05_mma_shared_disable_output_lane_cg1,
              notIntrinsic},
             // cg2
             {llvm::Intrinsic::nvvm_tcgen05_mma_shared_disable_output_lane_cg2,
              notIntrinsic}}},
           {{// cg1
             {
                 llvm::Intrinsic::
                     nvvm_tcgen05_mma_tensor_disable_output_lane_cg1,
                 llvm::Intrinsic::
                     nvvm_tcgen05_mma_tensor_disable_output_lane_cg1_ashift,
             },
             // cg2
             {
                 llvm::Intrinsic::
                     nvvm_tcgen05_mma_tensor_disable_output_lane_cg2,
                 llvm::Intrinsic::
                     nvvm_tcgen05_mma_tensor_disable_output_lane_cg2_ashift,
             }}}}},
         // with scale input D
         {{  // shared
           {{// cg1
             {llvm::Intrinsic::
                  nvvm_tcgen05_mma_shared_scale_d_disable_output_lane_cg1,
              notIntrinsic},
             // cg2
             {llvm::Intrinsic::
                  nvvm_tcgen05_mma_shared_scale_d_disable_output_lane_cg2,
              notIntrinsic}}},
           // tensor
           {{// cg1
             {llvm::Intrinsic::
                  nvvm_tcgen05_mma_tensor_scale_d_disable_output_lane_cg1,
              llvm::Intrinsic::
                  nvvm_tcgen05_mma_tensor_scale_d_disable_output_lane_cg1_ashift},
             // cg2
             {
                 llvm::Intrinsic::
                     nvvm_tcgen05_mma_tensor_scale_d_disable_output_lane_cg2,
                 llvm::Intrinsic::
                     nvvm_tcgen05_mma_tensor_scale_d_disable_output_lane_cg2_ashift,
             }}}}}}}}};

  llvm::Value *ScaleInputD = mt.lookupValue(thisOp.getScaleInputD());
  bool hasScaleInputD = ScaleInputD != nullptr;

  llvm::Value *DisableOutputLane =
      mt.lookupValue(thisOp.getDisableOutputLane());
  bool hasDisableOutputLane = DisableOutputLane != nullptr;

  const unsigned ctaGroup =
      static_cast<unsigned>(getNVVMCtaGroupKind(thisOp.getCtaGroup()));

  llvm::Intrinsic::ID ID =
      tcgen05MMAIDs[hasDisableOutputLane][hasScaleInputD][isATensor]
                   [ctaGroup - 1][thisOp.getAShift()];

  assert(ID != notIntrinsic && "Invalid intrinsic for Tcgen05MMAOp.");

  if (hasScaleInputD)
    args.push_back(ScaleInputD);

  if (hasDisableOutputLane)
    args.push_back(DisableOutputLane);

  args.push_back(builder.getInt32(static_cast<unsigned>(thisOp.getKind())));

  if (!hasDisableOutputLane)
    args.push_back(builder.getInt32(ctaGroup));

  args.push_back(
      builder.getInt32(static_cast<unsigned>(thisOp.getCollectorOp())));

  return {ID, args};
}

static LogicalResult
verifyTcgen05MMAOp(bool isATensor, mlir::Value disableOutputLane,
                   NVVM::CTAGroupKind ctaGroup, bool hasAShift,
                   NVVM::Tcgen05MMACollectorOp collectorOp, Location loc) {

  if (disableOutputLane) {
    mlir::VectorType disableOutputLaneType =
        cast<mlir::VectorType>(disableOutputLane.getType());
    if ((ctaGroup == NVVM::CTAGroupKind::CTA_1 &&
         disableOutputLaneType.getNumElements() != 4) ||
        (ctaGroup == NVVM::CTAGroupKind::CTA_2 &&
         disableOutputLaneType.getNumElements() != 8))
      return emitError(loc) << "Disable Output Lane of length "
                            << disableOutputLaneType.getNumElements()
                            << " is incompatible with CtaGroupAttr";
  }

  if (hasAShift && !isATensor)
    return emitError(
        loc, "A-shift can be applied only when matrix A is in tensor memory");

  if (hasAShift == true && (collectorOp == Tcgen05MMACollectorOp::FILL ||
                            collectorOp == Tcgen05MMACollectorOp::USE))
    return emitError(
        loc, "Cannot use collector buffer operation fill or use with ashift");

  return success();
}

LogicalResult Tcgen05MMAOp::verify() {
  return verifyTcgen05MMAOp(isa<LLVM::LLVMPointerType>(getMatrixA().getType()),
                            getDisableOutputLane(), getCtaGroup(), getAShift(),
                            getCollectorOp(), getLoc());
}

//===----------------------------------------------------------------------===//
// NVVM tcgen05.mma.sp functions
//===----------------------------------------------------------------------===//

mlir::NVVM::IDArgPair Tcgen05MMASparseOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {

  auto thisOp = cast<NVVM::Tcgen05MMASparseOp>(op);
  llvm::SmallVector<llvm::Value *> args;

  args.push_back(mt.lookupValue(thisOp.getMatrixD()));

  llvm::Value *A = mt.lookupValue(thisOp.getMatrixA());
  bool isATensor = isa<llvm::PointerType>(A->getType());
  args.push_back(A);

  args.push_back(mt.lookupValue(thisOp.getMatrixB()));
  args.push_back(mt.lookupValue(thisOp.getIdesc()));
  args.push_back(mt.lookupValue(thisOp.getEnableInputD()));
  args.push_back(mt.lookupValue(thisOp.getSparseMetadata()));

  using EnableAShiftArray = std::array<llvm::Intrinsic::ID, 2>;
  using CtaGroupArray = std::array<EnableAShiftArray, 2>;
  using IsATensorArray = std::array<CtaGroupArray, 2>;
  using HasScaleInputDArray = std::array<IsATensorArray, 2>;
  using HasDisableOutputLaneArray = std::array<HasScaleInputDArray, 2>;

  // [hasDisableOutputLane][hasScaleInputD][isATensor][CtaGroup][EnableAShift]
  static constexpr HasDisableOutputLaneArray tcgen05MMASparseIDs = {
      {  // without diable output lane
       {{// without scale input D
         {{
             // shared
             {{// cg1
               {llvm::Intrinsic::nvvm_tcgen05_mma_sp_shared, notIntrinsic},
               // cg2
               {llvm::Intrinsic::nvvm_tcgen05_mma_sp_shared, notIntrinsic}}},
             {{// tensor
               {
                   // cg1
                   llvm::Intrinsic::nvvm_tcgen05_mma_sp_tensor,
                   llvm::Intrinsic::nvvm_tcgen05_mma_sp_tensor_ashift,
               },
               {
                   // cg2
                   llvm::Intrinsic::nvvm_tcgen05_mma_sp_tensor,
                   llvm::Intrinsic::nvvm_tcgen05_mma_sp_tensor_ashift,
               }}},
         }},
         // with scale input D
         {{  // shared
           {{// cg1
             {llvm::Intrinsic::nvvm_tcgen05_mma_sp_shared_scale_d,
              notIntrinsic},
             // cg2
             {llvm::Intrinsic::nvvm_tcgen05_mma_sp_shared_scale_d,
              notIntrinsic}}},
           {{// tensor
             {
                 // cg1
                 llvm::Intrinsic::nvvm_tcgen05_mma_sp_tensor_scale_d,
                 llvm::Intrinsic::nvvm_tcgen05_mma_sp_tensor_scale_d_ashift,
             },
             {
                 // cg2
                 llvm::Intrinsic::nvvm_tcgen05_mma_sp_tensor_scale_d,
                 llvm::Intrinsic::nvvm_tcgen05_mma_sp_tensor_scale_d_ashift,
             }}}}}}},
       // with disable output lane
       {{    // without scale input D
         {{  // shared
           {{// cg1
             {llvm::Intrinsic::
                  nvvm_tcgen05_mma_sp_shared_disable_output_lane_cg1,
              notIntrinsic},
             // cg2
             {llvm::Intrinsic::
                  nvvm_tcgen05_mma_sp_shared_disable_output_lane_cg2,
              notIntrinsic}}},
           {{// cg1
             {
                 llvm::Intrinsic::
                     nvvm_tcgen05_mma_sp_tensor_disable_output_lane_cg1,
                 llvm::Intrinsic::
                     nvvm_tcgen05_mma_sp_tensor_disable_output_lane_cg1_ashift,
             },
             // cg2
             {
                 llvm::Intrinsic::
                     nvvm_tcgen05_mma_sp_tensor_disable_output_lane_cg2,
                 llvm::Intrinsic::
                     nvvm_tcgen05_mma_sp_tensor_disable_output_lane_cg2_ashift,
             }}}}},
         // with scale input D
         {{  // shared
           {{// cg1
             {llvm::Intrinsic::
                  nvvm_tcgen05_mma_sp_shared_scale_d_disable_output_lane_cg1,
              notIntrinsic},
             // cg2
             {llvm::Intrinsic::
                  nvvm_tcgen05_mma_sp_shared_scale_d_disable_output_lane_cg2,
              notIntrinsic}}},
           // tensor
           {{// cg1
             {llvm::Intrinsic::
                  nvvm_tcgen05_mma_sp_tensor_scale_d_disable_output_lane_cg1,
              llvm::Intrinsic::
                  nvvm_tcgen05_mma_sp_tensor_scale_d_disable_output_lane_cg1_ashift},
             // cg2
             {
                 llvm::Intrinsic::
                     nvvm_tcgen05_mma_sp_tensor_scale_d_disable_output_lane_cg2,
                 llvm::Intrinsic::
                     nvvm_tcgen05_mma_sp_tensor_scale_d_disable_output_lane_cg2_ashift,
             }}}}}}}}};

  llvm::Value *ScaleInputD = mt.lookupValue(thisOp.getScaleInputD());
  bool hasScaleInputD = ScaleInputD != nullptr;

  llvm::Value *DisableOutputLane =
      mt.lookupValue(thisOp.getDisableOutputLane());
  bool hasDisableOutputLane = DisableOutputLane != nullptr;

  unsigned ctaGroup =
      static_cast<unsigned>(getNVVMCtaGroupKind(thisOp.getCtaGroup()));

  llvm::Intrinsic::ID ID =
      tcgen05MMASparseIDs[hasDisableOutputLane][hasScaleInputD][isATensor]
                         [ctaGroup - 1][thisOp.getAShift()];

  assert(ID != notIntrinsic && "Invalid intrinsic for Tcgen05MMASparseOp.");

  if (hasScaleInputD)
    args.push_back(ScaleInputD);

  if (hasDisableOutputLane)
    args.push_back(DisableOutputLane);

  args.push_back(builder.getInt32(static_cast<unsigned>(thisOp.getKind())));

  if (!hasDisableOutputLane)
    args.push_back(builder.getInt32(ctaGroup));

  args.push_back(
      builder.getInt32(static_cast<unsigned>(thisOp.getCollectorOp())));

  return {ID, args};
}

LogicalResult Tcgen05MMASparseOp::verify() {
  return verifyTcgen05MMAOp(isa<LLVM::LLVMPointerType>(getMatrixA().getType()),
                            getDisableOutputLane(), getCtaGroup(), getAShift(),
                            getCollectorOp(), getLoc());
}

//===----------------------------------------------------------------------===//
// NVVM tcgen05.mma.block_scale functions
//===----------------------------------------------------------------------===//

mlir::NVVM::IDArgPair Tcgen05MMABlockScaleOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {

  auto thisOp = cast<NVVM::Tcgen05MMABlockScaleOp>(op);
  llvm::SmallVector<llvm::Value *> args;

  args.push_back(mt.lookupValue(thisOp.getMatrixD()));

  llvm::Value *A = mt.lookupValue(thisOp.getMatrixA());
  bool isATensor = isa<llvm::PointerType>(A->getType());
  args.push_back(A);

  args.push_back(mt.lookupValue(thisOp.getMatrixB()));
  args.push_back(mt.lookupValue(thisOp.getIdesc()));
  args.push_back(mt.lookupValue(thisOp.getEnableInputD()));
  args.push_back(mt.lookupValue(thisOp.getScaleA()));
  args.push_back(mt.lookupValue(thisOp.getScaleB()));
  args.push_back(builder.getInt32(
      static_cast<unsigned>(getNVVMCtaGroupKind(thisOp.getCtaGroup()))));
  args.push_back(
      builder.getInt32(static_cast<unsigned>(thisOp.getCollectorOp())));

  auto kind = thisOp.getKind();
  auto blockScale = thisOp.getBlockScale();
  llvm::Intrinsic::ID ID = [&]() {
    if (kind == NVVM::MMABlockScaleKind::MXF8F6F4) {
      if (blockScale == NVVM::Tcgen05MMABlockScale::DEFAULT) {
        return isATensor ? llvm::Intrinsic::
                               nvvm_tcgen05_mma_tensor_mxf8f6f4_block_scale
                         : llvm::Intrinsic::
                               nvvm_tcgen05_mma_shared_mxf8f6f4_block_scale;
      } else if (blockScale == NVVM::Tcgen05MMABlockScale::BLOCK32) {
        return isATensor
                   ? llvm::Intrinsic::
                         nvvm_tcgen05_mma_tensor_mxf8f6f4_block_scale_block32
                   : llvm::Intrinsic::
                         nvvm_tcgen05_mma_shared_mxf8f6f4_block_scale_block32;
      }
    } else if (kind == NVVM::MMABlockScaleKind::MXF4) {
      if (blockScale == NVVM::Tcgen05MMABlockScale::DEFAULT) {
        return isATensor
                   ? llvm::Intrinsic::nvvm_tcgen05_mma_tensor_mxf4_block_scale
                   : llvm::Intrinsic::nvvm_tcgen05_mma_shared_mxf4_block_scale;
      } else if (blockScale == NVVM::Tcgen05MMABlockScale::BLOCK32) {
        return isATensor ? llvm::Intrinsic::
                               nvvm_tcgen05_mma_tensor_mxf4_block_scale_block32
                         : llvm::Intrinsic::
                               nvvm_tcgen05_mma_shared_mxf4_block_scale_block32;
      }
    } else if (kind == NVVM::MMABlockScaleKind::MXF4NVF4) {
      if (blockScale == NVVM::Tcgen05MMABlockScale::BLOCK32) {
        return isATensor
                   ? llvm::Intrinsic::
                         nvvm_tcgen05_mma_tensor_mxf4nvf4_block_scale_block32
                   : llvm::Intrinsic::
                         nvvm_tcgen05_mma_shared_mxf4nvf4_block_scale_block32;

      } else if (blockScale == NVVM::Tcgen05MMABlockScale::BLOCK16) {
        return isATensor
                   ? llvm::Intrinsic::
                         nvvm_tcgen05_mma_tensor_mxf4nvf4_block_scale_block16
                   : llvm::Intrinsic::
                         nvvm_tcgen05_mma_shared_mxf4nvf4_block_scale_block16;
      }
    }
    llvm_unreachable("Invalid tcgen05.mma.block_scale attributes");
  }();

  return {ID, args};
}

static LogicalResult verifyTcgen05MMABlockScaleOp(
    NVVM::Tcgen05MMACollectorOp collectorOp, NVVM::MMABlockScaleKind kind,
    NVVM::Tcgen05MMABlockScale blockScale, Location loc) {

  if (blockScale == NVVM::Tcgen05MMABlockScale::DEFAULT &&
      kind == MMABlockScaleKind::MXF4NVF4)
    return emitError(loc, "mxf4nvf4 requires block scale attribute");

  if (blockScale == NVVM::Tcgen05MMABlockScale::BLOCK16 &&
      kind != MMABlockScaleKind::MXF4NVF4)
    return emitError(loc,
                     llvm::formatv("{} kind does not support block16 attribute",
                                   stringifyEnum(kind)));

  return success();
}

LogicalResult Tcgen05MMABlockScaleOp::verify() {
  return verifyTcgen05MMABlockScaleOp(getCollectorOp(), getKind(),
                                      getBlockScale(), getLoc());
}

//===----------------------------------------------------------------------===//
// NVVM tcgen05.mma.sp.block_scale functions
//===----------------------------------------------------------------------===//

mlir::NVVM::IDArgPair Tcgen05MMASparseBlockScaleOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {

  auto thisOp = cast<NVVM::Tcgen05MMASparseBlockScaleOp>(op);
  llvm::SmallVector<llvm::Value *> args;

  args.push_back(mt.lookupValue(thisOp.getMatrixD()));

  llvm::Value *A = mt.lookupValue(thisOp.getMatrixA());
  bool isATensor = isa<llvm::PointerType>(A->getType());
  args.push_back(A);

  args.push_back(mt.lookupValue(thisOp.getMatrixB()));
  args.push_back(mt.lookupValue(thisOp.getIdesc()));
  args.push_back(mt.lookupValue(thisOp.getEnableInputD()));
  args.push_back(mt.lookupValue(thisOp.getSparseMetadata()));
  args.push_back(mt.lookupValue(thisOp.getScaleA()));
  args.push_back(mt.lookupValue(thisOp.getScaleB()));
  args.push_back(builder.getInt32(
      static_cast<unsigned>(getNVVMCtaGroupKind(thisOp.getCtaGroup()))));
  args.push_back(
      builder.getInt32(static_cast<unsigned>(thisOp.getCollectorOp())));

  auto kind = thisOp.getKind();
  auto blockScale = thisOp.getBlockScale();
  llvm::Intrinsic::ID ID = [&]() {
    if (kind == NVVM::MMABlockScaleKind::MXF8F6F4) {
      if (blockScale == NVVM::Tcgen05MMABlockScale::DEFAULT) {
        return isATensor ? llvm::Intrinsic::
                               nvvm_tcgen05_mma_sp_tensor_mxf8f6f4_block_scale
                         : llvm::Intrinsic::
                               nvvm_tcgen05_mma_sp_shared_mxf8f6f4_block_scale;
      } else if (blockScale == NVVM::Tcgen05MMABlockScale::BLOCK32) {
        return isATensor
                   ? llvm::Intrinsic::
                         nvvm_tcgen05_mma_sp_tensor_mxf8f6f4_block_scale_block32
                   : llvm::Intrinsic::
                         nvvm_tcgen05_mma_sp_shared_mxf8f6f4_block_scale_block32;
      }
    } else if (kind == NVVM::MMABlockScaleKind::MXF4) {
      if (blockScale == NVVM::Tcgen05MMABlockScale::DEFAULT) {
        return isATensor ? llvm::Intrinsic::
                               nvvm_tcgen05_mma_sp_tensor_mxf4_block_scale
                         : llvm::Intrinsic::
                               nvvm_tcgen05_mma_sp_shared_mxf4_block_scale;
      } else if (blockScale == NVVM::Tcgen05MMABlockScale::BLOCK32) {
        return isATensor
                   ? llvm::Intrinsic::
                         nvvm_tcgen05_mma_sp_tensor_mxf4_block_scale_block32
                   : llvm::Intrinsic::
                         nvvm_tcgen05_mma_sp_shared_mxf4_block_scale_block32;
      }
    } else if (kind == NVVM::MMABlockScaleKind::MXF4NVF4) {
      if (blockScale == NVVM::Tcgen05MMABlockScale::BLOCK32) {
        return isATensor
                   ? llvm::Intrinsic::
                         nvvm_tcgen05_mma_sp_tensor_mxf4nvf4_block_scale_block32
                   : llvm::Intrinsic::
                         nvvm_tcgen05_mma_sp_shared_mxf4nvf4_block_scale_block32;

      } else if (blockScale == NVVM::Tcgen05MMABlockScale::BLOCK16) {
        return isATensor
                   ? llvm::Intrinsic::
                         nvvm_tcgen05_mma_sp_tensor_mxf4nvf4_block_scale_block16
                   : llvm::Intrinsic::
                         nvvm_tcgen05_mma_sp_shared_mxf4nvf4_block_scale_block16;
      }
    }
    llvm_unreachable("Invalid tcgen05.mma.sp.block_scale attributes");
  }();

  return {ID, args};
}

LogicalResult Tcgen05MMASparseBlockScaleOp::verify() {
  return verifyTcgen05MMABlockScaleOp(getCollectorOp(), getKind(),
                                      getBlockScale(), getLoc());
}

//===----------------------------------------------------------------------===//
// NVVM tcgen05.mma.ws functions
//===----------------------------------------------------------------------===//

mlir::NVVM::IDArgPair Tcgen05MMAWsOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {

  auto thisOp = cast<NVVM::Tcgen05MMAWsOp>(op);
  llvm::SmallVector<llvm::Value *> args;

  args.push_back(mt.lookupValue(thisOp.getMatrixD()));

  llvm::Value *A = mt.lookupValue(thisOp.getMatrixA());
  bool isATensor = isa<llvm::PointerType>(A->getType());
  args.push_back(A);

  args.push_back(mt.lookupValue(thisOp.getMatrixB()));
  args.push_back(mt.lookupValue(thisOp.getIdesc()));
  args.push_back(mt.lookupValue(thisOp.getEnableInputD()));

  mlir::Value ZeroColMask = thisOp.getZeroColMask();
  llvm::Intrinsic::ID ID = notIntrinsic;
  if (ZeroColMask) {
    args.push_back(mt.lookupValue(ZeroColMask));
    ID = isATensor ? llvm::Intrinsic::nvvm_tcgen05_mma_ws_tensor_zero_col_mask
                   : llvm::Intrinsic::nvvm_tcgen05_mma_ws_shared_zero_col_mask;
  } else
    ID = isATensor ? llvm::Intrinsic::nvvm_tcgen05_mma_ws_tensor
                   : llvm::Intrinsic::nvvm_tcgen05_mma_ws_shared;

  args.push_back(builder.getInt32(static_cast<unsigned>(thisOp.getKind())));
  args.push_back(
      builder.getInt32(static_cast<unsigned>(thisOp.getCollectorBBuffer())));
  args.push_back(
      builder.getInt32(static_cast<unsigned>(thisOp.getCollectorOp())));

  return {ID, args};
}

//===----------------------------------------------------------------------===//
// NVVM tcgen05.mma.ws.sp functions
//===----------------------------------------------------------------------===//

mlir::NVVM::IDArgPair Tcgen05MMAWsSparseOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {

  auto thisOp = cast<NVVM::Tcgen05MMAWsSparseOp>(op);
  llvm::SmallVector<llvm::Value *> args;

  args.push_back(mt.lookupValue(thisOp.getMatrixD()));

  llvm::Value *A = mt.lookupValue(thisOp.getMatrixA());
  bool isATensor = isa<llvm::PointerType>(A->getType());
  args.push_back(A);

  args.push_back(mt.lookupValue(thisOp.getMatrixB()));
  args.push_back(mt.lookupValue(thisOp.getIdesc()));
  args.push_back(mt.lookupValue(thisOp.getEnableInputD()));
  args.push_back(mt.lookupValue(thisOp.getSparseMetadata()));

  mlir::Value ZeroColMask = thisOp.getZeroColMask();
  llvm::Intrinsic::ID ID = notIntrinsic;
  if (ZeroColMask) {
    args.push_back(mt.lookupValue(ZeroColMask));
    ID = isATensor
             ? llvm::Intrinsic::nvvm_tcgen05_mma_ws_sp_tensor_zero_col_mask
             : llvm::Intrinsic::nvvm_tcgen05_mma_ws_sp_shared_zero_col_mask;
  } else
    ID = isATensor ? llvm::Intrinsic::nvvm_tcgen05_mma_ws_sp_tensor
                   : llvm::Intrinsic::nvvm_tcgen05_mma_ws_sp_shared;

  args.push_back(builder.getInt32(static_cast<unsigned>(thisOp.getKind())));
  args.push_back(
      builder.getInt32(static_cast<unsigned>(thisOp.getCollectorBBuffer())));
  args.push_back(
      builder.getInt32(static_cast<unsigned>(thisOp.getCollectorOp())));

  return {ID, args};
}

//===----------------------------------------------------------------------===//
// NVVMDialect initialization, type parsing, and registration.
//===----------------------------------------------------------------------===//

// TODO: This should be the llvm.nvvm dialect once this is supported.
void NVVMDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LLVMIR/NVVMOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/LLVMIR/NVVMOpsAttributes.cpp.inc"
      >();

  // Support unknown operations because not all NVVM operations are
  // registered.
  allowUnknownOperations();
  declarePromisedInterface<ConvertToLLVMPatternInterface, NVVMDialect>();
  declarePromisedInterface<gpu::TargetAttrInterface, NVVMTargetAttr>();
}

LogicalResult NVVMDialect::verifyOperationAttribute(Operation *op,
                                                    NamedAttribute attr) {
  StringAttr attrName = attr.getName();
  // Kernel function attribute should be attached to functions.
  if (attrName == NVVMDialect::getKernelFuncAttrName()) {
    if (!isa<LLVM::LLVMFuncOp>(op)) {
      return op->emitError() << "'" << NVVMDialect::getKernelFuncAttrName()
                             << "' attribute attached to unexpected op";
    }
  }
  // If maxntid / reqntid / cluster_dim exist, it must be an array with max 3
  // dim
  if (attrName == NVVMDialect::getMaxntidAttrName() ||
      attrName == NVVMDialect::getReqntidAttrName() ||
      attrName == NVVMDialect::getClusterDimAttrName()) {
    auto values = llvm::dyn_cast<DenseI32ArrayAttr>(attr.getValue());
    if (!values || values.empty() || values.size() > 3) {
      return op->emitError()
             << "'" << attrName
             << "' attribute must be integer array with maximum 3 index";
    }
  }
  // If minctasm / maxnreg / cluster_max_blocks exist, it must be an integer
  // attribute
  if (attrName == NVVMDialect::getMinctasmAttrName() ||
      attrName == NVVMDialect::getMaxnregAttrName() ||
      attrName == NVVMDialect::getClusterMaxBlocksAttrName()) {
    if (!llvm::dyn_cast<IntegerAttr>(attr.getValue())) {
      return op->emitError()
             << "'" << attrName << "' attribute must be integer constant";
    }
  }
  // blocksareclusters must be used along with reqntid and cluster_dim
  if (attrName == NVVMDialect::getBlocksAreClustersAttrName()) {
    if (!op->hasAttr(NVVMDialect::getReqntidAttrName()) ||
        !op->hasAttr(NVVMDialect::getClusterDimAttrName())) {
      return op->emitError()
             << "'" << attrName << "' attribute must be used along with "
             << "'" << NVVMDialect::getReqntidAttrName() << "' and "
             << "'" << NVVMDialect::getClusterDimAttrName() << "'";
    }
  }

  return success();
}

LogicalResult NVVMDialect::verifyRegionArgAttribute(Operation *op,
                                                    unsigned regionIndex,
                                                    unsigned argIndex,
                                                    NamedAttribute argAttr) {
  auto funcOp = dyn_cast<FunctionOpInterface>(op);
  if (!funcOp)
    return success();

  bool isKernel = op->hasAttr(NVVMDialect::getKernelFuncAttrName());
  StringAttr attrName = argAttr.getName();
  if (attrName == NVVM::NVVMDialect::getGridConstantAttrName()) {
    if (!isKernel) {
      return op->emitError()
             << "'" << attrName
             << "' attribute must be present only on kernel arguments";
    }
    if (!isa<UnitAttr>(argAttr.getValue()))
      return op->emitError() << "'" << attrName << "' must be a unit attribute";
    if (!funcOp.getArgAttr(argIndex, LLVM::LLVMDialect::getByValAttrName())) {
      return op->emitError()
             << "'" << attrName
             << "' attribute requires the argument to also have attribute '"
             << LLVM::LLVMDialect::getByValAttrName() << "'";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// NVVM Address Space Attr
//===----------------------------------------------------------------------===//

unsigned NVVMMemorySpaceAttr::getAddressSpace() const {
  return static_cast<unsigned>(getValue());
}

bool NVVMMemorySpaceAttr::isValidLoad(
    Type type, ptr::AtomicOrdering ordering, std::optional<int64_t> alignment,
    const ::mlir::DataLayout *dataLayout,
    function_ref<InFlightDiagnostic()> emitError) const {
  return LLVM::detail::isValidLoadStoreImpl(type, ordering, alignment,
                                            dataLayout, emitError);
}

bool NVVMMemorySpaceAttr::isValidStore(
    Type type, ptr::AtomicOrdering ordering, std::optional<int64_t> alignment,
    const ::mlir::DataLayout *dataLayout,
    function_ref<InFlightDiagnostic()> emitError) const {
  return LLVM::detail::isValidLoadStoreImpl(type, ordering, alignment,
                                            dataLayout, emitError);
}

bool NVVMMemorySpaceAttr::isValidAtomicOp(
    ptr::AtomicBinOp op, Type type, ptr::AtomicOrdering ordering,
    std::optional<int64_t> alignment, const ::mlir::DataLayout *dataLayout,
    function_ref<InFlightDiagnostic()> emitError) const {
  // TODO: update this method once `ptr.atomic_rmw` is implemented.
  assert(false && "unimplemented, see TODO in the source.");
  return false;
}

bool NVVMMemorySpaceAttr::isValidAtomicXchg(
    Type type, ptr::AtomicOrdering successOrdering,
    ptr::AtomicOrdering failureOrdering, std::optional<int64_t> alignment,
    const ::mlir::DataLayout *dataLayout,
    function_ref<InFlightDiagnostic()> emitError) const {
  // TODO: update this method once `ptr.atomic_cmpxchg` is implemented.
  assert(false && "unimplemented, see TODO in the source.");
  return false;
}

bool NVVMMemorySpaceAttr::isValidAddrSpaceCast(
    Type tgt, Type src, function_ref<InFlightDiagnostic()> emitError) const {
  // TODO: update this method once the `ptr.addrspace_cast` op is added to the
  // dialect.
  assert(false && "unimplemented, see TODO in the source.");
  return false;
}

bool NVVMMemorySpaceAttr::isValidPtrIntCast(
    Type intLikeTy, Type ptrLikeTy,
    function_ref<InFlightDiagnostic()> emitError) const {
  // TODO: update this method once the int-cast ops are added to the `ptr`
  // dialect.
  assert(false && "unimplemented, see TODO in the source.");
  return false;
}

//===----------------------------------------------------------------------===//
// NVVM target attribute.
//===----------------------------------------------------------------------===//
LogicalResult
NVVMTargetAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                       int optLevel, StringRef triple, StringRef chip,
                       StringRef features, DictionaryAttr flags,
                       ArrayAttr files, bool verifyTarget) {
  if (optLevel < 0 || optLevel > 3) {
    emitError() << "The optimization level must be a number between 0 and 3.";
    return failure();
  }
  if (triple.empty()) {
    emitError() << "The target triple cannot be empty.";
    return failure();
  }
  if (chip.empty()) {
    emitError() << "The target chip cannot be empty.";
    return failure();
  }
  if (files && !llvm::all_of(files, [](::mlir::Attribute attr) {
        return mlir::isa_and_nonnull<StringAttr>(attr);
      })) {
    emitError() << "All the elements in the `link` array must be strings.";
    return failure();
  }
  return success();
}

LogicalResult NVVMTargetAttr::verifyTarget(Operation *gpuModule) {
  if (!getVerifyTarget())
    return success();

  auto gpuModuleOp = llvm::dyn_cast<gpu::GPUModuleOp>(gpuModule);
  if (!gpuModuleOp) {
    return emitError(gpuModule->getLoc(),
                     "NVVM target attribute must be attached to a GPU module");
  }

  const NVVMCheckSMVersion targetSMVersion =
      NVVMCheckSMVersion::getTargetSMVersionFromStr(getChip());
  if (!targetSMVersion.isMinimumSMVersion()) {
    return emitError(gpuModule->getLoc(),
                     "Minimum NVVM target SM version is sm_20");
  }

  if (gpuModuleOp
          ->walk([&](Operation *op) {
            if (auto reqOp = llvm::dyn_cast<NVVM::RequiresSMInterface>(op)) {
              const NVVMCheckSMVersion requirement =
                  reqOp.getRequiredMinSMVersion();
              if (!requirement.isCompatibleWith(targetSMVersion)) {
                op->emitOpError() << "is not supported on " << getChip();
                return WalkResult::interrupt();
              }
            }
            return WalkResult::advance();
          })
          .wasInterrupted())
    return failure();

  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMOpsAttributes.cpp.inc"
