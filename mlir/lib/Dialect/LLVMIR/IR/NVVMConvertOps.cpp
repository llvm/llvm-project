//===- NVVMConvertOps.cpp - NVVM dialect conversion ops -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ops of op partition `Convert` of the NVVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/NVVMConvertOps.h"

#include "IR/NVVMOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/Support/Casting.h"

#include <utility>

using namespace mlir;
using namespace mlir::NVVM;

//===----------------------------------------------------------------------===//
// Verifier methods
//===----------------------------------------------------------------------===//

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
  bool hasPzo = getPzo();

  mlir::MLIRContext *ctx = getContext();
  bool hasScale = static_cast<bool>(getScaleFactor());

  return llvm::TypeSwitch<mlir::Type, LogicalResult>(getDstTy())
      .Case<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(
          [&](mlir::Type) -> LogicalResult {
            if (!(isRoundingModeRN || isRoundingModeRZ)) {
              return emitOpError("Only RN and RZ rounding modes are supported "
                                 "for conversions from f32x2 to ")
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
        if (hasRelu || hasPzo) {
          return emitOpError() << (hasRelu ? "relu" : "pzo")
                               << " not supported for conversions to "
                               << mlir::Float8E8M0FNUType::get(ctx) << " type";
        }
        if (hasScale)
          return emitOpError("scale not supported for conversions to ")
                 << mlir::Float8E8M0FNUType::get(ctx) << " type";
        return success();
      })
      .Case<mlir::Float8E5M3FNUType>([&](mlir::Type) -> LogicalResult {
        if (!(isRoundingModeRN || isRoundingModeRZ || isRoundingModeRP))
          return emitOpError("Only RN, RZ, and RP rounding modes are supported "
                             "for conversions from f32x2 to ")
                 << mlir::Float8E5M3FNUType::get(ctx) << " type";
        if (hasRelu || hasPzo)
          return emitOpError() << (hasRelu ? "relu" : "pzo")
                               << " not supported for conversions to "
                               << mlir::Float8E5M3FNUType::get(ctx) << " type";
        if (hasScale && isRoundingModeRP)
          return emitOpError("scale not supported for conversions to ")
                 << mlir::Float8E5M3FNUType::get(ctx)
                 << " type with RP rounding mode";
        return success();
      })
      .Default([&](mlir::Type) {
        return emitOpError("Only ")
               << mlir::Float8E4M3FNType::get(ctx) << ", "
               << mlir::Float8E5M2Type::get(ctx) << ", "
               << mlir::Float8E5M3FNUType::get(ctx) << ", and "
               << mlir::Float8E8M0FNUType::get(ctx)
               << " types are "
                  "supported for conversions from f32x2 to f8x2";
      });
}

LogicalResult ConvertF16x2ToF8x2Op::verify() {
  mlir::MLIRContext *ctx = getContext();
  bool hasScale = static_cast<bool>(getScaleFactor());

  if (llvm::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(getDstTy())) {
    if (getRnd() != FPRoundingMode::RN && getRnd() != FPRoundingMode::RZ)
      return emitOpError("Only RN and RZ rounding modes are supported for "
                         "conversions from f16x2 to ")
             << mlir::Float8E4M3FNType::get(ctx) << " and "
             << mlir::Float8E5M2Type::get(ctx) << " types";
    if (getSat() != SaturationMode::NONE)
      return emitOpError("Only NONE saturation mode is supported for "
                         "conversions from f16x2 to ")
             << mlir::Float8E4M3FNType::get(ctx) << " and "
             << mlir::Float8E5M2Type::get(ctx) << " types";
    return success();
  }
  if (llvm::isa<mlir::Float8E5M3FNUType>(getDstTy())) {
    if (getRnd() != FPRoundingMode::RN && getRnd() != FPRoundingMode::RZ &&
        getRnd() != FPRoundingMode::RP)
      return emitOpError("Only RN, RZ, and RP rounding modes are supported "
                         "for conversions from f16x2 to ")
             << mlir::Float8E5M3FNUType::get(ctx) << " type";
    if (getRelu() || getPzo())
      return emitOpError() << (getRelu() ? "relu" : "pzo")
                           << " not supported for conversions to "
                           << mlir::Float8E5M3FNUType::get(ctx) << " type";
    if (hasScale && getRnd() == FPRoundingMode::RP)
      return emitOpError("scale not supported for conversions to ")
             << mlir::Float8E5M3FNUType::get(ctx)
             << " type with RP rounding mode";
    return success();
  }
  {
    return emitOpError("Only ")
           << mlir::Float8E5M3FNUType::get(ctx) << ", "
           << mlir::Float8E4M3FNType::get(ctx) << ", and "
           << mlir::Float8E5M2Type::get(ctx)
           << " types are supported for conversions from f16x2 to f8x2.";
  }
}

LogicalResult ConvertBF16x2ToF8x2Op::verify() {
  using RndMode = NVVM::FPRoundingMode;
  using SatMode = NVVM::SaturationMode;

  bool isRoundingModeRN = getRnd() == RndMode::RN;
  bool isRoundingModeRZ = getRnd() == RndMode::RZ;
  bool isRoundingModeRP = getRnd() == RndMode::RP;
  bool isSatFinite = getSat() == SatMode::SATFINITE;
  bool hasRelu = getRelu();
  bool hasPzo = getPzo();

  mlir::MLIRContext *ctx = getContext();
  bool hasScale = static_cast<bool>(getScaleFactor());

  return llvm::TypeSwitch<mlir::Type, LogicalResult>(getDstTy())
      .Case<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(
          [&](mlir::Type) -> LogicalResult {
            if (!(isRoundingModeRN || isRoundingModeRZ))
              return emitOpError("Only RN and RZ rounding modes are supported "
                                 "for conversions from bf16x2 to ")
                     << mlir::Float8E4M3FNType::get(ctx) << " and "
                     << mlir::Float8E5M2Type::get(ctx) << " types";
            if (!isSatFinite)
              return emitOpError("Only SATFINITE saturation mode is supported "
                                 "for conversions from bf16x2 to ")
                     << mlir::Float8E4M3FNType::get(ctx) << " and "
                     << mlir::Float8E5M2Type::get(ctx) << " types";
            return success();
          })
      .Case<mlir::Float8E8M0FNUType>([&](mlir::Type) -> LogicalResult {
        if (!(isRoundingModeRZ || isRoundingModeRP))
          return emitOpError("Only RZ and RP rounding modes are supported for "
                             "conversions from bf16x2 to ")
                 << mlir::Float8E8M0FNUType::get(ctx) << " type";
        if (hasRelu || hasPzo)
          return emitOpError() << (hasRelu ? "relu" : "pzo")
                               << " not supported for conversions to "
                               << mlir::Float8E8M0FNUType::get(ctx) << " type";
        if (hasScale)
          return emitOpError("scale not supported for conversions to ")
                 << mlir::Float8E8M0FNUType::get(ctx) << " type";
        return success();
      })
      .Case<mlir::Float8E5M3FNUType>([&](mlir::Type) -> LogicalResult {
        if (!(isRoundingModeRN || isRoundingModeRZ || isRoundingModeRP))
          return emitOpError("Only RN, RZ, and RP rounding modes are supported "
                             "for conversions from bf16x2 to ")
                 << mlir::Float8E5M3FNUType::get(ctx) << " type";
        if (hasRelu || hasPzo)
          return emitOpError() << (hasRelu ? "relu" : "pzo")
                               << " not supported for conversions to "
                               << mlir::Float8E5M3FNUType::get(ctx) << " type";
        if (hasScale && isRoundingModeRP)
          return emitOpError("scale not supported for conversions to ")
                 << mlir::Float8E5M3FNUType::get(ctx)
                 << " type with RP rounding mode";
        return success();
      })
      .Default([&](mlir::Type) -> LogicalResult {
        llvm_unreachable("Invalid conversion in ConvertBF16x2ToF8x2Op");
        return failure();
      });
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
  if (llvm::isa<Float8E5M3FNUType>(getSrcType()) && getRelu())
    return emitOpError("relu not supported for conversions from ")
           << Float8E5M3FNUType::get(ctx) << " type.";
  return success();
}

LogicalResult ConvertF8x2ToBF16x2Op::verify() {
  mlir::MLIRContext *ctx = getContext();
  if (llvm::isa<Float8E8M0FNUType>(getSrcType())) {
    if (getSat() != SaturationMode::NONE)
      return emitOpError(
                 "Only NONE saturation mode is supported for conversions from ")
             << Float8E8M0FNUType::get(ctx) << " type";
    if (getScaleFactor())
      return emitOpError("scaleFactor not supported for conversions from ")
             << Float8E8M0FNUType::get(ctx) << " type";
    if (getRelu())
      return emitOpError("relu not supported for conversions from ")
             << Float8E8M0FNUType::get(ctx) << " type";
  }
  if (llvm::isa<Float8E5M3FNUType>(getSrcType()) && getRelu())
    return emitOpError("relu not supported for conversions from ")
           << Float8E5M3FNUType::get(ctx) << " type";

  return success();
}

//===----------------------------------------------------------------------===//
// Stochastic Rounding Conversion Ops
//===----------------------------------------------------------------------===//

static LogicalResult verifyConvertF32x2ToFP16x2Op(Twine dstType,
                                                  FPRoundingMode rnd,
                                                  bool hasRandomBits,
                                                  bool hasPzo, Operation *op) {
  static constexpr FPRoundingMode validRndModes[] = {
      FPRoundingMode::RN, FPRoundingMode::RZ, FPRoundingMode::RS};

  if (!llvm::is_contained(validRndModes, rnd)) {
    return op->emitOpError(
               "Only RN, RZ, and RS rounding modes are supported for "
               "conversions from f32x2 to ")
           << dstType << ".";
  }

  if (rnd == FPRoundingMode::RS) {
    if (!hasRandomBits)
      return op->emitOpError("random_bits is required for RS rounding mode.");
    if (hasPzo)
      return op->emitOpError("pzo not supported for RS rounding mode.");
  } else {
    if (hasRandomBits) {
      return op->emitOpError(
          "random_bits not supported for RN and RZ rounding modes.");
    }
  }

  return success();
}

LogicalResult ConvertF32x2ToF16x2Op::verify() {
  return verifyConvertF32x2ToFP16x2Op(
      "f16x2", getRnd(), getRandomBits() ? true : false, getPzo(), *this);
}

LogicalResult ConvertF32x2ToBF16x2Op::verify() {
  return verifyConvertF32x2ToFP16x2Op(
      "bf16x2", getRnd(), getRandomBits() ? true : false, getPzo(), *this);
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

//===----------------------------------------------------------------------===//
// getIntrinsicID/getIntrinsicIDAndArgs methods
//===----------------------------------------------------------------------===//

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

static void
appendN1ScaleArg(Value scaleFactor, LLVM::ModuleTranslation &mt,
                 llvm::IRBuilderBase &builder,
                 llvm::SmallVectorImpl<llvm::Value *> &args,
                 std::optional<uint16_t> defaultScale = std::nullopt) {
  if (scaleFactor) {
    args.push_back(
        builder.CreateZExt(mt.lookupValue(scaleFactor),
                           llvm::Type::getInt16Ty(builder.getContext())));
    return;
  }
  if (defaultScale)
    args.push_back(builder.getInt16(*defaultScale));
}

NVVM::IDArgPair
ConvertF32x2ToF4x2Op::getIntrinsicIDAndArgs(NVVM::ConvertF32x2ToF4x2Op op,
                                            LLVM::ModuleTranslation &mt,
                                            llvm::IRBuilderBase &builder) {
  llvm::SmallVector<llvm::Value *> args;
  args.push_back(mt.lookupValue(op.getA()));
  args.push_back(mt.lookupValue(op.getB()));
  appendN1ScaleArg(op.getScaleFactor(), mt, builder, args,
                   /*defaultScale=*/0x7f);
  args.push_back(builder.getInt1(op.getPzo()));

  unsigned idx =
      (static_cast<unsigned>(op.getRnd() == NVVM::FPRoundingMode::RZ) << 1) |
      static_cast<unsigned>(op.getRelu());
  static constexpr llvm::Intrinsic::ID scaledIds[] = {
      llvm::Intrinsic::nvvm_ff_to_e2m1x2_rn_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_ff_to_e2m1x2_rn_relu_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_ff_to_e2m1x2_rz_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_ff_to_e2m1x2_rz_relu_satfinite_scale_n1_ue8m0,
  };

  return {scaledIds[idx], std::move(args)};
}

NVVM::IDArgPair
ConvertF32x2ToF6x2Op::getIntrinsicIDAndArgs(NVVM::ConvertF32x2ToF6x2Op &op,
                                            LLVM::ModuleTranslation &mt,
                                            llvm::IRBuilderBase &builder) {
  unsigned idx =
      (static_cast<unsigned>(op.getRnd() == NVVM::FPRoundingMode::RZ) << 1) |
      static_cast<unsigned>(op.getRelu());
  static constexpr llvm::Intrinsic::ID e2m3x2ScaledIDs[] = {
      llvm::Intrinsic::nvvm_ff_to_e2m3x2_rn_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_ff_to_e2m3x2_rn_relu_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_ff_to_e2m3x2_rz_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_ff_to_e2m3x2_rz_relu_satfinite_scale_n1_ue8m0,
  };
  static constexpr llvm::Intrinsic::ID e3m2x2ScaledIDs[] = {
      llvm::Intrinsic::nvvm_ff_to_e3m2x2_rn_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_ff_to_e3m2x2_rn_relu_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_ff_to_e3m2x2_rz_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_ff_to_e3m2x2_rz_relu_satfinite_scale_n1_ue8m0,
  };

  llvm::Intrinsic::ID intId =
      llvm::TypeSwitch<mlir::Type, llvm::Intrinsic::ID>(op.getDstTy())
          .Case([&](mlir::Float6E2M3FNType) { return e2m3x2ScaledIDs[idx]; })
          .Case([&](mlir::Float6E3M2FNType) { return e3m2x2ScaledIDs[idx]; })
          .Default([](mlir::Type) {
            llvm_unreachable("Invalid conversion in ConvertF32x2ToF6x2Op");
            return llvm::Intrinsic::not_intrinsic;
          });

  llvm::SmallVector<llvm::Value *> args = {mt.lookupValue(op.getA()),
                                           mt.lookupValue(op.getB())};
  appendN1ScaleArg(op.getScaleFactor(), mt, builder, args,
                   /*defaultScale=*/0x7f);
  args.push_back(builder.getInt1(op.getPzo()));
  return {intId, std::move(args)};
}

NVVM::IDArgPair
ConvertF16x2ToF4x2Op::getIntrinsicIDAndArgs(NVVM::ConvertF16x2ToF4x2Op &op,
                                            LLVM::ModuleTranslation &mt,
                                            llvm::IRBuilderBase &builder) {
  mlir::Type dstTy = op.getDstTy();

  unsigned idx =
      (static_cast<unsigned>(op.getRnd() == NVVM::FPRoundingMode::RZ) << 1) |
      static_cast<unsigned>(op.getRelu());
  static constexpr llvm::Intrinsic::ID scaledIds[] = {
      llvm::Intrinsic::nvvm_f16x2_to_e2m1x2_rn_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_f16x2_to_e2m1x2_rn_relu_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_f16x2_to_e2m1x2_rz_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_f16x2_to_e2m1x2_rz_relu_satfinite_scale_n1_ue8m0,
  };

  if (!llvm::isa<mlir::Float4E2M1FNType>(dstTy))
    llvm_unreachable("Invalid conversion in ConvertF16x2ToF4x2Op");

  llvm::SmallVector<llvm::Value *> args;
  args.push_back(mt.lookupValue(op.getSrc()));
  appendN1ScaleArg(op.getScaleFactor(), mt, builder, args,
                   /*defaultScale=*/0x7f);
  args.push_back(builder.getInt1(op.getPzo()));

  return {scaledIds[idx], std::move(args)};
}

NVVM::IDArgPair
ConvertBF16x2ToF4x2Op::getIntrinsicIDAndArgs(NVVM::ConvertBF16x2ToF4x2Op &op,
                                             LLVM::ModuleTranslation &mt,
                                             llvm::IRBuilderBase &builder) {
  mlir::Type dstTy = op.getDstTy();

  unsigned idx =
      (static_cast<unsigned>(op.getRnd() == NVVM::FPRoundingMode::RZ) << 1) |
      static_cast<unsigned>(op.getRelu());
  static constexpr llvm::Intrinsic::ID scaledIds[] = {
      llvm::Intrinsic::nvvm_bf16x2_to_e2m1x2_rn_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_bf16x2_to_e2m1x2_rn_relu_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_bf16x2_to_e2m1x2_rz_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_bf16x2_to_e2m1x2_rz_relu_satfinite_scale_n1_ue8m0,
  };

  if (!llvm::isa<mlir::Float4E2M1FNType>(dstTy))
    llvm_unreachable("Invalid conversion in ConvertBF16x2ToF4x2Op");

  llvm::SmallVector<llvm::Value *> args;
  args.push_back(mt.lookupValue(op.getSrc()));
  appendN1ScaleArg(op.getScaleFactor(), mt, builder, args,
                   /*defaultScale=*/0x7f);
  args.push_back(builder.getInt1(op.getPzo()));

  return {scaledIds[idx], std::move(args)};
}

NVVM::IDArgPair
ConvertF16x2ToF6x2Op::getIntrinsicIDAndArgs(NVVM::ConvertF16x2ToF6x2Op &op,
                                            LLVM::ModuleTranslation &mt,
                                            llvm::IRBuilderBase &builder) {
  unsigned idx =
      (static_cast<unsigned>(op.getRnd() == NVVM::FPRoundingMode::RZ) << 1) |
      static_cast<unsigned>(op.getRelu());
  static constexpr llvm::Intrinsic::ID e2m3x2ScaledIDs[] = {
      llvm::Intrinsic::nvvm_f16x2_to_e2m3x2_rn_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_f16x2_to_e2m3x2_rn_relu_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_f16x2_to_e2m3x2_rz_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_f16x2_to_e2m3x2_rz_relu_satfinite_scale_n1_ue8m0,
  };
  static constexpr llvm::Intrinsic::ID e3m2x2ScaledIDs[] = {
      llvm::Intrinsic::nvvm_f16x2_to_e3m2x2_rn_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_f16x2_to_e3m2x2_rn_relu_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_f16x2_to_e3m2x2_rz_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_f16x2_to_e3m2x2_rz_relu_satfinite_scale_n1_ue8m0,
  };

  llvm::Intrinsic::ID intId =
      llvm::TypeSwitch<mlir::Type, llvm::Intrinsic::ID>(op.getDstTy())
          .Case<mlir::Float6E2M3FNType>(
              [&](mlir::Float6E2M3FNType) { return e2m3x2ScaledIDs[idx]; })
          .Case<mlir::Float6E3M2FNType>(
              [&](mlir::Float6E3M2FNType) { return e3m2x2ScaledIDs[idx]; })
          .Default([](mlir::Type) {
            llvm_unreachable("Invalid conversion in ConvertF16x2ToF6x2Op");
            return llvm::Intrinsic::not_intrinsic;
          });
  llvm::SmallVector<llvm::Value *> args = {mt.lookupValue(op.getSrc())};
  appendN1ScaleArg(op.getScaleFactor(), mt, builder, args,
                   /*defaultScale=*/0x7f);
  args.push_back(builder.getInt1(op.getPzo()));
  return {intId, std::move(args)};
}

NVVM::IDArgPair
ConvertBF16x2ToF6x2Op::getIntrinsicIDAndArgs(NVVM::ConvertBF16x2ToF6x2Op &op,
                                             LLVM::ModuleTranslation &mt,
                                             llvm::IRBuilderBase &builder) {
  unsigned idx =
      (static_cast<unsigned>(op.getRnd() == NVVM::FPRoundingMode::RZ) << 1) |
      static_cast<unsigned>(op.getRelu());
  static constexpr llvm::Intrinsic::ID e2m3x2ScaledIDs[] = {
      llvm::Intrinsic::nvvm_bf16x2_to_e2m3x2_rn_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_bf16x2_to_e2m3x2_rn_relu_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_bf16x2_to_e2m3x2_rz_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_bf16x2_to_e2m3x2_rz_relu_satfinite_scale_n1_ue8m0,
  };
  static constexpr llvm::Intrinsic::ID e3m2x2ScaledIDs[] = {
      llvm::Intrinsic::nvvm_bf16x2_to_e3m2x2_rn_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_bf16x2_to_e3m2x2_rn_relu_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_bf16x2_to_e3m2x2_rz_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_bf16x2_to_e3m2x2_rz_relu_satfinite_scale_n1_ue8m0,
  };

  llvm::Intrinsic::ID intId =
      llvm::TypeSwitch<mlir::Type, llvm::Intrinsic::ID>(op.getDstTy())
          .Case<mlir::Float6E2M3FNType>(
              [&](mlir::Float6E2M3FNType) { return e2m3x2ScaledIDs[idx]; })
          .Case<mlir::Float6E3M2FNType>(
              [&](mlir::Float6E3M2FNType) { return e3m2x2ScaledIDs[idx]; })
          .Default([](mlir::Type) {
            llvm_unreachable("Invalid conversion in ConvertBF16x2ToF6x2Op");
            return llvm::Intrinsic::not_intrinsic;
          });
  llvm::SmallVector<llvm::Value *> args = {mt.lookupValue(op.getSrc())};
  appendN1ScaleArg(op.getScaleFactor(), mt, builder, args,
                   /*defaultScale=*/0x7f);
  args.push_back(builder.getInt1(op.getPzo()));
  return {intId, std::move(args)};
}

static llvm::Intrinsic::ID
getUE5M3IntrinsicID(NVVM::FPRoundingMode rnd, bool hasSatFinite, bool hasScale,
                    ArrayRef<llvm::Intrinsic::ID> ids,
                    ArrayRef<llvm::Intrinsic::ID> scaledIds) {
  unsigned rndIdx;
  switch (rnd) {
  case NVVM::FPRoundingMode::RN:
    rndIdx = 0;
    break;
  case NVVM::FPRoundingMode::RZ:
    rndIdx = 1;
    break;
  case NVVM::FPRoundingMode::RP:
    rndIdx = 2;
    break;
  default:
    llvm_unreachable("Invalid rounding mode for UE5M3 conversion");
  }
  // satfinite selects the next rounding-mode group: stride 3 unscaled
  // (rn/rz/rp), stride 2 scaled (rn/rz; there is no scaled rp).
  if (hasScale)
    return scaledIds[hasSatFinite * 2 + rndIdx];
  return ids[hasSatFinite * 3 + rndIdx];
}

NVVM::IDArgPair
ConvertF32x2ToF8x2Op::getIntrinsicIDAndArgs(NVVM::ConvertF32x2ToF8x2Op &op,
                                            LLVM::ModuleTranslation &mt,
                                            llvm::IRBuilderBase &builder) {
  bool hasSatFinite = op.getSat() == NVVM::SaturationMode::SATFINITE;
  bool hasScale = static_cast<bool>(op.getScaleFactor());
  unsigned idx =
      (static_cast<unsigned>(op.getRnd() == NVVM::FPRoundingMode::RZ) << 1) |
      static_cast<unsigned>(op.getRelu());
  unsigned ue8m0Idx =
      (static_cast<unsigned>(hasSatFinite) << 1) |
      static_cast<unsigned>(op.getRnd() == NVVM::FPRoundingMode::RP);

  static constexpr llvm::Intrinsic::ID e4m3x2IDs[] = {
      llvm::Intrinsic::nvvm_ff_to_e4m3x2_rn,
      llvm::Intrinsic::nvvm_ff_to_e4m3x2_rn_relu,
      llvm::Intrinsic::nvvm_ff_to_e4m3x2_rz,
      llvm::Intrinsic::nvvm_ff_to_e4m3x2_rz_relu,
  };
  static constexpr llvm::Intrinsic::ID e5m2x2IDs[] = {
      llvm::Intrinsic::nvvm_ff_to_e5m2x2_rn,
      llvm::Intrinsic::nvvm_ff_to_e5m2x2_rn_relu,
      llvm::Intrinsic::nvvm_ff_to_e5m2x2_rz,
      llvm::Intrinsic::nvvm_ff_to_e5m2x2_rz_relu,
  };
  static constexpr llvm::Intrinsic::ID ue8m0x2IDs[] = {
      llvm::Intrinsic::nvvm_ff_to_ue8m0x2_rz,
      llvm::Intrinsic::nvvm_ff_to_ue8m0x2_rp,
      llvm::Intrinsic::nvvm_ff_to_ue8m0x2_rz_satfinite,
      llvm::Intrinsic::nvvm_ff_to_ue8m0x2_rp_satfinite,
  };
  static constexpr llvm::Intrinsic::ID e4m3x2ScaledIDs[] = {
      llvm::Intrinsic::nvvm_ff_to_e4m3x2_rn_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_ff_to_e4m3x2_rn_relu_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_ff_to_e4m3x2_rz_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_ff_to_e4m3x2_rz_relu_scale_n1_ue8m0,
  };
  static constexpr llvm::Intrinsic::ID e5m2x2ScaledIDs[] = {
      llvm::Intrinsic::nvvm_ff_to_e5m2x2_rn_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_ff_to_e5m2x2_rn_relu_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_ff_to_e5m2x2_rz_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_ff_to_e5m2x2_rz_relu_scale_n1_ue8m0,
  };
  static constexpr llvm::Intrinsic::ID ue5m3x2IDs[] = {
      llvm::Intrinsic::nvvm_ff_to_ue5m3x2_rn,
      llvm::Intrinsic::nvvm_ff_to_ue5m3x2_rz,
      llvm::Intrinsic::nvvm_ff_to_ue5m3x2_rp,
      llvm::Intrinsic::nvvm_ff_to_ue5m3x2_rn_satfinite,
      llvm::Intrinsic::nvvm_ff_to_ue5m3x2_rz_satfinite,
      llvm::Intrinsic::nvvm_ff_to_ue5m3x2_rp_satfinite,
  };
  static constexpr llvm::Intrinsic::ID ue5m3x2ScaledIDs[] = {
      llvm::Intrinsic::nvvm_ff_to_ue5m3x2_rn_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_ff_to_ue5m3x2_rz_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_ff_to_ue5m3x2_rn_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_ff_to_ue5m3x2_rz_satfinite_scale_n1_ue8m0,
  };

  llvm::Intrinsic::ID intId =
      llvm::TypeSwitch<mlir::Type, llvm::Intrinsic::ID>(op.getDstTy())
          .Case([&](mlir::Float8E4M3FNType) {
            return hasScale ? e4m3x2ScaledIDs[idx] : e4m3x2IDs[idx];
          })
          .Case([&](mlir::Float8E5M2Type) {
            return hasScale ? e5m2x2ScaledIDs[idx] : e5m2x2IDs[idx];
          })
          .Case([&](mlir::Float8E8M0FNUType) {
            if (op.getRnd() != NVVM::FPRoundingMode::RZ &&
                op.getRnd() != NVVM::FPRoundingMode::RP)
              llvm_unreachable("Invalid conversion in ConvertF32x2ToF8x2Op");
            return ue8m0x2IDs[ue8m0Idx];
          })
          .Case([&](mlir::Float8E5M3FNUType) {
            return getUE5M3IntrinsicID(op.getRnd(), hasSatFinite, hasScale,
                                       ue5m3x2IDs, ue5m3x2ScaledIDs);
          })
          .Default([](mlir::Type) {
            llvm_unreachable("Invalid conversion in ConvertF32x2ToF8x2Op");
            return llvm::Intrinsic::not_intrinsic;
          });

  llvm::SmallVector<llvm::Value *> args = {mt.lookupValue(op.getA()),
                                           mt.lookupValue(op.getB())};
  appendN1ScaleArg(op.getScaleFactor(), mt, builder, args);
  if (!isa<Float8E8M0FNUType, Float8E5M3FNUType>(op.getDstTy()))
    args.push_back(builder.getInt1(op.getPzo()));
  return {intId, std::move(args)};
}

NVVM::IDArgPair
ConvertF16x2ToF8x2Op::getIntrinsicIDAndArgs(NVVM::ConvertF16x2ToF8x2Op &op,
                                            LLVM::ModuleTranslation &mt,
                                            llvm::IRBuilderBase &builder) {
  bool hasScale = static_cast<bool>(op.getScaleFactor());
  unsigned idx =
      (static_cast<unsigned>(op.getRnd() == NVVM::FPRoundingMode::RZ) << 1) |
      static_cast<unsigned>(op.getRelu());
  static constexpr llvm::Intrinsic::ID e4m3x2IDs[] = {
      llvm::Intrinsic::nvvm_f16x2_to_e4m3x2_rn,
      llvm::Intrinsic::nvvm_f16x2_to_e4m3x2_rn_relu,
      llvm::Intrinsic::nvvm_f16x2_to_e4m3x2_rz,
      llvm::Intrinsic::nvvm_f16x2_to_e4m3x2_rz_relu,
  };
  static constexpr llvm::Intrinsic::ID e5m2x2IDs[] = {
      llvm::Intrinsic::nvvm_f16x2_to_e5m2x2_rn,
      llvm::Intrinsic::nvvm_f16x2_to_e5m2x2_rn_relu,
      llvm::Intrinsic::nvvm_f16x2_to_e5m2x2_rz,
      llvm::Intrinsic::nvvm_f16x2_to_e5m2x2_rz_relu,
  };
  static constexpr llvm::Intrinsic::ID e4m3x2ScaledIDs[] = {
      llvm::Intrinsic::nvvm_f16x2_to_e4m3x2_rn_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_f16x2_to_e4m3x2_rn_relu_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_f16x2_to_e4m3x2_rz_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_f16x2_to_e4m3x2_rz_relu_scale_n1_ue8m0,
  };
  static constexpr llvm::Intrinsic::ID e5m2x2ScaledIDs[] = {
      llvm::Intrinsic::nvvm_f16x2_to_e5m2x2_rn_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_f16x2_to_e5m2x2_rn_relu_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_f16x2_to_e5m2x2_rz_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_f16x2_to_e5m2x2_rz_relu_scale_n1_ue8m0,
  };
  static constexpr llvm::Intrinsic::ID ue5m3x2IDs[] = {
      llvm::Intrinsic::nvvm_f16x2_to_ue5m3x2_rn,
      llvm::Intrinsic::nvvm_f16x2_to_ue5m3x2_rz,
      llvm::Intrinsic::nvvm_f16x2_to_ue5m3x2_rp,
      llvm::Intrinsic::nvvm_f16x2_to_ue5m3x2_rn_satfinite,
      llvm::Intrinsic::nvvm_f16x2_to_ue5m3x2_rz_satfinite,
      llvm::Intrinsic::nvvm_f16x2_to_ue5m3x2_rp_satfinite,
  };
  static constexpr llvm::Intrinsic::ID ue5m3x2ScaledIDs[] = {
      llvm::Intrinsic::nvvm_f16x2_to_ue5m3x2_rn_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_f16x2_to_ue5m3x2_rz_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_f16x2_to_ue5m3x2_rn_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_f16x2_to_ue5m3x2_rz_satfinite_scale_n1_ue8m0,
  };

  llvm::Intrinsic::ID intId =
      llvm::TypeSwitch<mlir::Type, llvm::Intrinsic::ID>(op.getDstTy())
          .Case([&](mlir::Float8E4M3FNType) {
            return hasScale ? e4m3x2ScaledIDs[idx] : e4m3x2IDs[idx];
          })
          .Case([&](mlir::Float8E5M2Type) {
            return hasScale ? e5m2x2ScaledIDs[idx] : e5m2x2IDs[idx];
          })
          .Case([&](mlir::Float8E5M3FNUType) {
            return getUE5M3IntrinsicID(
                op.getRnd(), op.getSat() == NVVM::SaturationMode::SATFINITE,
                hasScale, ue5m3x2IDs, ue5m3x2ScaledIDs);
          })
          .Default([](mlir::Type) {
            llvm_unreachable("Invalid conversion in ConvertF16x2ToF8x2Op");
            return llvm::Intrinsic::not_intrinsic;
          });

  llvm::SmallVector<llvm::Value *> args = {mt.lookupValue(op.getA())};
  appendN1ScaleArg(op.getScaleFactor(), mt, builder, args);
  if (!isa<Float8E5M3FNUType>(op.getDstTy()))
    args.push_back(builder.getInt1(op.getPzo()));
  return {intId, std::move(args)};
}

NVVM::IDArgPair
ConvertBF16x2ToF8x2Op::getIntrinsicIDAndArgs(NVVM::ConvertBF16x2ToF8x2Op &op,
                                             LLVM::ModuleTranslation &mt,
                                             llvm::IRBuilderBase &builder) {
  bool hasSatFinite = op.getSat() == NVVM::SaturationMode::SATFINITE;
  bool hasScale = static_cast<bool>(op.getScaleFactor());
  unsigned idx =
      (static_cast<unsigned>(op.getRnd() == NVVM::FPRoundingMode::RZ) << 1) |
      static_cast<unsigned>(op.getRelu());
  unsigned ue8m0Idx =
      (static_cast<unsigned>(hasSatFinite) << 1) |
      static_cast<unsigned>(op.getRnd() == NVVM::FPRoundingMode::RP);

  static constexpr llvm::Intrinsic::ID e4m3x2IDs[] = {
      llvm::Intrinsic::nvvm_bf16x2_to_e4m3x2_rn_satfinite,
      llvm::Intrinsic::nvvm_bf16x2_to_e4m3x2_rn_relu_satfinite,
      llvm::Intrinsic::nvvm_bf16x2_to_e4m3x2_rz_satfinite,
      llvm::Intrinsic::nvvm_bf16x2_to_e4m3x2_rz_relu_satfinite,
  };
  static constexpr llvm::Intrinsic::ID e5m2x2IDs[] = {
      llvm::Intrinsic::nvvm_bf16x2_to_e5m2x2_rn_satfinite,
      llvm::Intrinsic::nvvm_bf16x2_to_e5m2x2_rn_relu_satfinite,
      llvm::Intrinsic::nvvm_bf16x2_to_e5m2x2_rz_satfinite,
      llvm::Intrinsic::nvvm_bf16x2_to_e5m2x2_rz_relu_satfinite,
  };
  static constexpr llvm::Intrinsic::ID ue8m0x2IDs[] = {
      llvm::Intrinsic::nvvm_bf16x2_to_ue8m0x2_rz,
      llvm::Intrinsic::nvvm_bf16x2_to_ue8m0x2_rp,
      llvm::Intrinsic::nvvm_bf16x2_to_ue8m0x2_rz_satfinite,
      llvm::Intrinsic::nvvm_bf16x2_to_ue8m0x2_rp_satfinite,
  };
  static constexpr llvm::Intrinsic::ID e4m3x2ScaledIDs[] = {
      llvm::Intrinsic::nvvm_bf16x2_to_e4m3x2_rn_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_bf16x2_to_e4m3x2_rn_relu_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_bf16x2_to_e4m3x2_rz_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_bf16x2_to_e4m3x2_rz_relu_satfinite_scale_n1_ue8m0,
  };
  static constexpr llvm::Intrinsic::ID e5m2x2ScaledIDs[] = {
      llvm::Intrinsic::nvvm_bf16x2_to_e5m2x2_rn_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_bf16x2_to_e5m2x2_rn_relu_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_bf16x2_to_e5m2x2_rz_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_bf16x2_to_e5m2x2_rz_relu_satfinite_scale_n1_ue8m0,
  };
  static constexpr llvm::Intrinsic::ID ue5m3x2IDs[] = {
      llvm::Intrinsic::nvvm_bf16x2_to_ue5m3x2_rn,
      llvm::Intrinsic::nvvm_bf16x2_to_ue5m3x2_rz,
      llvm::Intrinsic::nvvm_bf16x2_to_ue5m3x2_rp,
      llvm::Intrinsic::nvvm_bf16x2_to_ue5m3x2_rn_satfinite,
      llvm::Intrinsic::nvvm_bf16x2_to_ue5m3x2_rz_satfinite,
      llvm::Intrinsic::nvvm_bf16x2_to_ue5m3x2_rp_satfinite,
  };
  static constexpr llvm::Intrinsic::ID ue5m3x2ScaledIDs[] = {
      llvm::Intrinsic::nvvm_bf16x2_to_ue5m3x2_rn_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_bf16x2_to_ue5m3x2_rz_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_bf16x2_to_ue5m3x2_rn_satfinite_scale_n1_ue8m0,
      llvm::Intrinsic::nvvm_bf16x2_to_ue5m3x2_rz_satfinite_scale_n1_ue8m0,
  };

  llvm::Intrinsic::ID intId =
      llvm::TypeSwitch<mlir::Type, llvm::Intrinsic::ID>(op.getDstTy())
          .Case<mlir::Float8E4M3FNType>([&](mlir::Float8E4M3FNType) {
            return hasScale ? e4m3x2ScaledIDs[idx] : e4m3x2IDs[idx];
          })
          .Case<mlir::Float8E5M2Type>([&](mlir::Float8E5M2Type) {
            return hasScale ? e5m2x2ScaledIDs[idx] : e5m2x2IDs[idx];
          })
          .Case<mlir::Float8E8M0FNUType>(
              [&](mlir::Float8E8M0FNUType) { return ue8m0x2IDs[ue8m0Idx]; })
          .Case<mlir::Float8E5M3FNUType>([&](mlir::Float8E5M3FNUType) {
            return getUE5M3IntrinsicID(op.getRnd(), hasSatFinite, hasScale,
                                       ue5m3x2IDs, ue5m3x2ScaledIDs);
          })
          .Default([](mlir::Type) {
            llvm_unreachable("Invalid conversion in ConvertBF16x2ToF8x2Op");
            return llvm::Intrinsic::not_intrinsic;
          });

  llvm::SmallVector<llvm::Value *> args = {mt.lookupValue(op.getSrc())};
  appendN1ScaleArg(op.getScaleFactor(), mt, builder, args);
  if (!isa<Float8E8M0FNUType, Float8E5M3FNUType>(op.getDstTy()))
    args.push_back(builder.getInt1(op.getPzo()));
  return {intId, std::move(args)};
}

NVVM::IDArgPair ConvertF8x2ToF16x2Op::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto curOp = cast<NVVM::ConvertF8x2ToF16x2Op>(op);

  bool hasRelu = curOp.getRelu();

  llvm::Intrinsic::ID intId =
      llvm::TypeSwitch<mlir::Type, llvm::Intrinsic::ID>(curOp.getSrcType())
          .Case([&](Float8E4M3FNType type) {
            return hasRelu ? llvm::Intrinsic::nvvm_e4m3x2_to_f16x2_rn_relu
                           : llvm::Intrinsic::nvvm_e4m3x2_to_f16x2_rn;
          })
          .Case([&](Float8E5M2Type type) {
            return hasRelu ? llvm::Intrinsic::nvvm_e5m2x2_to_f16x2_rn_relu
                           : llvm::Intrinsic::nvvm_e5m2x2_to_f16x2_rn;
          })
          .Case([&](Float8E5M3FNUType type) {
            return llvm::Intrinsic::nvvm_ue5m3x2_to_f16x2_rn;
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
  bool hasScale = static_cast<bool>(curOp.getScaleFactor());
  bool hasSatfinite = curOp.getSat() == NVVM::SaturationMode::SATFINITE;
  bool hasRelu = curOp.getRelu();

  static constexpr llvm::Intrinsic::ID E4M3Ids[] = {
      llvm::Intrinsic::nvvm_e4m3x2_to_bf16x2_rn_scale_n2_ue8m0,
      llvm::Intrinsic::nvvm_e4m3x2_to_bf16x2_rn_relu_scale_n2_ue8m0,
      llvm::Intrinsic::nvvm_e4m3x2_to_bf16x2_rn_satfinite_scale_n2_ue8m0,
      llvm::Intrinsic::nvvm_e4m3x2_to_bf16x2_rn_relu_satfinite_scale_n2_ue8m0,
  };

  static constexpr llvm::Intrinsic::ID E5M2Ids[] = {
      llvm::Intrinsic::nvvm_e5m2x2_to_bf16x2_rn_scale_n2_ue8m0,
      llvm::Intrinsic::nvvm_e5m2x2_to_bf16x2_rn_relu_scale_n2_ue8m0,
      llvm::Intrinsic::nvvm_e5m2x2_to_bf16x2_rn_satfinite_scale_n2_ue8m0,
      llvm::Intrinsic::nvvm_e5m2x2_to_bf16x2_rn_relu_satfinite_scale_n2_ue8m0,
  };
  static constexpr llvm::Intrinsic::ID UE5M3Ids[] = {
      llvm::Intrinsic::nvvm_ue5m3x2_to_bf16x2_rn,
      llvm::Intrinsic::nvvm_ue5m3x2_to_bf16x2_rn,
      llvm::Intrinsic::nvvm_ue5m3x2_to_bf16x2_rn_satfinite,
      llvm::Intrinsic::nvvm_ue5m3x2_to_bf16x2_rn_satfinite,
  };
  static constexpr llvm::Intrinsic::ID UE5M3ScaledIds[] = {
      llvm::Intrinsic::nvvm_ue5m3x2_to_bf16x2_rn_scale_n2_ue8m0,
      llvm::Intrinsic::nvvm_ue5m3x2_to_bf16x2_rn_scale_n2_ue8m0,
      llvm::Intrinsic::nvvm_ue5m3x2_to_bf16x2_rn_satfinite_scale_n2_ue8m0,
      llvm::Intrinsic::nvvm_ue5m3x2_to_bf16x2_rn_satfinite_scale_n2_ue8m0,
  };

  llvm::Intrinsic::ID intId =
      llvm::TypeSwitch<mlir::Type, llvm::Intrinsic::ID>(curOp.getSrcType())
          .Case([&](Float8E8M0FNUType type) {
            return llvm::Intrinsic::nvvm_ue8m0x2_to_bf16x2;
          })
          .Case([&](Float8E4M3FNType type) {
            return E4M3Ids[hasSatfinite << 1 | hasRelu];
          })
          .Case([&](Float8E5M2Type type) {
            return E5M2Ids[hasSatfinite << 1 | hasRelu];
          })
          .Case([&](Float8E5M3FNUType type) {
            return (hasScale ? UE5M3ScaledIds : UE5M3Ids)[hasSatfinite << 1];
          })
          .Default([](mlir::Type type) {
            llvm_unreachable("Invalid type for ConvertF8x2ToBF16x2Op");
            return llvm::Intrinsic::not_intrinsic;
          });
  llvm::Value *packedI16 =
      builder.CreateBitCast(mt.lookupValue(curOp.getSrc()),
                            llvm::Type::getInt16Ty(builder.getContext()));

  llvm::SmallVector<llvm::Value *> args;
  args.push_back(packedI16);
  if (isa<Float8E5M3FNUType>(curOp.getSrcType())) {
    if (hasScale)
      args.push_back(mt.lookupValue(curOp.getScaleFactor()));
  } else if (!isa<Float8E8M0FNUType>(curOp.getSrcType())) {
    args.push_back(
        hasScale ? mt.lookupValue(curOp.getScaleFactor())
                 : builder.getInt16(0x7f7f)); // default scale factor (value of
                                              // 1 for both elements).
  }

  return {intId, std::move(args)};
}

NVVM::IDArgPair ConvertF6x2ToF16x2Op::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto curOp = cast<NVVM::ConvertF6x2ToF16x2Op>(op);

  bool hasRelu = curOp.getRelu();

  llvm::Intrinsic::ID intId =
      llvm::TypeSwitch<mlir::Type, llvm::Intrinsic::ID>(curOp.getSrcType())
          .Case([&](Float6E2M3FNType type) {
            return hasRelu ? llvm::Intrinsic::nvvm_e2m3x2_to_f16x2_rn_relu
                           : llvm::Intrinsic::nvvm_e2m3x2_to_f16x2_rn;
          })
          .Case([&](Float6E3M2FNType type) {
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

NVVM::IDArgPair ConvertF6x2ToBF16x2Op::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto curOp = cast<NVVM::ConvertF6x2ToBF16x2Op>(op);
  bool hasScale = static_cast<bool>(curOp.getScaleFactor());
  bool hasSatfinite = curOp.getSat() == NVVM::SaturationMode::SATFINITE;
  bool hasRelu = curOp.getRelu();

  static constexpr llvm::Intrinsic::ID E2M3Ids[] = {
      llvm::Intrinsic::nvvm_e2m3x2_to_bf16x2_rn_scale_n2_ue8m0,
      llvm::Intrinsic::nvvm_e2m3x2_to_bf16x2_rn_relu_scale_n2_ue8m0,
      llvm::Intrinsic::nvvm_e2m3x2_to_bf16x2_rn_satfinite_scale_n2_ue8m0,
      llvm::Intrinsic::nvvm_e2m3x2_to_bf16x2_rn_relu_satfinite_scale_n2_ue8m0,
  };

  static constexpr llvm::Intrinsic::ID E3M2Ids[] = {
      llvm::Intrinsic::nvvm_e3m2x2_to_bf16x2_rn_scale_n2_ue8m0,
      llvm::Intrinsic::nvvm_e3m2x2_to_bf16x2_rn_relu_scale_n2_ue8m0,
      llvm::Intrinsic::nvvm_e3m2x2_to_bf16x2_rn_satfinite_scale_n2_ue8m0,
      llvm::Intrinsic::nvvm_e3m2x2_to_bf16x2_rn_relu_satfinite_scale_n2_ue8m0,
  };

  unsigned idx = (hasSatfinite << 1) | hasRelu;
  llvm::Intrinsic::ID intId =
      llvm::TypeSwitch<mlir::Type, llvm::Intrinsic::ID>(curOp.getSrcType())
          .Case([&](Float6E2M3FNType type) { return E2M3Ids[idx]; })
          .Case([&](Float6E3M2FNType type) { return E3M2Ids[idx]; })
          .Default([](mlir::Type type) {
            llvm_unreachable("Invalid type for ConvertF6x2ToBF16x2Op");
            return llvm::Intrinsic::not_intrinsic;
          });

  llvm::Value *packedI16 =
      builder.CreateBitCast(mt.lookupValue(curOp.getSrc()),
                            llvm::Type::getInt16Ty(builder.getContext()));

  llvm::SmallVector<llvm::Value *> args;
  args.push_back(packedI16);
  args.push_back(
      hasScale
          ? mt.lookupValue(curOp.getScaleFactor())
          : builder.getInt16(
                0x7f7f)); // default scale factor (value of 1 for both elements)

  return {intId, std::move(args)};
}

NVVM::IDArgPair ConvertF4x2ToF16x2Op::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto curOp = cast<NVVM::ConvertF4x2ToF16x2Op>(op);

  bool hasRelu = curOp.getRelu();

  llvm::Intrinsic::ID intId =
      llvm::TypeSwitch<mlir::Type, llvm::Intrinsic::ID>(curOp.getSrcType())
          .Case([&](Float4E2M1FNType type) {
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

NVVM::IDArgPair ConvertF4x2ToBF16x2Op::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto curOp = cast<NVVM::ConvertF4x2ToBF16x2Op>(op);
  bool hasScale = static_cast<bool>(curOp.getScaleFactor());
  bool hasSatfinite = curOp.getSat() == NVVM::SaturationMode::SATFINITE;
  bool hasRelu = curOp.getRelu();

  static constexpr llvm::Intrinsic::ID E2M1Ids[] = {
      llvm::Intrinsic::nvvm_e2m1x2_to_bf16x2_rn_scale_n2_ue8m0,
      llvm::Intrinsic::nvvm_e2m1x2_to_bf16x2_rn_relu_scale_n2_ue8m0,
      llvm::Intrinsic::nvvm_e2m1x2_to_bf16x2_rn_satfinite_scale_n2_ue8m0,
      llvm::Intrinsic::nvvm_e2m1x2_to_bf16x2_rn_relu_satfinite_scale_n2_ue8m0,
  };

  unsigned idx = (hasSatfinite << 1) | hasRelu;
  llvm::Intrinsic::ID intId =
      llvm::TypeSwitch<mlir::Type, llvm::Intrinsic::ID>(curOp.getSrcType())
          .Case([&](Float4E2M1FNType type) { return E2M1Ids[idx]; })
          .Default([](mlir::Type type) {
            llvm_unreachable("Invalid type for ConvertF4x2ToBF16x2Op");
            return llvm::Intrinsic::not_intrinsic;
          });

  llvm::Value *extendedI16 =
      builder.CreateZExt(mt.lookupValue(curOp.getSrc()),
                         llvm::Type::getInt16Ty(builder.getContext()));

  llvm::SmallVector<llvm::Value *> args;
  args.push_back(extendedI16);
  args.push_back(
      hasScale
          ? mt.lookupValue(curOp.getScaleFactor())
          : builder.getInt16(
                0x7f7f)); // default scale factor (value of 1 for both elements)

  return {intId, std::move(args)};
}

NVVM::IDArgPair ConvertF32x2ToS2F6x2Op::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::ConvertF32x2ToS2F6x2Op>(op);
  bool hasRelu = thisOp.getRelu();
  bool hasScale = static_cast<bool>(thisOp.getScaleFactor());

  llvm::Intrinsic::ID id =
      hasRelu
          ? llvm::Intrinsic::nvvm_ff_to_s2f6x2_rn_relu_satfinite_scale_n2_ue8m0
          : llvm::Intrinsic::nvvm_ff_to_s2f6x2_rn_satfinite_scale_n2_ue8m0;

  // Fill the Intrinsic Args
  llvm::SmallVector<llvm::Value *> args;
  args.push_back(mt.lookupValue(thisOp.getA()));
  args.push_back(mt.lookupValue(thisOp.getB()));
  args.push_back(hasScale ? mt.lookupValue(thisOp.getScaleFactor())
                          : builder.getInt16(0x7f7f));
  return {id, std::move(args)};
}

NVVM::IDArgPair ConvertBF16x2ToS2F6x2Op::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::ConvertBF16x2ToS2F6x2Op>(op);
  bool hasRelu = thisOp.getRelu();
  bool hasScale = static_cast<bool>(thisOp.getScaleFactor());

  llvm::Intrinsic::ID id =
      hasRelu
          ? llvm::Intrinsic::
                nvvm_bf16x2_to_s2f6x2_rn_relu_satfinite_scale_n2_ue8m0
          : llvm::Intrinsic::nvvm_bf16x2_to_s2f6x2_rn_satfinite_scale_n2_ue8m0;

  // Fill the Intrinsic Args
  llvm::SmallVector<llvm::Value *> args;
  args.push_back(mt.lookupValue(thisOp.getSrc()));
  args.push_back(hasScale ? mt.lookupValue(thisOp.getScaleFactor())
                          : builder.getInt16(0x7f7f));
  return {id, std::move(args)};
}

NVVM::IDArgPair ConvertS2F6x2ToBF16x2Op::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::ConvertS2F6x2ToBF16x2Op>(op);
  bool hasRelu = thisOp.getRelu();
  bool hasScale = static_cast<bool>(thisOp.getScaleFactor());
  bool hasSat = thisOp.getSat() == NVVM::SaturationMode::SATFINITE;

  static constexpr llvm::Intrinsic::ID ids[] = {
      llvm::Intrinsic::nvvm_s2f6x2_to_bf16x2_rn_scale_n2_ue8m0,
      llvm::Intrinsic::nvvm_s2f6x2_to_bf16x2_rn_relu_scale_n2_ue8m0,
      llvm::Intrinsic::nvvm_s2f6x2_to_bf16x2_rn_satfinite_scale_n2_ue8m0,
      llvm::Intrinsic::nvvm_s2f6x2_to_bf16x2_rn_relu_satfinite_scale_n2_ue8m0,
  };

  unsigned idx = (hasSat << 1) | hasRelu;

  // Fill the Intrinsic Args
  llvm::SmallVector<llvm::Value *> args;
  llvm::Value *packedI16 =
      builder.CreateBitCast(mt.lookupValue(thisOp.getSrc()),
                            llvm::Type::getInt16Ty(builder.getContext()));
  args.push_back(packedI16);
  args.push_back(hasScale ? mt.lookupValue(thisOp.getScaleFactor())
                          : builder.getInt16(0x7f7f));

  return {ids[idx], std::move(args)};
}

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

  if (op.getRnd() != FPRoundingMode::RS)
    args.push_back(builder.getInt1(op.getPzo()));

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

  if (op.getRnd() != FPRoundingMode::RS)
    args.push_back(builder.getInt1(op.getPzo()));

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
      .Case([&](mlir::Float8E4M3FNType) {
        return hasRelu ? llvm::Intrinsic::nvvm_f32x4_to_e4m3x4_rs_relu_satfinite
                       : llvm::Intrinsic::nvvm_f32x4_to_e4m3x4_rs_satfinite;
      })
      .Case([&](mlir::Float8E5M2Type) {
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
      .Case([&](mlir::Float6E2M3FNType) {
        return hasRelu ? llvm::Intrinsic::nvvm_f32x4_to_e2m3x4_rs_relu_satfinite
                       : llvm::Intrinsic::nvvm_f32x4_to_e2m3x4_rs_satfinite;
      })
      .Case([&](mlir::Float6E3M2FNType) {
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
      .Case([&](mlir::Float4E2M1FNType) {
        return hasRelu ? llvm::Intrinsic::nvvm_f32x4_to_e2m1x4_rs_relu_satfinite
                       : llvm::Intrinsic::nvvm_f32x4_to_e2m1x4_rs_satfinite;
      })
      .Default([](mlir::Type) {
        llvm_unreachable("Invalid F4 type in ConvertF32x4ToF4x4Op");
        return llvm::Intrinsic::not_intrinsic;
      });
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMConvertOps.cpp.inc"
