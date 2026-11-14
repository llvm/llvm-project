//===- NVVMArithOps.cpp - NVVM dialect arithmetic ops ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ops of op partition `Arith` of the NVVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/NVVMArithOps.h"

#include "IR/NVVMOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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

using namespace mlir;
using namespace mlir::NVVM;

//===----------------------------------------------------------------------===//
// Verifier methods
//===----------------------------------------------------------------------===//

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

template <typename OpType>
static LogicalResult verifyFloatBinaryOp(OpType op, StringRef opKind) {
  mlir::NVVM::FPRoundingMode rndMode = op.getRnd();
  mlir::NVVM::SaturationMode satMode = op.getSat();
  bool isFTZ = op.getFtz();

  mlir::Type opType = op.getRes().getType();
  mlir::Type opBaseType = isa<VectorType>(opType)
                              ? cast<VectorType>(opType).getElementType()
                              : opType;

  if (opBaseType.isF64() && (satMode != NVVM::SaturationMode::NONE || isFTZ))
    return op.emitOpError("FTZ and saturation are not supported for ")
           << opKind << " involving f64 type";

  if (opBaseType.isF16() && !(rndMode == NVVM::FPRoundingMode::RN ||
                              rndMode == NVVM::FPRoundingMode::NONE))
    return op.emitOpError("only RN rounding mode is supported for f16 and "
                          "vector<2xf16> ")
           << opKind;

  if (opBaseType.isBF16()) {
    if (rndMode != NVVM::FPRoundingMode::RN &&
        rndMode != NVVM::FPRoundingMode::NONE)
      return op.emitOpError("only RN rounding mode is supported for bf16 and "
                            "vector<2xbf16> ")
             << opKind;
    if (satMode != NVVM::SaturationMode::NONE || isFTZ)
      return op.emitOpError("FTZ and saturation are not supported for bf16 and "
                            "vector<2xbf16> ")
             << opKind;
  }

  return success();
}

LogicalResult NVVM::AddFOp::verify() {
  return verifyFloatBinaryOp(*this, "additions");
}

LogicalResult NVVM::SubFOp::verify() {
  return verifyFloatBinaryOp(*this, "subtractions");
}

LogicalResult NVVM::MulFOp::verify() {
  return verifyFloatBinaryOp(*this, "multiplications");
}

LogicalResult NVVM::FmaOp::verify() {
  auto opType = getRes().getType();
  mlir::NVVM::FPRoundingMode rndMode = getRnd();
  mlir::NVVM::SaturationMode satMode = getSat();
  bool isFTZ = getFtz();
  bool isRelu = getRelu();
  bool hasOOB = getOob();

  auto getBaseFType = [](Type type) -> Type {
    if (isa<VectorType>(type))
      return cast<VectorType>(type).getElementType();
    return type;
  };

  auto opBaseType = getBaseFType(opType);

  if (rndMode == NVVM::FPRoundingMode::NONE)
    return emitOpError("rounding mode must be specified");

  if (isRelu && satMode == NVVM::SaturationMode::SAT)
    return emitOpError("relu and saturation are not supported together");

  if (hasOOB && (satMode == NVVM::SaturationMode::SAT || isFTZ))
    return emitOpError("oob is not supported with saturation or FTZ");

  if (!(opBaseType.isF16() || opBaseType.isBF16()) && (isRelu || hasOOB))
    return emitOpError("relu and oob are only supported for f16 and bf16");

  if (opBaseType.isF64() && (satMode != NVVM::SaturationMode::NONE || isFTZ))
    return emitOpError("FTZ and saturation are not supported for f64 type");

  if (opBaseType.isF16() && rndMode != NVVM::FPRoundingMode::RN)
    return emitOpError(
        "only RN rounding mode is supported for f16 and vector<2xf16>");

  if (opBaseType.isBF16()) {
    if (rndMode != NVVM::FPRoundingMode::RN)
      return emitOpError(
          "only RN rounding mode is supported for bf16 and vector<2xbf16>");
    if (satMode != NVVM::SaturationMode::NONE || isFTZ)
      return emitOpError(
          "FTZ and saturation are not supported for bf16 and vector<2xbf16>");
  }

  return success();
}

LogicalResult NVVM::SqrtOp::verify() {
  if (getRnd() == NVVM::FPRoundingMode::NONE)
    return emitOpError("rounding mode cannot be None");

  if (getRes().getType().isF64() && getFtz())
    return emitOpError("FTZ is not supported for f64");

  return success();
}

LogicalResult NVVM::DivFOp::verify() {
  bool isApprox = getApprox();
  bool isFull = getFull();
  bool isF64 = getRes().getType().isF64();
  bool isFtz = getFtz();
  NVVM::FPRoundingMode rndMode = getRnd();

  if (isApprox && isFull)
    return emitOpError("'approx' and 'full' are mutually exclusive");

  if (isApprox || isFull) {
    if (isF64)
      return emitOpError("'approx' and 'full' forms are f32-only");
    if (rndMode != NVVM::FPRoundingMode::NONE)
      return emitOpError(
          "'approx' and 'full' forms do not accept a rounding mode");
    return success();
  }

  // Rounded form below.
  if (rndMode == NVVM::FPRoundingMode::NONE)
    return emitOpError("rounding mode cannot be None for the rounded divide");
  if (isF64 && isFtz)
    return emitOpError("FTZ is not supported for f64");

  return success();
}

//===----------------------------------------------------------------------===//
// Canonicalization patterns
//===----------------------------------------------------------------------===//

struct ConvertFsubToFnegFadd : public OpRewritePattern<SubFOp> {
  using OpRewritePattern<SubFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubFOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value negRhs =
        LLVM::FNegOp::create(rewriter, loc, op.getRhs().getType(), op.getRhs());

    rewriter.replaceOpWithNewOp<AddFOp>(op, op.getType(), op.getLhs(), negRhs,
                                        op.getRnd(), op.getSat(), op.getFtz());
    return success();
  }
};

void SubFOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                         MLIRContext *context) {
  patterns.add<ConvertFsubToFnegFadd>(context);
}

//===----------------------------------------------------------------------===//
// getIntrinsicID/getIntrinsicIDAndArgs methods
//===----------------------------------------------------------------------===//

mlir::NVVM::IDArgPair
CosOp::getIntrinsicIDAndArgs(Operation &op, LLVM::ModuleTranslation &mt,
                             llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::CosOp>(op);
  llvm::Intrinsic::ID id = thisOp.getFtz()
                               ? llvm::Intrinsic::nvvm_cos_approx_ftz_f
                               : llvm::Intrinsic::nvvm_cos_approx_f;
  return {id, {mt.lookupValue(thisOp.getSrc())}};
}

mlir::NVVM::IDArgPair
SinOp::getIntrinsicIDAndArgs(Operation &op, LLVM::ModuleTranslation &mt,
                             llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::SinOp>(op);
  llvm::Intrinsic::ID id = thisOp.getFtz()
                               ? llvm::Intrinsic::nvvm_sin_approx_ftz_f
                               : llvm::Intrinsic::nvvm_sin_approx_f;
  return {id, {mt.lookupValue(thisOp.getSrc())}};
}

mlir::NVVM::IDArgPair
Log2Op::getIntrinsicIDAndArgs(Operation &op, LLVM::ModuleTranslation &mt,
                              llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::Log2Op>(op);
  llvm::Intrinsic::ID id = thisOp.getFtz()
                               ? llvm::Intrinsic::nvvm_lg2_approx_ftz_f
                               : llvm::Intrinsic::nvvm_lg2_approx_f;
  return {id, {mt.lookupValue(thisOp.getSrc())}};
}

mlir::NVVM::IDArgPair
Ex2Op::getIntrinsicIDAndArgs(Operation &op, LLVM::ModuleTranslation &mt,
                             llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::Ex2Op>(op);
  llvm::Intrinsic::ID id = thisOp.getFtz()
                               ? llvm::Intrinsic::nvvm_ex2_approx_ftz
                               : llvm::Intrinsic::nvvm_ex2_approx;
  return {id, {mt.lookupValue(thisOp.getSrc())}};
}

LogicalResult NVVM::Ex2Op::verify() {
  auto vectorType = dyn_cast<VectorType>(getSrc().getType());
  if (!vectorType)
    return success();

  if (vectorType.getElementType().isF16() && getFtz())
    return emitOpError("FTZ is not supported for vector<2xf16>");
  if (vectorType.getElementType().isBF16() && !getFtz())
    return emitOpError("FTZ is required for vector<2xbf16>");
  return success();
}

mlir::NVVM::IDArgPair
RsqrtOp::getIntrinsicIDAndArgs(Operation &op, LLVM::ModuleTranslation &mt,
                               llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::RsqrtOp>(op);
  Type t = thisOp.getRes().getType();
  bool isFtz = thisOp.getFtz();

  llvm::Intrinsic::ID id = [&] {
    if (t.isF32()) {
      return isFtz ? llvm::Intrinsic::nvvm_rsqrt_approx_ftz_f
                   : llvm::Intrinsic::nvvm_rsqrt_approx_f;
    }
    // f64
    return isFtz ? llvm::Intrinsic::nvvm_rsqrt_approx_ftz_d
                 : llvm::Intrinsic::nvvm_rsqrt_approx_d;
  }();

  return {id, {mt.lookupValue(thisOp.getSrc())}};
}

mlir::NVVM::IDArgPair
SqrtOp::getIntrinsicIDAndArgs(Operation &op, LLVM::ModuleTranslation &mt,
                              llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::SqrtOp>(op);
  Type t = thisOp.getRes().getType();
  NVVM::FPRoundingMode rndMode = thisOp.getRnd();
  bool isFtz = thisOp.getFtz();

  // RM is one of RN/RM/RP/RZ (verifier rejects NONE).
  // Subtracting 1 maps RN=1..RZ=4 to 0..3.
  unsigned rndIndex = static_cast<unsigned>(rndMode) - 1;

  static constexpr llvm::Intrinsic::ID f32IDs[] = {
      llvm::Intrinsic::nvvm_sqrt_rn_f,
      llvm::Intrinsic::nvvm_sqrt_rm_f,
      llvm::Intrinsic::nvvm_sqrt_rp_f,
      llvm::Intrinsic::nvvm_sqrt_rz_f,
  };
  static constexpr llvm::Intrinsic::ID f32FTZIDs[] = {
      llvm::Intrinsic::nvvm_sqrt_rn_ftz_f,
      llvm::Intrinsic::nvvm_sqrt_rm_ftz_f,
      llvm::Intrinsic::nvvm_sqrt_rp_ftz_f,
      llvm::Intrinsic::nvvm_sqrt_rz_ftz_f,
  };
  static constexpr llvm::Intrinsic::ID f64IDs[] = {
      llvm::Intrinsic::nvvm_sqrt_rn_d,
      llvm::Intrinsic::nvvm_sqrt_rm_d,
      llvm::Intrinsic::nvvm_sqrt_rp_d,
      llvm::Intrinsic::nvvm_sqrt_rz_d,
  };

  llvm::Intrinsic::ID id =
      t.isF32() ? (isFtz ? f32FTZIDs[rndIndex] : f32IDs[rndIndex])
                : f64IDs[rndIndex];

  return {id, {mt.lookupValue(thisOp.getSrc())}};
}

mlir::NVVM::IDArgPair
SqrtApproxOp::getIntrinsicIDAndArgs(Operation &op, LLVM::ModuleTranslation &mt,
                                    llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::SqrtApproxOp>(op);
  llvm::Intrinsic::ID id = thisOp.getFtz()
                               ? llvm::Intrinsic::nvvm_sqrt_approx_ftz_f
                               : llvm::Intrinsic::nvvm_sqrt_approx_f;
  return {id, {mt.lookupValue(thisOp.getSrc())}};
}

mlir::NVVM::IDArgPair
DivFOp::getIntrinsicIDAndArgs(Operation &op, LLVM::ModuleTranslation &mt,
                              llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::DivFOp>(op);
  bool isFtz = thisOp.getFtz();

  llvm::Intrinsic::ID id;

  if (thisOp.getApprox()) {
    id = isFtz ? llvm::Intrinsic::nvvm_div_approx_ftz_f
               : llvm::Intrinsic::nvvm_div_approx_f;
  } else if (thisOp.getFull()) {
    // Intrinsic Naming quirk: int_nvvm_div_full has no `_f` suffix (unlike
    // approx).
    id = isFtz ? llvm::Intrinsic::nvvm_div_full_ftz
               : llvm::Intrinsic::nvvm_div_full;
  } else {
    // Rounded form — three 4-entry tables indexed by (rndMode - 1).
    unsigned rndIndex = static_cast<unsigned>(thisOp.getRnd()) - 1;

    static constexpr llvm::Intrinsic::ID f32IDs[] = {
        llvm::Intrinsic::nvvm_div_rn_f,
        llvm::Intrinsic::nvvm_div_rm_f,
        llvm::Intrinsic::nvvm_div_rp_f,
        llvm::Intrinsic::nvvm_div_rz_f,
    };
    static constexpr llvm::Intrinsic::ID f32FTZIDs[] = {
        llvm::Intrinsic::nvvm_div_rn_ftz_f,
        llvm::Intrinsic::nvvm_div_rm_ftz_f,
        llvm::Intrinsic::nvvm_div_rp_ftz_f,
        llvm::Intrinsic::nvvm_div_rz_ftz_f,
    };
    static constexpr llvm::Intrinsic::ID f64IDs[] = {
        llvm::Intrinsic::nvvm_div_rn_d,
        llvm::Intrinsic::nvvm_div_rm_d,
        llvm::Intrinsic::nvvm_div_rp_d,
        llvm::Intrinsic::nvvm_div_rz_d,
    };
    Type t = thisOp.getRes().getType();
    id = t.isF32() ? (isFtz ? f32FTZIDs[rndIndex] : f32IDs[rndIndex])
                   : f64IDs[rndIndex];
  }

  return {id,
          {mt.lookupValue(thisOp.getLhs()), mt.lookupValue(thisOp.getRhs())}};
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

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMArithOps.cpp.inc"
