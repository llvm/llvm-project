//===- MathToFuncs.cpp - Math to outlined implementation conversion -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MathToFuncs/MathToFuncs.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMATHTOFUNCS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
// Pattern to convert vector operations to scalar operations.
template <typename Op>
struct VecOpToScalarOp : public OpRewritePattern<Op> {
public:
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final;
};

// Callback type for getting pre-generated FuncOp implementing
// a power operation of the given type.
using GetPowerFuncCallbackTy = function_ref<func::FuncOp(Type)>;

// Pattern to convert scalar IPowIOp into a call of outlined
// software implementation.
class IPowIOpLowering : public OpRewritePattern<math::IPowIOp> {
public:
  IPowIOpLowering(MLIRContext *context, GetPowerFuncCallbackTy cb)
      : OpRewritePattern<math::IPowIOp>(context), getFuncOpCallback(cb) {}

  /// Convert IPowI into a call to a local function implementing
  /// the power operation. The local function computes a scalar result,
  /// so vector forms of IPowI are linearized.
  LogicalResult matchAndRewrite(math::IPowIOp op,
                                PatternRewriter &rewriter) const final;

private:
  GetPowerFuncCallbackTy getFuncOpCallback;
};

// Pattern to convert scalar FPowIOp into a call of outlined
// software implementation.
class FPowIOpLowering : public OpRewritePattern<math::FPowIOp> {
public:
  FPowIOpLowering(MLIRContext *context, GetPowerFuncCallbackTy cb)
      : OpRewritePattern<math::FPowIOp>(context), getFuncOpCallback(cb) {}

  /// Convert FPowI into a call to a local function implementing
  /// the power operation. The local function computes a scalar result,
  /// so vector forms of FPowI are linearized.
  LogicalResult matchAndRewrite(math::FPowIOp op,
                                PatternRewriter &rewriter) const final;

private:
  GetPowerFuncCallbackTy getFuncOpCallback;
};
} // namespace

template <typename Op>
LogicalResult
VecOpToScalarOp<Op>::matchAndRewrite(Op op, PatternRewriter &rewriter) const {
  Type opType = op.getType();
  Location loc = op.getLoc();
  auto vecType = opType.template dyn_cast<VectorType>();

  if (!vecType)
    return rewriter.notifyMatchFailure(op, "not a vector operation");
  if (!vecType.hasRank())
    return rewriter.notifyMatchFailure(op, "unknown vector rank");
  ArrayRef<int64_t> shape = vecType.getShape();
  int64_t numElements = vecType.getNumElements();

  Type resultElementType = vecType.getElementType();
  Attribute initValueAttr;
  if (resultElementType.isa<FloatType>())
    initValueAttr = FloatAttr::get(resultElementType, 0.0);
  else
    initValueAttr = IntegerAttr::get(resultElementType, 0);
  Value result = rewriter.create<arith::ConstantOp>(
      loc, DenseElementsAttr::get(vecType, initValueAttr));
  SmallVector<int64_t> strides = computeStrides(shape);
  for (int64_t linearIndex = 0; linearIndex < numElements; ++linearIndex) {
    SmallVector<int64_t> positions = delinearize(linearIndex, strides);
    SmallVector<Value> operands;
    for (Value input : op->getOperands())
      operands.push_back(
          rewriter.create<vector::ExtractOp>(loc, input, positions));
    Value scalarOp =
        rewriter.create<Op>(loc, vecType.getElementType(), operands);
    result =
        rewriter.create<vector::InsertOp>(loc, scalarOp, result, positions);
  }
  rewriter.replaceOp(op, result);
  return success();
}

static FunctionType getElementalFuncTypeForOp(Operation *op) {
  SmallVector<Type, 1> resultTys(op->getNumResults());
  SmallVector<Type, 2> inputTys(op->getNumOperands());
  std::transform(op->result_type_begin(), op->result_type_end(),
                 resultTys.begin(),
                 [](Type ty) { return getElementTypeOrSelf(ty); });
  std::transform(op->operand_type_begin(), op->operand_type_end(),
                 inputTys.begin(),
                 [](Type ty) { return getElementTypeOrSelf(ty); });
  return FunctionType::get(op->getContext(), inputTys, resultTys);
}

/// Create linkonce_odr function to implement the power function with
/// the given \p elementType type inside \p module. The \p elementType
/// must be IntegerType, an the created function has
/// 'IntegerType (*)(IntegerType, IntegerType)' function type.
///
/// template <typename T>
/// T __mlir_math_ipowi_*(T b, T p) {
///   if (p == T(0))
///     return T(1);
///   if (p < T(0)) {
///     if (b == T(0))
///       return T(1) / T(0); // trigger div-by-zero
///     if (b == T(1))
///       return T(1);
///     if (b == T(-1)) {
///       if (p & T(1))
///         return T(-1);
///       return T(1);
///     }
///     return T(0);
///   }
///   T result = T(1);
///   while (true) {
///     if (p & T(1))
///       result *= b;
///     p >>= T(1);
///     if (p == T(0))
///       return result;
///     b *= b;
///   }
/// }
static func::FuncOp createElementIPowIFunc(ModuleOp *module, Type elementType) {
  assert(elementType.isa<IntegerType>() &&
         "non-integer element type for IPowIOp");

  ImplicitLocOpBuilder builder =
      ImplicitLocOpBuilder::atBlockEnd(module->getLoc(), module->getBody());

  std::string funcName("__mlir_math_ipowi");
  llvm::raw_string_ostream nameOS(funcName);
  nameOS << '_' << elementType;

  FunctionType funcType = FunctionType::get(
      builder.getContext(), {elementType, elementType}, elementType);
  auto funcOp = builder.create<func::FuncOp>(funcName, funcType);
  LLVM::linkage::Linkage inlineLinkage = LLVM::linkage::Linkage::LinkonceODR;
  Attribute linkage =
      LLVM::LinkageAttr::get(builder.getContext(), inlineLinkage);
  funcOp->setAttr("llvm.linkage", linkage);
  funcOp.setPrivate();

  Block *entryBlock = funcOp.addEntryBlock();
  Region *funcBody = entryBlock->getParent();

  Value bArg = funcOp.getArgument(0);
  Value pArg = funcOp.getArgument(1);
  builder.setInsertionPointToEnd(entryBlock);
  Value zeroValue = builder.create<arith::ConstantOp>(
      elementType, builder.getIntegerAttr(elementType, 0));
  Value oneValue = builder.create<arith::ConstantOp>(
      elementType, builder.getIntegerAttr(elementType, 1));
  Value minusOneValue = builder.create<arith::ConstantOp>(
      elementType,
      builder.getIntegerAttr(elementType,
                             APInt(elementType.getIntOrFloatBitWidth(), -1ULL,
                                   /*isSigned=*/true)));

  // if (p == T(0))
  //   return T(1);
  auto pIsZero =
      builder.create<arith::CmpIOp>(arith::CmpIPredicate::eq, pArg, zeroValue);
  Block *thenBlock = builder.createBlock(funcBody);
  builder.create<func::ReturnOp>(oneValue);
  Block *fallthroughBlock = builder.createBlock(funcBody);
  // Set up conditional branch for (p == T(0)).
  builder.setInsertionPointToEnd(pIsZero->getBlock());
  builder.create<cf::CondBranchOp>(pIsZero, thenBlock, fallthroughBlock);

  // if (p < T(0)) {
  builder.setInsertionPointToEnd(fallthroughBlock);
  auto pIsNeg =
      builder.create<arith::CmpIOp>(arith::CmpIPredicate::sle, pArg, zeroValue);
  //   if (b == T(0))
  builder.createBlock(funcBody);
  auto bIsZero =
      builder.create<arith::CmpIOp>(arith::CmpIPredicate::eq, bArg, zeroValue);
  //     return T(1) / T(0);
  thenBlock = builder.createBlock(funcBody);
  builder.create<func::ReturnOp>(
      builder.create<arith::DivSIOp>(oneValue, zeroValue).getResult());
  fallthroughBlock = builder.createBlock(funcBody);
  // Set up conditional branch for (b == T(0)).
  builder.setInsertionPointToEnd(bIsZero->getBlock());
  builder.create<cf::CondBranchOp>(bIsZero, thenBlock, fallthroughBlock);

  //   if (b == T(1))
  builder.setInsertionPointToEnd(fallthroughBlock);
  auto bIsOne =
      builder.create<arith::CmpIOp>(arith::CmpIPredicate::eq, bArg, oneValue);
  //    return T(1);
  thenBlock = builder.createBlock(funcBody);
  builder.create<func::ReturnOp>(oneValue);
  fallthroughBlock = builder.createBlock(funcBody);
  // Set up conditional branch for (b == T(1)).
  builder.setInsertionPointToEnd(bIsOne->getBlock());
  builder.create<cf::CondBranchOp>(bIsOne, thenBlock, fallthroughBlock);

  //   if (b == T(-1)) {
  builder.setInsertionPointToEnd(fallthroughBlock);
  auto bIsMinusOne = builder.create<arith::CmpIOp>(arith::CmpIPredicate::eq,
                                                   bArg, minusOneValue);
  //     if (p & T(1))
  builder.createBlock(funcBody);
  auto pIsOdd = builder.create<arith::CmpIOp>(
      arith::CmpIPredicate::ne, builder.create<arith::AndIOp>(pArg, oneValue),
      zeroValue);
  //       return T(-1);
  thenBlock = builder.createBlock(funcBody);
  builder.create<func::ReturnOp>(minusOneValue);
  fallthroughBlock = builder.createBlock(funcBody);
  // Set up conditional branch for (p & T(1)).
  builder.setInsertionPointToEnd(pIsOdd->getBlock());
  builder.create<cf::CondBranchOp>(pIsOdd, thenBlock, fallthroughBlock);

  //     return T(1);
  //   } // b == T(-1)
  builder.setInsertionPointToEnd(fallthroughBlock);
  builder.create<func::ReturnOp>(oneValue);
  fallthroughBlock = builder.createBlock(funcBody);
  // Set up conditional branch for (b == T(-1)).
  builder.setInsertionPointToEnd(bIsMinusOne->getBlock());
  builder.create<cf::CondBranchOp>(bIsMinusOne, pIsOdd->getBlock(),
                                   fallthroughBlock);

  //   return T(0);
  // } // (p < T(0))
  builder.setInsertionPointToEnd(fallthroughBlock);
  builder.create<func::ReturnOp>(zeroValue);
  Block *loopHeader = builder.createBlock(
      funcBody, funcBody->end(), {elementType, elementType, elementType},
      {builder.getLoc(), builder.getLoc(), builder.getLoc()});
  // Set up conditional branch for (p < T(0)).
  builder.setInsertionPointToEnd(pIsNeg->getBlock());
  // Set initial values of 'result', 'b' and 'p' for the loop.
  builder.create<cf::CondBranchOp>(pIsNeg, bIsZero->getBlock(), loopHeader,
                                   ValueRange{oneValue, bArg, pArg});

  // T result = T(1);
  // while (true) {
  //   if (p & T(1))
  //     result *= b;
  //   p >>= T(1);
  //   if (p == T(0))
  //     return result;
  //   b *= b;
  // }
  Value resultTmp = loopHeader->getArgument(0);
  Value baseTmp = loopHeader->getArgument(1);
  Value powerTmp = loopHeader->getArgument(2);
  builder.setInsertionPointToEnd(loopHeader);

  //   if (p & T(1))
  auto powerTmpIsOdd = builder.create<arith::CmpIOp>(
      arith::CmpIPredicate::ne,
      builder.create<arith::AndIOp>(powerTmp, oneValue), zeroValue);
  thenBlock = builder.createBlock(funcBody);
  //     result *= b;
  Value newResultTmp = builder.create<arith::MulIOp>(resultTmp, baseTmp);
  fallthroughBlock = builder.createBlock(funcBody, funcBody->end(), elementType,
                                         builder.getLoc());
  builder.setInsertionPointToEnd(thenBlock);
  builder.create<cf::BranchOp>(newResultTmp, fallthroughBlock);
  // Set up conditional branch for (p & T(1)).
  builder.setInsertionPointToEnd(powerTmpIsOdd->getBlock());
  builder.create<cf::CondBranchOp>(powerTmpIsOdd, thenBlock, fallthroughBlock,
                                   resultTmp);
  // Merged 'result'.
  newResultTmp = fallthroughBlock->getArgument(0);

  //   p >>= T(1);
  builder.setInsertionPointToEnd(fallthroughBlock);
  Value newPowerTmp = builder.create<arith::ShRUIOp>(powerTmp, oneValue);

  //   if (p == T(0))
  auto newPowerIsZero = builder.create<arith::CmpIOp>(arith::CmpIPredicate::eq,
                                                      newPowerTmp, zeroValue);
  //     return result;
  thenBlock = builder.createBlock(funcBody);
  builder.create<func::ReturnOp>(newResultTmp);
  fallthroughBlock = builder.createBlock(funcBody);
  // Set up conditional branch for (p == T(0)).
  builder.setInsertionPointToEnd(newPowerIsZero->getBlock());
  builder.create<cf::CondBranchOp>(newPowerIsZero, thenBlock, fallthroughBlock);

  //   b *= b;
  // }
  builder.setInsertionPointToEnd(fallthroughBlock);
  Value newBaseTmp = builder.create<arith::MulIOp>(baseTmp, baseTmp);
  // Pass new values for 'result', 'b' and 'p' to the loop header.
  builder.create<cf::BranchOp>(
      ValueRange{newResultTmp, newBaseTmp, newPowerTmp}, loopHeader);
  return funcOp;
}

/// Convert IPowI into a call to a local function implementing
/// the power operation. The local function computes a scalar result,
/// so vector forms of IPowI are linearized.
LogicalResult
IPowIOpLowering::matchAndRewrite(math::IPowIOp op,
                                 PatternRewriter &rewriter) const {
  auto baseType = op.getOperands()[0].getType().dyn_cast<IntegerType>();

  if (!baseType)
    return rewriter.notifyMatchFailure(op, "non-integer base operand");

  // The outlined software implementation must have been already
  // generated.
  func::FuncOp elementFunc = getFuncOpCallback(baseType);
  if (!elementFunc)
    return rewriter.notifyMatchFailure(op, "missing software implementation");

  rewriter.replaceOpWithNewOp<func::CallOp>(op, elementFunc, op.getOperands());
  return success();
}

/// Create linkonce_odr function to implement the power function with
/// the given \p funcType type inside \p module. The \p funcType must be
/// 'FloatType (*)(FloatType, IntegerType)' function type.
///
/// template <typename T>
/// Tb __mlir_math_fpowi_*(Tb b, Tp p) {
///   if (p == Tp{0})
///     return Tb{1};
///   bool isNegativePower{p < Tp{0}}
///   bool isMin{p == std::numeric_limits<Tp>::min()};
///   if (isMin) {
///     p = std::numeric_limits<Tp>::max();
///   } else if (isNegativePower) {
///     p = -p;
///   }
///   Tb result = Tb{1};
///   Tb origBase = Tb{b};
///   while (true) {
///     if (p & Tp{1})
///       result *= b;
///     p >>= Tp{1};
///     if (p == Tp{0})
///       break;
///     b *= b;
///   }
///   if (isMin) {
///     result *= origBase;
///   }
///   if (isNegativePower) {
///     result = Tb{1} / result;
///   }
///   return result;
/// }
static func::FuncOp createElementFPowIFunc(ModuleOp *module,
                                           FunctionType funcType) {
  auto baseType = funcType.getInput(0).cast<FloatType>();
  auto powType = funcType.getInput(1).cast<IntegerType>();
  ImplicitLocOpBuilder builder =
      ImplicitLocOpBuilder::atBlockEnd(module->getLoc(), module->getBody());

  std::string funcName("__mlir_math_fpowi");
  llvm::raw_string_ostream nameOS(funcName);
  nameOS << '_' << baseType;
  nameOS << '_' << powType;
  auto funcOp = builder.create<func::FuncOp>(funcName, funcType);
  LLVM::linkage::Linkage inlineLinkage = LLVM::linkage::Linkage::LinkonceODR;
  Attribute linkage =
      LLVM::LinkageAttr::get(builder.getContext(), inlineLinkage);
  funcOp->setAttr("llvm.linkage", linkage);
  funcOp.setPrivate();

  Block *entryBlock = funcOp.addEntryBlock();
  Region *funcBody = entryBlock->getParent();

  Value bArg = funcOp.getArgument(0);
  Value pArg = funcOp.getArgument(1);
  builder.setInsertionPointToEnd(entryBlock);
  Value oneBValue = builder.create<arith::ConstantOp>(
      baseType, builder.getFloatAttr(baseType, 1.0));
  Value zeroPValue = builder.create<arith::ConstantOp>(
      powType, builder.getIntegerAttr(powType, 0));
  Value onePValue = builder.create<arith::ConstantOp>(
      powType, builder.getIntegerAttr(powType, 1));
  Value minPValue = builder.create<arith::ConstantOp>(
      powType, builder.getIntegerAttr(powType, llvm::APInt::getSignedMinValue(
                                                   powType.getWidth())));
  Value maxPValue = builder.create<arith::ConstantOp>(
      powType, builder.getIntegerAttr(powType, llvm::APInt::getSignedMaxValue(
                                                   powType.getWidth())));

  // if (p == Tp{0})
  //   return Tb{1};
  auto pIsZero =
      builder.create<arith::CmpIOp>(arith::CmpIPredicate::eq, pArg, zeroPValue);
  Block *thenBlock = builder.createBlock(funcBody);
  builder.create<func::ReturnOp>(oneBValue);
  Block *fallthroughBlock = builder.createBlock(funcBody);
  // Set up conditional branch for (p == Tp{0}).
  builder.setInsertionPointToEnd(pIsZero->getBlock());
  builder.create<cf::CondBranchOp>(pIsZero, thenBlock, fallthroughBlock);

  builder.setInsertionPointToEnd(fallthroughBlock);
  // bool isNegativePower{p < Tp{0}}
  auto pIsNeg = builder.create<arith::CmpIOp>(arith::CmpIPredicate::sle, pArg,
                                              zeroPValue);
  // bool isMin{p == std::numeric_limits<Tp>::min()};
  auto pIsMin =
      builder.create<arith::CmpIOp>(arith::CmpIPredicate::eq, pArg, minPValue);

  // if (isMin) {
  //   p = std::numeric_limits<Tp>::max();
  // } else if (isNegativePower) {
  //   p = -p;
  // }
  Value negP = builder.create<arith::SubIOp>(zeroPValue, pArg);
  auto pInit = builder.create<arith::SelectOp>(pIsNeg, negP, pArg);
  pInit = builder.create<arith::SelectOp>(pIsMin, maxPValue, pInit);

  // Tb result = Tb{1};
  // Tb origBase = Tb{b};
  // while (true) {
  //   if (p & Tp{1})
  //     result *= b;
  //   p >>= Tp{1};
  //   if (p == Tp{0})
  //     break;
  //   b *= b;
  // }
  Block *loopHeader = builder.createBlock(
      funcBody, funcBody->end(), {baseType, baseType, powType},
      {builder.getLoc(), builder.getLoc(), builder.getLoc()});
  // Set initial values of 'result', 'b' and 'p' for the loop.
  builder.setInsertionPointToEnd(pInit->getBlock());
  builder.create<cf::BranchOp>(loopHeader, ValueRange{oneBValue, bArg, pInit});

  // Create loop body.
  Value resultTmp = loopHeader->getArgument(0);
  Value baseTmp = loopHeader->getArgument(1);
  Value powerTmp = loopHeader->getArgument(2);
  builder.setInsertionPointToEnd(loopHeader);

  //   if (p & Tp{1})
  auto powerTmpIsOdd = builder.create<arith::CmpIOp>(
      arith::CmpIPredicate::ne,
      builder.create<arith::AndIOp>(powerTmp, onePValue), zeroPValue);
  thenBlock = builder.createBlock(funcBody);
  //     result *= b;
  Value newResultTmp = builder.create<arith::MulFOp>(resultTmp, baseTmp);
  fallthroughBlock = builder.createBlock(funcBody, funcBody->end(), baseType,
                                         builder.getLoc());
  builder.setInsertionPointToEnd(thenBlock);
  builder.create<cf::BranchOp>(newResultTmp, fallthroughBlock);
  // Set up conditional branch for (p & Tp{1}).
  builder.setInsertionPointToEnd(powerTmpIsOdd->getBlock());
  builder.create<cf::CondBranchOp>(powerTmpIsOdd, thenBlock, fallthroughBlock,
                                   resultTmp);
  // Merged 'result'.
  newResultTmp = fallthroughBlock->getArgument(0);

  //   p >>= Tp{1};
  builder.setInsertionPointToEnd(fallthroughBlock);
  Value newPowerTmp = builder.create<arith::ShRUIOp>(powerTmp, onePValue);

  //   if (p == Tp{0})
  auto newPowerIsZero = builder.create<arith::CmpIOp>(arith::CmpIPredicate::eq,
                                                      newPowerTmp, zeroPValue);
  //     break;
  //
  // The conditional branch is finalized below with a jump to
  // the loop exit block.
  fallthroughBlock = builder.createBlock(funcBody);

  //   b *= b;
  // }
  builder.setInsertionPointToEnd(fallthroughBlock);
  Value newBaseTmp = builder.create<arith::MulFOp>(baseTmp, baseTmp);
  // Pass new values for 'result', 'b' and 'p' to the loop header.
  builder.create<cf::BranchOp>(
      ValueRange{newResultTmp, newBaseTmp, newPowerTmp}, loopHeader);

  // Set up conditional branch for early loop exit:
  //   if (p == Tp{0})
  //     break;
  Block *loopExit = builder.createBlock(funcBody, funcBody->end(), baseType,
                                        builder.getLoc());
  builder.setInsertionPointToEnd(newPowerIsZero->getBlock());
  builder.create<cf::CondBranchOp>(newPowerIsZero, loopExit, newResultTmp,
                                   fallthroughBlock, ValueRange{});

  // if (isMin) {
  //   result *= origBase;
  // }
  newResultTmp = loopExit->getArgument(0);
  thenBlock = builder.createBlock(funcBody);
  fallthroughBlock = builder.createBlock(funcBody, funcBody->end(), baseType,
                                         builder.getLoc());
  builder.setInsertionPointToEnd(loopExit);
  builder.create<cf::CondBranchOp>(pIsMin, thenBlock, fallthroughBlock,
                                   newResultTmp);
  builder.setInsertionPointToEnd(thenBlock);
  newResultTmp = builder.create<arith::MulFOp>(newResultTmp, bArg);
  builder.create<cf::BranchOp>(newResultTmp, fallthroughBlock);

  /// if (isNegativePower) {
  ///   result = Tb{1} / result;
  /// }
  newResultTmp = fallthroughBlock->getArgument(0);
  thenBlock = builder.createBlock(funcBody);
  Block *returnBlock = builder.createBlock(funcBody, funcBody->end(), baseType,
                                           builder.getLoc());
  builder.setInsertionPointToEnd(fallthroughBlock);
  builder.create<cf::CondBranchOp>(pIsNeg, thenBlock, returnBlock,
                                   newResultTmp);
  builder.setInsertionPointToEnd(thenBlock);
  newResultTmp = builder.create<arith::DivFOp>(oneBValue, newResultTmp);
  builder.create<cf::BranchOp>(newResultTmp, returnBlock);

  // return result;
  builder.setInsertionPointToEnd(returnBlock);
  builder.create<func::ReturnOp>(returnBlock->getArgument(0));

  return funcOp;
}

/// Convert FPowI into a call to a local function implementing
/// the power operation. The local function computes a scalar result,
/// so vector forms of FPowI are linearized.
LogicalResult
FPowIOpLowering::matchAndRewrite(math::FPowIOp op,
                                 PatternRewriter &rewriter) const {
  if (op.getType().template dyn_cast<VectorType>())
    return rewriter.notifyMatchFailure(op, "non-scalar operation");

  FunctionType funcType = getElementalFuncTypeForOp(op);

  // The outlined software implementation must have been already
  // generated.
  func::FuncOp elementFunc = getFuncOpCallback(funcType);
  if (!elementFunc)
    return rewriter.notifyMatchFailure(op, "missing software implementation");

  rewriter.replaceOpWithNewOp<func::CallOp>(op, elementFunc, op.getOperands());
  return success();
}

namespace {
struct ConvertMathToFuncsPass
    : public impl::ConvertMathToFuncsBase<ConvertMathToFuncsPass> {
  ConvertMathToFuncsPass() = default;
  ConvertMathToFuncsPass(const ConvertMathToFuncsOptions &options)
      : impl::ConvertMathToFuncsBase<ConvertMathToFuncsPass>(options) {}

  void runOnOperation() override;

private:
  // Return true, if this FPowI operation must be converted
  // because the width of its exponent's type is greater than
  // or equal to minWidthOfFPowIExponent option value.
  bool isFPowIConvertible(math::FPowIOp op);

  // Generate outlined implementations for power operations
  // and store them in powerFuncs map.
  void preprocessPowOperations();

  // A map between function types deduced from power operations
  // and the corresponding outlined software implementations
  // of these operations.
  DenseMap<Type, func::FuncOp> powerFuncs;
};
} // namespace

bool ConvertMathToFuncsPass::isFPowIConvertible(math::FPowIOp op) {
  auto expTy =
      getElementTypeOrSelf(op.getRhs().getType()).dyn_cast<IntegerType>();
  return (expTy && expTy.getWidth() >= minWidthOfFPowIExponent);
}

void ConvertMathToFuncsPass::preprocessPowOperations() {
  ModuleOp module = getOperation();

  module.walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<math::IPowIOp>([&](math::IPowIOp op) {
          Type resultType = getElementTypeOrSelf(op.getResult().getType());

          // Generate the software implementation of this operation,
          // if it has not been generated yet.
          auto entry = powerFuncs.try_emplace(resultType, func::FuncOp{});
          if (entry.second)
            entry.first->second = createElementIPowIFunc(&module, resultType);
        })
        .Case<math::FPowIOp>([&](math::FPowIOp op) {
          if (!isFPowIConvertible(op))
            return;

          FunctionType funcType = getElementalFuncTypeForOp(op);

          // Generate the software implementation of this operation,
          // if it has not been generated yet.
          // FPowI implementations are mapped via the FunctionType
          // created from the operation's result and operands.
          auto entry = powerFuncs.try_emplace(funcType, func::FuncOp{});
          if (entry.second)
            entry.first->second = createElementFPowIFunc(&module, funcType);
        });
  });
}

void ConvertMathToFuncsPass::runOnOperation() {
  ModuleOp module = getOperation();

  // Create outlined implementations for power operations.
  preprocessPowOperations();

  RewritePatternSet patterns(&getContext());
  patterns.add<VecOpToScalarOp<math::IPowIOp>, VecOpToScalarOp<math::FPowIOp>>(
      patterns.getContext());

  // For the given Type Returns FuncOp stored in powerFuncs map.
  auto getPowerFuncOpByType = [&](Type type) -> func::FuncOp {
    auto it = powerFuncs.find(type);
    if (it == powerFuncs.end())
      return {};

    return it->second;
  };
  patterns.add<IPowIOpLowering, FPowIOpLowering>(patterns.getContext(),
                                                 getPowerFuncOpByType);

  ConversionTarget target(getContext());
  target.addLegalDialect<arith::ArithDialect, cf::ControlFlowDialect,
                         func::FuncDialect, vector::VectorDialect>();
  target.addIllegalOp<math::IPowIOp>();
  target.addDynamicallyLegalOp<math::FPowIOp>(
      [this](math::FPowIOp op) { return !isFPowIConvertible(op); });
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
