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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMATHTOFUNCS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "math-to-funcs"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace {
// Pattern to convert vector operations to scalar operations.
template <typename Op>
struct VecOpToScalarOp : public OpRewritePattern<Op> {
public:
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final;
};

// Callback type for getting pre-generated FuncOp implementing
// an operation of the given type.
using GetFuncCallbackTy = function_ref<func::FuncOp(Operation *, Type)>;

// Pattern to convert scalar IPowIOp into a call of outlined
// software implementation.
class IPowIOpLowering : public OpRewritePattern<math::IPowIOp> {
public:
  IPowIOpLowering(MLIRContext *context, GetFuncCallbackTy cb)
      : OpRewritePattern<math::IPowIOp>(context), getFuncOpCallback(cb) {}

  /// Convert IPowI into a call to a local function implementing
  /// the power operation. The local function computes a scalar result,
  /// so vector forms of IPowI are linearized.
  LogicalResult matchAndRewrite(math::IPowIOp op,
                                PatternRewriter &rewriter) const final;

private:
  GetFuncCallbackTy getFuncOpCallback;
};

// Pattern to convert scalar FPowIOp into a call of outlined
// software implementation.
class FPowIOpLowering : public OpRewritePattern<math::FPowIOp> {
public:
  FPowIOpLowering(MLIRContext *context, GetFuncCallbackTy cb)
      : OpRewritePattern<math::FPowIOp>(context), getFuncOpCallback(cb) {}

  /// Convert FPowI into a call to a local function implementing
  /// the power operation. The local function computes a scalar result,
  /// so vector forms of FPowI are linearized.
  LogicalResult matchAndRewrite(math::FPowIOp op,
                                PatternRewriter &rewriter) const final;

private:
  GetFuncCallbackTy getFuncOpCallback;
};

// Pattern to convert scalar ctlz into a call of outlined software
// implementation.
class CtlzOpLowering : public OpRewritePattern<math::CountLeadingZerosOp> {
public:
  CtlzOpLowering(MLIRContext *context, GetFuncCallbackTy cb)
      : OpRewritePattern<math::CountLeadingZerosOp>(context),
        getFuncOpCallback(cb) {}

  /// Convert ctlz into a call to a local function implementing
  /// the count leading zeros operation.
  LogicalResult matchAndRewrite(math::CountLeadingZerosOp op,
                                PatternRewriter &rewriter) const final;

private:
  GetFuncCallbackTy getFuncOpCallback;
};
} // namespace

template <typename Op>
LogicalResult
VecOpToScalarOp<Op>::matchAndRewrite(Op op, PatternRewriter &rewriter) const {
  Type opType = op.getType();
  Location loc = op.getLoc();
  auto vecType = dyn_cast<VectorType>(opType);

  if (!vecType)
    return rewriter.notifyMatchFailure(op, "not a vector operation");
  if (!vecType.hasRank())
    return rewriter.notifyMatchFailure(op, "unknown vector rank");
  ArrayRef<int64_t> shape = vecType.getShape();
  int64_t numElements = vecType.getNumElements();

  Type resultElementType = vecType.getElementType();
  Attribute initValueAttr;
  if (isa<FloatType>(resultElementType))
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
  assert(isa<IntegerType>(elementType) &&
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
  auto baseType = dyn_cast<IntegerType>(op.getOperands()[0].getType());

  if (!baseType)
    return rewriter.notifyMatchFailure(op, "non-integer base operand");

  // The outlined software implementation must have been already
  // generated.
  func::FuncOp elementFunc = getFuncOpCallback(op, baseType);
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
  auto baseType = cast<FloatType>(funcType.getInput(0));
  auto powType = cast<IntegerType>(funcType.getInput(1));
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
  if (dyn_cast<VectorType>(op.getType()))
    return rewriter.notifyMatchFailure(op, "non-scalar operation");

  FunctionType funcType = getElementalFuncTypeForOp(op);

  // The outlined software implementation must have been already
  // generated.
  func::FuncOp elementFunc = getFuncOpCallback(op, funcType);
  if (!elementFunc)
    return rewriter.notifyMatchFailure(op, "missing software implementation");

  rewriter.replaceOpWithNewOp<func::CallOp>(op, elementFunc, op.getOperands());
  return success();
}

/// Create function to implement the ctlz function the given \p elementType type
/// inside \p module. The \p elementType must be IntegerType, an the created
/// function has 'IntegerType (*)(IntegerType)' function type.
///
/// template <typename T>
/// T __mlir_math_ctlz_*(T x) {
///     bits = sizeof(x) * 8;
///     if (x == 0)
///       return bits;
///
///     uint32_t n = 0;
///     for (int i = 1; i < bits; ++i) {
///         if (x < 0) continue;
///         n++;
///         x <<= 1;
///     }
///     return n;
/// }
///
/// Converts to (for i32):
///
/// func.func private @__mlir_math_ctlz_i32(%arg: i32) -> i32 {
///   %c_32 = arith.constant 32 : index
///   %c_0 = arith.constant 0 : i32
///   %arg_eq_zero = arith.cmpi eq, %arg, %c_0 : i1
///   %out = scf.if %arg_eq_zero {
///     scf.yield %c_32 : i32
///   } else {
///     %c_1index = arith.constant 1 : index
///     %c_1i32 = arith.constant 1 : i32
///     %n = arith.constant 0 : i32
///     %arg_out, %n_out = scf.for %i = %c_1index to %c_32 step %c_1index
///         iter_args(%arg_iter = %arg, %n_iter = %n) -> (i32, i32) {
///       %cond = arith.cmpi slt, %arg_iter, %c_0 : i32
///       %yield_val = scf.if %cond {
///         scf.yield %arg_iter, %n_iter : i32, i32
///       } else {
///         %arg_next = arith.shli %arg_iter, %c_1i32 : i32
///         %n_next = arith.addi %n_iter, %c_1i32 : i32
///         scf.yield %arg_next, %n_next : i32, i32
///       }
///       scf.yield %yield_val: i32, i32
///     }
///     scf.yield %n_out : i32
///   }
///   return %out: i32
/// }
static func::FuncOp createCtlzFunc(ModuleOp *module, Type elementType) {
  if (!isa<IntegerType>(elementType)) {
    LLVM_DEBUG({
      DBGS() << "non-integer element type for CtlzFunc; type was: ";
      elementType.print(llvm::dbgs());
    });
    llvm_unreachable("non-integer element type");
  }
  int64_t bitWidth = elementType.getIntOrFloatBitWidth();

  Location loc = module->getLoc();
  ImplicitLocOpBuilder builder =
      ImplicitLocOpBuilder::atBlockEnd(loc, module->getBody());

  std::string funcName("__mlir_math_ctlz");
  llvm::raw_string_ostream nameOS(funcName);
  nameOS << '_' << elementType;
  FunctionType funcType =
      FunctionType::get(builder.getContext(), {elementType}, elementType);
  auto funcOp = builder.create<func::FuncOp>(funcName, funcType);

  // LinkonceODR ensures that there is only one implementation of this function
  // across all math.ctlz functions that are lowered in this way.
  LLVM::linkage::Linkage inlineLinkage = LLVM::linkage::Linkage::LinkonceODR;
  Attribute linkage =
      LLVM::LinkageAttr::get(builder.getContext(), inlineLinkage);
  funcOp->setAttr("llvm.linkage", linkage);
  funcOp.setPrivate();

  // set the insertion point to the start of the function
  Block *funcBody = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(funcBody);

  Value arg = funcOp.getArgument(0);
  Type indexType = builder.getIndexType();
  Value bitWidthValue = builder.create<arith::ConstantOp>(
      elementType, builder.getIntegerAttr(elementType, bitWidth));
  Value zeroValue = builder.create<arith::ConstantOp>(
      elementType, builder.getIntegerAttr(elementType, 0));

  Value inputEqZero =
      builder.create<arith::CmpIOp>(arith::CmpIPredicate::eq, arg, zeroValue);

  // if input == 0, return bit width, else enter loop.
  scf::IfOp ifOp = builder.create<scf::IfOp>(
      elementType, inputEqZero, /*addThenBlock=*/true, /*addElseBlock=*/true);
  ifOp.getThenBodyBuilder().create<scf::YieldOp>(loc, bitWidthValue);

  auto elseBuilder =
      ImplicitLocOpBuilder::atBlockEnd(loc, &ifOp.getElseRegion().front());

  Value oneIndex = elseBuilder.create<arith::ConstantOp>(
      indexType, elseBuilder.getIndexAttr(1));
  Value oneValue = elseBuilder.create<arith::ConstantOp>(
      elementType, elseBuilder.getIntegerAttr(elementType, 1));
  Value bitWidthIndex = elseBuilder.create<arith::ConstantOp>(
      indexType, elseBuilder.getIndexAttr(bitWidth));
  Value nValue = elseBuilder.create<arith::ConstantOp>(
      elementType, elseBuilder.getIntegerAttr(elementType, 0));

  auto loop = elseBuilder.create<scf::ForOp>(
      oneIndex, bitWidthIndex, oneIndex,
      // Initial values for two loop induction variables, the arg which is being
      // shifted left in each iteration, and the n value which tracks the count
      // of leading zeros.
      ValueRange{arg, nValue},
      // Callback to build the body of the for loop
      //   if (arg < 0) {
      //     continue;
      //   } else {
      //     n++;
      //     arg <<= 1;
      //   }
      [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
        Value argIter = args[0];
        Value nIter = args[1];

        Value argIsNonNegative = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, argIter, zeroValue);
        scf::IfOp ifOp = b.create<scf::IfOp>(
            loc, argIsNonNegative,
            [&](OpBuilder &b, Location loc) {
              // If arg is negative, continue (effectively, break)
              b.create<scf::YieldOp>(loc, ValueRange{argIter, nIter});
            },
            [&](OpBuilder &b, Location loc) {
              // Otherwise, increment n and shift arg left.
              Value nNext = b.create<arith::AddIOp>(loc, nIter, oneValue);
              Value argNext = b.create<arith::ShLIOp>(loc, argIter, oneValue);
              b.create<scf::YieldOp>(loc, ValueRange{argNext, nNext});
            });
        b.create<scf::YieldOp>(loc, ifOp.getResults());
      });
  elseBuilder.create<scf::YieldOp>(loop.getResult(1));

  builder.create<func::ReturnOp>(ifOp.getResult(0));
  return funcOp;
}

/// Convert ctlz into a call to a local function implementing the ctlz
/// operation.
LogicalResult CtlzOpLowering::matchAndRewrite(math::CountLeadingZerosOp op,
                                              PatternRewriter &rewriter) const {
  if (dyn_cast<VectorType>(op.getType()))
    return rewriter.notifyMatchFailure(op, "non-scalar operation");

  Type type = getElementTypeOrSelf(op.getResult().getType());
  func::FuncOp elementFunc = getFuncOpCallback(op, type);
  if (!elementFunc)
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << "Missing software implementation for op " << op->getName()
           << " and type " << type;
    });

  rewriter.replaceOpWithNewOp<func::CallOp>(op, elementFunc, op.getOperand());
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

  // Reture true, if operation is integer type.
  bool isConvertible(Operation *op);

  // Generate outlined implementations for power operations
  // and store them in funcImpls map.
  void generateOpImplementations();

  // A map between pairs of (operation, type) deduced from operations that this
  // pass will convert, and the corresponding outlined software implementations
  // of these operations for the given type.
  DenseMap<std::pair<OperationName, Type>, func::FuncOp> funcImpls;
};
} // namespace

bool ConvertMathToFuncsPass::isFPowIConvertible(math::FPowIOp op) {
  auto expTy =
      dyn_cast<IntegerType>(getElementTypeOrSelf(op.getRhs().getType()));
  return (expTy && expTy.getWidth() >= minWidthOfFPowIExponent);
}

bool ConvertMathToFuncsPass::isConvertible(Operation *op) {
  return isa<IntegerType>(getElementTypeOrSelf(op->getResult(0).getType()));
}

void ConvertMathToFuncsPass::generateOpImplementations() {
  ModuleOp module = getOperation();

  module.walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<math::CountLeadingZerosOp>([&](math::CountLeadingZerosOp op) {
          if (!convertCtlz || !isConvertible(op))
            return;
          Type resultType = getElementTypeOrSelf(op.getResult().getType());

          // Generate the software implementation of this operation,
          // if it has not been generated yet.
          auto key = std::pair(op->getName(), resultType);
          auto entry = funcImpls.try_emplace(key, func::FuncOp{});
          if (entry.second)
            entry.first->second = createCtlzFunc(&module, resultType);
        })
        .Case<math::IPowIOp>([&](math::IPowIOp op) {
          if (!isConvertible(op))
            return;

          Type resultType = getElementTypeOrSelf(op.getResult().getType());

          // Generate the software implementation of this operation,
          // if it has not been generated yet.
          auto key = std::pair(op->getName(), resultType);
          auto entry = funcImpls.try_emplace(key, func::FuncOp{});
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
          auto key = std::pair(op->getName(), funcType);
          auto entry = funcImpls.try_emplace(key, func::FuncOp{});
          if (entry.second)
            entry.first->second = createElementFPowIFunc(&module, funcType);
        });
  });
}

void ConvertMathToFuncsPass::runOnOperation() {
  ModuleOp module = getOperation();

  // Create outlined implementations for power operations.
  generateOpImplementations();

  RewritePatternSet patterns(&getContext());
  patterns.add<VecOpToScalarOp<math::IPowIOp>, VecOpToScalarOp<math::FPowIOp>,
               VecOpToScalarOp<math::CountLeadingZerosOp>>(
      patterns.getContext());

  // For the given Type Returns FuncOp stored in funcImpls map.
  auto getFuncOpByType = [&](Operation *op, Type type) -> func::FuncOp {
    auto it = funcImpls.find(std::pair(op->getName(), type));
    if (it == funcImpls.end())
      return {};

    return it->second;
  };
  patterns.add<IPowIOpLowering, FPowIOpLowering>(patterns.getContext(),
                                                 getFuncOpByType);

  if (convertCtlz)
    patterns.add<CtlzOpLowering>(patterns.getContext(), getFuncOpByType);

  ConversionTarget target(getContext());
  target.addLegalDialect<arith::ArithDialect, cf::ControlFlowDialect,
                         func::FuncDialect, scf::SCFDialect,
                         vector::VectorDialect>();

  target.addDynamicallyLegalOp<math::IPowIOp>(
      [this](math::IPowIOp op) { return !isConvertible(op); });
  if (convertCtlz) {
    target.addDynamicallyLegalOp<math::CountLeadingZerosOp>(
        [this](math::CountLeadingZerosOp op) { return !isConvertible(op); });
  }
  target.addDynamicallyLegalOp<math::FPowIOp>(
      [this](math::FPowIOp op) { return !isFPowIConvertible(op); });
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
