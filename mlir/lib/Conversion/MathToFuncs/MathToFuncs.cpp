//===- MathToFuncs.cpp - Math to outlined implementation conversion -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MathToFuncs/MathToFuncs.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
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
struct IPowIOpLowering : public OpRewritePattern<math::IPowIOp> {

private:
  GetPowerFuncCallbackTy getFuncOpCallback;

public:
  IPowIOpLowering(MLIRContext *context, GetPowerFuncCallbackTy cb)
      : OpRewritePattern<math::IPowIOp>(context), getFuncOpCallback(cb) {}

  /// Convert IPowI into a call to a local function implementing
  /// the power operation. The local function computes a scalar result,
  /// so vector forms of IPowI are linearized.
  LogicalResult matchAndRewrite(math::IPowIOp op,
                                PatternRewriter &rewriter) const final;
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

  Value result = rewriter.create<arith::ConstantOp>(
      loc, DenseElementsAttr::get(
               vecType, IntegerAttr::get(vecType.getElementType(), 0)));
  SmallVector<int64_t> ones(shape.size(), 1);
  SmallVector<int64_t> strides = computeStrides(shape, ones);
  for (int64_t linearIndex = 0; linearIndex < numElements; ++linearIndex) {
    SmallVector<int64_t> positions = delinearize(strides, linearIndex);
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

/// Create linkonce_odr function to implement the power function with
/// the given \p funcType type inside \p module. \p funcType must be
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

  //  IntegerType elementType = funcType.getInput(0).cast<IntegerType>();
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

namespace {
struct ConvertMathToFuncsPass
    : public impl::ConvertMathToFuncsBase<ConvertMathToFuncsPass> {
  ConvertMathToFuncsPass() = default;

  void runOnOperation() override;

private:
  // Generate outlined implementations for power operations
  // and store them in powerFuncs map.
  void preprocessPowOperations();

  // A map between function types deduced from power operations
  // and the corresponding outlined software implementations
  // of these operations.
  DenseMap<Type, func::FuncOp> powerFuncs;
};
} // namespace

void ConvertMathToFuncsPass::preprocessPowOperations() {
  ModuleOp module = getOperation();

  module.walk([&](Operation *op) {
    TypeSwitch<Operation *>(op).Case<math::IPowIOp>([&](math::IPowIOp op) {
      Type resultType = getElementTypeOrSelf(op.getResult().getType());

      // Generate the software implementation of this operation,
      // if it has not been generated yet.
      auto entry = powerFuncs.try_emplace(resultType, func::FuncOp{});
      if (entry.second)
        entry.first->second = createElementIPowIFunc(&module, resultType);
    });
  });
}

void ConvertMathToFuncsPass::runOnOperation() {
  ModuleOp module = getOperation();

  // Create outlined implementations for power operations.
  preprocessPowOperations();

  RewritePatternSet patterns(&getContext());
  patterns.add<VecOpToScalarOp<math::IPowIOp>>(patterns.getContext());

  // For the given Type Returns FuncOp stored in powerFuncs map.
  auto getPowerFuncOpByType = [&](Type type) -> func::FuncOp {
    auto it = powerFuncs.find(type);
    if (it == powerFuncs.end())
      return {};

    return it->second;
  };
  patterns.add<IPowIOpLowering>(patterns.getContext(), getPowerFuncOpByType);

  ConversionTarget target(getContext());
  target.addLegalDialect<arith::ArithmeticDialect, cf::ControlFlowDialect,
                         func::FuncDialect, vector::VectorDialect>();
  target.addIllegalOp<math::IPowIOp>();
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::createConvertMathToFuncsPass() {
  return std::make_unique<ConvertMathToFuncsPass>();
}
