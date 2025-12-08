//===- ArithToAPFloat.cpp - Arithmetic to APFloat Conversion --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToAPFloat/ArithToAPFloat.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_ARITHTOAPFLOATCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::func;

static FuncOp createFnDecl(OpBuilder &b, SymbolOpInterface symTable,
                           StringRef name, FunctionType funcT, bool setPrivate,
                           SymbolTableCollection *symbolTables = nullptr) {
  OpBuilder::InsertionGuard g(b);
  assert(!symTable->getRegion(0).empty() && "expected non-empty region");
  b.setInsertionPointToStart(&symTable->getRegion(0).front());
  FuncOp funcOp = FuncOp::create(b, symTable->getLoc(), name, funcT);
  if (setPrivate)
    funcOp.setPrivate();
  if (symbolTables) {
    SymbolTable &symbolTable = symbolTables->getSymbolTable(symTable);
    symbolTable.insert(funcOp, symTable->getRegion(0).front().begin());
  }
  return funcOp;
}

/// Helper function to look up or create the symbol for a runtime library
/// function with the given parameter types. Returns an int64_t, unless a
/// different result type is specified.
static FailureOr<FuncOp>
lookupOrCreateApFloatFn(OpBuilder &b, SymbolOpInterface symTable,
                        StringRef name, TypeRange paramTypes,
                        SymbolTableCollection *symbolTables = nullptr,
                        Type resultType = {}) {
  if (!resultType)
    resultType = IntegerType::get(symTable->getContext(), 64);
  std::string funcName = (llvm::Twine("_mlir_apfloat_") + name).str();
  auto funcT = FunctionType::get(b.getContext(), paramTypes, {resultType});
  FailureOr<FuncOp> func =
      lookupFnDecl(symTable, funcName, funcT, symbolTables);
  // Failed due to type mismatch.
  if (failed(func))
    return func;
  // Successfully matched existing decl.
  if (*func)
    return *func;

  return createFnDecl(b, symTable, funcName, funcT,
                      /*setPrivate=*/true, symbolTables);
}

/// Helper function to look up or create the symbol for a runtime library
/// function for a binary arithmetic operation.
///
/// Parameter 1: APFloat semantics
/// Parameter 2: Left-hand side operand
/// Parameter 3: Right-hand side operand
///
/// This function will return a failure if the function is found but has an
/// unexpected signature.
///
static FailureOr<FuncOp>
lookupOrCreateBinaryFn(OpBuilder &b, SymbolOpInterface symTable, StringRef name,
                       SymbolTableCollection *symbolTables = nullptr) {
  auto i32Type = IntegerType::get(symTable->getContext(), 32);
  auto i64Type = IntegerType::get(symTable->getContext(), 64);
  return lookupOrCreateApFloatFn(b, symTable, name, {i32Type, i64Type, i64Type},
                                 symbolTables);
}

static Value getSemanticsValue(OpBuilder &b, Location loc, FloatType floatTy) {
  int32_t sem = llvm::APFloatBase::SemanticsToEnum(floatTy.getFloatSemantics());
  return arith::ConstantOp::create(b, loc, b.getI32Type(),
                                   b.getIntegerAttr(b.getI32Type(), sem));
}

/// Given two operands of vector type and vector result type (with the same
/// shape), call the given function for each pair of scalar operands and
/// package the result into a vector. If the given operands and result type are
/// not vectors, call the function directly. The second operand is optional.
template <typename Fn, typename... Values>
static Value forEachScalarValue(RewriterBase &rewriter, Location loc,
                                Value operand1, Value operand2, Type resultType,
                                Fn fn) {
  auto vecTy1 = dyn_cast<VectorType>(operand1.getType());
  if (operand2) {
    // Sanity check: Operand types must match.
    assert(vecTy1 == dyn_cast<VectorType>(operand2.getType()) &&
           "expected same vector types");
  }
  if (!vecTy1) {
    // Not a vector. Call the function directly.
    return fn(operand1, operand2, resultType);
  }

  // Prepare scalar operands.
  ResultRange sclars1 =
      vector::ToElementsOp::create(rewriter, loc, operand1)->getResults();
  SmallVector<Value> scalars2;
  if (!operand2) {
    // No second operand. Create a vector of empty values.
    scalars2.assign(vecTy1.getNumElements(), Value());
  } else {
    llvm::append_range(
        scalars2,
        vector::ToElementsOp::create(rewriter, loc, operand2)->getResults());
  }

  // Call the function for each pair of scalar operands.
  auto resultVecType = cast<VectorType>(resultType);
  SmallVector<Value> results;
  for (auto [scalar1, scalar2] : llvm::zip_equal(sclars1, scalars2)) {
    Value result = fn(scalar1, scalar2, resultVecType.getElementType());
    results.push_back(result);
  }

  // Package the results into a vector.
  return vector::FromElementsOp::create(
      rewriter, loc,
      vecTy1.cloneWith(/*shape=*/std::nullopt, results.front().getType()),
      results);
}

/// Check preconditions for the conversion:
/// 1. All operands / results must be integers or floats (or vectors thereof).
/// 2. The bitwidth of the operands / results must be <= 64.
static LogicalResult checkPreconditions(RewriterBase &rewriter, Operation *op) {
  for (Value value : llvm::concat<Value>(op->getOperands(), op->getResults())) {
    Type type = value.getType();
    if (auto vecTy = dyn_cast<VectorType>(type)) {
      type = vecTy.getElementType();
    }
    if (!type.isIntOrFloat()) {
      return rewriter.notifyMatchFailure(
          op, "only integers and floats (or vectors thereof) are supported");
    }
    if (type.getIntOrFloatBitWidth() > 64)
      return rewriter.notifyMatchFailure(op,
                                         "bitwidth > 64 bits is not supported");
  }
  return success();
}

/// Rewrite a binary arithmetic operation to an APFloat function call.
template <typename OpTy>
struct BinaryArithOpToAPFloatConversion final : OpRewritePattern<OpTy> {
  BinaryArithOpToAPFloatConversion(MLIRContext *context,
                                   const char *APFloatName,
                                   SymbolOpInterface symTable,
                                   PatternBenefit benefit = 1)
      : OpRewritePattern<OpTy>(context, benefit), symTable(symTable),
        APFloatName(APFloatName) {};

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (failed(checkPreconditions(rewriter, op)))
      return failure();

    // Get APFloat function from runtime library.
    FailureOr<FuncOp> fn =
        lookupOrCreateBinaryFn(rewriter, symTable, APFloatName);
    if (failed(fn))
      return fn;

    // Scalarize and convert to APFloat runtime calls.
    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    Value repl = forEachScalarValue(
        rewriter, loc, op.getLhs(), op.getRhs(), op.getType(),
        [&](Value lhs, Value rhs, Type resultType) {
          // Cast operands to 64-bit integers.
          auto floatTy = cast<FloatType>(resultType);
          auto intWType = rewriter.getIntegerType(floatTy.getWidth());
          auto int64Type = rewriter.getI64Type();
          Value lhsBits = arith::ExtUIOp::create(
              rewriter, loc, int64Type,
              arith::BitcastOp::create(rewriter, loc, intWType, lhs));
          Value rhsBits = arith::ExtUIOp::create(
              rewriter, loc, int64Type,
              arith::BitcastOp::create(rewriter, loc, intWType, rhs));

          // Call APFloat function.
          Value semValue = getSemanticsValue(rewriter, loc, floatTy);
          SmallVector<Value> params = {semValue, lhsBits, rhsBits};
          auto resultOp = func::CallOp::create(rewriter, loc,
                                               TypeRange(rewriter.getI64Type()),
                                               SymbolRefAttr::get(*fn), params);

          // Truncate result to the original width.
          Value truncatedBits = arith::TruncIOp::create(rewriter, loc, intWType,
                                                        resultOp->getResult(0));
          return arith::BitcastOp::create(rewriter, loc, floatTy,
                                          truncatedBits);
        });
    rewriter.replaceOp(op, repl);
    return success();
  }

  SymbolOpInterface symTable;
  const char *APFloatName;
};

template <typename OpTy>
struct FpToFpConversion final : OpRewritePattern<OpTy> {
  FpToFpConversion(MLIRContext *context, SymbolOpInterface symTable,
                   PatternBenefit benefit = 1)
      : OpRewritePattern<OpTy>(context, benefit), symTable(symTable) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (failed(checkPreconditions(rewriter, op)))
      return failure();

    // Get APFloat function from runtime library.
    auto i32Type = IntegerType::get(symTable->getContext(), 32);
    auto i64Type = IntegerType::get(symTable->getContext(), 64);
    FailureOr<FuncOp> fn = lookupOrCreateApFloatFn(
        rewriter, symTable, "convert", {i32Type, i32Type, i64Type});
    if (failed(fn))
      return fn;

    // Scalarize and convert to APFloat runtime calls.
    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    Value repl = forEachScalarValue(
        rewriter, loc, op.getOperand(), /*operand2=*/Value(), op.getType(),
        [&](Value operand1, Value operand2, Type resultType) {
          // Cast operands to 64-bit integers.
          auto inFloatTy = cast<FloatType>(operand1.getType());
          auto inIntWType = rewriter.getIntegerType(inFloatTy.getWidth());
          Value operandBits = arith::ExtUIOp::create(
              rewriter, loc, i64Type,
              arith::BitcastOp::create(rewriter, loc, inIntWType, operand1));

          // Call APFloat function.
          Value inSemValue = getSemanticsValue(rewriter, loc, inFloatTy);
          auto outFloatTy = cast<FloatType>(resultType);
          Value outSemValue = getSemanticsValue(rewriter, loc, outFloatTy);
          std::array<Value, 3> params = {inSemValue, outSemValue, operandBits};
          auto resultOp = func::CallOp::create(rewriter, loc,
                                               TypeRange(rewriter.getI64Type()),
                                               SymbolRefAttr::get(*fn), params);

          // Truncate result to the original width.
          auto outIntWType = rewriter.getIntegerType(outFloatTy.getWidth());
          Value truncatedBits = arith::TruncIOp::create(
              rewriter, loc, outIntWType, resultOp->getResult(0));
          return arith::BitcastOp::create(rewriter, loc, outFloatTy,
                                          truncatedBits);
        });
    rewriter.replaceOp(op, repl);
    return success();
  }

  SymbolOpInterface symTable;
};

template <typename OpTy>
struct FpToIntConversion final : OpRewritePattern<OpTy> {
  FpToIntConversion(MLIRContext *context, SymbolOpInterface symTable,
                    bool isUnsigned, PatternBenefit benefit = 1)
      : OpRewritePattern<OpTy>(context, benefit), symTable(symTable),
        isUnsigned(isUnsigned) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (failed(checkPreconditions(rewriter, op)))
      return failure();

    // Get APFloat function from runtime library.
    auto i1Type = IntegerType::get(symTable->getContext(), 1);
    auto i32Type = IntegerType::get(symTable->getContext(), 32);
    auto i64Type = IntegerType::get(symTable->getContext(), 64);
    FailureOr<FuncOp> fn =
        lookupOrCreateApFloatFn(rewriter, symTable, "convert_to_int",
                                {i32Type, i32Type, i1Type, i64Type});
    if (failed(fn))
      return fn;

    // Scalarize and convert to APFloat runtime calls.
    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    Value repl = forEachScalarValue(
        rewriter, loc, op.getOperand(), /*operand2=*/Value(), op.getType(),
        [&](Value operand1, Value operand2, Type resultType) {
          // Cast operands to 64-bit integers.
          auto inFloatTy = cast<FloatType>(operand1.getType());
          auto inIntWType = rewriter.getIntegerType(inFloatTy.getWidth());
          Value operandBits = arith::ExtUIOp::create(
              rewriter, loc, i64Type,
              arith::BitcastOp::create(rewriter, loc, inIntWType, operand1));

          // Call APFloat function.
          Value inSemValue = getSemanticsValue(rewriter, loc, inFloatTy);
          auto outIntTy = cast<IntegerType>(resultType);
          Value outWidthValue = arith::ConstantOp::create(
              rewriter, loc, i32Type,
              rewriter.getIntegerAttr(i32Type, outIntTy.getWidth()));
          Value isUnsignedValue = arith::ConstantOp::create(
              rewriter, loc, i1Type,
              rewriter.getIntegerAttr(i1Type, isUnsigned));
          SmallVector<Value> params = {inSemValue, outWidthValue,
                                       isUnsignedValue, operandBits};
          auto resultOp = func::CallOp::create(rewriter, loc,
                                               TypeRange(rewriter.getI64Type()),
                                               SymbolRefAttr::get(*fn), params);

          // Truncate result to the original width.
          return arith::TruncIOp::create(rewriter, loc, outIntTy,
                                         resultOp->getResult(0));
        });
    rewriter.replaceOp(op, repl);
    return success();
  }

  SymbolOpInterface symTable;
  bool isUnsigned;
};

template <typename OpTy>
struct IntToFpConversion final : OpRewritePattern<OpTy> {
  IntToFpConversion(MLIRContext *context, SymbolOpInterface symTable,
                    bool isUnsigned, PatternBenefit benefit = 1)
      : OpRewritePattern<OpTy>(context, benefit), symTable(symTable),
        isUnsigned(isUnsigned) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (failed(checkPreconditions(rewriter, op)))
      return failure();

    // Get APFloat function from runtime library.
    auto i1Type = IntegerType::get(symTable->getContext(), 1);
    auto i32Type = IntegerType::get(symTable->getContext(), 32);
    auto i64Type = IntegerType::get(symTable->getContext(), 64);
    FailureOr<FuncOp> fn =
        lookupOrCreateApFloatFn(rewriter, symTable, "convert_from_int",
                                {i32Type, i32Type, i1Type, i64Type});
    if (failed(fn))
      return fn;

    // Scalarize and convert to APFloat runtime calls.
    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    Value repl = forEachScalarValue(
        rewriter, loc, op.getOperand(), /*operand2=*/Value(), op.getType(),
        [&](Value operand1, Value operand2, Type resultType) {
          // Cast operands to 64-bit integers.
          auto inIntTy = cast<IntegerType>(operand1.getType());
          Value operandBits = operand1;
          if (operandBits.getType().getIntOrFloatBitWidth() < 64) {
            if (isUnsigned) {
              operandBits =
                  arith::ExtUIOp::create(rewriter, loc, i64Type, operandBits);
            } else {
              operandBits =
                  arith::ExtSIOp::create(rewriter, loc, i64Type, operandBits);
            }
          }

          // Call APFloat function.
          auto outFloatTy = cast<FloatType>(resultType);
          Value outSemValue = getSemanticsValue(rewriter, loc, outFloatTy);
          Value inWidthValue = arith::ConstantOp::create(
              rewriter, loc, i32Type,
              rewriter.getIntegerAttr(i32Type, inIntTy.getWidth()));
          Value isUnsignedValue = arith::ConstantOp::create(
              rewriter, loc, i1Type,
              rewriter.getIntegerAttr(i1Type, isUnsigned));
          SmallVector<Value> params = {outSemValue, inWidthValue,
                                       isUnsignedValue, operandBits};
          auto resultOp = func::CallOp::create(rewriter, loc,
                                               TypeRange(rewriter.getI64Type()),
                                               SymbolRefAttr::get(*fn), params);

          // Truncate result to the original width.
          auto outIntWType = rewriter.getIntegerType(outFloatTy.getWidth());
          Value truncatedBits = arith::TruncIOp::create(
              rewriter, loc, outIntWType, resultOp->getResult(0));
          return arith::BitcastOp::create(rewriter, loc, outFloatTy,
                                          truncatedBits);
        });
    rewriter.replaceOp(op, repl);
    return success();
  }

  SymbolOpInterface symTable;
  bool isUnsigned;
};

struct CmpFOpToAPFloatConversion final : OpRewritePattern<arith::CmpFOp> {
  CmpFOpToAPFloatConversion(MLIRContext *context, SymbolOpInterface symTable,
                            PatternBenefit benefit = 1)
      : OpRewritePattern<arith::CmpFOp>(context, benefit), symTable(symTable) {}

  LogicalResult matchAndRewrite(arith::CmpFOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(checkPreconditions(rewriter, op)))
      return failure();

    // Get APFloat function from runtime library.
    auto i1Type = IntegerType::get(symTable->getContext(), 1);
    auto i8Type = IntegerType::get(symTable->getContext(), 8);
    auto i32Type = IntegerType::get(symTable->getContext(), 32);
    auto i64Type = IntegerType::get(symTable->getContext(), 64);
    FailureOr<FuncOp> fn =
        lookupOrCreateApFloatFn(rewriter, symTable, "compare",
                                {i32Type, i64Type, i64Type}, nullptr, i8Type);
    if (failed(fn))
      return fn;

    // Scalarize and convert to APFloat runtime calls.
    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    Value repl = forEachScalarValue(
        rewriter, loc, op.getLhs(), op.getRhs(), op.getType(),
        [&](Value lhs, Value rhs, Type resultType) {
          // Cast operands to 64-bit integers.
          auto floatTy = cast<FloatType>(lhs.getType());
          auto intWType = rewriter.getIntegerType(floatTy.getWidth());
          Value lhsBits = arith::ExtUIOp::create(
              rewriter, loc, i64Type,
              arith::BitcastOp::create(rewriter, loc, intWType, lhs));
          Value rhsBits = arith::ExtUIOp::create(
              rewriter, loc, i64Type,
              arith::BitcastOp::create(rewriter, loc, intWType, rhs));

          // Call APFloat function.
          Value semValue = getSemanticsValue(rewriter, loc, floatTy);
          SmallVector<Value> params = {semValue, lhsBits, rhsBits};
          Value comparisonResult =
              func::CallOp::create(rewriter, loc, TypeRange(i8Type),
                                   SymbolRefAttr::get(*fn), params)
                  ->getResult(0);

          // Generate an i1 SSA value that is "true" if the comparison result
          // matches the given `val`.
          auto checkResult = [&](llvm::APFloat::cmpResult val) {
            return arith::CmpIOp::create(
                rewriter, loc, arith::CmpIPredicate::eq, comparisonResult,
                arith::ConstantOp::create(
                    rewriter, loc, i8Type,
                    rewriter.getIntegerAttr(i8Type, static_cast<int8_t>(val)))
                    .getResult());
          };
          // Generate an i1 SSA value that is "true" if the comparison result
          // matches any of the given `vals`.
          std::function<Value(ArrayRef<llvm::APFloat::cmpResult>)>
              checkResults = [&](ArrayRef<llvm::APFloat::cmpResult> vals) {
                Value first = checkResult(vals.front());
                if (vals.size() == 1)
                  return first;
                Value rest = checkResults(vals.drop_front());
                return arith::OrIOp::create(rewriter, loc, first, rest)
                    .getResult();
              };

          // This switch-case statement was taken from arith::applyCmpPredicate.
          Value result;
          switch (op.getPredicate()) {
          case arith::CmpFPredicate::AlwaysFalse:
            result =
                arith::ConstantOp::create(rewriter, loc, i1Type,
                                          rewriter.getIntegerAttr(i1Type, 0))
                    .getResult();
            break;
          case arith::CmpFPredicate::OEQ:
            result = checkResult(llvm::APFloat::cmpEqual);
            break;
          case arith::CmpFPredicate::OGT:
            result = checkResult(llvm::APFloat::cmpGreaterThan);
            break;
          case arith::CmpFPredicate::OGE:
            result = checkResults(
                {llvm::APFloat::cmpGreaterThan, llvm::APFloat::cmpEqual});
            break;
          case arith::CmpFPredicate::OLT:
            result = checkResult(llvm::APFloat::cmpLessThan);
            break;
          case arith::CmpFPredicate::OLE:
            result = checkResults(
                {llvm::APFloat::cmpLessThan, llvm::APFloat::cmpEqual});
            break;
          case arith::CmpFPredicate::ONE:
            // Not cmpUnordered and not cmpUnordered.
            result = checkResults(
                {llvm::APFloat::cmpLessThan, llvm::APFloat::cmpGreaterThan});
            break;
          case arith::CmpFPredicate::ORD:
            // Not cmpUnordered.
            result = checkResults({llvm::APFloat::cmpLessThan,
                                   llvm::APFloat::cmpGreaterThan,
                                   llvm::APFloat::cmpEqual});
            break;
          case arith::CmpFPredicate::UEQ:
            result = checkResults(
                {llvm::APFloat::cmpUnordered, llvm::APFloat::cmpEqual});
            break;
          case arith::CmpFPredicate::UGT:
            result = checkResults(
                {llvm::APFloat::cmpUnordered, llvm::APFloat::cmpGreaterThan});
            break;
          case arith::CmpFPredicate::UGE:
            result = checkResults({llvm::APFloat::cmpUnordered,
                                   llvm::APFloat::cmpGreaterThan,
                                   llvm::APFloat::cmpEqual});
            break;
          case arith::CmpFPredicate::ULT:
            result = checkResults(
                {llvm::APFloat::cmpUnordered, llvm::APFloat::cmpLessThan});
            break;
          case arith::CmpFPredicate::ULE:
            result = checkResults({llvm::APFloat::cmpUnordered,
                                   llvm::APFloat::cmpLessThan,
                                   llvm::APFloat::cmpEqual});
            break;
          case arith::CmpFPredicate::UNE:
            // Not cmpEqual.
            result = checkResults({llvm::APFloat::cmpLessThan,
                                   llvm::APFloat::cmpGreaterThan,
                                   llvm::APFloat::cmpUnordered});
            break;
          case arith::CmpFPredicate::UNO:
            result = checkResult(llvm::APFloat::cmpUnordered);
            break;
          case arith::CmpFPredicate::AlwaysTrue:
            result =
                arith::ConstantOp::create(rewriter, loc, i1Type,
                                          rewriter.getIntegerAttr(i1Type, 1))
                    .getResult();
            break;
          }
          return result;
        });
    rewriter.replaceOp(op, repl);
    return success();
  }

  SymbolOpInterface symTable;
};

struct NegFOpToAPFloatConversion final : OpRewritePattern<arith::NegFOp> {
  NegFOpToAPFloatConversion(MLIRContext *context, SymbolOpInterface symTable,
                            PatternBenefit benefit = 1)
      : OpRewritePattern<arith::NegFOp>(context, benefit), symTable(symTable) {}

  LogicalResult matchAndRewrite(arith::NegFOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(checkPreconditions(rewriter, op)))
      return failure();

    // Get APFloat function from runtime library.
    auto i32Type = IntegerType::get(symTable->getContext(), 32);
    auto i64Type = IntegerType::get(symTable->getContext(), 64);
    FailureOr<FuncOp> fn =
        lookupOrCreateApFloatFn(rewriter, symTable, "neg", {i32Type, i64Type});
    if (failed(fn))
      return fn;

    // Scalarize and convert to APFloat runtime calls.
    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    Value repl = forEachScalarValue(
        rewriter, loc, op.getOperand(), /*operand2=*/Value(), op.getType(),
        [&](Value operand1, Value operand2, Type resultType) {
          // Cast operands to 64-bit integers.
          auto floatTy = cast<FloatType>(operand1.getType());
          auto intWType = rewriter.getIntegerType(floatTy.getWidth());
          Value operandBits = arith::ExtUIOp::create(
              rewriter, loc, i64Type,
              arith::BitcastOp::create(rewriter, loc, intWType, operand1));

          // Call APFloat function.
          Value semValue = getSemanticsValue(rewriter, loc, floatTy);
          SmallVector<Value> params = {semValue, operandBits};
          Value negatedBits =
              func::CallOp::create(rewriter, loc, TypeRange(i64Type),
                                   SymbolRefAttr::get(*fn), params)
                  ->getResult(0);

          // Truncate result to the original width.
          Value truncatedBits =
              arith::TruncIOp::create(rewriter, loc, intWType, negatedBits);
          return arith::BitcastOp::create(rewriter, loc, floatTy,
                                          truncatedBits);
        });
    rewriter.replaceOp(op, repl);
    return success();
  }

  SymbolOpInterface symTable;
};

namespace {
struct ArithToAPFloatConversionPass final
    : impl::ArithToAPFloatConversionPassBase<ArithToAPFloatConversionPass> {
  using Base::Base;

  void runOnOperation() override;
};

void ArithToAPFloatConversionPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.add<BinaryArithOpToAPFloatConversion<arith::AddFOp>>(context, "add",
                                                                getOperation());
  patterns.add<BinaryArithOpToAPFloatConversion<arith::SubFOp>>(
      context, "subtract", getOperation());
  patterns.add<BinaryArithOpToAPFloatConversion<arith::MulFOp>>(
      context, "multiply", getOperation());
  patterns.add<BinaryArithOpToAPFloatConversion<arith::DivFOp>>(
      context, "divide", getOperation());
  patterns.add<BinaryArithOpToAPFloatConversion<arith::RemFOp>>(
      context, "remainder", getOperation());
  patterns.add<BinaryArithOpToAPFloatConversion<arith::MinNumFOp>>(
      context, "minnum", getOperation());
  patterns.add<BinaryArithOpToAPFloatConversion<arith::MaxNumFOp>>(
      context, "maxnum", getOperation());
  patterns.add<BinaryArithOpToAPFloatConversion<arith::MinimumFOp>>(
      context, "minimum", getOperation());
  patterns.add<BinaryArithOpToAPFloatConversion<arith::MaximumFOp>>(
      context, "maximum", getOperation());
  patterns
      .add<FpToFpConversion<arith::ExtFOp>, FpToFpConversion<arith::TruncFOp>,
           CmpFOpToAPFloatConversion, NegFOpToAPFloatConversion>(
          context, getOperation());
  patterns.add<FpToIntConversion<arith::FPToSIOp>>(context, getOperation(),
                                                   /*isUnsigned=*/false);
  patterns.add<FpToIntConversion<arith::FPToUIOp>>(context, getOperation(),
                                                   /*isUnsigned=*/true);
  patterns.add<IntToFpConversion<arith::SIToFPOp>>(context, getOperation(),
                                                   /*isUnsigned=*/false);
  patterns.add<IntToFpConversion<arith::UIToFPOp>>(context, getOperation(),
                                                   /*isUnsigned=*/true);
  LogicalResult result = success();
  ScopedDiagnosticHandler scopedHandler(context, [&result](Diagnostic &diag) {
    if (diag.getSeverity() == DiagnosticSeverity::Error) {
      result = failure();
    }
    // NB: if you don't return failure, no other diag handlers will fire (see
    // mlir/lib/IR/Diagnostics.cpp:DiagnosticEngineImpl::emit).
    return failure();
  });
  walkAndApplyPatterns(getOperation(), std::move(patterns));
  if (failed(result))
    return signalPassFailure();
}
} // namespace
