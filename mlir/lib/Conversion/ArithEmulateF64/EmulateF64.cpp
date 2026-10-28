//===- EmulateF64.cpp - Double precision on 64-bit integers ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emulates f64 for targets without double precision (AIR, AGX): f64 values
// become their bit patterns in i64, in every type (memrefs, vectors, function
// signatures), and f64 operations become calls of the library of
// SoftF64/soft_f64.c (IEEE 754 binary64, rounding to nearest even, with
// denormals), which is inserted, as LLVM dialect functions, into the symbol
// table that uses it. Negation, absolute value and copysign are bit
// operations. An f64 operation without an emulation (math.exp, ...) is an
// error.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithEmulateF64/ArithEmulateF64.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APFloat.h"

namespace mlir {
#define GEN_PASS_DEF_ARITHEMULATEF64
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
#include "SoftF64/SoftF64.inc"

constexpr uint64_t kSign = 0x8000000000000000ULL;

bool isF64(Type t) { return getElementTypeOrSelf(t).isF64(); }

/// f64 as i64, in scalars, vectors and memrefs.
struct F64Converter : TypeConverter {
  F64Converter() {
    addConversion([](Type t) { return t; });
    addConversion([](Float64Type t) -> Type {
      return IntegerType::get(t.getContext(), 64);
    });
    addConversion([](VectorType t) -> Type {
      if (!t.getElementType().isF64())
        return t;
      return t.clone(IntegerType::get(t.getContext(), 64));
    });
    addConversion([](MemRefType t) -> Type {
      if (!t.getElementType().isF64())
        return t;
      return MemRefType::get(t.getShape(), IntegerType::get(t.getContext(), 64),
                             t.getLayout(), t.getMemorySpace());
    });
  }
};

/// The library of soft_f64.c, parsed once per pass run, from which functions
/// are copied into the symbol tables that call them.
struct Library {
  OwningOpRef<ModuleOp> module;

  LogicalResult load(MLIRContext *ctx) {
    std::string text;
    for (const char *line : kSoftF64Library)
      text += line;
    module = parseSourceString<ModuleOp>(text, ctx);
    return module ? success() : failure();
  }

  /// A call of the library function `name`, which is copied into the symbol
  /// table of `anchor` on first use.
  Value call(OpBuilder &b, Operation *anchor, StringRef name, Type result,
             ValueRange args) const {
    Operation *table = anchor->getParentWithTrait<OpTrait::SymbolTable>();
    auto fn = dyn_cast_or_null<LLVM::LLVMFuncOp>(
        SymbolTable::lookupSymbolIn(table, name));
    if (!fn) {
      auto source = cast<LLVM::LLVMFuncOp>(
          SymbolTable::lookupSymbolIn(module.get(), name));
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(&table->getRegion(0).front());
      fn = cast<LLVM::LLVMFuncOp>(b.clone(*source));
      fn.setLinkage(LLVM::Linkage::Internal);
    }
    // The library touches no memory (which lets the calls stand in
    // memref.generic_atomic_rmw bodies).
    auto none = LLVM::MemoryEffectsAttr::get(
        b.getContext(), {LLVM::ModRefInfo::NoModRef, LLVM::ModRefInfo::NoModRef,
                         LLVM::ModRefInfo::NoModRef});
    fn.setMemoryEffectsAttr(none);
    auto call = LLVM::CallOp::create(b, anchor->getLoc(), fn, args);
    call.setMemoryEffectsAttr(none);
    return call.getResult();
  }
};

/// Applies `scalar` to each element of the (rank-1) vector operands, or to
/// the scalar operands.
template <typename Fn>
FailureOr<Value> elementwise(OpBuilder &b, Location loc, Type resultType,
                             ValueRange operands, Fn scalar) {
  auto vt = dyn_cast<VectorType>(resultType);
  if (!vt)
    return Value(scalar(operands));
  if (vt.getRank() != 1 || vt.isScalable())
    return failure();
  Value result = arith::ConstantOp::create(b, loc, vt, b.getZeroAttr(vt));
  for (int64_t i = 0, e = vt.getDimSize(0); i < e; ++i) {
    SmallVector<Value> elements;
    for (Value v : operands)
      elements.push_back(vector::ExtractOp::create(b, loc, v, i));
    result = vector::InsertOp::create(b, loc, scalar(elements), result, i);
  }
  return result;
}

/// Base of the patterns of one f64 operation: the converted operands, the
/// converted result type and the library.
template <typename Op>
struct F64Pattern : OpConversionPattern<Op> {
  F64Pattern(const TypeConverter &tc, MLIRContext *ctx, const Library &lib)
      : OpConversionPattern<Op>(tc, ctx), lib(lib) {}
  const Library &lib;
};

/// arith.constant of f64: the bits.
struct ConstantPattern : F64Pattern<arith::ConstantOp> {
  using F64Pattern::F64Pattern;
  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isF64(op.getType()))
      return failure();
    Type type = getTypeConverter()->convertType(op.getType());
    TypedAttr value;
    if (auto f = dyn_cast<FloatAttr>(op.getValue()))
      value = rewriter.getIntegerAttr(type,
                                      f.getValue().bitcastToAPInt().getZExtValue());
    else if (auto d = dyn_cast<DenseFPElementsAttr>(op.getValue()))
      value = d.bitcast(rewriter.getI64Type());
    else
      return rewriter.notifyMatchFailure(op, "f64 constant attribute");
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, type, value);
    return success();
  }
};

/// An f64 operation that is one library call per element.
template <typename Op>
struct CallPattern : F64Pattern<Op> {
  CallPattern(const TypeConverter &tc, MLIRContext *ctx, const Library &lib,
              StringRef name)
      : F64Pattern<Op>(tc, ctx, lib), name(name) {}
  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isF64(op.getType()))
      return failure();
    Type type = this->getTypeConverter()->convertType(op.getType());
    Type i64 = rewriter.getI64Type();
    FailureOr<Value> r = elementwise(
        rewriter, op.getLoc(), type, adaptor.getOperands(),
        [&](ValueRange args) {
          return this->lib.call(rewriter, op, name, i64, args);
        });
    if (failed(r))
      return rewriter.notifyMatchFailure(op, "vector shape");
    rewriter.replaceOp(op, *r);
    return success();
  }
  std::string name;
};

/// Sign bit operations: negf, absf and copysign.
template <typename Op>
struct SignPattern : F64Pattern<Op> {
  using F64Pattern<Op>::F64Pattern;
  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isF64(op.getType()))
      return failure();
    Location loc = op.getLoc();
    Type type = this->getTypeConverter()->convertType(op.getType());
    auto mask = [&](uint64_t bits) -> Value {
      return arith::ConstantOp::create(
          rewriter, loc, type,
          isa<VectorType>(type)
              ? cast<TypedAttr>(DenseElementsAttr::get(
                    cast<ShapedType>(type), rewriter.getI64IntegerAttr(bits)))
              : rewriter.getI64IntegerAttr(bits));
    };
    ValueRange ops = adaptor.getOperands();
    Value r;
    if (std::is_same_v<Op, arith::NegFOp>)
      r = arith::XOrIOp::create(rewriter, loc, ops[0], mask(kSign));
    else if (std::is_same_v<Op, math::AbsFOp>)
      r = arith::AndIOp::create(rewriter, loc, ops[0], mask(~kSign));
    else
      r = arith::OrIOp::create(
          rewriter, loc,
          arith::AndIOp::create(rewriter, loc, ops[0], mask(~kSign)),
          arith::AndIOp::create(rewriter, loc, ops[1], mask(kSign)));
    rewriter.replaceOp(op, r);
    return success();
  }
};

/// arith.cmpf on f64, by the library's eq, lt and unord.
struct CmpFPattern : F64Pattern<arith::CmpFOp> {
  using F64Pattern::F64Pattern;
  LogicalResult
  matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isF64(op.getLhs().getType()))
      return failure();
    Location loc = op.getLoc();
    Type i32 = rewriter.getI32Type();
    FailureOr<Value> r = elementwise(
        rewriter, loc, op.getType(), {adaptor.getLhs(), adaptor.getRhs()},
        [&](ValueRange ab) -> Value {
          auto test = [&](StringRef fn, Value x, Value y) -> Value {
            Value v = lib.call(rewriter, op, fn, i32, {x, y});
            Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
            return arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne,
                                         v, zero);
          };
          auto Or = [&](Value x, Value y) -> Value {
            return arith::OrIOp::create(rewriter, loc, x, y);
          };
          auto Not = [&](Value x) -> Value {
            Value one = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
            return arith::XOrIOp::create(rewriter, loc, x, one);
          };
          Value a = ab[0], b = ab[1];
          using P = arith::CmpFPredicate;
          switch (op.getPredicate()) {
          case P::AlwaysFalse:
            return arith::ConstantIntOp::create(rewriter, loc, 0, 1);
          case P::AlwaysTrue:
            return arith::ConstantIntOp::create(rewriter, loc, 1, 1);
          case P::OEQ:
            return test("__air_f64_eq", a, b);
          case P::UNE:
            return Not(test("__air_f64_eq", a, b));
          case P::OLT:
            return test("__air_f64_lt", a, b);
          case P::OGT:
            return test("__air_f64_lt", b, a);
          case P::OLE:
            return Or(test("__air_f64_lt", a, b), test("__air_f64_eq", a, b));
          case P::OGE:
            return Or(test("__air_f64_lt", b, a), test("__air_f64_eq", a, b));
          case P::UGE:
            return Not(test("__air_f64_lt", a, b));
          case P::ULE:
            return Not(test("__air_f64_lt", b, a));
          case P::ULT:
            return Or(test("__air_f64_unord", a, b), test("__air_f64_lt", a, b));
          case P::UGT:
            return Or(test("__air_f64_unord", a, b), test("__air_f64_lt", b, a));
          case P::UEQ:
            return Or(test("__air_f64_unord", a, b), test("__air_f64_eq", a, b));
          case P::ONE:
            return Not(Or(test("__air_f64_unord", a, b),
                          test("__air_f64_eq", a, b)));
          case P::ORD:
            return Not(test("__air_f64_unord", a, b));
          case P::UNO:
            return test("__air_f64_unord", a, b);
          }
          llvm_unreachable("unknown predicate");
        });
    if (failed(r))
      return rewriter.notifyMatchFailure(op, "vector shape");
    rewriter.replaceOp(op, *r);
    return success();
  }
};

/// maximumf / minimumf (NaN-propagating) and maxnumf / minnumf (NaN-
/// ignoring); -0 is below +0.
template <typename Op, bool IsMax, bool PropagatesNaN>
struct MinMaxPattern : F64Pattern<Op> {
  using F64Pattern<Op>::F64Pattern;
  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isF64(op.getType()))
      return failure();
    Location loc = op.getLoc();
    Type i32 = rewriter.getI32Type();
    Type type = this->getTypeConverter()->convertType(op.getType());
    FailureOr<Value> r = elementwise(
        rewriter, loc, type, adaptor.getOperands(), [&](ValueRange ab) -> Value {
          Value a = ab[0], b = ab[1];
          auto test = [&](StringRef fn, Value x, Value y) -> Value {
            Value v = this->lib.call(rewriter, op, fn, i32, {x, y});
            Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
            return arith::CmpIOp::create(rewriter, loc,
                                         arith::CmpIPredicate::ne, v, zero);
          };
          auto isNaN = [&](Value x) { return test("__air_f64_unord", x, x); };
          // Equal values (or zeros): the sign bits ANDed for the maximum,
          // ORed for the minimum.
          Value tie = IsMax ? Value(arith::AndIOp::create(rewriter, loc, a, b))
                            : Value(arith::OrIOp::create(rewriter, loc, a, b));
          Value aLess = test("__air_f64_lt", a, b);
          Value bLess = test("__air_f64_lt", b, a);
          Value pick = arith::SelectOp::create(
              rewriter, loc, aLess, IsMax ? b : a,
              arith::SelectOp::create(rewriter, loc, bLess, IsMax ? a : b, tie));
          Value quiet = arith::ConstantIntOp::create(rewriter, loc,
                                                     0x0008000000000000LL, 64);
          if (PropagatesNaN) {
            Value nanA = arith::OrIOp::create(rewriter, loc, a, quiet);
            Value nanB = arith::OrIOp::create(rewriter, loc, b, quiet);
            return arith::SelectOp::create(
                rewriter, loc, isNaN(a), nanA,
                arith::SelectOp::create(rewriter, loc, isNaN(b), nanB, pick));
          }
          return arith::SelectOp::create(
              rewriter, loc, isNaN(a), b,
              arith::SelectOp::create(rewriter, loc, isNaN(b), a, pick));
        });
    if (failed(r))
      return rewriter.notifyMatchFailure(op, "vector shape");
    rewriter.replaceOp(op, *r);
    return success();
  }
};

/// sitofp / uitofp to f64 and fptosi / fptoui from f64, through 64 bits.
template <typename Op, bool Signed, bool ToF64>
struct IntConversionPattern : F64Pattern<Op> {
  using F64Pattern<Op>::F64Pattern;
  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isF64(ToF64 ? op.getType() : op.getIn().getType()))
      return failure();
    Location loc = op.getLoc();
    Type i64 = rewriter.getI64Type();
    Type type = this->getTypeConverter()->convertType(op.getType());
    FailureOr<Value> r = elementwise(
        rewriter, loc, type, adaptor.getOperands(), [&](ValueRange in) -> Value {
          Value x = in[0];
          if (ToF64) {
            if (x.getType().getIntOrFloatBitWidth() < 64)
              x = Signed ? Value(arith::ExtSIOp::create(rewriter, loc, i64, x))
                         : Value(arith::ExtUIOp::create(rewriter, loc, i64, x));
            return this->lib.call(rewriter, op,
                                  Signed ? "__air_f64_from_i64"
                                         : "__air_f64_from_u64",
                                  i64, x);
          }
          Value v = this->lib.call(
              rewriter, op, Signed ? "__air_f64_to_i64" : "__air_f64_to_u64",
              i64, x);
          Type elt = getElementTypeOrSelf(type);
          if (elt.getIntOrFloatBitWidth() < 64)
            v = arith::TruncIOp::create(rewriter, loc, elt, v);
          return v;
        });
    if (failed(r))
      return rewriter.notifyMatchFailure(op, "vector shape");
    rewriter.replaceOp(op, *r);
    return success();
  }
};

/// extf to f64 (exact, through f32 for the narrower types) and truncf from
/// f64 to f32.
struct ExtFPattern : F64Pattern<arith::ExtFOp> {
  using F64Pattern::F64Pattern;
  LogicalResult
  matchAndRewrite(arith::ExtFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isF64(op.getType()))
      return failure();
    Location loc = op.getLoc();
    Type type = getTypeConverter()->convertType(op.getType());
    FailureOr<Value> r = elementwise(
        rewriter, loc, type, adaptor.getOperands(), [&](ValueRange in) -> Value {
          Value x = in[0];
          if (!x.getType().isF32())
            x = arith::ExtFOp::create(rewriter, loc, rewriter.getF32Type(), x);
          Value bits =
              arith::BitcastOp::create(rewriter, loc, rewriter.getI32Type(), x);
          return lib.call(rewriter, op, "__air_f64_from_f32",
                          rewriter.getI64Type(), bits);
        });
    if (failed(r))
      return rewriter.notifyMatchFailure(op, "vector shape");
    rewriter.replaceOp(op, *r);
    return success();
  }
};

struct TruncFPattern : F64Pattern<arith::TruncFOp> {
  using F64Pattern::F64Pattern;
  LogicalResult
  matchAndRewrite(arith::TruncFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isF64(op.getIn().getType()))
      return failure();
    if (!getElementTypeOrSelf(op.getType()).isF32())
      return op.emitOpError("from f64 to a type other than f32 is not "
                            "emulated (it would round twice)");
    Location loc = op.getLoc();
    FailureOr<Value> r = elementwise(
        rewriter, loc, op.getType(), adaptor.getOperands(),
        [&](ValueRange in) -> Value {
          Value bits = lib.call(rewriter, op, "__air_f64_to_f32",
                                rewriter.getI32Type(), in[0]);
          return arith::BitcastOp::create(rewriter, loc, rewriter.getF32Type(),
                                          bits);
        });
    if (failed(r))
      return rewriter.notifyMatchFailure(op, "vector shape");
    rewriter.replaceOp(op, *r);
    return success();
  }
};

/// arith.bitcast between f64 and i64: the value itself.
struct BitcastPattern : F64Pattern<arith::BitcastOp> {
  using F64Pattern::F64Pattern;
  LogicalResult
  matchAndRewrite(arith::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isF64(op.getType()) && !isF64(op.getIn().getType()))
      return failure();
    Type type = getTypeConverter()->convertType(op.getType());
    if (adaptor.getIn().getType() == type)
      rewriter.replaceOp(op, adaptor.getIn());
    else
      rewriter.replaceOpWithNewOp<arith::BitcastOp>(op, type, adaptor.getIn());
    return success();
  }
};

/// math.ceil and math.trunc from the library's floor: ceil(x) = -floor(-x),
/// trunc(x) = x < 0 ? ceil(x) : floor(x) (on the sign bit).
template <typename Op, bool IsCeil>
struct CeilTruncPattern : F64Pattern<Op> {
  using F64Pattern<Op>::F64Pattern;
  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isF64(op.getType()))
      return failure();
    Location loc = op.getLoc();
    Type i64 = rewriter.getI64Type();
    Type type = this->getTypeConverter()->convertType(op.getType());
    FailureOr<Value> r = elementwise(
        rewriter, loc, type, adaptor.getOperands(), [&](ValueRange in) -> Value {
          Value x = in[0];
          Value sign = arith::ConstantIntOp::create(rewriter, loc, kSign, 64);
          auto ceil = [&](Value v) -> Value {
            Value neg = arith::XOrIOp::create(rewriter, loc, v, sign);
            Value f = this->lib.call(rewriter, op, "__air_f64_floor", i64, neg);
            return arith::XOrIOp::create(rewriter, loc, f, sign);
          };
          if (IsCeil)
            return ceil(x);
          Value floor = this->lib.call(rewriter, op, "__air_f64_floor", i64, x);
          Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 64);
          Value negative = arith::CmpIOp::create(
              rewriter, loc, arith::CmpIPredicate::slt, x, zero);
          return arith::SelectOp::create(rewriter, loc, negative, ceil(x), floor);
        });
    if (failed(r))
      return rewriter.notifyMatchFailure(op, "vector shape");
    rewriter.replaceOp(op, *r);
    return success();
  }
};

/// memref.atomic_rmw on f64 memory: a compare-and-swap loop on the element's
/// address around the emulated operation (memref.generic_atomic_rmw does not
/// take the library's calls).
struct AtomicRMWPattern : F64Pattern<memref::AtomicRMWOp> {
  using F64Pattern::F64Pattern;
  LogicalResult
  matchAndRewrite(memref::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isF64(op.getValue().getType()))
      return failure();
    StringRef fn;
    switch (op.getKind()) {
    case arith::AtomicRMWKind::addf:
      fn = "__air_f64_add";
      break;
    case arith::AtomicRMWKind::mulf:
      fn = "__air_f64_mul";
      break;
    case arith::AtomicRMWKind::assign:
      break;
    default:
      return op.emitOpError("of this kind on f64 is not emulated");
    }
    auto space = dyn_cast_or_null<IntegerAttr>(
        op.getMemRefType().getMemorySpace());
    if (op.getMemRefType().getMemorySpace() && !space)
      return op.emitOpError("on f64 in a memory space that is not a number");
    Location loc = op.getLoc();
    Type i64 = rewriter.getI64Type();
    Value memref = adaptor.getMemref();
    // The element's address: aligned pointer + 8 * (offset + sum(i * stride)).
    auto md = memref::ExtractStridedMetadataOp::create(rewriter, loc, memref);
    Value linear = md.getOffset();
    for (auto [i, stride] : llvm::zip(adaptor.getIndices(), md.getStrides()))
      linear = arith::AddIOp::create(
          rewriter, loc, linear,
          arith::MulIOp::create(rewriter, loc, i, stride));
    Value base =
        memref::ExtractAlignedPointerAsIndexOp::create(rewriter, loc, memref);
    Value eight = arith::ConstantIndexOp::create(rewriter, loc, 8);
    Value addr = arith::AddIOp::create(
        rewriter, loc, base, arith::MulIOp::create(rewriter, loc, linear, eight));
    Value addr64 = arith::IndexCastOp::create(rewriter, loc, i64, addr);
    auto ptrType = LLVM::LLVMPointerType::get(
        rewriter.getContext(), space ? space.getInt() : 0);
    Value ptr = LLVM::IntToPtrOp::create(rewriter, loc, ptrType, addr64);
    Value init = LLVM::LoadOp::create(rewriter, loc, i64, ptr);
    auto loop = scf::WhileOp::create(rewriter, loc, TypeRange{i64}, init);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      Block *before = rewriter.createBlock(&loop.getBefore(), {}, {i64}, {loc});
      Value old = before->getArgument(0);
      Value updated = fn.empty() ? adaptor.getValue()
                                 : lib.call(rewriter, op, fn, i64,
                                            {old, adaptor.getValue()});
      auto cas = LLVM::AtomicCmpXchgOp::create(
          rewriter, loc, ptr, old, updated, LLVM::AtomicOrdering::monotonic,
          LLVM::AtomicOrdering::monotonic);
      Value seen = LLVM::ExtractValueOp::create(rewriter, loc, cas, 0);
      Value ok = LLVM::ExtractValueOp::create(rewriter, loc, cas, 1);
      Value retry = arith::XOrIOp::create(
          rewriter, loc, ok, arith::ConstantIntOp::create(rewriter, loc, 1, 1));
      scf::ConditionOp::create(rewriter, loc, retry, seen);
      Block *after = rewriter.createBlock(&loop.getAfter(), {}, {i64}, {loc});
      scf::YieldOp::create(rewriter, loc, after->getArgument(0));
    }
    // The value the successful exchange found: the old value.
    rewriter.replaceOp(op, loop.getResult(0));
    return success();
  }
};

/// Any other operation with f64 in its types: the same operation on the
/// converted types, for operations that only move values (memref.load,
/// vector.extract, arith.select, func.call, gpu.return ...).
struct StructuralPattern : ConversionPattern {
  StructuralPattern(const TypeConverter &tc, MLIRContext *ctx)
      : ConversionPattern(tc, MatchAnyOpTypeTag(), /*benefit=*/0, ctx) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumRegions() || isa<FunctionOpInterface>(op))
      return failure();
    Dialect *dialect = op->getDialect();
    StringRef ns = dialect ? dialect->getNamespace() : "";
    // Arithmetic is only emulated by the patterns above.
    if (ns == "math" ||
        (ns == "arith" && !isa<arith::SelectOp, arith::ConstantOp>(op)))
      return op->emitOpError("on f64 is not emulated");
    SmallVector<Type> types;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(), types)))
      return failure();
    Operation *newOp =
        rewriter.create(op->getLoc(), op->getName().getIdentifier(), operands,
                        types, op->getAttrs(), op->getSuccessors());
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct EmulateF64Pass final : impl::ArithEmulateF64Base<EmulateF64Pass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    Library lib;
    if (failed(lib.load(ctx))) {
      getOperation()->emitError("cannot parse the soft-f64 library");
      return signalPassFailure();
    }
    // SIMD-group reductions of f64 become 64-bit shuffles and the arith ops
    // that combine them (emulated below): a reduction is not a structural op,
    // its integer form would add the bit patterns.
    SmallVector<Operation *> reductions;
    getOperation()->walk([&](gpu::SubgroupReduceOp op) {
      if (getElementTypeOrSelf(op.getType()).isF64())
        reductions.push_back(op);
    });
    if (!reductions.empty()) {
      RewritePatternSet shuffles(ctx);
      populateGpuLowerSubgroupReduceToShufflePatterns(
          shuffles, /*subgroupSize=*/32, /*shuffleBitwidth=*/64);
      populateGpuLowerClusteredSubgroupReduceToShufflePatterns(
          shuffles, /*subgroupSize=*/32, /*shuffleBitwidth=*/64);
      bool changed = false;
      if (failed(applyOpPatternsGreedily(reductions, std::move(shuffles),
                                         GreedyRewriteConfig(), &changed)))
        return signalPassFailure();
    }

    F64Converter converter;
    ConversionTarget target(*ctx);
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      if (auto fn = dyn_cast<FunctionOpInterface>(op))
        return converter.isLegal(fn.getArgumentTypes()) &&
               converter.isLegal(fn.getResultTypes()) &&
               converter.isLegal(&fn.getFunctionBody());
      return converter.isLegal(op);
    });
    RewritePatternSet patterns(ctx);
    patterns.add<StructuralPattern>(converter, ctx);
    patterns.add<ConstantPattern, CmpFPattern, ExtFPattern, TruncFPattern,
                 BitcastPattern, AtomicRMWPattern,
                 SignPattern<arith::NegFOp>, SignPattern<math::AbsFOp>,
                 SignPattern<math::CopySignOp>,
                 MinMaxPattern<arith::MaximumFOp, true, true>,
                 MinMaxPattern<arith::MinimumFOp, false, true>,
                 MinMaxPattern<arith::MaxNumFOp, true, false>,
                 MinMaxPattern<arith::MinNumFOp, false, false>,
                 IntConversionPattern<arith::SIToFPOp, true, true>,
                 IntConversionPattern<arith::UIToFPOp, false, true>,
                 IntConversionPattern<arith::FPToSIOp, true, false>,
                 IntConversionPattern<arith::FPToUIOp, false, false>,
                 CeilTruncPattern<math::CeilOp, true>,
                 CeilTruncPattern<math::TruncOp, false>>(converter, ctx, lib);
    patterns.add<CallPattern<arith::AddFOp>>(converter, ctx, lib,
                                             "__air_f64_add");
    patterns.add<CallPattern<arith::SubFOp>>(converter, ctx, lib,
                                             "__air_f64_sub");
    patterns.add<CallPattern<arith::MulFOp>>(converter, ctx, lib,
                                             "__air_f64_mul");
    patterns.add<CallPattern<arith::DivFOp>>(converter, ctx, lib,
                                             "__air_f64_div");
    patterns.add<CallPattern<math::FmaOp>>(converter, ctx, lib,
                                           "__air_f64_fma");
    patterns.add<CallPattern<math::SqrtOp>>(converter, ctx, lib,
                                            "__air_f64_sqrt");
    patterns.add<CallPattern<math::FloorOp>>(converter, ctx, lib,
                                             "__air_f64_floor");
    patterns.add<CallPattern<math::RoundEvenOp>>(converter, ctx, lib,
                                                 "__air_f64_roundeven");
    populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, converter);
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
