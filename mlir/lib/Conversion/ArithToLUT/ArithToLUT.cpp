//===- ArithToLUT.cpp - Replace arith.extf narrow-float→f32 with LUT ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// For each arith.extf %v : narrowFP to f32 (scalar or vector), this pass emits
// a 2^N-entry f32 global constant (one per distinct source format) and replaces
// the op with a LUT lookup. Scalars use memref.load; vectors use vector.gather.
// e.g.
// ```
//   ...
//   %f32_x = arith.extf %f8_x : f8E4M3FN to f32
//   ...
// ```
// results in this sequence:
// ```
//   memref.global "private" constant @__extf_lut_f8E4M3FN : memref<256xf32>
//       = dense<"0x000000000000003B0000803B...">
//   ...
//   func.func @foo (...) {
//     ...
//     %tbl  = memref.get_global @__extf_lut_f8E4M3FN  : memref<256xf32>
//     %i8   = arith.bitcast     %v    : f8X   -> i8
//     %idx  = arith.index_castui %i8  : i8    -> index
//     %res  = memref.load       %tbl[%idx]  : memref<256xf32>
// ```
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToLUT/ArithToLUT.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTARITHEXTFTOLUT
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

// Returns true for the narrow float types that support LUT-based extf lowering.
static bool isSupportedNarrowFloatType(Type t) {
  return isa<Float4E2M1FNType, Float6E2M3FNType, Float6E3M2FNType,
             Float8E4M3FNType, Float8E5M2Type, Float8E4M3FNUZType,
             Float8E5M2FNUZType, Float8E4M3B11FNUZType, Float8E3M4Type,
             Float8E4M3Type>(t);
}

// Returns a name for the global LUT by appending the MLIR textual
// representation of the given f8 type to a fixed prefix.
static std::string lutSymbolName(FloatType srcType) {
  std::string name = "__extf_lut_";
  llvm::raw_string_ostream os(name);
  srcType.print(os);
  return name;
}

// Precomputes 2^N f32 values by enumerating every N-bit pattern for srcType.
static SmallVector<float> buildExtFLUT(FloatType srcType) {
  const llvm::fltSemantics &sem = srcType.getFloatSemantics();
  unsigned bitWidth = srcType.getWidth();
  unsigned numEntries = 1u << bitWidth;
  SmallVector<float> table;
  table.reserve(numEntries);
  for (unsigned i = 0; i < numEntries; ++i) {
    APFloat val(sem, APInt(bitWidth, i));
    bool losesInfo = false;
    val.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven,
                &losesInfo);
    table.push_back(val.convertToFloat());
  }
  return table;
}

// Inserts (or returns existing) memref.global constant for the given f8 type.
static memref::GlobalOp
getOrCreateLUT(Operation *symbolTableOp, FloatType srcType,
               llvm::DenseMap<Type, memref::GlobalOp> &cache) {
  auto it = cache.find(srcType);
  if (it != cache.end())
    return it->second;

  // Choose a symbol name for the LUT global, handling potential collisions.
  //
  // Start with the canonical name (e.g. "__extf_lut_f8E4M3FN") and look it up:
  //  - Not found: the name is free; exit the loop and create a new global.
  //  - Found a memref.global: a compatible LUT already exists (e.g. the pass
  //    was applied twice, or the user pre-defined the table); reuse it.
  //  - Found a non-memref.global symbol: a symbol of a different op kind
  //    already holds this name (e.g. a function named "__extf_lut_f8E4M3FN").
  //    Retry with a numeric suffix ("__extf_lut_f8E4M3FN_0", "_1", …) until
  //    we find a free slot.
  std::string baseSymName = lutSymbolName(srcType);
  std::string symName = baseSymName;
  for (unsigned suffix = 0;; ++suffix) {
    auto *existing = SymbolTable::lookupSymbolIn(symbolTableOp, symName);
    if (!existing)
      break;
    if (auto globalOp = dyn_cast<memref::GlobalOp>(existing)) {
      cache[srcType] = globalOp;
      return globalOp;
    }
    symName = baseSymName + "_" + std::to_string(suffix);
  }

  OpBuilder builder(symbolTableOp->getContext());
  builder.setInsertionPointToStart(&symbolTableOp->getRegion(0).front());
  auto f32Ty = builder.getF32Type();
  int64_t numEntries = 1LL << srcType.getWidth();
  auto memrefTy = MemRefType::get({numEntries}, f32Ty);
  auto tensorTy = RankedTensorType::get({numEntries}, f32Ty);

  SmallVector<float> values = buildExtFLUT(srcType);
  auto denseAttr = DenseElementsAttr::get(tensorTy, ArrayRef<float>(values));

  auto global = memref::GlobalOp::create(
      builder, symbolTableOp->getLoc(),
      /*sym_name=*/symName,
      /*sym_visibility=*/builder.getStringAttr("private"),
      /*type=*/memrefTy,
      /*initial_value=*/denseAttr,
      /*constant=*/true,
      /*alignment=*/builder.getI64IntegerAttr(64));
  cache[srcType] = global;
  return global;
}

struct ExtFToLUTPattern : public OpRewritePattern<arith::ExtFOp> {
  ExtFToLUTPattern(MLIRContext *ctx,
                   llvm::DenseMap<Type, memref::GlobalOp> &lutCache)
      : OpRewritePattern(ctx), lutCache(lutCache) {}

  LogicalResult matchAndRewrite(arith::ExtFOp op,
                                PatternRewriter &rewriter) const override {
    Type srcTy = op.getIn().getType();
    Type dstTy = op.getType();
    Location loc = op.getLoc();

    // Scalar narrowFP → f32
    if (isSupportedNarrowFloatType(srcTy) && isa<Float32Type>(dstTy)) {
      auto srcFloatTy = cast<FloatType>(srcTy);
      Operation *symbolTableOp = SymbolTable::getNearestSymbolTable(op);
      memref::GlobalOp global =
          getOrCreateLUT(symbolTableOp, srcFloatTy, lutCache);
      auto memrefTy = cast<MemRefType>(global.getType());

      Value tbl = memref::GetGlobalOp::create(rewriter, loc, memrefTy,
                                              global.getSymName());
      Value iNval = arith::BitcastOp::create(
          rewriter, loc, rewriter.getIntegerType(srcFloatTy.getWidth()),
          op.getIn());
      Value idx = arith::IndexCastUIOp::create(rewriter, loc,
                                               rewriter.getIndexType(), iNval);
      Value result = memref::LoadOp::create(rewriter, loc, tbl, idx);
      rewriter.replaceOp(op, result);
      return success();
    }

    // Vector<NxNarrowFP> → vector<NxF32>
    auto srcVecTy = dyn_cast<VectorType>(srcTy);
    auto dstVecTy = dyn_cast<VectorType>(dstTy);
    if (!srcVecTy || !dstVecTy)
      return failure();
    if (!isSupportedNarrowFloatType(srcVecTy.getElementType()) ||
        !isa<Float32Type>(dstVecTy.getElementType()))
      return failure();

    auto srcElemTy = cast<FloatType>(srcVecTy.getElementType());
    Operation *symbolTableOp = SymbolTable::getNearestSymbolTable(op);
    memref::GlobalOp global =
        getOrCreateLUT(symbolTableOp, srcElemTy, lutCache);
    auto memrefTy = cast<MemRefType>(global.getType());

    Value tbl = memref::GetGlobalOp::create(rewriter, loc, memrefTy,
                                            global.getSymName());
    // bitcast vector<NxNarrowFP> -> vector<NxiW>
    unsigned bitWidth = srcElemTy.getWidth();
    auto iNVecTy =
        VectorType::get(srcVecTy.getShape(), rewriter.getIntegerType(bitWidth));
    Value iNvec = arith::BitcastOp::create(rewriter, loc, iNVecTy, op.getIn());

    // extui vector<NxiW> -> vector<Nxi32> (gather offsets)
    auto i32VecTy = VectorType::get(srcVecTy.getShape(), rewriter.getI32Type());
    Value offsets = arith::ExtUIOp::create(rewriter, loc, i32VecTy, iNvec);

    // all-true mask
    auto maskVecTy = VectorType::get(srcVecTy.getShape(), rewriter.getI1Type());
    Value mask = vector::ConstantMaskOp::create(
        rewriter, loc, maskVecTy, vector::ConstantMaskKind::AllTrue);

    // passthru: dense<0.0> vector<Nxf32>
    Value passthru = arith::ConstantOp::create(
        rewriter, loc, DenseElementsAttr::get(dstVecTy, ArrayRef<float>{0.0f}));

    // base index into the LUT
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);

    Value result = vector::GatherOp::create(
        rewriter, loc, dstVecTy, tbl, ValueRange{c0}, offsets, mask, passthru);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  llvm::DenseMap<Type, memref::GlobalOp> &lutCache;
};

namespace {
struct ConvertArithExtFToLUTPass
    : public impl::ConvertArithExtFToLUTBase<ConvertArithExtFToLUTPass> {

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = &getContext();

    // Cache so each format gets exactly one global, inserted once.
    llvm::DenseMap<Type, memref::GlobalOp> lutCache;

    RewritePatternSet patterns(ctx);
    patterns.add<ExtFToLUTPattern>(ctx, lutCache);

    walkAndApplyPatterns(moduleOp, std::move(patterns));
  }
};
} // namespace
