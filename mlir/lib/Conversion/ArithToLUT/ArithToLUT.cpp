//===- ArithToLUT.cpp - Replace arith.extf f8→f32 with LUT lookup ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// For each arith.extf %v : f8X to f32, this pass emits a 256-entry f32 global
// constant (one per distinct f8 format) and replaces the op with LUT:
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
//     %tbl   = memref.get_global @__extf_lut_f8E4M3FN  : memref<256xf32>
//     %i8    = arith.bitcast  %v   : f8X   -> i8
//     %ui32  = arith.extui    %i8  : i8    -> i32
//     %idx   = arith.index_cast %ui32 : i32 -> index
//     %res   = memref.load    %tbl[%idx]  : memref<256xf32>
// ```
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToLUT/ArithToLUT.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTARITHFP8EXTFTOLUT
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Returns true for the f8 float types that need LUT-based extf lowering.
static bool isSupportedF8Type(Type t) {
  return isa<Float8E4M3FNType, Float8E5M2Type, Float8E4M3FNUZType,
             Float8E5M2FNUZType, Float8E4M3B11FNUZType, Float8E3M4Type,
             Float8E4M3Type>(t);
}

// Returns a stable, symbol-safe name for the global LUT of the given f8 type.

// Returns a name for the global LUT by appending the MLIR textual
// // representation of the given f8 type to a fixed prefix.
static std::string lutSymbolName(FloatType srcType) {
  std::string name = "__extf_lut_";
  llvm::raw_string_ostream os(name);
  srcType.print(os);
  return name;
}

// Precomputes 256 f32 values by enumerating every 8-bit pattern for srcType.
static SmallVector<float, 256> buildExtFLUT(FloatType srcType) {
  const llvm::fltSemantics &sem = srcType.getFloatSemantics();
  SmallVector<float, 256> table;
  table.reserve(256);
  for (unsigned i = 0; i < 256; ++i) {
    APFloat val(sem, APInt(8, i));
    bool losesInfo = false;
    val.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven,
                &losesInfo);
    table.push_back(val.convertToFloat());
  }
  return table;
}

// Inserts (or returns existing) memref.global constant for the given f8 type.
static memref::GlobalOp
getOrCreateLUT(ModuleOp module, FloatType srcType,
               llvm::DenseMap<Type, memref::GlobalOp> &cache) {
  auto it = cache.find(srcType);
  if (it != cache.end())
    return it->second;

  std::string symName = lutSymbolName(srcType);
  if (auto existing = module.lookupSymbol<memref::GlobalOp>(symName))
    return existing;

  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  auto f32Ty = builder.getF32Type();
  auto memrefTy = MemRefType::get({256}, f32Ty);
  auto tensorTy = RankedTensorType::get({256}, f32Ty);

  SmallVector<float, 256> values = buildExtFLUT(srcType);
  auto denseAttr = DenseElementsAttr::get(tensorTy, ArrayRef<float>(values));

  auto global = memref::GlobalOp::create(
      builder, module.getLoc(),
      /*sym_name=*/symName,
      /*sym_visibility=*/builder.getStringAttr("private"),
      /*type=*/memrefTy,
      /*initial_value=*/denseAttr,
      /*constant=*/true,
      /*alignment=*/builder.getI64IntegerAttr(64));
  cache[srcType] = global;
  return global;
}

//===----------------------------------------------------------------------===//
// Rewrite pattern
//===----------------------------------------------------------------------===//

struct ExtFToLUTPattern : public OpRewritePattern<arith::ExtFOp> {
  ExtFToLUTPattern(MLIRContext *ctx,
                   llvm::DenseMap<Type, memref::GlobalOp> &lutCache)
      : OpRewritePattern(ctx), lutCache(lutCache) {}

  LogicalResult matchAndRewrite(arith::ExtFOp op,
                                PatternRewriter &rewriter) const override {
    Type srcTy = op.getIn().getType();
    Type dstTy = op.getType();

    if (!isSupportedF8Type(srcTy) || !isa<Float32Type>(dstTy))
      return failure();

    auto srcFloatTy = cast<FloatType>(srcTy);
    auto module = op->getParentOfType<ModuleOp>();
    memref::GlobalOp global = getOrCreateLUT(module, srcFloatTy, lutCache);

    Location loc = op.getLoc();
    auto memrefTy = cast<MemRefType>(global.getType());

    // %tbl = memref.get_global @__extf_lut_<fmt>
    Value tbl = memref::GetGlobalOp::create(rewriter, loc, memrefTy,
                                            global.getSymName());
    // %i8 = arith.bitcast %in : f8X -> i8
    Value i8val = arith::BitcastOp::create(rewriter, loc,
                                           rewriter.getIntegerType(8),
                                           op.getIn());

    // %ui32 = arith.extui %i8 : i8 -> i32
    Value ui32val =
        arith::ExtUIOp::create(rewriter, loc, rewriter.getI32Type(), i8val);

    // %idx = arith.index_cast %ui32 : i32 -> index
    Value idx = arith::IndexCastOp::create(rewriter, loc,
                                           rewriter.getIndexType(), ui32val);

    // %res = memref.load %tbl[%idx]
    Value result = memref::LoadOp::create(rewriter, loc, tbl, idx);

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  llvm::DenseMap<Type, memref::GlobalOp> &lutCache;
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertArithFP8ExtFToLUTPass
    : public impl::ConvertArithFP8ExtFToLUTBase<ConvertArithFP8ExtFToLUTPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // Cache so each format gets exactly one global, inserted once.
    llvm::DenseMap<Type, memref::GlobalOp> lutCache;

    RewritePatternSet patterns(ctx);
    patterns.add<ExtFToLUTPattern>(ctx, lutCache);

    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
