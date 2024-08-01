//===- EnableAMXTileBinding.cpp - Enable tile binding for Intel AMX -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass enables the tile register binding semantic for Intel® Advanced
// Matrix Extensions (Intel® AMX). Intuitively, this pass analyses the tile
// binding hints set by users, legalize the hints and automatically configures
// needed hardware context. The AMX tile register usage in lowered intrinsics
// would strictly respect the given hints, enforced in lowering pass
// `--convert-vector-to-llvm`.
//
// Note that if this pass is not invoked prior to `--convert-vector-to-llvm`,
// the AMX lowering would ignore the binding info and fallback to original
// scheme.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMX/AMXDialect.h"
#include "mlir/Dialect/AMX/Analysis/AMXBindingAnalysis.h"
#include "mlir/Dialect/AMX/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/FormattedStream.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#define DEBUG_TYPE "enable-amx-tile-binding"

namespace mlir {
namespace amx {

#define GEN_PASS_DEF_ENABLEAMXTILEBINDING
#include "mlir/Dialect/AMX/Passes.h.inc"

class TileStoreBindingRewriter : public OpRewritePattern<TileStoreOp> {
private:
  TileBindingAnalysis &analysis;

public:
  using OpRewritePattern<TileStoreOp>::OpRewritePattern;

  TileStoreBindingRewriter(MLIRContext *context, TileBindingAnalysis &ana)
      : OpRewritePattern(context), analysis{ana} {}

  LogicalResult matchAndRewrite(TileStoreOp op,
                                PatternRewriter &rewriter) const final {
    auto val = op.getVal();
    auto srcIndex = analysis.getBinding(val);
    if (srcIndex < 0)
      return failure();
    auto existingAccIndex = op.getSrcRegIndex();
    if (existingAccIndex && *existingAccIndex != srcIndex)
      return failure();

    rewriter.replaceOpWithNewOp<TileStoreOp>(
        op, op.getBase(), op.getIndices(), val,
        rewriter.getI8IntegerAttr(srcIndex));
    return success();
  }
};

class TileMulFBindingRewriter : public OpRewritePattern<TileMulFOp> {
private:
  TileBindingAnalysis &analysis;

public:
  using OpRewritePattern<TileMulFOp>::OpRewritePattern;

  TileMulFBindingRewriter(MLIRContext *context, TileBindingAnalysis &ana)
      : OpRewritePattern(context), analysis{ana} {}

  LogicalResult matchAndRewrite(TileMulFOp op,
                                PatternRewriter &rewriter) const final {
    auto lhsVal = op.getLhs();
    auto rhsVal = op.getRhs();
    auto accVal = op.getAcc();
    auto lhsIndex = analysis.getBinding(lhsVal);
    auto rhsIndex = analysis.getBinding(rhsVal);
    auto accIndex = analysis.getBinding(accVal);
    if (lhsIndex < 0 || rhsIndex < 0 || accIndex < 0)
      return failure();
    auto existingLhsIndex = op.getLhsRegIndex();
    auto existingRhsIndex = op.getRhsRegIndex();
    auto existingAccIndex = op.getAccRegIndex();
    if ((existingLhsIndex && *existingLhsIndex != lhsIndex) ||
        (existingRhsIndex && *existingRhsIndex != rhsIndex) ||
        (existingAccIndex && *existingAccIndex != accIndex))
      return failure();

    rewriter.replaceOpWithNewOp<TileMulFOp>(
        op, op.getRes().getType(), lhsVal, rhsVal, accVal,
        rewriter.getI8IntegerAttr(lhsIndex),
        rewriter.getI8IntegerAttr(rhsIndex),
        rewriter.getI8IntegerAttr(accIndex));
    return success();
  }
};

class TileMulIBindingRewriter : public OpRewritePattern<TileMulIOp> {
private:
  TileBindingAnalysis &analysis;

public:
  using OpRewritePattern<TileMulIOp>::OpRewritePattern;

  TileMulIBindingRewriter(MLIRContext *context, TileBindingAnalysis &ana)
      : OpRewritePattern(context), analysis{ana} {}

  LogicalResult matchAndRewrite(TileMulIOp op,
                                PatternRewriter &rewriter) const final {
    auto lhsVal = op.getLhs();
    auto rhsVal = op.getRhs();
    auto accVal = op.getAcc();
    auto lhsIndex = analysis.getBinding(lhsVal);
    auto rhsIndex = analysis.getBinding(rhsVal);
    auto accIndex = analysis.getBinding(accVal);
    if (lhsIndex < 0 || rhsIndex < 0 || accIndex < 0)
      return failure();
    auto existingLhsIndex = op.getLhsRegIndex();
    auto existingRhsIndex = op.getRhsRegIndex();
    auto existingAccIndex = op.getAccRegIndex();
    if ((existingLhsIndex && *existingLhsIndex != lhsIndex) ||
        (existingRhsIndex && *existingRhsIndex != rhsIndex) ||
        (existingAccIndex && *existingAccIndex != accIndex))
      return failure();

    rewriter.replaceOpWithNewOp<TileMulIOp>(
        op, op.getRes().getType(), lhsVal, rhsVal, accVal, op.getIsZextLhs(),
        op.getIsZextRhs(), rewriter.getI8IntegerAttr(lhsIndex),
        rewriter.getI8IntegerAttr(rhsIndex),
        rewriter.getI8IntegerAttr(accIndex));
    return success();
  }
};

static inline void uint8ArrayToHex(std::string &out, uint8_t array[],
                                   int size) {
  llvm::raw_string_ostream os(out);
  for (int index = 0; index < size; index++) {
    os << llvm::format_hex_no_prefix(array[index], 2, true);
  }
}

struct EnableAMXTileBindingPass
    : public impl::EnableAMXTileBindingBase<EnableAMXTileBindingPass> {
private:
  LLVM::GlobalOp
  getOrCreateGlobalPalette(const TileScopeAnalysis::PaletteInfo &pi) {
    assert(!pi.overflow && "Expecting valid palette");
// Pack struct so it can fit into a single 64-byte cache line.
#pragma pack(push, 1)
    struct {
      uint8_t paletteId;
      uint8_t startRow;
      uint8_t reserved[14];
      uint16_t cols[16];
      uint8_t rows[16];
    } paletteConfig;
#pragma pack(pop)

    size_t paletteArraySize = 64;
    auto paletteAsArray = reinterpret_cast<uint8_t *>(&paletteConfig);
    memset(paletteAsArray, 0x0, paletteArraySize);
    // Intel AMX: The only legal non-INIT value for palette_id is 1.
    // TODO(haixin): fetch from CPUID ?
    paletteConfig.paletteId = 1;
    for (int index = 0; index < 8; index++) {
      const auto &regShape = pi.palette[index];
      paletteConfig.rows[index] = regShape.first;
      paletteConfig.cols[index] = regShape.second;
    }

    std::string paletteSymName = "g_intel_amx_palette_";
    uint8ArrayToHex(paletteSymName, paletteAsArray, paletteArraySize);

    ModuleOp module = getOperation()->template getParentOfType<ModuleOp>();
    if (auto global = module.lookupSymbol<LLVM::GlobalOp>(paletteSymName))
      return global;
    // Create a global symbol containing palette config.
    auto ctx = module->getContext();
    OpBuilder builder(module);
    builder.setInsertionPointToStart(module.getBody());

    SmallVector<uint8_t> elementVals;
    for (size_t index = 0; index < paletteArraySize; index++)
      elementVals.push_back(paletteAsArray[index]);
    auto dataAttrType = RankedTensorType::get(
        {static_cast<int64_t>(elementVals.size())}, builder.getI8Type());
    auto dataAttr =
        DenseElementsAttr::get(dataAttrType, llvm::ArrayRef(elementVals));
    auto arrayType =
        LLVM::LLVMArrayType::get(IntegerType::get(ctx, 8), elementVals.size());
    auto global = builder.create<LLVM::GlobalOp>(
        module.getLoc(), arrayType, /*isConstant*/ true, LLVM::Linkage::Private,
        paletteSymName, dataAttr, /*alignment=*/64);
    return global;
  }

public:
  void runOnOperation() override {
    // 0. Get AnalyseInfo for each concerned Value (Does not allow mixed used of
    // tmul & normal vector operations).
    TileBindingAnalysis &bindingAna = getAnalysis<TileBindingAnalysis>();
    if (!bindingAna.isValid())
      return;

    LLVM_DEBUG(llvm::dbgs() << ">>> After binding analysis\n");
    // 1. Set propagated binding info to AMX Ops.
    RewritePatternSet patterns(&getContext());
    patterns.add<TileStoreBindingRewriter>(&getContext(), bindingAna);
    patterns.add<TileMulFBindingRewriter>(&getContext(), bindingAna);
    patterns.add<TileMulIBindingRewriter>(&getContext(), bindingAna);
    FrozenRewritePatternSet patternSet(std::move(patterns));

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), patternSet, config)))
      return;

    LLVM_DEBUG(llvm::dbgs() << ">>> After propagating binding info\n");

    // 2. Analyse tile scopes & expand them maximally.
    TileScopeAnalysis &scopeAna = getAnalysis<TileScopeAnalysis>();
    if (!scopeAna.isValid())
      return;

    LLVM_DEBUG(llvm::dbgs() << ">>> After tile scope analysis\n");

    // 3. Insert tile config/release according to tile scopes.
    OpBuilder builder(getOperation());
    for (auto &scope : scopeAna.getTileScopes()) {
      LLVM_DEBUG(llvm::dbgs() << ">>> Processing tile scope: "
                              << *scope.seg.begin() << "\n");
      assert(!scope.pi.overflow && "Expecting legal AMX palette info");
      auto paletteGlobal = getOrCreateGlobalPalette(scope.pi);
      assert(paletteGlobal && "Failed to create global palette");

      Operation *begin = &(*scope.seg.begin());
      Location loc = begin->getLoc();

      builder.setInsertionPoint(begin);
      Value paletteGlobalPtr =
          builder.create<LLVM::AddressOfOp>(loc, paletteGlobal);
      builder.create<amx::x86_amx_ldtilecfg_plain>(loc, paletteGlobalPtr);

      Operation *end = &(*scope.seg.end());
      loc = end->getLoc();
      builder.setInsertionPointAfter(end);
      builder.create<amx::x86_amx_tilerelease_plain>(loc);
    }
    markAnalysesPreserved<TileScopeAnalysis>();
  }
};

} // namespace amx
} // namespace mlir
