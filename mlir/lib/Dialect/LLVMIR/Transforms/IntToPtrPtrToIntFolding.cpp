//===- IntToPtrPtrToIntFolding.cpp - IntToPtr/PtrToInt folding ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that folds inttoptr/ptrtoint operation sequences.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/Transforms/IntToPtrPtrToIntFolding.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "fold-llvm-inttoptr-ptrtoint"

namespace mlir {
namespace LLVM {

#define GEN_PASS_DEF_FOLDINTTOPTRPTRTOINTPASS
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h.inc"

} // namespace LLVM
} // namespace mlir

using namespace mlir;

namespace {

/// Return the bitwidth of a pointer or integer type. If the type is a pointer,
/// return the bitwidth of the address space from `addrSpaceBWs`, if available.
/// Return failure if the address space bitwidth is not available.
static FailureOr<unsigned> getIntOrPtrBW(Type type,
                                         ArrayRef<unsigned> addrSpaceBWs) {
  if (auto ptrType = dyn_cast<LLVM::LLVMPointerType>(type)) {
    unsigned addrSpace = ptrType.getAddressSpace();
    if (addrSpace < addrSpaceBWs.size() && addrSpaceBWs[addrSpace] != 0)
      return addrSpaceBWs[addrSpace];
    return failure();
  }

  auto integerType = cast<IntegerType>(type);
  return integerType.getWidth();
}

/// Check if folding inttoptr/ptrtoint is valid. Check that the original type
/// matches the result type of the end-to-end conversion and that the input
/// value is not truncated along the conversion chain.
static LogicalResult canFoldIntToPtrPtrToInt(Type originalType,
                                             Type intermediateType,
                                             Type resultType,
                                             ArrayRef<unsigned> addrSpaceBWs) {
  // Check if the original type matches the result type.
  // TODO: Support address space conversions?
  // TODO: Support int trunc/ext?
  if (originalType != resultType)
    return failure();

  // Make sure there is no data truncation with respect to the original type at
  // any point during the conversion. Truncating the intermediate data is fine
  // as long as the original data is not truncated.
  auto originalBW = getIntOrPtrBW(originalType, addrSpaceBWs);
  if (failed(originalBW))
    return failure();

  auto intermediateBW = getIntOrPtrBW(intermediateType, addrSpaceBWs);
  if (failed(intermediateBW))
    return failure();

  if (*originalBW > *intermediateBW)
    return failure();
  return success();
}

/// Folds inttoptr(ptrtoint(x)) -> x or ptrtoint(inttoptr(x)) -> x.
template <typename SrcConvOp, typename DstConvOp>
class FoldIntToPtrPtrToInt : public OpRewritePattern<DstConvOp> {
public:
  FoldIntToPtrPtrToInt(MLIRContext *context, ArrayRef<unsigned> addrSpaceBWs)
      : OpRewritePattern<DstConvOp>(context), addrSpaceBWs(addrSpaceBWs) {}

  LogicalResult matchAndRewrite(DstConvOp dstConvOp,
                                PatternRewriter &rewriter) const override {
    auto srcConvOp = dstConvOp.getArg().template getDefiningOp<SrcConvOp>();
    if (!srcConvOp)
      return failure();

    // Check if folding is valid based on type matching and bitwidth
    // information.
    if (failed(canFoldIntToPtrPtrToInt(srcConvOp.getArg().getType(),
                                       srcConvOp.getType(), dstConvOp.getType(),
                                       addrSpaceBWs))) {
      return failure();
    }

    rewriter.replaceOp(dstConvOp, srcConvOp.getArg());
    return success();
  }

private:
  SmallVector<unsigned> addrSpaceBWs;
};

/// Pass that folds inttoptr/ptrtoint operation sequences.
struct FoldIntToPtrPtrToIntPass
    : public LLVM::impl::FoldIntToPtrPtrToIntPassBase<
          FoldIntToPtrPtrToIntPass> {
  using Base =
      LLVM::impl::FoldIntToPtrPtrToIntPassBase<FoldIntToPtrPtrToIntPass>;
  using Base::FoldIntToPtrPtrToIntPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    LLVM::populateIntToPtrPtrToIntFoldingPatterns(patterns, addrSpaceBWs);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::LLVM::populateIntToPtrPtrToIntFoldingPatterns(
    RewritePatternSet &patterns, ArrayRef<unsigned> addrSpaceBWs) {
  patterns.add<FoldIntToPtrPtrToInt<LLVM::PtrToIntOp, LLVM::IntToPtrOp>,
               FoldIntToPtrPtrToInt<LLVM::IntToPtrOp, LLVM::PtrToIntOp>>(
      patterns.getContext(), addrSpaceBWs);
}
