//==============================================================================
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==============================================================================
//
// Pass to lower the ripple dialect to the llvm dialect.
//
//==============================================================================

#include "mlir/Conversion/RippleToLLVM/RippleToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Ripple/Ripple.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::ripple;

namespace mlir {
#define GEN_PASS_DEF_RIPPLETOLLVM
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
class SetShapeOpLowering : public ConversionPattern {
public:
  explicit SetShapeOpLowering(MLIRContext *context)
      : ConversionPattern(ripple::SetShapeOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto *context = rewriter.getContext();
    auto loc = op->getLoc();

    size_t numOperands =
        operands.size(); // realistically, assert this is \in [2, 11]
    if ((numOperands < 2) || (numOperands > 11))
      return failure();

    size_t correction = 11 - numOperands;
    SmallVector<Value, 11> padding;
    llvm::append_range(padding, operands);

    auto shapeIType = operands[1].getType();

    auto zeroOp = rewriter.create<arith::ConstantOp>(
        loc, shapeIType, rewriter.getZeroAttr(shapeIType));

    for (size_t i = 0; i < correction; i++)
      padding.push_back(zeroOp.getResult());

    auto callIntrOpRef = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, cast<mlir::Type>(LLVM::LLVMPointerType::get(context)),
        mlir::StringAttr::get(context, "llvm.ripple.block.setshape"),
        ValueRange(padding));

    rewriter.replaceOp(op, callIntrOpRef);
    return success();
  }
};

class GetSizeOpLowering : public ConversionPattern {
public:
  explicit GetSizeOpLowering(MLIRContext *context)
      : ConversionPattern(ripple::GetSizeOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto *context = rewriter.getContext();
    auto loc = op->getLoc();

    if (operands.size() != 2)
      return failure();

    auto dimOperand = operands[1];
    auto retType = dimOperand.getType();

    auto callIntrOpRef = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, retType,
        mlir::StringAttr::get(context, "llvm.ripple.block.getsize"),
        ValueRange(operands));

    rewriter.replaceOp(op, callIntrOpRef);
    return success();
  }
};

class IndexOpLowering : public ConversionPattern {
public:
  explicit IndexOpLowering(MLIRContext *context)
      : ConversionPattern(ripple::IndexOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto *context = rewriter.getContext();
    auto loc = op->getLoc();

    if (operands.size() != 2)
      return failure();

    auto dimOperand = operands[1];
    auto retType = dimOperand.getType();

    auto callIntrOpRef = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, retType, mlir::StringAttr::get(context, "llvm.ripple.block.index"),
        ValueRange(operands));

    rewriter.replaceOp(op, callIntrOpRef);
    return success();
  }
};

class BroadcastOpLowering : public ConversionPattern {
public:
  explicit BroadcastOpLowering(MLIRContext *context)
      : ConversionPattern(ripple::BroadcastOp::getOperationName(), 1, context) {
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto *context = rewriter.getContext();
    auto loc = op->getLoc();

    if (operands.size() != 3)
      return failure();

    auto retType = operands[2].getType();

    auto callIntrOpRef = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, retType, mlir::StringAttr::get(context, "llvm.ripple.broadcast"),
        ValueRange(operands));

    rewriter.replaceOp(op, callIntrOpRef);
    return success();
  }
};

class BroadcastPtrOpLowering : public ConversionPattern {
public:
  explicit BroadcastPtrOpLowering(MLIRContext *context)
      : ConversionPattern(ripple::BroadcastPtrOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto *context = rewriter.getContext();
    auto loc = op->getLoc();

    if (operands.size() != 3)
      return failure();

    auto retType = operands[2].getType();

    auto callIntrOpRef = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, retType, mlir::StringAttr::get(context, "llvm.ripple.broadcast"),
        ValueRange(operands));

    rewriter.replaceOp(op, callIntrOpRef);
    return success();
  }
};

class SliceOpLowering : public ConversionPattern {
public:
  explicit SliceOpLowering(MLIRContext *context)
      : ConversionPattern(ripple::SliceOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto *context = rewriter.getContext();
    auto loc = op->getLoc();

    size_t numOperands =
        operands.size(); // realistically, assert this is \in [2, 11]
    if ((numOperands < 2) || (numOperands > 11))
      return failure();

    size_t correction = 11 - numOperands;
    SmallVector<Value, 11> padding;
    llvm::append_range(padding, operands);

    auto zeroOp = rewriter.create<arith::ConstantOp>(
        loc, IntegerType::get(context, 64), rewriter.getI64IntegerAttr(0));

    for (size_t i = 0; i < correction; i++)
      padding.push_back(zeroOp.getResult());

    auto retType = operands[0].getType();

    auto callIntrOpRef = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, retType, mlir::StringAttr::get(context, "llvm.ripple.slice"),
        ValueRange(padding));

    rewriter.replaceOp(op, callIntrOpRef);
    return success();
  }
};

class ShuffleOpLowering : public ConversionPattern {
public:
  explicit ShuffleOpLowering(MLIRContext *context)
      : ConversionPattern(ripple::ShuffleOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto *context = rewriter.getContext();
    auto loc = op->getLoc();

    if (operands.size() != 2)
      return failure();

    auto shuffleType = operands[0].getType();

    auto zeroOp = rewriter.create<arith::ConstantOp>(
        loc, IntegerType::get(context, 1),
        rewriter.getZeroAttr(IntegerType::get(context, 1)));

    auto blankOp = rewriter.create<arith::ConstantOp>(
        loc, shuffleType, rewriter.getZeroAttr(shuffleType));

    SmallVector<Value, 4> newOperands;
    newOperands.push_back(operands[0]);
    newOperands.push_back(blankOp.getResult());
    newOperands.push_back(zeroOp.getResult());
    newOperands.push_back(operands[1]);

    auto callIntrOpRef = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, shuffleType, mlir::StringAttr::get(context, "llvm.ripple.shuffle"),
        ValueRange(newOperands));

    rewriter.replaceOp(op, callIntrOpRef);

    return success();
  }
};

class ShufflePairOpLowering : public ConversionPattern {
public:
  explicit ShufflePairOpLowering(MLIRContext *context)
      : ConversionPattern(ripple::ShufflePairOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto *context = rewriter.getContext();
    auto loc = op->getLoc();

    if (operands.size() != 3)
      return failure();

    auto shuffleType = operands[0].getType();

    auto oneOp = rewriter.create<arith::ConstantOp>(
        loc, IntegerType::get(context, 1),
        rewriter.getOneAttr(IntegerType::get(context, 1)));

    SmallVector<Value, 4> newOperands;
    newOperands.push_back(operands[0]);
    newOperands.push_back(operands[1]);
    newOperands.push_back(oneOp.getResult());
    newOperands.push_back(operands[2]);

    if (shuffleType.isInteger()) {
      auto callIntrOpRef = rewriter.create<LLVM::CallIntrinsicOp>(
          loc, shuffleType,
          mlir::StringAttr::get(context, "llvm.ripple.ishuffle"),
          ValueRange(newOperands));

      rewriter.replaceOp(op, callIntrOpRef);
    } else if (shuffleType.isFloat()) {
      auto callIntrOpRef = rewriter.create<LLVM::CallIntrinsicOp>(
          loc, shuffleType,
          mlir::StringAttr::get(context, "llvm.ripple.fshuffle"),
          ValueRange(newOperands));

      rewriter.replaceOp(op, callIntrOpRef);
    } else {
      return failure();
    }

    return success();
  }
};

class ReduceAddOpLowering : public ConversionPattern {
public:
  explicit ReduceAddOpLowering(MLIRContext *context)
    : ConversionPattern(ripple::ReduceAddOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto *context = rewriter.getContext();
    auto loc = op->getLoc();

    if (operands.size() != 2)
      return failure();

    auto retType = operands[1].getType();

    if (retType.isInteger()) {
      auto callIntrOpRef = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, retType, mlir::StringAttr::get(context, "llvm.ripple.reduce.add"),
        ValueRange(operands));

      rewriter.replaceOp(op, callIntrOpRef);
    } else if (retType.isFloat()) {
      auto callIntrOpRef = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, retType, mlir::StringAttr::get(context, "llvm.ripple.reduce.fadd"),
        ValueRange(operands));

      rewriter.replaceOp(op, callIntrOpRef);
    } else {
      return failure();
    }

    return success();
  }
};

class ReduceMulOpLowering : public ConversionPattern {
public:
  explicit ReduceMulOpLowering(MLIRContext *context)
    : ConversionPattern(ripple::ReduceMulOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto *context = rewriter.getContext();
    auto loc = op->getLoc();

    if (operands.size() != 2)
      return failure();

    auto retType = operands[1].getType();

    if (retType.isInteger()) {
      auto callIntrOpRef = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, retType, mlir::StringAttr::get(context, "llvm.ripple.reduce.mul"),
        ValueRange(operands));

      rewriter.replaceOp(op, callIntrOpRef);
    } else if (retType.isFloat()) {
      auto callIntrOpRef = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, retType, mlir::StringAttr::get(context, "llvm.ripple.reduce.fmul"),
        ValueRange(operands));

      rewriter.replaceOp(op, callIntrOpRef);
    } else {
      return failure();
    }

    return success();
  }
};

class ReduceAndOpLowering : public ConversionPattern {
public:
  explicit ReduceAndOpLowering(MLIRContext *context)
    : ConversionPattern(ripple::ReduceAndOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto *context = rewriter.getContext();
    auto loc = op->getLoc();

    if (operands.size() != 2)
      return failure();

    auto retType = operands[1].getType();

    auto callIntrOpRef = rewriter.create<LLVM::CallIntrinsicOp>(
      loc, retType, mlir::StringAttr::get(context, "llvm.ripple.reduce.and"),
      ValueRange(operands));

    rewriter.replaceOp(op, callIntrOpRef);
  }
};

class ReduceOrOpLowering : public ConversionPattern {
public:
  explicit ReduceOrOpLowering(MLIRContext *context)
    : ConversionPattern(ripple::ReduceAndOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto *context = rewriter.getContext();
    auto loc = op->getLoc();

    if (operands.size() != 2)
      return failure();

    auto retType = operands[1].getType();

    auto callIntrOpRef = rewriter.create<LLVM::CallIntrinsicOp>(
      loc, retType, mlir::StringAttr::get(context, "llvm.ripple.reduce.or"),
      ValueRange(operands));

    rewriter.replaceOp(op, callIntrOpRef);
  }
};

class ReduceXorOpLowering : public ConversionPattern {
public:
  explicit ReduceXorOpLowering(MLIRContext *context)
    : ConversionPattern(ripple::ReduceAndOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto *context = rewriter.getContext();
    auto loc = op->getLoc();

    if (operands.size() != 2)
      return failure();

    auto retType = operands[1].getType();

    auto callIntrOpRef = rewriter.create<LLVM::CallIntrinsicOp>(
      loc, retType, mlir::StringAttr::get(context, "llvm.ripple.reduce.xor"),
      ValueRange(operands));

    rewriter.replaceOp(op, callIntrOpRef);
  }
};

class ReduceMaxOpLowering : public ConversionPattern {
public:
  explicit ReduceMaxOpLowering(MLIRContext *context)
    : ConversionPattern(ripple::ReduceMaxOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto *context = rewriter.getContext();
    auto loc = op->getLoc();

    if (operands.size() != 2)
      return failure();

    auto retType = operands[1].getType();

    if (retType.isInteger()) {
      if (retType.isSignedInteger()){
        auto callIntrOpRef = rewriter.create<LLVM::CallIntrinsicOp>(
          loc, retType, mlir::StringAttr::get(context, "llvm.ripple.reduce.smax"),
          ValueRange(operands));

        rewriter.replaceOp(op, callIntrOpRef);
      } else {
        // NOTE: This makes the assumption that a *signless* integer is also
        // semantically an unsigned integer.
        auto callIntrOpRef = rewriter.create<LLVM::CallIntrinsicOp>(
          loc, retType, mlir::StringAttr::get(context, "llvm.ripple.reduce.umax"),
          ValueRange(operands));

        rewriter.replaceOp(op, callIntrOpRef);
      }
    } else if (retType.isFloat()) {
      auto callIntrOpRef = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, retType, mlir::StringAttr::get(context, "llvm.ripple.reduce.fmax"),
        ValueRange(operands));

      rewriter.replaceOp(op, callIntrOpRef);
    } else {
      return failure();
    }

    return success();
  }
};

class ReduceMinOpLowering : public ConversionPattern {
public:
  explicit ReduceMinOpLowering(MLIRContext *context)
    : ConversionPattern(ripple::ReduceMinOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto *context = rewriter.getContext();
    auto loc = op->getLoc();

    if (operands.size() != 2)
      return failure();

    auto retType = operands[1].getType();

    if (retType.isInteger()) {
      if (retType.isSignedInteger()){
        auto callIntrOpRef = rewriter.create<LLVM::CallIntrinsicOp>(
          loc, retType, mlir::StringAttr::get(context, "llvm.ripple.reduce.smin"),
          ValueRange(operands));

        rewriter.replaceOp(op, callIntrOpRef);
      } else {
        // NOTE: This makes the assumption that a *signless* integer is also
        // semantically an unsigned integer.
        auto callIntrOpRef = rewriter.create<LLVM::CallIntrinsicOp>(
          loc, retType, mlir::StringAttr::get(context, "llvm.ripple.reduce.umin"),
          ValueRange(operands));

        rewriter.replaceOp(op, callIntrOpRef);
      }
    } else if (retType.isFloat()) {
      auto callIntrOpRef = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, retType, mlir::StringAttr::get(context, "llvm.ripple.reduce.fmin"),
        ValueRange(operands));

      rewriter.replaceOp(op, callIntrOpRef);
    } else {
      return failure();
    }

    return success();
  }
};
} // namespace

namespace {
struct RippleToLLVMPass : public impl::RippleToLLVMBase<RippleToLLVMPass> {
  using Base::Base;

  void runOnOperation() override;
};
} // namespace

void RippleToLLVMPass::runOnOperation() {
  LLVMConversionTarget target(getContext());
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addLegalDialect<mlir::arith::ArithDialect>();
  target.addLegalDialect<mlir::ptr::PtrDialect>();
  target.addLegalDialect<mlir::index::IndexDialect>();
  target.addLegalDialect<mlir::memref::MemRefDialect>();
  target.addLegalOp<ModuleOp>();

  LLVMTypeConverter typeConverter(&getContext());

  RewritePatternSet patterns(&getContext());
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  patterns.insert<SetShapeOpLowering>(&getContext());
  patterns.insert<GetSizeOpLowering>(&getContext());
  patterns.insert<IndexOpLowering>(&getContext());
  patterns.insert<BroadcastOpLowering>(&getContext());
  patterns.insert<BroadcastPtrOpLowering>(&getContext());
  patterns.insert<SliceOpLowering>(&getContext());
  patterns.insert<ShuffleOpLowering>(&getContext());
  patterns.insert<ShufflePairOpLowering>(&getContext());

  patterns.insert<ReduceAddOpLowering>(&getContext());
  patterns.insert<ReduceMulOpLowering>(&getContext());
  patterns.insert<ReduceAndOpLowering>(&getContext());
  patterns.insert<ReduceOrOpLowering>(&getContext());
  patterns.insert<ReduceXorOpLowering>(&getContext());
  patterns.insert<ReduceMaxOpLowering>(&getContext());
  patterns.insert<ReduceMinOpLowering>(&getContext());

  auto *module = getOperation();
  if (failed(applyFullConversion(module, target,
                                 std::move(patterns) /*, &typeConverter */)))
    signalPassFailure();
}

std::unique_ptr<OperationPass<>> mlir::createRippleToLLVMPass() {
  return std::make_unique<RippleToLLVMPass>();
}
