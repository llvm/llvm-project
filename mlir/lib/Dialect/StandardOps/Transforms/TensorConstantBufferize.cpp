//===- Bufferize.cpp - Bufferization for std ops --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements bufferization of tensor-valued arith.constant ops.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/BufferUtils.h"
#include "mlir/Transforms/Bufferize.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

memref::GlobalOp GlobalCreator::getGlobalFor(arith::ConstantOp constantOp) {
  auto type = constantOp.getType().cast<RankedTensorType>();

  BufferizeTypeConverter typeConverter;

  // If we already have a global for this constant value, no need to do
  // anything else.
  auto it = globals.find(constantOp.value());
  if (it != globals.end())
    return cast<memref::GlobalOp>(it->second);

  // Create a builder without an insertion point. We will insert using the
  // symbol table to guarantee unique names.
  OpBuilder globalBuilder(moduleOp.getContext());
  SymbolTable symbolTable(moduleOp);

  // Create a pretty name.
  SmallString<64> buf;
  llvm::raw_svector_ostream os(buf);
  interleave(type.getShape(), os, "x");
  os << "x" << type.getElementType();

  // Add an optional alignment to the global memref.
  IntegerAttr memrefAlignment =
      alignment > 0 ? IntegerAttr::get(globalBuilder.getI64Type(), alignment)
                    : IntegerAttr();

  auto global = globalBuilder.create<memref::GlobalOp>(
      constantOp.getLoc(), (Twine("__constant_") + os.str()).str(),
      /*sym_visibility=*/globalBuilder.getStringAttr("private"),
      /*type=*/typeConverter.convertType(type).cast<MemRefType>(),
      /*initial_value=*/constantOp.value().cast<ElementsAttr>(),
      /*constant=*/true,
      /*alignment=*/memrefAlignment);
  symbolTable.insert(global);
  // The symbol table inserts at the end of the module, but globals are a bit
  // nicer if they are at the beginning.
  global->moveBefore(&moduleOp.front());
  globals[constantOp.value()] = global;
  return global;
}

namespace {
class BufferizeTensorConstantOp
    : public OpConversionPattern<arith::ConstantOp> {
public:
  BufferizeTensorConstantOp(GlobalCreator &globals,
                            TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<arith::ConstantOp>(typeConverter, context,
                                               /*benefit=*/1),
        globals(globals) {}

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = op.getType().dyn_cast<RankedTensorType>();
    if (!type)
      return failure();

    auto globalMemref = globals.getGlobalFor(op);
    rewriter.replaceOpWithNewOp<memref::GetGlobalOp>(op, globalMemref.type(),
                                                     globalMemref.getName());
    return success();
  }
  GlobalCreator &globals;
};
} // namespace

void mlir::populateTensorConstantBufferizePatterns(
    GlobalCreator &globalCreator, BufferizeTypeConverter &typeConverter,
    RewritePatternSet &patterns) {
  patterns.add<BufferizeTensorConstantOp>(globalCreator, typeConverter,
                                          patterns.getContext());
}

namespace {
class TensorConstantBufferizePass
    : public TensorConstantBufferizeBase<TensorConstantBufferizePass> {
public:
  explicit TensorConstantBufferizePass(unsigned alignment) {
    if (alignment)
      this->alignment = alignment;
  }

  void runOnOperation() override {
    auto module = getOperation();
    GlobalCreator globals(module, alignment);

    auto *context = &getContext();
    BufferizeTypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    target.addLegalDialect<memref::MemRefDialect>();
    populateTensorConstantBufferizePatterns(globals, typeConverter, patterns);
    target.addDynamicallyLegalOp<arith::ConstantOp>([&](arith::ConstantOp op) {
      return typeConverter.isLegal(op.getType());
    });
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass>
mlir::createTensorConstantBufferizePass(unsigned alignment) {
  return std::make_unique<TensorConstantBufferizePass>(alignment);
}
