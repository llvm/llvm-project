//===- Bufferize.cpp - Bufferization for std ops --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements bufferization of tensor-valued std.constant ops.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/Bufferize.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
// This class creates global ops for all tensor-valued constants in the program.
// It creates them with pretty names and makes sure that duplicate globals
// aren't created.
class GlobalCreator {
public:
  explicit GlobalCreator(ModuleOp module);
  memref::GlobalOp getGlobalFor(Attribute attr) {
    assert(globals.find(attr) != globals.end() && "unknown constant attr");
    return globals[attr];
  }

private:
  DenseMap<Attribute, memref::GlobalOp> globals;
};

GlobalCreator::GlobalCreator(ModuleOp module) {
  BufferizeTypeConverter typeConverter;
  // Create a builder without an insertion point. We will insert using the
  // symbol table to guarantee unique names.
  OpBuilder globalBuilder(module.getContext());
  SymbolTable symbolTable(module);
  module.walk([&](ConstantOp op) {
    // We only want tensor constants for now.
    auto type = op.getType().dyn_cast<RankedTensorType>();
    if (!type)
      return;
    // If we already have a global for this constant value, no need to do
    // anything else.
    auto it = globals.find(op.getValue());
    if (it != globals.end())
      return;

    // Create a pretty name.
    SmallString<64> buf;
    llvm::raw_svector_ostream os(buf);
    interleave(type.getShape(), os, "x");
    os << "x" << type.getElementType();

    auto global = globalBuilder.create<memref::GlobalOp>(
        op.getLoc(), (Twine("__constant_") + os.str()).str(),
        /*sym_visibility=*/globalBuilder.getStringAttr("private"),
        /*type=*/typeConverter.convertType(type),
        /*initial_value=*/op.getValue().cast<ElementsAttr>(),
        /*constant=*/true);
    symbolTable.insert(global);
    // The symbol table inserts at the end of the module, but globals are a bit
    // nicer if they are at the beginning.
    global->moveBefore(&module.front());
    globals[op.getValue()] = global;
  });
}
} // namespace

namespace {
class BufferizeTensorConstantOp : public OpConversionPattern<ConstantOp> {
public:
  BufferizeTensorConstantOp(GlobalCreator &globals,
                            TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<ConstantOp>(typeConverter, context, /*benefit=*/1),
        globals(globals) {}

  LogicalResult
  matchAndRewrite(ConstantOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = op.getType().dyn_cast<RankedTensorType>();
    if (!type)
      return failure();

    auto globalMemref = globals.getGlobalFor(op.value());
    rewriter.replaceOpWithNewOp<memref::GetGlobalOp>(op, globalMemref.type(),
                                                     globalMemref.getName());
    return success();
  }
  GlobalCreator &globals;
};
} // namespace

namespace {
struct TensorConstantBufferizePass
    : public TensorConstantBufferizeBase<TensorConstantBufferizePass> {
  void runOnOperation() override {
    auto module = getOperation();
    GlobalCreator globals(module);

    auto *context = &getContext();
    BufferizeTypeConverter typeConverter;
    OwningRewritePatternList patterns;
    ConversionTarget target(*context);

    target.addLegalDialect<memref::MemRefDialect>();
    patterns.insert<BufferizeTensorConstantOp>(globals, typeConverter, context);
    target.addDynamicallyLegalOp<ConstantOp>(
        [&](ConstantOp op) { return typeConverter.isLegal(op.getType()); });
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createTensorConstantBufferizePass() {
  return std::make_unique<TensorConstantBufferizePass>();
}
