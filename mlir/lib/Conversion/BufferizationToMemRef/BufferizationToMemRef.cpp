//===- BufferizationToMemRef.cpp - Bufferization to MemRef conversion -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert Bufferization dialect to MemRef
// dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTBUFFERIZATIONTOMEMREF
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
/// The CloneOpConversion transforms all bufferization clone operations into
/// memref alloc and memref copy operations. In the dynamic-shape case, it also
/// emits additional dim and constant operations to determine the shape. This
/// conversion does not resolve memory leaks if it is used alone.
struct CloneOpConversion : public OpConversionPattern<bufferization::CloneOp> {
  using OpConversionPattern<bufferization::CloneOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(bufferization::CloneOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check for unranked memref types which are currently not supported.
    Type type = op.getType();
    if (isa<UnrankedMemRefType>(type)) {
      return rewriter.notifyMatchFailure(
          op, "UnrankedMemRefType is not supported.");
    }
    MemRefType memrefType = cast<MemRefType>(type);
    MemRefLayoutAttrInterface layout;
    auto allocType =
        MemRefType::get(memrefType.getShape(), memrefType.getElementType(),
                        layout, memrefType.getMemorySpace());
    // Since this implementation always allocates, certain result types of the
    // clone op cannot be lowered.
    if (!memref::CastOp::areCastCompatible({allocType}, {memrefType}))
      return failure();

    // Transform a clone operation into alloc + copy operation and pay
    // attention to the shape dimensions.
    Location loc = op->getLoc();
    SmallVector<Value, 4> dynamicOperands;
    for (int i = 0; i < memrefType.getRank(); ++i) {
      if (!memrefType.isDynamicDim(i))
        continue;
      Value dim = rewriter.createOrFold<memref::DimOp>(loc, op.getInput(), i);
      dynamicOperands.push_back(dim);
    }

    // Allocate a memref with identity layout.
    Value alloc = rewriter.create<memref::AllocOp>(op->getLoc(), allocType,
                                                   dynamicOperands);
    // Cast the allocation to the specified type if needed.
    if (memrefType != allocType)
      alloc = rewriter.create<memref::CastOp>(op->getLoc(), memrefType, alloc);
    rewriter.replaceOp(op, alloc);
    rewriter.create<memref::CopyOp>(loc, op.getInput(), alloc);
    return success();
  }
};

/// The DeallocOpConversion transforms all bufferization dealloc operations into
/// memref dealloc operations potentially guarded by scf if operations.
/// Additionally, memref extract_aligned_pointer_as_index and arith operations
/// are inserted to compute the guard conditions. We distinguish multiple cases
/// to provide an overall more efficient lowering. In the general case, a helper
/// func is created to avoid quadratic code size explosion (relative to the
/// number of operands of the dealloc operation). For examples of each case,
/// refer to the documentation of the member functions of this class.
class DeallocOpConversion
    : public OpConversionPattern<bufferization::DeallocOp> {

  /// Lower a simple case avoiding the helper function. Ideally, static analysis
  /// can provide enough aliasing information to split the dealloc operations up
  /// into this simple case as much as possible before running this pass.
  ///
  /// Example:
  /// ```
  /// %0 = bufferization.dealloc (%arg0 : memref<2xf32>) if (%arg1)
  /// ```
  /// is lowered to
  /// ```
  /// scf.if %arg1 {
  ///   memref.dealloc %arg0 : memref<2xf32>
  /// }
  /// %0 = arith.constant false
  /// ```
  LogicalResult
  rewriteOneMemrefNoRetainCase(bufferization::DeallocOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
    rewriter.create<scf::IfOp>(op.getLoc(), adaptor.getConditions()[0],
                               [&](OpBuilder &builder, Location loc) {
                                 builder.create<memref::DeallocOp>(
                                     loc, adaptor.getMemrefs()[0]);
                                 builder.create<scf::YieldOp>(loc);
                               });
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op,
                                                   rewriter.getBoolAttr(false));
    return success();
  }

  /// Lowering that supports all features the dealloc operation has to offer. It
  /// computes the base pointer of each memref (as an index), stores them in a
  /// new memref and passes it to the helper function generated in
  /// 'buildDeallocationHelperFunction'. The two return values are used as
  /// condition for the scf if operation containing the memref deallocate and as
  /// replacement for the original bufferization dealloc respectively.
  ///
  /// Example:
  /// ```
  /// %0:2 = bufferization.dealloc (%arg0, %arg1 : memref<2xf32>, memref<5xf32>)
  ///                           if (%arg3, %arg4) retain (%arg2 : memref<1xf32>)
  /// ```
  /// lowers to (simplified):
  /// ```
  /// %c0 = arith.constant 0 : index
  /// %c1 = arith.constant 1 : index
  /// %alloc = memref.alloc() : memref<2xindex>
  /// %alloc_0 = memref.alloc() : memref<1xindex>
  /// %intptr = memref.extract_aligned_pointer_as_index %arg0
  /// memref.store %intptr, %alloc[%c0] : memref<2xindex>
  /// %intptr_1 = memref.extract_aligned_pointer_as_index %arg1
  /// memref.store %intptr_1, %alloc[%c1] : memref<2xindex>
  /// %intptr_2 = memref.extract_aligned_pointer_as_index %arg2
  /// memref.store %intptr_2, %alloc_0[%c0] : memref<1xindex>
  /// %cast = memref.cast %alloc : memref<2xindex> to memref<?xindex>
  /// %cast_4 = memref.cast %alloc_0 : memref<1xindex> to memref<?xindex>
  /// %0:2 = call @dealloc_helper(%cast, %cast_4, %c0)
  /// %1 = arith.andi %0#0, %arg3 : i1
  /// %2 = arith.andi %0#1, %arg3 : i1
  /// scf.if %1 {
  ///   memref.dealloc %arg0 : memref<2xf32>
  /// }
  /// %3:2 = call @dealloc_helper(%cast, %cast_4, %c1)
  /// %4 = arith.andi %3#0, %arg4 : i1
  /// %5 = arith.andi %3#1, %arg4 : i1
  /// scf.if %4 {
  ///   memref.dealloc %arg1 : memref<5xf32>
  /// }
  /// memref.dealloc %alloc : memref<2xindex>
  /// memref.dealloc %alloc_0 : memref<1xindex>
  /// // replace %0#0 with %2
  /// // replace %0#1 with %5
  /// ```
  LogicalResult rewriteGeneralCase(bufferization::DeallocOp op,
                                   OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
    // Allocate two memrefs holding the base pointer indices of the list of
    // memrefs to be deallocated and the ones to be retained. These can then be
    // passed to the helper function and the for-loops can iterate over them.
    // Without storing them to memrefs, we could not use for-loops but only a
    // completely unrolled version of it, potentially leading to code-size
    // blow-up.
    Value toDeallocMemref = rewriter.create<memref::AllocOp>(
        op.getLoc(), MemRefType::get({(int64_t)adaptor.getMemrefs().size()},
                                     rewriter.getIndexType()));
    Value toRetainMemref = rewriter.create<memref::AllocOp>(
        op.getLoc(), MemRefType::get({(int64_t)adaptor.getRetained().size()},
                                     rewriter.getIndexType()));

    auto getConstValue = [&](uint64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(op.getLoc(),
                                                rewriter.getIndexAttr(value));
    };

    // Extract the base pointers of the memrefs as indices to check for aliasing
    // at runtime.
    for (auto [i, toDealloc] : llvm::enumerate(adaptor.getMemrefs())) {
      Value memrefAsIdx =
          rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(op.getLoc(),
                                                                  toDealloc);
      rewriter.create<memref::StoreOp>(op.getLoc(), memrefAsIdx,
                                       toDeallocMemref, getConstValue(i));
    }
    for (auto [i, toRetain] : llvm::enumerate(adaptor.getRetained())) {
      Value memrefAsIdx =
          rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(op.getLoc(),
                                                                  toRetain);
      rewriter.create<memref::StoreOp>(op.getLoc(), memrefAsIdx, toRetainMemref,
                                       getConstValue(i));
    }

    // Cast the allocated memrefs to dynamic shape because we want only one
    // helper function no matter how many operands the bufferization.dealloc
    // has.
    Value castedDeallocMemref = rewriter.create<memref::CastOp>(
        op->getLoc(),
        MemRefType::get({ShapedType::kDynamic}, rewriter.getIndexType()),
        toDeallocMemref);
    Value castedRetainMemref = rewriter.create<memref::CastOp>(
        op->getLoc(),
        MemRefType::get({ShapedType::kDynamic}, rewriter.getIndexType()),
        toRetainMemref);

    SmallVector<Value> replacements;
    for (unsigned i = 0, e = adaptor.getMemrefs().size(); i < e; ++i) {
      auto callOp = rewriter.create<func::CallOp>(
          op.getLoc(), deallocHelperFunc,
          SmallVector<Value>{castedDeallocMemref, castedRetainMemref,
                             getConstValue(i)});
      Value shouldDealloc = rewriter.create<arith::AndIOp>(
          op.getLoc(), callOp.getResult(0), adaptor.getConditions()[i]);
      Value ownership = rewriter.create<arith::AndIOp>(
          op.getLoc(), callOp.getResult(1), adaptor.getConditions()[i]);
      replacements.push_back(ownership);
      rewriter.create<scf::IfOp>(
          op.getLoc(), shouldDealloc, [&](OpBuilder &builder, Location loc) {
            builder.create<memref::DeallocOp>(loc, adaptor.getMemrefs()[i]);
            builder.create<scf::YieldOp>(loc);
          });
    }

    // Deallocate above allocated memrefs again to avoid memory leaks.
    // Deallocation will not be run on code after this stage.
    rewriter.create<memref::DeallocOp>(op.getLoc(), toDeallocMemref);
    rewriter.create<memref::DeallocOp>(op.getLoc(), toRetainMemref);

    rewriter.replaceOp(op, replacements);
    return success();
  }

public:
  DeallocOpConversion(MLIRContext *context, func::FuncOp deallocHelperFunc)
      : OpConversionPattern<bufferization::DeallocOp>(context),
        deallocHelperFunc(deallocHelperFunc) {}

  LogicalResult
  matchAndRewrite(bufferization::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Lower the trivial case.
    if (adaptor.getMemrefs().empty())
      return rewriter.eraseOp(op), success();

    if (adaptor.getMemrefs().size() == 1 && adaptor.getRetained().empty())
      return rewriteOneMemrefNoRetainCase(op, adaptor, rewriter);

    return rewriteGeneralCase(op, adaptor, rewriter);
  }

  /// Build a helper function per compilation unit that can be called at
  /// bufferization dealloc sites to determine aliasing and ownership.
  ///
  /// The generated function takes two memrefs of indices and one index value as
  /// arguments and returns two boolean values:
  ///   * The first memref argument A should contain the result of the
  ///   extract_aligned_pointer_as_index operation applied to the memrefs to be
  ///   deallocated
  ///   * The second memref argument B should contain the result of the
  ///   extract_aligned_pointer_as_index operation applied to the memrefs to be
  ///   retained
  ///   * The index argument I represents the currently processed index of
  ///   memref A and is needed because aliasing with all previously deallocated
  ///   memrefs has to be checked to avoid double deallocation
  ///   * The first result indicates whether the memref at position I should be
  ///   deallocated
  ///   * The second result provides the updated ownership value corresponding
  ///   the the memref at position I
  ///
  /// This helper function is supposed to be called for each element in the list
  /// of memrefs to be deallocated to determine the deallocation need and new
  /// ownership indicator, but does not perform the deallocation itself.
  ///
  /// The first scf for loop in the body computes whether the memref at index I
  /// aliases with any memref in the list of retained memrefs.
  /// The second loop additionally checks whether one of the previously
  /// deallocated memrefs aliases with the currently processed one.
  ///
  /// Generated code:
  /// ```
  /// func.func @dealloc_helper(%arg0: memref<?xindex>,
  ///                           %arg1: memref<?xindex>,
  ///                           %arg2: index) -> (i1, i1) {
  ///   %c0 = arith.constant 0 : index
  ///   %c1 = arith.constant 1 : index
  ///   %true = arith.constant true
  ///   %dim = memref.dim %arg1, %c0 : memref<?xindex>
  ///   %0 = memref.load %arg0[%arg2] : memref<?xindex>
  ///   %1 = scf.for %i = %c0 to %dim step %c1 iter_args(%arg4 = %true) -> (i1){
  ///     %4 = memref.load %arg1[%i] : memref<?xindex>
  ///     %5 = arith.cmpi ne, %4, %0 : index
  ///     %6 = arith.andi %arg4, %5 : i1
  ///     scf.yield %6 : i1
  ///   }
  ///   %2 = scf.for %i = %c0 to %arg2 step %c1 iter_args(%arg4 = %1) -> (i1) {
  ///     %4 = memref.load %arg0[%i] : memref<?xindex>
  ///     %5 = arith.cmpi ne, %4, %0 : index
  ///     %6 = arith.andi %arg4, %5 : i1
  ///     scf.yield %6 : i1
  ///   }
  ///   %3 = arith.xori %1, %true : i1
  ///   return %2, %3 : i1, i1
  /// }
  /// ```
  static func::FuncOp
  buildDeallocationHelperFunction(OpBuilder &builder, Location loc,
                                  SymbolTable &symbolTable) {
    Type idxType = builder.getIndexType();
    Type memrefArgType = MemRefType::get({ShapedType::kDynamic}, idxType);
    SmallVector<Type> argTypes{memrefArgType, memrefArgType, idxType};
    builder.clearInsertionPoint();

    // Generate the func operation itself.
    auto helperFuncOp = func::FuncOp::create(
        loc, "dealloc_helper",
        builder.getFunctionType(argTypes,
                                {builder.getI1Type(), builder.getI1Type()}));
    symbolTable.insert(helperFuncOp);
    auto &block = helperFuncOp.getFunctionBody().emplaceBlock();
    block.addArguments(argTypes, SmallVector<Location>(argTypes.size(), loc));

    builder.setInsertionPointToStart(&block);
    Value toDeallocMemref = helperFuncOp.getArguments()[0];
    Value toRetainMemref = helperFuncOp.getArguments()[1];
    Value idxArg = helperFuncOp.getArguments()[2];

    // Insert some prerequisites.
    Value c0 = builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(0));
    Value c1 = builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(1));
    Value trueValue =
        builder.create<arith::ConstantOp>(loc, builder.getBoolAttr(true));
    Value toRetainSize = builder.create<memref::DimOp>(loc, toRetainMemref, c0);
    Value toDealloc =
        builder.create<memref::LoadOp>(loc, toDeallocMemref, idxArg);

    // Build the first for loop that computes aliasing with retained memrefs.
    Value noRetainAlias =
        builder
            .create<scf::ForOp>(
                loc, c0, toRetainSize, c1, trueValue,
                [&](OpBuilder &builder, Location loc, Value i,
                    ValueRange iterArgs) {
                  Value retainValue =
                      builder.create<memref::LoadOp>(loc, toRetainMemref, i);
                  Value doesntAlias = builder.create<arith::CmpIOp>(
                      loc, arith::CmpIPredicate::ne, retainValue, toDealloc);
                  Value yieldValue = builder.create<arith::AndIOp>(
                      loc, iterArgs[0], doesntAlias);
                  builder.create<scf::YieldOp>(loc, yieldValue);
                })
            .getResult(0);

    // Build the second for loop that adds aliasing with previously deallocated
    // memrefs.
    Value noAlias =
        builder
            .create<scf::ForOp>(
                loc, c0, idxArg, c1, noRetainAlias,
                [&](OpBuilder &builder, Location loc, Value i,
                    ValueRange iterArgs) {
                  Value prevDeallocValue =
                      builder.create<memref::LoadOp>(loc, toDeallocMemref, i);
                  Value doesntAlias = builder.create<arith::CmpIOp>(
                      loc, arith::CmpIPredicate::ne, prevDeallocValue,
                      toDealloc);
                  Value yieldValue = builder.create<arith::AndIOp>(
                      loc, iterArgs[0], doesntAlias);
                  builder.create<scf::YieldOp>(loc, yieldValue);
                })
            .getResult(0);

    Value ownership =
        builder.create<arith::XOrIOp>(loc, noRetainAlias, trueValue);
    builder.create<func::ReturnOp>(loc, SmallVector<Value>{noAlias, ownership});

    return helperFuncOp;
  }

private:
  func::FuncOp deallocHelperFunc;
};
} // namespace

namespace {
struct BufferizationToMemRefPass
    : public impl::ConvertBufferizationToMemRefBase<BufferizationToMemRefPass> {
  BufferizationToMemRefPass() = default;

  void runOnOperation() override {
    ModuleOp module = cast<ModuleOp>(getOperation());
    OpBuilder builder =
        OpBuilder::atBlockBegin(&module.getBodyRegion().front());
    SymbolTable symbolTable(module);

    // Build dealloc helper function if there are deallocs.
    func::FuncOp helperFuncOp;
    getOperation()->walk([&](bufferization::DeallocOp deallocOp) {
      if (deallocOp.getMemrefs().size() > 1 ||
          !deallocOp.getRetained().empty()) {
        helperFuncOp = DeallocOpConversion::buildDeallocationHelperFunction(
            builder, getOperation()->getLoc(), symbolTable);
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<CloneOpConversion>(patterns.getContext());
    patterns.add<DeallocOpConversion>(patterns.getContext(), helperFuncOp);

    ConversionTarget target(getContext());
    target.addLegalDialect<memref::MemRefDialect, arith::ArithDialect,
                           scf::SCFDialect, func::FuncDialect>();
    target.addIllegalDialect<bufferization::BufferizationDialect>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createBufferizationToMemRefPass() {
  return std::make_unique<BufferizationToMemRefPass>();
}
