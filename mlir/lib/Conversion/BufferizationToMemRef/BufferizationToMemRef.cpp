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

  /// Lower a simple case without any retained values and a single memref to
  /// avoiding the helper function. Ideally, static analysis can provide enough
  /// aliasing information to split the dealloc operations up into this simple
  /// case as much as possible before running this pass.
  ///
  /// Example:
  /// ```
  /// bufferization.dealloc (%arg0 : memref<2xf32>) if (%arg1)
  /// ```
  /// is lowered to
  /// ```
  /// scf.if %arg1 {
  ///   memref.dealloc %arg0 : memref<2xf32>
  /// }
  /// ```
  LogicalResult
  rewriteOneMemrefNoRetainCase(bufferization::DeallocOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
    assert(adaptor.getMemrefs().size() == 1 && "expected only one memref");
    assert(adaptor.getRetained().empty() && "expected no retained memrefs");

    rewriter.replaceOpWithNewOp<scf::IfOp>(
        op, adaptor.getConditions()[0], [&](OpBuilder &builder, Location loc) {
          builder.create<memref::DeallocOp>(loc, adaptor.getMemrefs()[0]);
          builder.create<scf::YieldOp>(loc);
        });
    return success();
  }

  /// A special case lowering for the deallocation operation with exactly one
  /// memref, but arbitrary number of retained values. This avoids the helper
  /// function that the general case needs and thus also avoids storing indices
  /// to specifically allocated memrefs. The size of the code produced by this
  /// lowering is linear to the number of retained values.
  ///
  /// Example:
  /// ```mlir
  /// %0:2 = bufferization.dealloc (%m : memref<2xf32>) if (%cond)
  //                        retain (%r0, %r1 : memref<1xf32>, memref<2xf32>)
  /// return %0#0, %0#1 : i1, i1
  /// ```
  /// ```mlir
  /// %m_base_pointer = memref.extract_aligned_pointer_as_index %m
  /// %r0_base_pointer = memref.extract_aligned_pointer_as_index %r0
  /// %r0_does_not_alias = arith.cmpi ne, %m_base_pointer, %r0_base_pointer
  /// %r1_base_pointer = memref.extract_aligned_pointer_as_index %r1
  /// %r1_does_not_alias = arith.cmpi ne, %m_base_pointer, %r1_base_pointer
  /// %not_retained = arith.andi %r0_does_not_alias, %r1_does_not_alias : i1
  /// %should_dealloc = arith.andi %not_retained, %cond : i1
  /// scf.if %should_dealloc {
  ///   memref.dealloc %m : memref<2xf32>
  /// }
  /// %true = arith.constant true
  /// %r0_does_alias = arith.xori %r0_does_not_alias, %true : i1
  /// %r0_ownership = arith.andi %r0_does_alias, %cond : i1
  /// %r1_does_alias = arith.xori %r1_does_not_alias, %true : i1
  /// %r1_ownership = arith.andi %r1_does_alias, %cond : i1
  /// return %r0_ownership, %r1_ownership : i1, i1
  /// ```
  LogicalResult rewriteOneMemrefMultipleRetainCase(
      bufferization::DeallocOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const {
    assert(adaptor.getMemrefs().size() == 1 && "expected only one memref");

    // Compute the base pointer indices, compare all retained indices to the
    // memref index to check if they alias.
    SmallVector<Value> doesNotAliasList;
    Value memrefAsIdx = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        op->getLoc(), adaptor.getMemrefs()[0]);
    for (Value retained : adaptor.getRetained()) {
      Value retainedAsIdx =
          rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(op->getLoc(),
                                                                  retained);
      Value doesNotAlias = rewriter.create<arith::CmpIOp>(
          op->getLoc(), arith::CmpIPredicate::ne, memrefAsIdx, retainedAsIdx);
      doesNotAliasList.push_back(doesNotAlias);
    }

    // AND-reduce the list of booleans from above.
    Value prev = doesNotAliasList.front();
    for (Value doesNotAlias : ArrayRef(doesNotAliasList).drop_front())
      prev = rewriter.create<arith::AndIOp>(op->getLoc(), prev, doesNotAlias);

    // Also consider the condition given by the dealloc operation and perform a
    // conditional deallocation guarded by that value.
    Value shouldDealloc = rewriter.create<arith::AndIOp>(
        op->getLoc(), prev, adaptor.getConditions()[0]);

    rewriter.create<scf::IfOp>(
        op.getLoc(), shouldDealloc, [&](OpBuilder &builder, Location loc) {
          builder.create<memref::DeallocOp>(loc, adaptor.getMemrefs()[0]);
          builder.create<scf::YieldOp>(loc);
        });

    // Compute the replacement values for the dealloc operation results. This
    // inserts an already canonicalized form of
    // `select(does_alias_with_memref(r), memref_cond, false)` for each retained
    // value r.
    SmallVector<Value> replacements;
    Value trueVal = rewriter.create<arith::ConstantOp>(
        op->getLoc(), rewriter.getBoolAttr(true));
    for (Value doesNotAlias : doesNotAliasList) {
      Value aliases =
          rewriter.create<arith::XOrIOp>(op->getLoc(), doesNotAlias, trueVal);
      Value result = rewriter.create<arith::AndIOp>(op->getLoc(), aliases,
                                                    adaptor.getConditions()[0]);
      replacements.push_back(result);
    }

    rewriter.replaceOp(op, replacements);

    return success();
  }

  /// Lowering that supports all features the dealloc operation has to offer. It
  /// computes the base pointer of each memref (as an index), stores it in a
  /// new memref helper structure and passes it to the helper function generated
  /// in 'buildDeallocationHelperFunction'. The results are stored in two lists
  /// (represented as memrefs) of booleans passed as arguments. The first list
  /// stores whether the corresponding condition should be deallocated, the
  /// second list stores the ownership of the retained values which can be used
  /// to replace the result values of the `bufferization.dealloc` operation.
  ///
  /// Example:
  /// ```
  /// %0:2 = bufferization.dealloc (%m0, %m1 : memref<2xf32>, memref<5xf32>)
  ///                           if (%cond0, %cond1)
  ///                       retain (%r0, %r1 : memref<1xf32>, memref<2xf32>)
  /// ```
  /// lowers to (simplified):
  /// ```
  /// %c0 = arith.constant 0 : index
  /// %c1 = arith.constant 1 : index
  /// %dealloc_base_pointer_list = memref.alloc() : memref<2xindex>
  /// %cond_list = memref.alloc() : memref<2xi1>
  /// %retain_base_pointer_list = memref.alloc() : memref<2xindex>
  /// %m0_base_pointer = memref.extract_aligned_pointer_as_index %m0
  /// memref.store %m0_base_pointer, %dealloc_base_pointer_list[%c0]
  /// %m1_base_pointer = memref.extract_aligned_pointer_as_index %m1
  /// memref.store %m1_base_pointer, %dealloc_base_pointer_list[%c1]
  /// memref.store %cond0, %cond_list[%c0]
  /// memref.store %cond1, %cond_list[%c1]
  /// %r0_base_pointer = memref.extract_aligned_pointer_as_index %r0
  /// memref.store %r0_base_pointer, %retain_base_pointer_list[%c0]
  /// %r1_base_pointer = memref.extract_aligned_pointer_as_index %r1
  /// memref.store %r1_base_pointer, %retain_base_pointer_list[%c1]
  /// %dyn_dealloc_base_pointer_list = memref.cast %dealloc_base_pointer_list :
  ///    memref<2xindex> to memref<?xindex>
  /// %dyn_cond_list = memref.cast %cond_list : memref<2xi1> to memref<?xi1>
  /// %dyn_retain_base_pointer_list = memref.cast %retain_base_pointer_list :
  ///    memref<2xindex> to memref<?xindex>
  /// %dealloc_cond_out = memref.alloc() : memref<2xi1>
  /// %ownership_out = memref.alloc() : memref<2xi1>
  /// %dyn_dealloc_cond_out = memref.cast %dealloc_cond_out :
  ///    memref<2xi1> to memref<?xi1>
  /// %dyn_ownership_out = memref.cast %ownership_out :
  ///    memref<2xi1> to memref<?xi1>
  /// call @dealloc_helper(%dyn_dealloc_base_pointer_list,
  ///                      %dyn_retain_base_pointer_list,
  ///                      %dyn_cond_list,
  ///                      %dyn_dealloc_cond_out,
  ///                      %dyn_ownership_out) : (...)
  /// %m0_dealloc_cond = memref.load %dyn_dealloc_cond_out[%c0] : memref<2xi1>
  /// scf.if %m0_dealloc_cond {
  ///   memref.dealloc %m0 : memref<2xf32>
  /// }
  /// %m1_dealloc_cond = memref.load %dyn_dealloc_cond_out[%c1] : memref<2xi1>
  /// scf.if %m1_dealloc_cond {
  ///   memref.dealloc %m1 : memref<5xf32>
  /// }
  /// %r0_ownership = memref.load %dyn_ownership_out[%c0] : memref<2xi1>
  /// %r1_ownership = memref.load %dyn_ownership_out[%c1] : memref<2xi1>
  /// memref.dealloc %dealloc_base_pointer_list : memref<2xindex>
  /// memref.dealloc %retain_base_pointer_list : memref<2xindex>
  /// memref.dealloc %cond_list : memref<2xi1>
  /// memref.dealloc %dealloc_cond_out : memref<2xi1>
  /// memref.dealloc %ownership_out : memref<2xi1>
  /// // replace %0#0 with %r0_ownership
  /// // replace %0#1 with %r1_ownership
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
    Value conditionMemref = rewriter.create<memref::AllocOp>(
        op.getLoc(), MemRefType::get({(int64_t)adaptor.getConditions().size()},
                                     rewriter.getI1Type()));
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

    for (auto [i, cond] : llvm::enumerate(adaptor.getConditions()))
      rewriter.create<memref::StoreOp>(op.getLoc(), cond, conditionMemref,
                                       getConstValue(i));

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
    Value castedCondsMemref = rewriter.create<memref::CastOp>(
        op->getLoc(),
        MemRefType::get({ShapedType::kDynamic}, rewriter.getI1Type()),
        conditionMemref);
    Value castedRetainMemref = rewriter.create<memref::CastOp>(
        op->getLoc(),
        MemRefType::get({ShapedType::kDynamic}, rewriter.getIndexType()),
        toRetainMemref);

    Value deallocCondsMemref = rewriter.create<memref::AllocOp>(
        op.getLoc(), MemRefType::get({(int64_t)adaptor.getMemrefs().size()},
                                     rewriter.getI1Type()));
    Value retainCondsMemref = rewriter.create<memref::AllocOp>(
        op.getLoc(), MemRefType::get({(int64_t)adaptor.getRetained().size()},
                                     rewriter.getI1Type()));

    Value castedDeallocCondsMemref = rewriter.create<memref::CastOp>(
        op->getLoc(),
        MemRefType::get({ShapedType::kDynamic}, rewriter.getI1Type()),
        deallocCondsMemref);
    Value castedRetainCondsMemref = rewriter.create<memref::CastOp>(
        op->getLoc(),
        MemRefType::get({ShapedType::kDynamic}, rewriter.getI1Type()),
        retainCondsMemref);

    rewriter.create<func::CallOp>(
        op.getLoc(), deallocHelperFunc,
        SmallVector<Value>{castedDeallocMemref, castedRetainMemref,
                           castedCondsMemref, castedDeallocCondsMemref,
                           castedRetainCondsMemref});

    for (unsigned i = 0, e = adaptor.getMemrefs().size(); i < e; ++i) {
      Value idxValue = getConstValue(i);
      Value shouldDealloc = rewriter.create<memref::LoadOp>(
          op.getLoc(), deallocCondsMemref, idxValue);
      rewriter.create<scf::IfOp>(
          op.getLoc(), shouldDealloc, [&](OpBuilder &builder, Location loc) {
            builder.create<memref::DeallocOp>(loc, adaptor.getMemrefs()[i]);
            builder.create<scf::YieldOp>(loc);
          });
    }

    SmallVector<Value> replacements;
    for (unsigned i = 0, e = adaptor.getRetained().size(); i < e; ++i) {
      Value idxValue = getConstValue(i);
      Value ownership = rewriter.create<memref::LoadOp>(
          op.getLoc(), retainCondsMemref, idxValue);
      replacements.push_back(ownership);
    }

    // Deallocate above allocated memrefs again to avoid memory leaks.
    // Deallocation will not be run on code after this stage.
    rewriter.create<memref::DeallocOp>(op.getLoc(), toDeallocMemref);
    rewriter.create<memref::DeallocOp>(op.getLoc(), toRetainMemref);
    rewriter.create<memref::DeallocOp>(op.getLoc(), conditionMemref);
    rewriter.create<memref::DeallocOp>(op.getLoc(), deallocCondsMemref);
    rewriter.create<memref::DeallocOp>(op.getLoc(), retainCondsMemref);

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
    if (adaptor.getMemrefs().empty()) {
      Value falseVal = rewriter.create<arith::ConstantOp>(
          op.getLoc(), rewriter.getBoolAttr(false));
      rewriter.replaceOp(
          op, SmallVector<Value>(adaptor.getRetained().size(), falseVal));
      return success();
    }

    if (adaptor.getMemrefs().size() == 1 && adaptor.getRetained().empty())
      return rewriteOneMemrefNoRetainCase(op, adaptor, rewriter);

    if (adaptor.getMemrefs().size() == 1)
      return rewriteOneMemrefMultipleRetainCase(op, adaptor, rewriter);

    if (!deallocHelperFunc)
      return op->emitError(
          "library function required for generic lowering, but cannot be "
          "automatically inserted when operating on functions");

    return rewriteGeneralCase(op, adaptor, rewriter);
  }

  /// Build a helper function per compilation unit that can be called at
  /// bufferization dealloc sites to determine aliasing and ownership.
  ///
  /// The generated function takes two memrefs of indices and three memrefs of
  /// booleans as arguments:
  ///   * The first argument A should contain the result of the
  ///   extract_aligned_pointer_as_index operation applied to the memrefs to be
  ///   deallocated
  ///   * The second argument B should contain the result of the
  ///   extract_aligned_pointer_as_index operation applied to the memrefs to be
  ///   retained
  ///   * The third argument C should contain the conditions as passed directly
  ///   to the deallocation operation.
  ///   * The fourth argument D is used to pass results to the caller. Those
  ///   represent the condition under which the memref at the corresponding
  ///   position in A should be deallocated.
  ///   * The fifth argument E is used to pass results to the caller. It
  ///   provides the ownership value corresponding the the memref at the same
  ///   position in B
  ///
  /// This helper function is supposed to be called once for each
  /// `bufferization.dealloc` operation to determine the deallocation need and
  /// new ownership indicator for the retained values, but does not perform the
  /// deallocation itself.
  ///
  /// Generated code:
  /// ```
  /// func.func @dealloc_helper(
  ///     %dyn_dealloc_base_pointer_list: memref<?xindex>,
  ///     %dyn_retain_base_pointer_list: memref<?xindex>,
  ///     %dyn_cond_list: memref<?xi1>,
  ///     %dyn_dealloc_cond_out: memref<?xi1>,
  ///     %dyn_ownership_out: memref<?xi1>) {
  ///   %c0 = arith.constant 0 : index
  ///   %c1 = arith.constant 1 : index
  ///   %true = arith.constant true
  ///   %false = arith.constant false
  ///   %num_dealloc_memrefs = memref.dim %dyn_dealloc_base_pointer_list, %c0
  ///   %num_retain_memrefs = memref.dim %dyn_retain_base_pointer_list, %c0
  ///   // Zero initialize result buffer.
  ///   scf.for %i = %c0 to %num_retain_memrefs step %c1 {
  ///     memref.store %false, %dyn_ownership_out[%i] : memref<?xi1>
  ///   }
  ///   scf.for %i = %c0 to %num_dealloc_memrefs step %c1 {
  ///     %dealloc_bp = memref.load %dyn_dealloc_base_pointer_list[%i]
  ///     %cond = memref.load %dyn_cond_list[%i]
  ///     // Check for aliasing with retained memrefs.
  ///     %does_not_alias_retained = scf.for %j = %c0 to %num_retain_memrefs
  ///         step %c1 iter_args(%does_not_alias_aggregated = %true) -> (i1) {
  ///       %retain_bp = memref.load %dyn_retain_base_pointer_list[%j]
  ///       %does_alias = arith.cmpi eq, %retain_bp, %dealloc_bp : index
  ///       scf.if %does_alias {
  ///         %curr_ownership = memref.load %dyn_ownership_out[%j]
  ///         %updated_ownership = arith.ori %curr_ownership, %cond : i1
  ///         memref.store %updated_ownership, %dyn_ownership_out[%j]
  ///       }
  ///       %does_not_alias = arith.cmpi ne, %retain_bp, %dealloc_bp : index
  ///       %updated_aggregate = arith.andi %does_not_alias_aggregated,
  ///                                       %does_not_alias : i1
  ///       scf.yield %updated_aggregate : i1
  ///     }
  ///     // Check for aliasing with dealloc memrefs in the list before the
  ///     // current one, i.e.,
  ///     // `fix i, forall j < i: check_aliasing(%dyn_dealloc_base_pointer[j],
  ///     // %dyn_dealloc_base_pointer[i])`
  ///     %does_not_alias_any = scf.for %j = %c0 to %i step %c1
  ///        iter_args(%does_not_alias_agg = %does_not_alias_retained) -> (i1) {
  ///       %prev_dealloc_bp = memref.load %dyn_dealloc_base_pointer_list[%j]
  ///       %does_not_alias = arith.cmpi ne, %prev_dealloc_bp, %dealloc_bp
  ///       %updated_alias_agg = arith.andi %does_not_alias_agg, %does_not_alias
  ///       scf.yield %updated_alias_agg : i1
  ///     }
  ///     %dealloc_cond = arith.andi %does_not_alias_any, %cond : i1
  ///     memref.store %dealloc_cond, %dyn_dealloc_cond_out[%i] : memref<?xi1>
  ///   }
  ///   return
  /// }
  /// ```
  static func::FuncOp
  buildDeallocationHelperFunction(OpBuilder &builder, Location loc,
                                  SymbolTable &symbolTable) {
    Type indexMemrefType =
        MemRefType::get({ShapedType::kDynamic}, builder.getIndexType());
    Type boolMemrefType =
        MemRefType::get({ShapedType::kDynamic}, builder.getI1Type());
    SmallVector<Type> argTypes{indexMemrefType, indexMemrefType, boolMemrefType,
                               boolMemrefType, boolMemrefType};
    builder.clearInsertionPoint();

    // Generate the func operation itself.
    auto helperFuncOp = func::FuncOp::create(
        loc, "dealloc_helper", builder.getFunctionType(argTypes, {}));
    symbolTable.insert(helperFuncOp);
    auto &block = helperFuncOp.getFunctionBody().emplaceBlock();
    block.addArguments(argTypes, SmallVector<Location>(argTypes.size(), loc));

    builder.setInsertionPointToStart(&block);
    Value toDeallocMemref = helperFuncOp.getArguments()[0];
    Value toRetainMemref = helperFuncOp.getArguments()[1];
    Value conditionMemref = helperFuncOp.getArguments()[2];
    Value deallocCondsMemref = helperFuncOp.getArguments()[3];
    Value retainCondsMemref = helperFuncOp.getArguments()[4];

    // Insert some prerequisites.
    Value c0 = builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(0));
    Value c1 = builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(1));
    Value trueValue =
        builder.create<arith::ConstantOp>(loc, builder.getBoolAttr(true));
    Value falseValue =
        builder.create<arith::ConstantOp>(loc, builder.getBoolAttr(false));
    Value toDeallocSize =
        builder.create<memref::DimOp>(loc, toDeallocMemref, c0);
    Value toRetainSize = builder.create<memref::DimOp>(loc, toRetainMemref, c0);

    builder.create<scf::ForOp>(
        loc, c0, toRetainSize, c1, std::nullopt,
        [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
          builder.create<memref::StoreOp>(loc, falseValue, retainCondsMemref,
                                          i);
          builder.create<scf::YieldOp>(loc);
        });

    builder.create<scf::ForOp>(
        loc, c0, toDeallocSize, c1, std::nullopt,
        [&](OpBuilder &builder, Location loc, Value outerIter,
            ValueRange iterArgs) {
          Value toDealloc =
              builder.create<memref::LoadOp>(loc, toDeallocMemref, outerIter);
          Value cond =
              builder.create<memref::LoadOp>(loc, conditionMemref, outerIter);

          // Build the first for loop that computes aliasing with retained
          // memrefs.
          Value noRetainAlias =
              builder
                  .create<scf::ForOp>(
                      loc, c0, toRetainSize, c1, trueValue,
                      [&](OpBuilder &builder, Location loc, Value i,
                          ValueRange iterArgs) {
                        Value retainValue = builder.create<memref::LoadOp>(
                            loc, toRetainMemref, i);
                        Value doesAlias = builder.create<arith::CmpIOp>(
                            loc, arith::CmpIPredicate::eq, retainValue,
                            toDealloc);
                        builder.create<scf::IfOp>(
                            loc, doesAlias,
                            [&](OpBuilder &builder, Location loc) {
                              Value retainCondValue =
                                  builder.create<memref::LoadOp>(
                                      loc, retainCondsMemref, i);
                              Value aggregatedRetainCond =
                                  builder.create<arith::OrIOp>(
                                      loc, retainCondValue, cond);
                              builder.create<memref::StoreOp>(
                                  loc, aggregatedRetainCond, retainCondsMemref,
                                  i);
                              builder.create<scf::YieldOp>(loc);
                            });
                        Value doesntAlias = builder.create<arith::CmpIOp>(
                            loc, arith::CmpIPredicate::ne, retainValue,
                            toDealloc);
                        Value yieldValue = builder.create<arith::AndIOp>(
                            loc, iterArgs[0], doesntAlias);
                        builder.create<scf::YieldOp>(loc, yieldValue);
                      })
                  .getResult(0);

          // Build the second for loop that adds aliasing with previously
          // deallocated memrefs.
          Value noAlias =
              builder
                  .create<scf::ForOp>(
                      loc, c0, outerIter, c1, noRetainAlias,
                      [&](OpBuilder &builder, Location loc, Value i,
                          ValueRange iterArgs) {
                        Value prevDeallocValue = builder.create<memref::LoadOp>(
                            loc, toDeallocMemref, i);
                        Value doesntAlias = builder.create<arith::CmpIOp>(
                            loc, arith::CmpIPredicate::ne, prevDeallocValue,
                            toDealloc);
                        Value yieldValue = builder.create<arith::AndIOp>(
                            loc, iterArgs[0], doesntAlias);
                        builder.create<scf::YieldOp>(loc, yieldValue);
                      })
                  .getResult(0);

          Value shouldDealoc =
              builder.create<arith::AndIOp>(loc, noAlias, cond);
          builder.create<memref::StoreOp>(loc, shouldDealoc, deallocCondsMemref,
                                          outerIter);
          builder.create<scf::YieldOp>(loc);
        });

    builder.create<func::ReturnOp>(loc);
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
    if (!isa<ModuleOp, FunctionOpInterface>(getOperation())) {
      emitError(getOperation()->getLoc(),
                "root operation must be a builtin.module or a function");
      signalPassFailure();
      return;
    }

    func::FuncOp helperFuncOp;
    if (auto module = dyn_cast<ModuleOp>(getOperation())) {
      OpBuilder builder =
          OpBuilder::atBlockBegin(&module.getBodyRegion().front());
      SymbolTable symbolTable(module);

      // Build dealloc helper function if there are deallocs.
      getOperation()->walk([&](bufferization::DeallocOp deallocOp) {
        if (deallocOp.getMemrefs().size() > 1) {
          helperFuncOp = DeallocOpConversion::buildDeallocationHelperFunction(
              builder, getOperation()->getLoc(), symbolTable);
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
    }

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

std::unique_ptr<Pass> mlir::createBufferizationToMemRefPass() {
  return std::make_unique<BufferizationToMemRefPass>();
}
