//===- SparseBufferRewriting.cpp - Sparse buffer rewriting rules ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements rewriting rules that are specific to sparse tensor
// primitives with memref operands.
//
//===----------------------------------------------------------------------===//

#include "CodegenUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

//===---------------------------------------------------------------------===//
// Helper methods for the actual rewriting rules.
//===---------------------------------------------------------------------===//

constexpr uint64_t loIdx = 0;
constexpr uint64_t hiIdx = 1;
constexpr uint64_t xStartIdx = 2;

typedef function_ref<void(OpBuilder &, ModuleOp, func::FuncOp, size_t)>
    FuncGeneratorType;

/// Constructs a function name with this format to facilitate quick sort:
///   <namePrefix><dim>_<x type>_<y0 type>..._<yn type>
static void getMangledSortHelperFuncName(llvm::raw_svector_ostream &nameOstream,
                                         StringRef namePrefix, size_t dim,
                                         ValueRange operands) {
  nameOstream
      << namePrefix << dim << "_"
      << operands[xStartIdx].getType().cast<MemRefType>().getElementType();

  for (Value v : operands.drop_front(xStartIdx + dim))
    nameOstream << "_" << v.getType().cast<MemRefType>().getElementType();
}

/// Looks up a function that is appropriate for the given operands being
/// sorted, and creates such a function if it doesn't exist yet.
static FlatSymbolRefAttr
getMangledSortHelperFunc(OpBuilder &builder, func::FuncOp insertPoint,
                         TypeRange resultTypes, StringRef namePrefix,
                         size_t dim, ValueRange operands,
                         FuncGeneratorType createFunc) {
  SmallString<32> nameBuffer;
  llvm::raw_svector_ostream nameOstream(nameBuffer);
  getMangledSortHelperFuncName(nameOstream, namePrefix, dim, operands);

  ModuleOp module = insertPoint->getParentOfType<ModuleOp>();
  MLIRContext *context = module.getContext();
  auto result = SymbolRefAttr::get(context, nameOstream.str());
  auto func = module.lookupSymbol<func::FuncOp>(result.getAttr());

  if (!func) {
    // Create the function.
    OpBuilder::InsertionGuard insertionGuard(builder);
    builder.setInsertionPoint(insertPoint);
    Location loc = insertPoint.getLoc();
    func = builder.create<func::FuncOp>(
        loc, nameOstream.str(),
        FunctionType::get(context, operands.getTypes(), resultTypes));
    func.setPrivate();
    createFunc(builder, module, func, dim);
  }

  return result;
}

/// Creates a function for swapping the values in index i and j for all the
/// buffers.
//
// The generate IR corresponds to this C like algorithm:
//   if (i != j) {
//     swap(x0[i], x0[j]);
//     swap(x1[i], x1[j]);
//     ...
//     swap(xn[i], xn[j]);
//     swap(y0[i], y0[j]);
//     ...
//     swap(yn[i], yn[j]);
//   }
static void createMaySwapFunc(OpBuilder &builder, ModuleOp unused,
                              func::FuncOp func, size_t dim) {
  OpBuilder::InsertionGuard insertionGuard(builder);

  Block *entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  Location loc = func.getLoc();
  ValueRange args = entryBlock->getArguments();
  Value i = args[0];
  Value j = args[1];
  Value cond =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, i, j);
  scf::IfOp ifOp = builder.create<scf::IfOp>(loc, cond, /*else=*/false);

  // If i!=j swap values in the buffers.
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  for (auto arg : args.drop_front(xStartIdx)) {
    Value vi = builder.create<memref::LoadOp>(loc, arg, i);
    Value vj = builder.create<memref::LoadOp>(loc, arg, j);
    builder.create<memref::StoreOp>(loc, vj, arg, i);
    builder.create<memref::StoreOp>(loc, vi, arg, j);
  }

  builder.setInsertionPointAfter(ifOp);
  builder.create<func::ReturnOp>(loc);
}

/// Generates an if-statement to compare x[i] and x[j].
static scf::IfOp createLessThanCompare(OpBuilder &builder, Location loc,
                                       Value i, Value j, Value x,
                                       bool isLastDim) {
  Value f = constantI1(builder, loc, false);
  Value t = constantI1(builder, loc, true);
  Value vi = builder.create<memref::LoadOp>(loc, x, i);
  Value vj = builder.create<memref::LoadOp>(loc, x, j);

  Value cond =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, vi, vj);
  scf::IfOp ifOp =
      builder.create<scf::IfOp>(loc, f.getType(), cond, /*else=*/true);
  // If (x[i] < x[j]).
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  builder.create<scf::YieldOp>(loc, t);

  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  if (isLastDim == 1) {
    // Finish checking all dimensions.
    builder.create<scf::YieldOp>(loc, f);
  } else {
    cond =
        builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, vj, vi);
    scf::IfOp ifOp2 =
        builder.create<scf::IfOp>(loc, f.getType(), cond, /*else=*/true);
    // Otherwise if (x[j] < x[i]).
    builder.setInsertionPointToStart(&ifOp2.getThenRegion().front());
    builder.create<scf::YieldOp>(loc, f);

    // Otherwise check the remaining dimensions.
    builder.setInsertionPointAfter(ifOp2);
    builder.create<scf::YieldOp>(loc, ifOp2.getResult(0));
    // Set up the insertion point for the nested if-stmt that checks the
    // remaining dimensions.
    builder.setInsertionPointToStart(&ifOp2.getElseRegion().front());
  }

  return ifOp;
}

/// Creates a function to compare the xs values in index i and j for all the
/// dimensions. The function returns true iff xs[i] < xs[j].
//
// The generate IR corresponds to this C like algorithm:
//   if (x0[i] < x0[j])
//     return true;
//   else if (x0[j] < x0[i])
//     return false;
//   else
//     if (x1[i] < x1[j])
//       return true;
//     else if (x1[j] < x1[i]))
//       and so on ...
static void createLessThanFunc(OpBuilder &builder, ModuleOp unused,
                               func::FuncOp func, size_t dim) {
  OpBuilder::InsertionGuard insertionGuard(builder);

  Block *entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  Location loc = func.getLoc();
  ValueRange args = entryBlock->getArguments();

  scf::IfOp topIfOp;
  for (const auto &item : llvm::enumerate(args.slice(xStartIdx, dim))) {
    scf::IfOp ifOp =
        createLessThanCompare(builder, loc, args[0], args[1], item.value(),
                              (item.index() == dim - 1));
    if (item.index() == 0) {
      topIfOp = ifOp;
    } else {
      OpBuilder::InsertionGuard insertionGuard(builder);
      builder.setInsertionPointAfter(ifOp);
      builder.create<scf::YieldOp>(loc, ifOp.getResult(0));
    }
  }

  builder.setInsertionPointAfter(topIfOp);
  builder.create<func::ReturnOp>(loc, topIfOp.getResult(0));
}

/// Creates a function to perform quick sort partition on the values in the
/// range of index [lo, hi), assuming lo < hi.
//
// The generated IR corresponds to this C like algorithm:
// int partition(lo, hi, data) {
//   pivot = data[hi - 1];
//   i = (lo – 1)  // RHS of the pivot found so far.
//   for (j = lo; j < hi - 1; j++){
//     if (data[j] < pivot){
//       i++;
//       swap data[i] and data[j]
//     }
//   }
//   i++
//   swap data[i] and data[hi-1])
//   return i
// }
static void createPartitionFunc(OpBuilder &builder, ModuleOp module,
                                func::FuncOp func, size_t dim) {
  OpBuilder::InsertionGuard insertionGuard(builder);

  Block *entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  MLIRContext *context = module.getContext();
  Location loc = func.getLoc();
  ValueRange args = entryBlock->getArguments();
  Value lo = args[loIdx];
  Value c1 = constantIndex(builder, loc, 1);
  Value i = builder.create<arith::SubIOp>(loc, lo, c1);
  Value him1 = builder.create<arith::SubIOp>(loc, args[hiIdx], c1);
  scf::ForOp forOp =
      builder.create<scf::ForOp>(loc, lo, him1, c1, ValueRange{i});

  // Start the for-stmt body.
  builder.setInsertionPointToStart(forOp.getBody());
  Value j = forOp.getInductionVar();
  SmallVector<Value, 6> compareOperands{j, him1};
  ValueRange xs = args.slice(xStartIdx, dim);
  compareOperands.append(xs.begin(), xs.end());
  Type i1Type = IntegerType::get(context, 1, IntegerType::Signless);
  FlatSymbolRefAttr lessThanFunc =
      getMangledSortHelperFunc(builder, func, {i1Type}, "_sparse_less_than_",
                               dim, compareOperands, createLessThanFunc);
  Value cond = builder
                   .create<func::CallOp>(loc, lessThanFunc, TypeRange{i1Type},
                                         compareOperands)
                   .getResult(0);
  scf::IfOp ifOp =
      builder.create<scf::IfOp>(loc, i.getType(), cond, /*else=*/true);

  // The if-stmt true branch: i++; swap(data[i], data[j]); yield i.
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  Value i1 =
      builder.create<arith::AddIOp>(loc, forOp.getRegionIterArgs().front(), c1);
  SmallVector<Value, 6> swapOperands{i1, j};
  swapOperands.append(args.begin() + xStartIdx, args.end());
  FlatSymbolRefAttr swapFunc =
      getMangledSortHelperFunc(builder, func, TypeRange(), "_sparse_may_swap_",
                               dim, swapOperands, createMaySwapFunc);
  builder.create<func::CallOp>(loc, swapFunc, TypeRange(), swapOperands);
  builder.create<scf::YieldOp>(loc, i1);

  // The if-stmt false branch: yield i.
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  builder.create<scf::YieldOp>(loc, forOp.getRegionIterArgs().front());

  // After the if-stmt, yield the updated i value to end the for-stmt body.
  builder.setInsertionPointAfter(ifOp);
  builder.create<scf::YieldOp>(loc, ifOp.getResult(0));

  // After the for-stmt: i++; swap(data[i], data[him1]); return i.
  builder.setInsertionPointAfter(forOp);
  i1 = builder.create<arith::AddIOp>(loc, forOp.getResult(0), c1);
  swapOperands[0] = i1;
  swapOperands[1] = him1;
  builder.create<func::CallOp>(loc, swapFunc, TypeRange(), swapOperands);
  builder.create<func::ReturnOp>(loc, i1);
}

/// Creates a function to perform quick sort on the value in the range of
/// index [lo, hi).
//
// The generate IR corresponds to this C like algorithm:
// void quickSort(lo, hi, data) {
//   if (lo < hi) {
//        p = partition(low, high, data);
//        quickSort(lo, p, data);
//        quickSort(p + 1, hi, data);
//   }
// }
static void createSortFunc(OpBuilder &builder, ModuleOp module,
                           func::FuncOp func, size_t dim) {
  OpBuilder::InsertionGuard insertionGuard(builder);
  Block *entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  MLIRContext *context = module.getContext();
  Location loc = func.getLoc();
  ValueRange args = entryBlock->getArguments();
  Value lo = args[loIdx];
  Value hi = args[hiIdx];
  Value cond =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, lo, hi);
  scf::IfOp ifOp = builder.create<scf::IfOp>(loc, cond, /*else=*/false);

  // The if-stmt true branch.
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  FlatSymbolRefAttr partitionFunc = getMangledSortHelperFunc(
      builder, func, {IndexType::get(context)}, "_sparse_partition_", dim, args,
      createPartitionFunc);
  auto p = builder.create<func::CallOp>(
      loc, partitionFunc, TypeRange{IndexType::get(context)}, ValueRange(args));

  SmallVector<Value, 6> lowOperands{lo, p.getResult(0)};
  lowOperands.append(args.begin() + xStartIdx, args.end());
  builder.create<func::CallOp>(loc, func, lowOperands);

  SmallVector<Value, 6> highOperands{
      builder.create<arith::AddIOp>(loc, p.getResult(0),
                                    constantIndex(builder, loc, 1)),
      hi};
  highOperands.append(args.begin() + xStartIdx, args.end());
  builder.create<func::CallOp>(loc, func, highOperands);

  // After the if-stmt.
  builder.setInsertionPointAfter(ifOp);
  builder.create<func::ReturnOp>(loc);
}

//===---------------------------------------------------------------------===//
// The actual sparse buffer rewriting rules.
//===---------------------------------------------------------------------===//

namespace {

/// Sparse rewriting rule for the push_back operator.
struct PushBackRewriter : OpRewritePattern<PushBackOp> {
public:
  using OpRewritePattern<PushBackOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(PushBackOp op,
                                PatternRewriter &rewriter) const override {
    // Rewrite push_back(buffer, value) to:
    // if (size(buffer) >= capacity(buffer))
    //    new_capacity = capacity(buffer)*2
    //    new_buffer = realloc(buffer, new_capacity)
    // buffer = new_buffer
    // store(buffer, value)
    // size(buffer)++
    //
    // The capacity check is skipped when the attribute inbounds is presented.
    Location loc = op->getLoc();
    Value c0 = constantIndex(rewriter, loc, 0);
    Value buffer = op.getInBuffer();
    Value capacity = rewriter.create<memref::DimOp>(loc, buffer, c0);
    Value idx = constantIndex(rewriter, loc, op.getIdx().getZExtValue());
    Value bufferSizes = op.getBufferSizes();
    Value size = rewriter.create<memref::LoadOp>(loc, bufferSizes, idx);
    Value value = op.getValue();

    if (!op.getInbounds()) {
      Value cond = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::uge, size, capacity);

      auto bufferType =
          MemRefType::get({ShapedType::kDynamicSize}, value.getType());
      scf::IfOp ifOp = rewriter.create<scf::IfOp>(loc, bufferType, cond,
                                                  /*else=*/true);
      // True branch.
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      Value c2 = constantIndex(rewriter, loc, 2);
      capacity = rewriter.create<arith::MulIOp>(loc, capacity, c2);
      Value newBuffer =
          rewriter.create<memref::ReallocOp>(loc, bufferType, buffer, capacity);
      rewriter.create<scf::YieldOp>(loc, newBuffer);

      // False branch.
      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
      rewriter.create<scf::YieldOp>(loc, buffer);

      // Prepare for adding the value to the end of the buffer.
      rewriter.setInsertionPointAfter(ifOp);
      buffer = ifOp.getResult(0);
    }

    // Add the value to the end of the buffer.
    rewriter.create<memref::StoreOp>(loc, value, buffer, size);

    // Increment the size of the buffer by 1.
    Value c1 = constantIndex(rewriter, loc, 1);
    size = rewriter.create<arith::AddIOp>(loc, size, c1);
    rewriter.create<memref::StoreOp>(loc, size, bufferSizes, idx);

    rewriter.replaceOp(op, buffer);
    return success();
  }
};

/// Sparse rewriting rule for the sort operator.
struct SortRewriter : public OpRewritePattern<SortOp> {
public:
  using OpRewritePattern<SortOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SortOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value, 6> operands{constantIndex(rewriter, loc, 0), op.getN()};

    // Convert `values` to have dynamic shape and append them to `operands`.
    auto addValues = [&](ValueRange values) {
      for (Value v : values) {
        auto mtp = v.getType().cast<MemRefType>();
        if (!mtp.isDynamicDim(0)) {
          auto new_mtp =
              MemRefType::get({ShapedType::kDynamicSize}, mtp.getElementType());
          v = rewriter.create<memref::CastOp>(loc, new_mtp, v);
        }
        operands.push_back(v);
      }
    };
    ValueRange xs = op.getXs();
    addValues(xs);
    addValues(op.getYs());
    auto insertPoint = op->getParentOfType<func::FuncOp>();
    FlatSymbolRefAttr func = getMangledSortHelperFunc(
        rewriter, insertPoint, TypeRange(), "_sparse_sort_", xs.size(),
        operands, createSortFunc);
    rewriter.replaceOpWithNewOp<func::CallOp>(op, func, TypeRange(), operands);
    return success();
  }
};

} // namespace

//===---------------------------------------------------------------------===//
// Methods that add patterns described in this file to a pattern list.
//===---------------------------------------------------------------------===//

void mlir::populateSparseBufferRewriting(RewritePatternSet &patterns) {
  patterns.add<PushBackRewriter, SortRewriter>(patterns.getContext());
}
