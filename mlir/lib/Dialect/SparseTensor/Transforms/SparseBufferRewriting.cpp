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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
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

static constexpr uint64_t loIdx = 0;
static constexpr uint64_t hiIdx = 1;
static constexpr uint64_t xStartIdx = 2;

static constexpr const char kLessThanFuncNamePrefix[] = "_sparse_less_than_";
static constexpr const char kCompareEqFuncNamePrefix[] = "_sparse_compare_eq_";
static constexpr const char kPartitionFuncNamePrefix[] = "_sparse_partition_";
static constexpr const char kBinarySearchFuncNamePrefix[] =
    "_sparse_binary_search_";
static constexpr const char kHybridQuickSortFuncNamePrefix[] =
    "_sparse_hybrid_qsort_";
static constexpr const char kSortStableFuncNamePrefix[] =
    "_sparse_sort_stable_";
static constexpr const char kShiftDownFuncNamePrefix[] = "_sparse_shift_down_";
static constexpr const char kHeapSortFuncNamePrefix[] = "_sparse_heap_sort_";
static constexpr const char kQuickSortFuncNamePrefix[] = "_sparse_qsort_";

using FuncGeneratorType = function_ref<void(
    OpBuilder &, ModuleOp, func::FuncOp, uint64_t, uint64_t, bool, uint32_t)>;

/// Constructs a function name with this format to facilitate quick sort:
///   <namePrefix><nx>_<x type>_<y0 type>..._<yn type> for sort
///   <namePrefix><nx>_<x type>_coo_<ny>_<y0 type>..._<yn type> for sort_coo
static void getMangledSortHelperFuncName(llvm::raw_svector_ostream &nameOstream,
                                         StringRef namePrefix, uint64_t nx,
                                         uint64_t ny, bool isCoo,
                                         ValueRange operands) {
  nameOstream << namePrefix << nx << "_"
              << getMemRefType(operands[xStartIdx]).getElementType();

  if (isCoo)
    nameOstream << "_coo_" << ny;

  uint64_t yBufferOffset = isCoo ? 1 : nx;
  for (Value v : operands.drop_front(xStartIdx + yBufferOffset))
    nameOstream << "_" << getMemRefType(v).getElementType();
}

/// Looks up a function that is appropriate for the given operands being
/// sorted, and creates such a function if it doesn't exist yet. The
/// parameters `nx` and `ny` tell the number of x and y values provided
/// by the buffer in xStartIdx, and `isCoo` indicates whether the instruction
/// being processed is a sparse_tensor.sort or sparse_tensor.sort_coo.
//
// All sorting function generators take (lo, hi, xs, ys) in `operands` as
// parameters for the sorting functions. Other parameters, such as the recursive
// call depth, are appended to the end of the parameter list as
// "trailing parameters".
static FlatSymbolRefAttr
getMangledSortHelperFunc(OpBuilder &builder, func::FuncOp insertPoint,
                         TypeRange resultTypes, StringRef namePrefix,
                         uint64_t nx, uint64_t ny, bool isCoo,
                         ValueRange operands, FuncGeneratorType createFunc,
                         uint32_t nTrailingP = 0) {
  SmallString<32> nameBuffer;
  llvm::raw_svector_ostream nameOstream(nameBuffer);
  getMangledSortHelperFuncName(nameOstream, namePrefix, nx, ny, isCoo,
                               operands.drop_back(nTrailingP));

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
    createFunc(builder, module, func, nx, ny, isCoo, nTrailingP);
  }

  return result;
}

/// Creates a code block to process each pair of (xs[i], xs[j]) for sorting.
/// The code to process the value pairs is generated by `bodyBuilder`.
static void forEachIJPairInXs(
    OpBuilder &builder, Location loc, ValueRange args, uint64_t nx, uint64_t ny,
    bool isCoo, function_ref<void(uint64_t, Value, Value, Value)> bodyBuilder) {
  Value iOffset, jOffset;
  if (isCoo) {
    Value cstep = constantIndex(builder, loc, nx + ny);
    iOffset = builder.create<arith::MulIOp>(loc, args[0], cstep);
    jOffset = builder.create<arith::MulIOp>(loc, args[1], cstep);
  }
  for (uint64_t k = 0; k < nx; k++) {
    scf::IfOp ifOp;
    Value i, j, buffer;
    if (isCoo) {
      Value ck = constantIndex(builder, loc, k);
      i = builder.create<arith::AddIOp>(loc, ck, iOffset);
      j = builder.create<arith::AddIOp>(loc, ck, jOffset);
      buffer = args[xStartIdx];
    } else {
      i = args[0];
      j = args[1];
      buffer = args[xStartIdx + k];
    }
    bodyBuilder(k, i, j, buffer);
  }
}

/// Creates a code block to process each pair of (xys[i], xys[j]) for sorting.
/// The code to process the value pairs is generated by `bodyBuilder`.
static void forEachIJPairInAllBuffers(
    OpBuilder &builder, Location loc, ValueRange args, uint64_t nx, uint64_t ny,
    bool isCoo, function_ref<void(uint64_t, Value, Value, Value)> bodyBuilder) {

  // Create code for the first (nx + ny) buffers. When isCoo==true, these
  // logical buffers are all from the xy buffer of the sort_coo operator.
  forEachIJPairInXs(builder, loc, args, nx + ny, 0, isCoo, bodyBuilder);

  uint64_t numHandledBuffers = isCoo ? 1 : nx + ny;

  // Create code for the remaining buffers.
  Value i = args[0];
  Value j = args[1];
  for (const auto &arg :
       llvm::enumerate(args.drop_front(xStartIdx + numHandledBuffers))) {
    bodyBuilder(arg.index() + nx + ny, i, j, arg.value());
  }
}

/// Creates a code block for swapping the values in index i and j for all the
/// buffers.
//
// The generated IR corresponds to this C like algorithm:
//     swap(x0[i], x0[j]);
//     swap(x1[i], x1[j]);
//     ...
//     swap(xn[i], xn[j]);
//     swap(y0[i], y0[j]);
//     ...
//     swap(yn[i], yn[j]);
static void createSwap(OpBuilder &builder, Location loc, ValueRange args,
                       uint64_t nx, uint64_t ny, bool isCoo) {
  auto swapOnePair = [&](uint64_t unused, Value i, Value j, Value buffer) {
    Value vi = builder.create<memref::LoadOp>(loc, buffer, i);
    Value vj = builder.create<memref::LoadOp>(loc, buffer, j);
    builder.create<memref::StoreOp>(loc, vj, buffer, i);
    builder.create<memref::StoreOp>(loc, vi, buffer, j);
  };

  forEachIJPairInAllBuffers(builder, loc, args, nx, ny, isCoo, swapOnePair);
}

/// Creates a function to compare all the (xs[i], xs[j]) pairs. The method to
/// compare each pair is create via `compareBuilder`.
static void createCompareFuncImplementation(
    OpBuilder &builder, ModuleOp unused, func::FuncOp func, uint64_t nx,
    uint64_t ny, bool isCoo,
    function_ref<scf::IfOp(OpBuilder &, Location, Value, Value, Value, bool)>
        compareBuilder) {
  OpBuilder::InsertionGuard insertionGuard(builder);

  Block *entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  Location loc = func.getLoc();
  ValueRange args = entryBlock->getArguments();

  scf::IfOp topIfOp;
  auto bodyBuilder = [&](uint64_t k, Value i, Value j, Value buffer) {
    scf::IfOp ifOp = compareBuilder(builder, loc, i, j, buffer, (k == nx - 1));
    if (k == 0) {
      topIfOp = ifOp;
    } else {
      OpBuilder::InsertionGuard insertionGuard(builder);
      builder.setInsertionPointAfter(ifOp);
      builder.create<scf::YieldOp>(loc, ifOp.getResult(0));
    }
  };

  forEachIJPairInXs(builder, loc, args, nx, ny, isCoo, bodyBuilder);

  builder.setInsertionPointAfter(topIfOp);
  builder.create<func::ReturnOp>(loc, topIfOp.getResult(0));
}

/// Generates an if-statement to compare whether x[i] is equal to x[j].
static scf::IfOp createEqCompare(OpBuilder &builder, Location loc, Value i,
                                 Value j, Value x, bool isLastDim) {
  Value f = constantI1(builder, loc, false);
  Value t = constantI1(builder, loc, true);
  Value vi = builder.create<memref::LoadOp>(loc, x, i);
  Value vj = builder.create<memref::LoadOp>(loc, x, j);

  Value cond =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, vi, vj);
  scf::IfOp ifOp =
      builder.create<scf::IfOp>(loc, f.getType(), cond, /*else=*/true);

  // x[1] != x[j]:
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  builder.create<scf::YieldOp>(loc, f);

  // x[i] == x[j]:
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  if (isLastDim == 1) {
    // Finish checking all dimensions.
    builder.create<scf::YieldOp>(loc, t);
  }

  return ifOp;
}

/// Creates a function to compare whether xs[i] is equal to xs[j].
//
// The generate IR corresponds to this C like algorithm:
//   if (x0[i] != x0[j])
//     return false;
//   else
//     if (x1[i] != x1[j])
//       return false;
//     else if (x2[2] != x2[j]))
//       and so on ...
static void createEqCompareFunc(OpBuilder &builder, ModuleOp unused,
                                func::FuncOp func, uint64_t nx, uint64_t ny,
                                bool isCoo, uint32_t nTrailingP = 0) {
  // Compare functions don't use trailing parameters.
  (void)nTrailingP;
  assert(nTrailingP == 0);
  createCompareFuncImplementation(builder, unused, func, nx, ny, isCoo,
                                  createEqCompare);
}

/// Generates an if-statement to compare whether x[i] is less than x[j].
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

/// Creates a function to compare whether xs[i] is less than xs[j].
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
                               func::FuncOp func, uint64_t nx, uint64_t ny,
                               bool isCoo, uint32_t nTrailingP = 0) {
  // Compare functions don't use trailing parameters.
  (void)nTrailingP;
  assert(nTrailingP == 0);
  createCompareFuncImplementation(builder, unused, func, nx, ny, isCoo,
                                  createLessThanCompare);
}

/// Creates a function to use a binary search to find the insertion point for
/// inserting xs[hi] to the sorted values xs[lo..hi).
//
// The generate IR corresponds to this C like algorithm:
//   p = hi
//   while (lo < hi)
//      mid = (lo + hi) >> 1
//      if (xs[p] < xs[mid])
//        hi = mid
//      else
//        lo = mid - 1
//   return lo;
//
static void createBinarySearchFunc(OpBuilder &builder, ModuleOp module,
                                   func::FuncOp func, uint64_t nx, uint64_t ny,
                                   bool isCoo, uint32_t nTrailingP = 0) {
  // Binary search doesn't use trailing parameters.
  (void)nTrailingP;
  assert(nTrailingP == 0);
  OpBuilder::InsertionGuard insertionGuard(builder);
  Block *entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  Location loc = func.getLoc();
  ValueRange args = entryBlock->getArguments();
  Value p = args[hiIdx];
  SmallVector<Type, 2> types(2, p.getType()); // Only two types.
  scf::WhileOp whileOp = builder.create<scf::WhileOp>(
      loc, types, SmallVector<Value, 2>{args[loIdx], args[hiIdx]});

  // The before-region of the WhileOp.
  Block *before =
      builder.createBlock(&whileOp.getBefore(), {}, types, {loc, loc});
  builder.setInsertionPointToEnd(before);
  Value cond1 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                              before->getArgument(0),
                                              before->getArgument(1));
  builder.create<scf::ConditionOp>(loc, cond1, before->getArguments());

  // The after-region of the WhileOp.
  Block *after =
      builder.createBlock(&whileOp.getAfter(), {}, types, {loc, loc});
  builder.setInsertionPointToEnd(after);
  Value lo = after->getArgument(0);
  Value hi = after->getArgument(1);
  // Compute mid = (lo + hi) >> 1.
  Value c1 = constantIndex(builder, loc, 1);
  Value mid = builder.create<arith::ShRUIOp>(
      loc, builder.create<arith::AddIOp>(loc, lo, hi), c1);
  Value midp1 = builder.create<arith::AddIOp>(loc, mid, c1);

  // Compare xs[p] < xs[mid].
  SmallVector<Value> compareOperands{p, mid};
  uint64_t numXBuffers = isCoo ? 1 : nx;
  compareOperands.append(args.begin() + xStartIdx,
                         args.begin() + xStartIdx + numXBuffers);
  Type i1Type = IntegerType::get(module.getContext(), 1, IntegerType::Signless);
  FlatSymbolRefAttr lessThanFunc = getMangledSortHelperFunc(
      builder, func, {i1Type}, kLessThanFuncNamePrefix, nx, ny, isCoo,
      compareOperands, createLessThanFunc, nTrailingP);
  Value cond2 = builder
                    .create<func::CallOp>(loc, lessThanFunc, TypeRange{i1Type},
                                          compareOperands)
                    .getResult(0);

  // Update lo and hi for the WhileOp as follows:
  //   if (xs[p] < xs[mid]))
  //     hi = mid;
  //   else
  //     lo = mid + 1;
  Value newLo = builder.create<arith::SelectOp>(loc, cond2, lo, midp1);
  Value newHi = builder.create<arith::SelectOp>(loc, cond2, mid, hi);
  builder.create<scf::YieldOp>(loc, ValueRange{newLo, newHi});

  builder.setInsertionPointAfter(whileOp);
  builder.create<func::ReturnOp>(loc, whileOp.getResult(0));
}

/// Creates code to advance i in a loop based on xs[p] as follows:
///   while (xs[i] < xs[p]) i += step (step > 0)
/// or
///   while (xs[i] > xs[p]) i += step (step < 0)
/// The routine returns i as well as a boolean value to indicate whether
/// xs[i] == xs[p].
static std::pair<Value, Value>
createScanLoop(OpBuilder &builder, ModuleOp module, func::FuncOp func,
               ValueRange xs, Value i, Value p, uint64_t nx, uint64_t ny,
               bool isCoo, int step) {
  Location loc = func.getLoc();
  scf::WhileOp whileOp =
      builder.create<scf::WhileOp>(loc, TypeRange{i.getType()}, ValueRange{i});

  Block *before =
      builder.createBlock(&whileOp.getBefore(), {}, {i.getType()}, {loc});
  builder.setInsertionPointToEnd(before);
  SmallVector<Value> compareOperands;
  if (step > 0) {
    compareOperands.push_back(before->getArgument(0));
    compareOperands.push_back(p);
  } else {
    assert(step < 0);
    compareOperands.push_back(p);
    compareOperands.push_back(before->getArgument(0));
  }
  compareOperands.append(xs.begin(), xs.end());
  MLIRContext *context = module.getContext();
  Type i1Type = IntegerType::get(context, 1, IntegerType::Signless);
  FlatSymbolRefAttr lessThanFunc = getMangledSortHelperFunc(
      builder, func, {i1Type}, kLessThanFuncNamePrefix, nx, ny, isCoo,
      compareOperands, createLessThanFunc);
  Value cond = builder
                   .create<func::CallOp>(loc, lessThanFunc, TypeRange{i1Type},
                                         compareOperands)
                   .getResult(0);
  builder.create<scf::ConditionOp>(loc, cond, before->getArguments());

  Block *after =
      builder.createBlock(&whileOp.getAfter(), {}, {i.getType()}, {loc});
  builder.setInsertionPointToEnd(after);
  Value cs = constantIndex(builder, loc, step);
  i = builder.create<arith::AddIOp>(loc, after->getArgument(0), cs);
  builder.create<scf::YieldOp>(loc, ValueRange{i});
  i = whileOp.getResult(0);

  builder.setInsertionPointAfter(whileOp);
  compareOperands[0] = i;
  compareOperands[1] = p;
  FlatSymbolRefAttr compareEqFunc = getMangledSortHelperFunc(
      builder, func, {i1Type}, kCompareEqFuncNamePrefix, nx, ny, isCoo,
      compareOperands, createEqCompareFunc);
  Value compareEq =
      builder
          .create<func::CallOp>(loc, compareEqFunc, TypeRange{i1Type},
                                compareOperands)
          .getResult(0);

  return std::make_pair(whileOp.getResult(0), compareEq);
}

/// Creates a code block to swap the values so that data[mi] is the median among
/// data[lo], data[hi], and data[mi].
//  The generated code corresponds to this C-like algorithm:
//  median = mi
//  if (data[mi] < data[lo]).                               (if1)
//    if (data[hi] < data[lo])                              (if2)
//       median = data[hi] < data[mi] ? mi : hi
//    else
//       median = lo
//  else
//    if data[hi] < data[mi]                                (if3)
//      median = data[hi] < data[lo] ? lo : hi
//  if median != mi swap data[median] with data[mi]
static void createChoosePivot(OpBuilder &builder, ModuleOp module,
                              func::FuncOp func, uint64_t nx, uint64_t ny,
                              bool isCoo, Value lo, Value hi, Value mi,
                              ValueRange args) {
  SmallVector<Value> compareOperands{mi, lo};
  uint64_t numXBuffers = isCoo ? 1 : nx;
  compareOperands.append(args.begin() + xStartIdx,
                         args.begin() + xStartIdx + numXBuffers);
  Type i1Type = IntegerType::get(module.getContext(), 1, IntegerType::Signless);
  SmallVector<Type, 1> cmpTypes{i1Type};
  FlatSymbolRefAttr lessThanFunc = getMangledSortHelperFunc(
      builder, func, cmpTypes, kLessThanFuncNamePrefix, nx, ny, isCoo,
      compareOperands, createLessThanFunc);
  Location loc = func.getLoc();
  // Compare data[mi] < data[lo].
  Value cond1 =
      builder.create<func::CallOp>(loc, lessThanFunc, cmpTypes, compareOperands)
          .getResult(0);
  SmallVector<Type, 1> ifTypes{lo.getType()};
  scf::IfOp ifOp1 =
      builder.create<scf::IfOp>(loc, ifTypes, cond1, /*else=*/true);

  // Generate an if-stmt to find the median value, assuming we already know that
  // data[b] < data[a] and we haven't compare data[c] yet.
  auto createFindMedian = [&](Value a, Value b, Value c) -> scf::IfOp {
    compareOperands[0] = c;
    compareOperands[1] = a;
    // Compare data[c]] < data[a].
    Value cond2 =
        builder
            .create<func::CallOp>(loc, lessThanFunc, cmpTypes, compareOperands)
            .getResult(0);
    scf::IfOp ifOp2 =
        builder.create<scf::IfOp>(loc, ifTypes, cond2, /*else=*/true);
    builder.setInsertionPointToStart(&ifOp2.getThenRegion().front());
    compareOperands[0] = c;
    compareOperands[1] = b;
    // Compare data[c] < data[b].
    Value cond3 =
        builder
            .create<func::CallOp>(loc, lessThanFunc, cmpTypes, compareOperands)
            .getResult(0);
    builder.create<scf::YieldOp>(
        loc, ValueRange{builder.create<arith::SelectOp>(loc, cond3, b, c)});
    builder.setInsertionPointToStart(&ifOp2.getElseRegion().front());
    builder.create<scf::YieldOp>(loc, ValueRange{a});
    return ifOp2;
  };

  builder.setInsertionPointToStart(&ifOp1.getThenRegion().front());
  scf::IfOp ifOp2 = createFindMedian(lo, mi, hi);
  builder.setInsertionPointAfter(ifOp2);
  builder.create<scf::YieldOp>(loc, ValueRange{ifOp2.getResult(0)});

  builder.setInsertionPointToStart(&ifOp1.getElseRegion().front());
  scf::IfOp ifOp3 = createFindMedian(mi, lo, hi);

  builder.setInsertionPointAfter(ifOp3);
  builder.create<scf::YieldOp>(loc, ValueRange{ifOp3.getResult(0)});

  builder.setInsertionPointAfter(ifOp1);
  Value median = ifOp1.getResult(0);
  Value cond =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, mi, median);
  scf::IfOp ifOp =
      builder.create<scf::IfOp>(loc, TypeRange(), cond, /*else=*/false);

  SmallVector<Value> swapOperands{median, mi};
  swapOperands.append(args.begin() + xStartIdx, args.end());
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  createSwap(builder, loc, swapOperands, nx, ny, isCoo);
  builder.setInsertionPointAfter(ifOp);
}

/// Creates a function to perform quick sort partition on the values in the
/// range of index [lo, hi), assuming lo < hi.
//
// The generated IR corresponds to this C like algorithm:
// int partition(lo, hi, xs) {
//   p = (lo+hi)/2  // pivot index
//   i = lo
//   j = hi-1
//   while (i < j) do {
//     while (xs[i] < xs[p]) i ++;
//     i_eq = (xs[i] == xs[p]);
//     while (xs[j] > xs[p]) j --;
//     j_eq = (xs[j] == xs[p]);
//     if (i < j) {
//       swap(xs[i], xs[j])
//       if (i == p) {
//         p = j;
//       } else if (j == p) {
//         p = i;
//       }
//       if (i_eq && j_eq) {
//         ++i;
//         --j;
//       }
//     }
//   }
//   return p
//   }
static void createPartitionFunc(OpBuilder &builder, ModuleOp module,
                                func::FuncOp func, uint64_t nx, uint64_t ny,
                                bool isCoo, uint32_t nTrailingP = 0) {
  // Quick sort partition doesn't use trailing parameters.
  (void)nTrailingP;
  assert(nTrailingP == 0);
  OpBuilder::InsertionGuard insertionGuard(builder);

  Block *entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  Location loc = func.getLoc();
  ValueRange args = entryBlock->getArguments();
  Value lo = args[loIdx];
  Value hi = args[hiIdx];
  Value sum = builder.create<arith::AddIOp>(loc, lo, hi);
  Value c1 = constantIndex(builder, loc, 1);
  Value p = builder.create<arith::ShRUIOp>(loc, sum, c1);

  Value i = lo;
  Value j = builder.create<arith::SubIOp>(loc, hi, c1);
  createChoosePivot(builder, module, func, nx, ny, isCoo, i, j, p, args);
  SmallVector<Value, 3> operands{i, j, p}; // Exactly three values.
  SmallVector<Type, 3> types{i.getType(), j.getType(), p.getType()};
  scf::WhileOp whileOp = builder.create<scf::WhileOp>(loc, types, operands);

  // The before-region of the WhileOp.
  Block *before =
      builder.createBlock(&whileOp.getBefore(), {}, types, {loc, loc, loc});
  builder.setInsertionPointToEnd(before);
  Value cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                             before->getArgument(0),
                                             before->getArgument(1));
  builder.create<scf::ConditionOp>(loc, cond, before->getArguments());

  // The after-region of the WhileOp.
  Block *after =
      builder.createBlock(&whileOp.getAfter(), {}, types, {loc, loc, loc});
  builder.setInsertionPointToEnd(after);
  i = after->getArgument(0);
  j = after->getArgument(1);
  p = after->getArgument(2);

  uint64_t numXBuffers = isCoo ? 1 : nx;
  auto [iresult, iCompareEq] =
      createScanLoop(builder, module, func, args.slice(xStartIdx, numXBuffers),
                     i, p, nx, ny, isCoo, 1);
  i = iresult;
  auto [jresult, jCompareEq] =
      createScanLoop(builder, module, func, args.slice(xStartIdx, numXBuffers),
                     j, p, nx, ny, isCoo, -1);
  j = jresult;

  // If i < j:
  cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, i, j);
  scf::IfOp ifOp = builder.create<scf::IfOp>(loc, types, cond, /*else=*/true);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  SmallVector<Value> swapOperands{i, j};
  swapOperands.append(args.begin() + xStartIdx, args.end());
  createSwap(builder, loc, swapOperands, nx, ny, isCoo);
  // If the pivot is moved, update p with the new pivot.
  Value icond =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, i, p);
  scf::IfOp ifOpI = builder.create<scf::IfOp>(loc, TypeRange{p.getType()},
                                              icond, /*else=*/true);
  builder.setInsertionPointToStart(&ifOpI.getThenRegion().front());
  builder.create<scf::YieldOp>(loc, ValueRange{j});
  builder.setInsertionPointToStart(&ifOpI.getElseRegion().front());
  Value jcond =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, j, p);
  scf::IfOp ifOpJ = builder.create<scf::IfOp>(loc, TypeRange{p.getType()},
                                              jcond, /*else=*/true);
  builder.setInsertionPointToStart(&ifOpJ.getThenRegion().front());
  builder.create<scf::YieldOp>(loc, ValueRange{i});
  builder.setInsertionPointToStart(&ifOpJ.getElseRegion().front());
  builder.create<scf::YieldOp>(loc, ValueRange{p});
  builder.setInsertionPointAfter(ifOpJ);
  builder.create<scf::YieldOp>(loc, ifOpJ.getResults());
  builder.setInsertionPointAfter(ifOpI);
  Value compareEqIJ =
      builder.create<arith::AndIOp>(loc, iCompareEq, jCompareEq);
  scf::IfOp ifOp2 = builder.create<scf::IfOp>(
      loc, TypeRange{i.getType(), j.getType()}, compareEqIJ, /*else=*/true);
  builder.setInsertionPointToStart(&ifOp2.getThenRegion().front());
  Value i2 = builder.create<arith::AddIOp>(loc, i, c1);
  Value j2 = builder.create<arith::SubIOp>(loc, j, c1);
  builder.create<scf::YieldOp>(loc, ValueRange{i2, j2});
  builder.setInsertionPointToStart(&ifOp2.getElseRegion().front());
  builder.create<scf::YieldOp>(loc, ValueRange{i, j});
  builder.setInsertionPointAfter(ifOp2);
  builder.create<scf::YieldOp>(
      loc,
      ValueRange{ifOp2.getResult(0), ifOp2.getResult(1), ifOpI.getResult(0)});

  // False branch for if i < j:
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  builder.create<scf::YieldOp>(loc, ValueRange{i, j, p});

  // Return for the whileOp.
  builder.setInsertionPointAfter(ifOp);
  builder.create<scf::YieldOp>(loc, ifOp.getResults());

  // Return for the function.
  builder.setInsertionPointAfter(whileOp);
  builder.create<func::ReturnOp>(loc, whileOp.getResult(2));
}

/// Computes (n-2)/n, assuming n has index type.
static Value createSubTwoDividedByTwo(OpBuilder &builder, Location loc,
                                      Value n) {
  Value i2 = constantIndex(builder, loc, 2);
  Value res = builder.create<arith::SubIOp>(loc, n, i2);
  Value i1 = constantIndex(builder, loc, 1);
  return builder.create<arith::ShRUIOp>(loc, res, i1);
}

/// Creates a function to heapify the subtree with root `start` within the full
/// binary tree in the range of index [first, first + n).
//
// The generated IR corresponds to this C like algorithm:
// void shiftDown(first, start, n, data) {
//   if (n >= 2) {
//     child = start - first
//     if ((n-2)/2 >= child) {
//       // Left child exists.
//       child = child * 2 + 1 // Initialize the bigger child to left child.
//       childIndex = child + first
//       if (child+1 < n && data[childIndex] < data[childIndex+1])
//         // Right child exits and is bigger.
//         childIndex++; child++;
//       // Shift data[start] down to where it belongs in the subtree.
//       while (data[start] < data[childIndex) {
//         swap(data[start], data[childIndex])
//         start = childIndex
//         if ((n - 2)/2 >= child) {
//           // Left child exists.
//           child = 2*child + 1
//           childIndex = child + 1
//           if (child + 1) < n && data[childIndex] < data[childIndex+1]
//             childIndex++; child++;
//         }
//       }
//     }
//   }
// }
//
static void createShiftDownFunc(OpBuilder &builder, ModuleOp module,
                                func::FuncOp func, uint64_t nx, uint64_t ny,
                                bool isCoo, uint32_t nTrailingP) {
  // The value n is passed in as a trailing parameter.
  assert(nTrailingP == 1);
  OpBuilder::InsertionGuard insertionGuard(builder);
  Block *entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  Location loc = func.getLoc();
  Value n = entryBlock->getArguments().back();
  ValueRange args = entryBlock->getArguments().drop_back();
  Value first = args[loIdx];
  Value start = args[hiIdx];

  // If (n >= 2).
  Value c2 = constantIndex(builder, loc, 2);
  Value condN =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, n, c2);
  scf::IfOp ifN = builder.create<scf::IfOp>(loc, condN, /*else=*/false);
  builder.setInsertionPointToStart(&ifN.getThenRegion().front());
  Value child = builder.create<arith::SubIOp>(loc, start, first);

  // If ((n-2)/2 >= child).
  Value t = createSubTwoDividedByTwo(builder, loc, n);
  Value condNc =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, t, child);
  scf::IfOp ifNc = builder.create<scf::IfOp>(loc, condNc, /*else=*/false);

  builder.setInsertionPointToStart(&ifNc.getThenRegion().front());
  Value c1 = constantIndex(builder, loc, 1);
  SmallVector<Value> compareOperands{start, start};
  uint64_t numXBuffers = isCoo ? 1 : nx;
  compareOperands.append(args.begin() + xStartIdx,
                         args.begin() + xStartIdx + numXBuffers);
  Type i1Type = IntegerType::get(module.getContext(), 1, IntegerType::Signless);
  FlatSymbolRefAttr lessThanFunc = getMangledSortHelperFunc(
      builder, func, {i1Type}, kLessThanFuncNamePrefix, nx, ny, isCoo,
      compareOperands, createLessThanFunc);

  // Generate code to inspect the children of 'r' and return the larger child
  // as follows:
  //   child = r * 2 + 1 // Left child.
  //   childIndex = child + first
  //   if (child+1 < n && data[childIndex] < data[childIndex+1])
  //     childIndex ++; child ++ // Right child is bigger.
  auto getLargerChild = [&](Value r) -> std::pair<Value, Value> {
    Value lChild = builder.create<arith::ShLIOp>(loc, r, c1);
    lChild = builder.create<arith::AddIOp>(loc, lChild, c1);
    Value lChildIdx = builder.create<arith::AddIOp>(loc, lChild, first);
    Value rChild = builder.create<arith::AddIOp>(loc, lChild, c1);
    Value cond1 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                                rChild, n);
    SmallVector<Type, 2> ifTypes(2, r.getType());
    scf::IfOp if1 =
        builder.create<scf::IfOp>(loc, ifTypes, cond1, /*else=*/true);
    builder.setInsertionPointToStart(&if1.getThenRegion().front());
    Value rChildIdx = builder.create<arith::AddIOp>(loc, rChild, first);
    // Compare data[left] < data[right].
    compareOperands[0] = lChildIdx;
    compareOperands[1] = rChildIdx;
    Value cond2 = builder
                      .create<func::CallOp>(loc, lessThanFunc,
                                            TypeRange{i1Type}, compareOperands)
                      .getResult(0);
    scf::IfOp if2 =
        builder.create<scf::IfOp>(loc, ifTypes, cond2, /*else=*/true);
    builder.setInsertionPointToStart(&if2.getThenRegion().front());
    builder.create<scf::YieldOp>(loc, ValueRange{rChild, rChildIdx});
    builder.setInsertionPointToStart(&if2.getElseRegion().front());
    builder.create<scf::YieldOp>(loc, ValueRange{lChild, lChildIdx});
    builder.setInsertionPointAfter(if2);
    builder.create<scf::YieldOp>(loc, if2.getResults());
    builder.setInsertionPointToStart(&if1.getElseRegion().front());
    builder.create<scf::YieldOp>(loc, ValueRange{lChild, lChildIdx});
    builder.setInsertionPointAfter(if1);
    return std::make_pair(if1.getResult(0), if1.getResult(1));
  };

  Value childIdx;
  std::tie(child, childIdx) = getLargerChild(child);

  // While (data[start] < data[childIndex]).
  SmallVector<Type, 3> types(3, child.getType());
  scf::WhileOp whileOp = builder.create<scf::WhileOp>(
      loc, types, SmallVector<Value, 2>{start, child, childIdx});

  // The before-region of the WhileOp.
  SmallVector<Location, 3> locs(3, loc);
  Block *before = builder.createBlock(&whileOp.getBefore(), {}, types, locs);
  builder.setInsertionPointToEnd(before);
  start = before->getArgument(0);
  childIdx = before->getArgument(2);
  compareOperands[0] = start;
  compareOperands[1] = childIdx;
  Value cond = builder
                   .create<func::CallOp>(loc, lessThanFunc, TypeRange{i1Type},
                                         compareOperands)
                   .getResult(0);
  builder.create<scf::ConditionOp>(loc, cond, before->getArguments());

  // The after-region of the WhileOp.
  Block *after = builder.createBlock(&whileOp.getAfter(), {}, types, locs);
  start = after->getArgument(0);
  child = after->getArgument(1);
  childIdx = after->getArgument(2);
  SmallVector<Value> swapOperands{start, childIdx};
  swapOperands.append(args.begin() + xStartIdx, args.end());
  createSwap(builder, loc, swapOperands, nx, ny, isCoo);
  start = childIdx;
  Value cond2 =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, t, child);
  scf::IfOp if2 = builder.create<scf::IfOp>(
      loc, TypeRange{child.getType(), child.getType()}, cond2, /*else=*/true);
  builder.setInsertionPointToStart(&if2.getThenRegion().front());
  auto [newChild, newChildIdx] = getLargerChild(child);
  builder.create<scf::YieldOp>(loc, ValueRange{newChild, newChildIdx});
  builder.setInsertionPointToStart(&if2.getElseRegion().front());
  builder.create<scf::YieldOp>(loc, ValueRange{child, childIdx});
  builder.setInsertionPointAfter(if2);
  builder.create<scf::YieldOp>(
      loc, ValueRange{start, if2.getResult(0), if2.getResult(1)});

  builder.setInsertionPointAfter(ifN);
  builder.create<func::ReturnOp>(loc);
}

/// Creates a function to perform heap sort on the values in the range of index
/// [lo, hi) with the assumption hi - lo >= 2.
//
// The generate IR corresponds to this C like algorithm:
// void heapSort(lo, hi, data) {
//   n = hi - lo
//   for i = (n-2)/2 downto 0
//     shiftDown(lo, lo+i, n)
//
//   for l = n downto 2
//      swap(lo, lo+l-1)
//      shiftdown(lo, lo, l-1)
// }
static void createHeapSortFunc(OpBuilder &builder, ModuleOp module,
                               func::FuncOp func, uint64_t nx, uint64_t ny,
                               bool isCoo, uint32_t nTrailingP) {
  // Heap sort function doesn't have trailing parameters.
  (void)nTrailingP;
  assert(nTrailingP == 0);
  OpBuilder::InsertionGuard insertionGuard(builder);
  Block *entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  Location loc = func.getLoc();
  ValueRange args = entryBlock->getArguments();
  Value lo = args[loIdx];
  Value hi = args[hiIdx];
  Value n = builder.create<arith::SubIOp>(loc, hi, lo);

  // For i = (n-2)/2 downto 0.
  Value c0 = constantIndex(builder, loc, 0);
  Value c1 = constantIndex(builder, loc, 1);
  Value s = createSubTwoDividedByTwo(builder, loc, n);
  Value up = builder.create<arith::AddIOp>(loc, s, c1);
  scf::ForOp forI = builder.create<scf::ForOp>(loc, c0, up, c1);
  builder.setInsertionPointToStart(forI.getBody());
  Value i = builder.create<arith::SubIOp>(loc, s, forI.getInductionVar());
  Value lopi = builder.create<arith::AddIOp>(loc, lo, i);
  SmallVector<Value> shiftDownOperands = {lo, lopi};
  shiftDownOperands.append(args.begin() + xStartIdx, args.end());
  shiftDownOperands.push_back(n);
  FlatSymbolRefAttr shiftDownFunc = getMangledSortHelperFunc(
      builder, func, TypeRange(), kShiftDownFuncNamePrefix, nx, ny, isCoo,
      shiftDownOperands, createShiftDownFunc, /*nTrailingP=*/1);
  builder.create<func::CallOp>(loc, shiftDownFunc, TypeRange(),
                               shiftDownOperands);

  builder.setInsertionPointAfter(forI);
  // For l = n downto 2.
  up = builder.create<arith::SubIOp>(loc, n, c1);
  scf::ForOp forL = builder.create<scf::ForOp>(loc, c0, up, c1);
  builder.setInsertionPointToStart(forL.getBody());
  Value l = builder.create<arith::SubIOp>(loc, n, forL.getInductionVar());
  Value loplm1 = builder.create<arith::AddIOp>(loc, lo, l);
  loplm1 = builder.create<arith::SubIOp>(loc, loplm1, c1);
  SmallVector<Value> swapOperands{lo, loplm1};
  swapOperands.append(args.begin() + xStartIdx, args.end());
  createSwap(builder, loc, swapOperands, nx, ny, isCoo);
  shiftDownOperands[1] = lo;
  shiftDownOperands[shiftDownOperands.size() - 1] =
      builder.create<arith::SubIOp>(loc, l, c1);
  builder.create<func::CallOp>(loc, shiftDownFunc, TypeRange(),
                               shiftDownOperands);

  builder.setInsertionPointAfter(forL);
  builder.create<func::ReturnOp>(loc);
}

static void createQuickSort(OpBuilder &builder, ModuleOp module,
                            func::FuncOp func, ValueRange args, uint64_t nx,
                            uint64_t ny, bool isCoo, uint32_t nTrailingP) {
  MLIRContext *context = module.getContext();
  Location loc = func.getLoc();
  Value lo = args[loIdx];
  Value hi = args[hiIdx];
  FlatSymbolRefAttr partitionFunc = getMangledSortHelperFunc(
      builder, func, {IndexType::get(context)}, kPartitionFuncNamePrefix, nx,
      ny, isCoo, args.drop_back(nTrailingP), createPartitionFunc);
  auto p = builder.create<func::CallOp>(loc, partitionFunc,
                                        TypeRange{IndexType::get(context)},
                                        args.drop_back(nTrailingP));

  SmallVector<Value> lowOperands{lo, p.getResult(0)};
  lowOperands.append(args.begin() + xStartIdx, args.end());
  builder.create<func::CallOp>(loc, func, lowOperands);

  SmallVector<Value> highOperands{
      builder.create<arith::AddIOp>(loc, p.getResult(0),
                                    constantIndex(builder, loc, 1)),
      hi};
  highOperands.append(args.begin() + xStartIdx, args.end());
  builder.create<func::CallOp>(loc, func, highOperands);
}

/// Creates a function to perform insertion sort on the values in the range of
/// index [lo, hi).
//
// The generate IR corresponds to this C like algorithm:
// void insertionSort(lo, hi, data) {
//   for (i = lo+1; i < hi; i++) {
//      d = data[i];
//      p = binarySearch(lo, i-1, data)
//      for (j = 0; j > i - p; j++)
//        data[i-j] = data[i-j-1]
//      data[p] = d
//   }
// }
static void createSortStableFunc(OpBuilder &builder, ModuleOp module,
                                 func::FuncOp func, uint64_t nx, uint64_t ny,
                                 bool isCoo, uint32_t nTrailingP) {
  // Stable sort function doesn't use trailing parameters.
  (void)nTrailingP;
  assert(nTrailingP == 0);
  OpBuilder::InsertionGuard insertionGuard(builder);
  Block *entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  MLIRContext *context = module.getContext();
  Location loc = func.getLoc();
  ValueRange args = entryBlock->getArguments();
  Value c1 = constantIndex(builder, loc, 1);
  Value lo = args[loIdx];
  Value hi = args[hiIdx];
  Value lop1 = builder.create<arith::AddIOp>(loc, lo, c1);

  // Start the outer for-stmt with induction variable i.
  scf::ForOp forOpI = builder.create<scf::ForOp>(loc, lop1, hi, c1);
  builder.setInsertionPointToStart(forOpI.getBody());
  Value i = forOpI.getInductionVar();

  // Binary search to find the insertion point p.
  SmallVector<Value> operands{lo, i};
  operands.append(args.begin() + xStartIdx, args.end());
  FlatSymbolRefAttr searchFunc = getMangledSortHelperFunc(
      builder, func, {IndexType::get(context)}, kBinarySearchFuncNamePrefix, nx,
      ny, isCoo, operands, createBinarySearchFunc);
  Value p = builder
                .create<func::CallOp>(loc, searchFunc, TypeRange{c1.getType()},
                                      operands)
                .getResult(0);

  // Move the value at data[i] to a temporary location.
  operands[0] = operands[1] = i;
  SmallVector<Value> d;
  forEachIJPairInAllBuffers(
      builder, loc, operands, nx, ny, isCoo,
      [&](uint64_t unused, Value i, Value unused2, Value buffer) {
        d.push_back(builder.create<memref::LoadOp>(loc, buffer, i));
      });

  // Start the inner for-stmt with induction variable j, for moving data[p..i)
  // to data[p+1..i+1).
  Value imp = builder.create<arith::SubIOp>(loc, i, p);
  Value c0 = constantIndex(builder, loc, 0);
  scf::ForOp forOpJ = builder.create<scf::ForOp>(loc, c0, imp, c1);
  builder.setInsertionPointToStart(forOpJ.getBody());
  Value j = forOpJ.getInductionVar();
  Value imj = builder.create<arith::SubIOp>(loc, i, j);
  operands[1] = imj;
  operands[0] = builder.create<arith::SubIOp>(loc, imj, c1);
  forEachIJPairInAllBuffers(
      builder, loc, operands, nx, ny, isCoo,
      [&](uint64_t unused, Value imjm1, Value imj, Value buffer) {
        Value t = builder.create<memref::LoadOp>(loc, buffer, imjm1);
        builder.create<memref::StoreOp>(loc, t, buffer, imj);
      });

  // Store the value at data[i] to data[p].
  builder.setInsertionPointAfter(forOpJ);
  operands[0] = operands[1] = p;
  forEachIJPairInAllBuffers(
      builder, loc, operands, nx, ny, isCoo,
      [&](uint64_t k, Value p, Value usused, Value buffer) {
        builder.create<memref::StoreOp>(loc, d[k], buffer, p);
      });

  builder.setInsertionPointAfter(forOpI);
  builder.create<func::ReturnOp>(loc);
}

/// Creates a function to perform quick sort or a hybrid quick sort on the
/// values in the range of index [lo, hi).
//
//
// When nTrailingP == 0, the generated IR corresponds to this C like algorithm:
// void quickSort(lo, hi, data) {
//   if (lo + 1 < hi) {
//        p = partition(low, high, data);
//        quickSort(lo, p, data);
//        quickSort(p + 1, hi, data);
//   }
// }
//
// When nTrailingP == 1, the generated IR corresponds to this C like algorithm:
// void hybridQuickSort(lo, hi, data, depthLimit) {
//   if (lo + 1 < hi) {
//     len = hi - lo;
//     if (len <= limit) {
//       insertionSort(lo, hi, data);
//     } else {
//       depthLimit --;
//       if (depthLimit <= 0) {
//         heapSort(lo, hi, data);
//       } else {
//          p = partition(low, high, data);
//          quickSort(lo, p, data);
//          quickSort(p + 1, hi, data);
//       }
//       depthLimit ++;
//     }
//   }
// }
//
static void createQuickSortFunc(OpBuilder &builder, ModuleOp module,
                                func::FuncOp func, uint64_t nx, uint64_t ny,
                                bool isCoo, uint32_t nTrailingP) {
  assert(nTrailingP == 1 || nTrailingP == 0);
  bool isHybrid = (nTrailingP == 1);
  OpBuilder::InsertionGuard insertionGuard(builder);
  Block *entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  Location loc = func.getLoc();
  ValueRange args = entryBlock->getArguments();
  Value lo = args[loIdx];
  Value hi = args[hiIdx];
  Value loCmp =
      builder.create<arith::AddIOp>(loc, lo, constantIndex(builder, loc, 1));
  Value cond =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, loCmp, hi);
  scf::IfOp ifOp = builder.create<scf::IfOp>(loc, cond, /*else=*/false);

  // The if-stmt true branch.
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  Value pDepthLimit;
  Value savedDepthLimit;
  scf::IfOp depthIf;

  if (isHybrid) {
    Value len = builder.create<arith::SubIOp>(loc, hi, lo);
    Value lenLimit = constantIndex(builder, loc, 30);
    Value lenCond = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ule, len, lenLimit);
    scf::IfOp lenIf = builder.create<scf::IfOp>(loc, lenCond, /*else=*/true);

    // When len <= limit.
    builder.setInsertionPointToStart(&lenIf.getThenRegion().front());
    FlatSymbolRefAttr insertionSortFunc = getMangledSortHelperFunc(
        builder, func, TypeRange(), kSortStableFuncNamePrefix, nx, ny, isCoo,
        args.drop_back(nTrailingP), createSortStableFunc);
    builder.create<func::CallOp>(loc, insertionSortFunc, TypeRange(),
                                 ValueRange(args.drop_back(nTrailingP)));

    // When len > limit.
    builder.setInsertionPointToStart(&lenIf.getElseRegion().front());
    pDepthLimit = args.back();
    savedDepthLimit = builder.create<memref::LoadOp>(loc, pDepthLimit);
    Value depthLimit = builder.create<arith::SubIOp>(
        loc, savedDepthLimit, constantI64(builder, loc, 1));
    builder.create<memref::StoreOp>(loc, depthLimit, pDepthLimit);
    Value depthCond =
        builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule,
                                      depthLimit, constantI64(builder, loc, 0));
    depthIf = builder.create<scf::IfOp>(loc, depthCond, /*else=*/true);

    // When depth exceeds limit.
    builder.setInsertionPointToStart(&depthIf.getThenRegion().front());
    FlatSymbolRefAttr heapSortFunc = getMangledSortHelperFunc(
        builder, func, TypeRange(), kHeapSortFuncNamePrefix, nx, ny, isCoo,
        args.drop_back(nTrailingP), createHeapSortFunc);
    builder.create<func::CallOp>(loc, heapSortFunc, TypeRange(),
                                 ValueRange(args.drop_back(nTrailingP)));

    // When depth doesn't exceed limit.
    builder.setInsertionPointToStart(&depthIf.getElseRegion().front());
  }

  createQuickSort(builder, module, func, args, nx, ny, isCoo, nTrailingP);

  if (isHybrid) {
    // Restore depthLimit.
    builder.setInsertionPointAfter(depthIf);
    builder.create<memref::StoreOp>(loc, savedDepthLimit, pDepthLimit);
  }

  // After the if-stmt.
  builder.setInsertionPointAfter(ifOp);
  builder.create<func::ReturnOp>(loc);
}

/// Implements the rewriting for operator sort and sort_coo.
template <typename OpTy>
LogicalResult matchAndRewriteSortOp(OpTy op, ValueRange xys, uint64_t nx,
                                    uint64_t ny, bool isCoo,
                                    PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  SmallVector<Value> operands{constantIndex(rewriter, loc, 0), op.getN()};

  // Convert `values` to have dynamic shape and append them to `operands`.
  for (Value v : xys) {
    auto mtp = getMemRefType(v);
    if (!mtp.isDynamicDim(0)) {
      auto newMtp =
          MemRefType::get({ShapedType::kDynamic}, mtp.getElementType());
      v = rewriter.create<memref::CastOp>(loc, newMtp, v);
    }
    operands.push_back(v);
  }

  auto insertPoint = op->template getParentOfType<func::FuncOp>();
  SmallString<32> funcName;
  FuncGeneratorType funcGenerator;
  uint32_t nTrailingP = 0;
  switch (op.getAlgorithm()) {
  case SparseTensorSortKind::HybridQuickSort: {
    funcName = kHybridQuickSortFuncNamePrefix;
    funcGenerator = createQuickSortFunc;
    nTrailingP = 1;
    Value pDepthLimit = rewriter.create<memref::AllocaOp>(
        loc, MemRefType::get({}, rewriter.getI64Type()));
    operands.push_back(pDepthLimit);
    // As a heuristics, set depthLimit = 2 * log2(n).
    Value lo = operands[loIdx];
    Value hi = operands[hiIdx];
    Value len = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(),
        rewriter.create<arith::SubIOp>(loc, hi, lo));
    Value depthLimit = rewriter.create<arith::SubIOp>(
        loc, constantI64(rewriter, loc, 64),
        rewriter.create<math::CountLeadingZerosOp>(loc, len));
    depthLimit = rewriter.create<arith::ShLIOp>(loc, depthLimit,
                                                constantI64(rewriter, loc, 1));
    rewriter.create<memref::StoreOp>(loc, depthLimit, pDepthLimit);
    break;
  }
  case SparseTensorSortKind::QuickSort:
    funcName = kQuickSortFuncNamePrefix;
    funcGenerator = createQuickSortFunc;
    break;
  case SparseTensorSortKind::InsertionSortStable:
    funcName = kSortStableFuncNamePrefix;
    funcGenerator = createSortStableFunc;
    break;
  case SparseTensorSortKind::HeapSort:
    funcName = kHeapSortFuncNamePrefix;
    funcGenerator = createHeapSortFunc;
    break;
  }

  FlatSymbolRefAttr func =
      getMangledSortHelperFunc(rewriter, insertPoint, TypeRange(), funcName, nx,
                               ny, isCoo, operands, funcGenerator, nTrailingP);
  rewriter.replaceOpWithNewOp<func::CallOp>(op, func, TypeRange(), operands);
  return success();
}

//===---------------------------------------------------------------------===//
// The actual sparse buffer rewriting rules.
//===---------------------------------------------------------------------===//

namespace {

/// Sparse rewriting rule for the push_back operator.
struct PushBackRewriter : OpRewritePattern<PushBackOp> {
public:
  using OpRewritePattern<PushBackOp>::OpRewritePattern;
  PushBackRewriter(MLIRContext *context, bool enableInit)
      : OpRewritePattern(context), enableBufferInitialization(enableInit) {}
  LogicalResult matchAndRewrite(PushBackOp op,
                                PatternRewriter &rewriter) const override {
    // Rewrite push_back(buffer, value, n) to:
    // new_size = size(buffer) + n
    // if (new_size > capacity(buffer))
    //    while new_size > new_capacity
    //      new_capacity = new_capacity*2
    //    new_buffer = realloc(buffer, new_capacity)
    // buffer = new_buffer
    // subBuffer = subviewof(buffer)
    // linalg.fill subBuffer value
    //
    // size(buffer) += n
    //
    // The capacity check is skipped when the attribute inbounds is presented.
    Location loc = op->getLoc();
    Value c0 = constantIndex(rewriter, loc, 0);
    Value buffer = op.getInBuffer();
    Value capacity = rewriter.create<memref::DimOp>(loc, buffer, c0);
    Value size = op.getCurSize();
    Value value = op.getValue();

    Value n = op.getN() ? op.getN() : constantIndex(rewriter, loc, 1);
    Value newSize = rewriter.create<arith::AddIOp>(loc, size, n);
    auto nValue = dyn_cast_or_null<arith::ConstantIndexOp>(n.getDefiningOp());
    bool nIsOne = (nValue && nValue.value() == 1);

    if (!op.getInbounds()) {
      Value cond = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ugt, newSize, capacity);

      Value c2 = constantIndex(rewriter, loc, 2);
      auto bufferType =
          MemRefType::get({ShapedType::kDynamic}, value.getType());
      scf::IfOp ifOp = rewriter.create<scf::IfOp>(loc, bufferType, cond,
                                                  /*else=*/true);
      // True branch.
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      if (nIsOne) {
        capacity = rewriter.create<arith::MulIOp>(loc, capacity, c2);
      } else {
        // Use a do-while loop to calculate the new capacity as follows:
        //   do { new_capacity *= 2 } while (size > new_capacity)
        scf::WhileOp whileOp =
            rewriter.create<scf::WhileOp>(loc, capacity.getType(), capacity);

        // The before-region of the WhileOp.
        Block *before = rewriter.createBlock(&whileOp.getBefore(), {},
                                             {capacity.getType()}, {loc});
        rewriter.setInsertionPointToEnd(before);

        capacity =
            rewriter.create<arith::MulIOp>(loc, before->getArgument(0), c2);
        cond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                              newSize, capacity);
        rewriter.create<scf::ConditionOp>(loc, cond, ValueRange{capacity});
        // The after-region of the WhileOp.
        Block *after = rewriter.createBlock(&whileOp.getAfter(), {},
                                            {capacity.getType()}, {loc});
        rewriter.setInsertionPointToEnd(after);
        rewriter.create<scf::YieldOp>(loc, after->getArguments());

        rewriter.setInsertionPointAfter(whileOp);
        capacity = whileOp.getResult(0);
      }

      Value newBuffer =
          rewriter.create<memref::ReallocOp>(loc, bufferType, buffer, capacity);
      if (enableBufferInitialization) {
        Value fillSize = rewriter.create<arith::SubIOp>(loc, capacity, newSize);
        Value fillValue = constantZero(rewriter, loc, value.getType());
        Value subBuffer = rewriter.create<memref::SubViewOp>(
            loc, newBuffer, /*offset=*/ValueRange{newSize},
            /*size=*/ValueRange{fillSize},
            /*step=*/ValueRange{constantIndex(rewriter, loc, 1)});
        rewriter.create<linalg::FillOp>(loc, fillValue, subBuffer);
      }
      rewriter.create<scf::YieldOp>(loc, newBuffer);

      // False branch.
      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
      rewriter.create<scf::YieldOp>(loc, buffer);

      // Prepare for adding the value to the end of the buffer.
      rewriter.setInsertionPointAfter(ifOp);
      buffer = ifOp.getResult(0);
    }

    // Add the value to the end of the buffer.
    if (nIsOne) {
      rewriter.create<memref::StoreOp>(loc, value, buffer, size);
    } else {
      Value subBuffer = rewriter.create<memref::SubViewOp>(
          loc, buffer, /*offset=*/ValueRange{size}, /*size=*/ValueRange{n},
          /*step=*/ValueRange{constantIndex(rewriter, loc, 1)});
      rewriter.create<linalg::FillOp>(loc, value, subBuffer);
    }

    // Update the buffer size.
    rewriter.replaceOp(op, {buffer, newSize});
    return success();
  }

private:
  bool enableBufferInitialization;
};

/// Sparse rewriting rule for the sort operator.
struct SortRewriter : public OpRewritePattern<SortOp> {
public:
  using OpRewritePattern<SortOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SortOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> xys(op.getXs());
    xys.append(op.getYs().begin(), op.getYs().end());
    return matchAndRewriteSortOp(op, xys, op.getXs().size(), /*ny=*/0,
                                 /*isCoo=*/false, rewriter);
  }
};

/// Sparse rewriting rule for the sort_coo operator.
struct SortCooRewriter : public OpRewritePattern<SortCooOp> {
public:
  using OpRewritePattern<SortCooOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SortCooOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> xys;
    xys.push_back(op.getXy());
    xys.append(op.getYs().begin(), op.getYs().end());
    uint64_t nx = 1;
    if (auto nxAttr = op.getNxAttr())
      nx = nxAttr.getInt();

    uint64_t ny = 0;
    if (auto nyAttr = op.getNyAttr())
      ny = nyAttr.getInt();

    return matchAndRewriteSortOp(op, xys, nx, ny,
                                 /*isCoo=*/true, rewriter);
  }
};

} // namespace

//===---------------------------------------------------------------------===//
// Methods that add patterns described in this file to a pattern list.
//===---------------------------------------------------------------------===//

void mlir::populateSparseBufferRewriting(RewritePatternSet &patterns,
                                         bool enableBufferInitialization) {
  patterns.add<PushBackRewriter>(patterns.getContext(),
                                 enableBufferInitialization);
  patterns.add<SortRewriter, SortCooRewriter>(patterns.getContext());
}
