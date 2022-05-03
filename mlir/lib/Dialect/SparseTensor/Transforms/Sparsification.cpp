//===- Sparsification.cpp - Implementation of sparsification --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements converting sparse tensor types to actual sparse code.
//
//===----------------------------------------------------------------------===//

#include "CodegenUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/SparseTensor/Utils/Merger.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TensorEncoding.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

//===----------------------------------------------------------------------===//
// Declarations of data structures.
//===----------------------------------------------------------------------===//

namespace {

// Iteration graph sorting.
enum SortMask { kSparseOnly = 0x0, kIncludeDense = 0x1, kIncludeUndef = 0x2 };

// Reduction kinds.
enum Reduction { kNoReduc, kSum, kProduct, kAnd, kOr, kXor };

// Code generation.
struct CodeGen {
  CodeGen(SparsificationOptions o, unsigned numTensors, unsigned numLoops,
          OpOperand *op, unsigned nest)
      : options(o), loops(numLoops), sizes(numLoops), buffers(numTensors),
        pointers(numTensors, std::vector<Value>(numLoops)),
        indices(numTensors, std::vector<Value>(numLoops)),
        highs(numTensors, std::vector<Value>(numLoops)),
        pidxs(numTensors, std::vector<Value>(numLoops)),
        idxs(numTensors, std::vector<Value>(numLoops)), redVal(), sparseOut(op),
        outerParNest(nest), lexIdx(), expValues(), expFilled(), expAdded(),
        expCount(), curVecMask() {}
  /// Sparsification options.
  SparsificationOptions options;
  /// Universal dense indices and upper bounds (by index). The loops array
  /// is updated with the value of the universal dense index in the current
  /// loop. The sizes array is set once with the inferred dimension sizes.
  std::vector<Value> loops;
  std::vector<Value> sizes;
  /// Buffers for storing dense and sparse numerical values (by tensor).
  /// This array is set once during bufferization of all tensors.
  std::vector<Value> buffers;
  /// Sparse storage schemes (1-D): pointers and indices (by tensor and index).
  /// This array is set once during bufferization of all sparse tensors.
  std::vector<std::vector<Value>> pointers;
  std::vector<std::vector<Value>> indices;
  /// Sparse iteration information (by tensor and index). These arrays
  /// are updated to remain current within the current loop.
  std::vector<std::vector<Value>> highs;
  std::vector<std::vector<Value>> pidxs;
  std::vector<std::vector<Value>> idxs;
  /// Current reduction, updated during code generation. When indices of a
  /// reduction are exhausted, all inner loops can use a scalarized reduction.
  unsigned redExp = -1u;
  Value redVal;
  Reduction redKind = kNoReduc;
  // Sparse tensor as output. Implemented either through direct injective
  // insertion in lexicographic index order (where indices are updated
  // in the temporary array `lexIdx`) or through access pattern expansion
  // in the innermost loop nest (`expValues` through `expCount`).
  OpOperand *sparseOut;
  unsigned outerParNest;
  Value lexIdx;
  Value expValues;
  Value expFilled;
  Value expAdded;
  Value expCount;
  // Current vector length and mask.
  unsigned curVecLength = 1;
  Value curVecMask;
};

} // namespace

//===----------------------------------------------------------------------===//
// Sparse compiler analysis methods.
//===----------------------------------------------------------------------===//

/// Helper method to apply dimension ordering permutation.
static unsigned perm(const SparseTensorEncodingAttr &enc, unsigned d) {
  if (enc) {
    auto order = enc.getDimOrdering();
    if (order) {
      assert(order.isPermutation());
      return order.getDimPosition(d);
    }
  }
  return d;
}

/// Helper method to translate dim level type to internal representation.
static Dim toDim(const SparseTensorEncodingAttr &enc, unsigned d) {
  if (enc) {
    SparseTensorEncodingAttr::DimLevelType tp = enc.getDimLevelType()[d];
    if (tp == SparseTensorEncodingAttr::DimLevelType::Compressed)
      return Dim::kSparse;
    if (tp == SparseTensorEncodingAttr::DimLevelType::Singleton)
      return Dim::kSingle;
  }
  return Dim::kDense;
}

/// Helper method to inspect affine expressions. Rejects cases where the
/// same index is used more than once. Also rejects affine expressions
/// that are not a direct index for annotated tensors.
// TODO: accept more affine cases for sparse tensors
static bool findAffine(Merger &merger, unsigned tensor, AffineExpr a, Dim dim,
                       bool isDense) {
  switch (a.getKind()) {
  case AffineExprKind::DimId: {
    unsigned idx = a.cast<AffineDimExpr>().getPosition();
    if (!merger.isDim(tensor, idx, Dim::kUndef))
      return false; // used more than once
    merger.setDim(tensor, idx, dim);
    return true;
  }
  case AffineExprKind::Add:
  case AffineExprKind::Mul: {
    if (!isDense)
      return false;
    auto binOp = a.cast<AffineBinaryOpExpr>();
    return findAffine(merger, tensor, binOp.getLHS(), dim, isDense) &&
           findAffine(merger, tensor, binOp.getRHS(), dim, isDense);
  }
  case AffineExprKind::Constant:
    return isDense;
  default:
    return false;
  }
}

/// Helper method to inspect sparse encodings in the tensor types.
/// Fills the per-dimension sparsity information for all tensors.
/// Returns true if the sparse annotations and affine subscript
/// expressions of all tensors are admissable. Returns false if
/// no annotations are found or inadmissable constructs occur.
static bool findSparseAnnotations(Merger &merger, linalg::GenericOp op) {
  bool annotated = false;
  for (OpOperand *t : op.getInputAndOutputOperands()) {
    auto map = op.getTiedIndexingMap(t);
    auto enc = getSparseTensorEncoding(t->get().getType());
    if (enc)
      annotated = true;
    assert(map.getNumResults() == op.getRank(t));
    for (unsigned d = 0, rank = map.getNumResults(); d < rank; d++) {
      unsigned tensor = t->getOperandNumber();
      AffineExpr a = map.getResult(perm(enc, d));
      if (!findAffine(merger, tensor, a, toDim(enc, d), !enc))
        return false; // inadmissable affine expression
    }
  }
  return annotated;
}

/// A DFS helper to compute a topological sort. Note that recursion is
/// bounded by the number of implicit loops, which is always small.
/// Returns false when a cycle is detected.
static bool topSortDFS(unsigned i, std::vector<unsigned> &visit,
                       std::vector<unsigned> &topSort,
                       std::vector<std::vector<bool>> &adjM) {
  if (visit[i] != 0)
    return visit[i] != 1; // 1 denotes cycle!
  visit[i] = 1;
  for (unsigned j = 0, e = visit.size(); j < e; j++)
    if (adjM[i][j])
      if (!topSortDFS(j, visit, topSort, adjM))
        return false;
  visit[i] = 2;
  topSort.push_back(i);
  return true;
}

/// Helper method to add all constraints from the indices in one affine
/// expression before all indices in the other affine expression. For
/// example i0+i1 < i2+i3+1 yields i0<i2, i0<i3, i1<i2, and i1<i3.
static void addAffineOrderings(std::vector<std::vector<bool>> &adjM,
                               AffineExpr a, AffineExpr b, unsigned fidx) {
  switch (a.getKind()) {
  case AffineExprKind::DimId: {
    unsigned idx = a.cast<AffineDimExpr>().getPosition();
    if (b)
      addAffineOrderings(adjM, b, AffineExpr(), idx);
    else
      adjM[fidx][idx] = true;
    break;
  }
  case AffineExprKind::Add:
  case AffineExprKind::Mul: {
    auto binOp = a.cast<AffineBinaryOpExpr>();
    addAffineOrderings(adjM, binOp.getLHS(), b, fidx);
    addAffineOrderings(adjM, binOp.getRHS(), b, fidx);
    break;
  }
  default:
    break;
  }
}

/// Computes a topologically sorted iteration graph for the linalg operation.
/// Ensures all tensors are visited in natural index order. This is essential
/// for sparse storage formats since these only support access along fixed
/// dimensions. Even for dense storage formats, however, the natural index
/// order yields innermost unit-stride access with better spatial locality.
static bool computeIterationGraph(Merger &merger, linalg::GenericOp op,
                                  std::vector<unsigned> &topSort,
                                  unsigned mask) {
  // Set up an n x n from/to adjacency matrix of the iteration graph
  // for the implicit loop indices i_0 .. i_n-1.
  unsigned n = op.getNumLoops();
  std::vector<std::vector<bool>> adjM(n, std::vector<bool>(n, false));

  // Iterate over the indexing maps of every tensor in the tensor expression.
  for (OpOperand *t : op.getInputAndOutputOperands()) {
    auto map = op.getTiedIndexingMap(t);
    auto enc = getSparseTensorEncoding(t->get().getType());
    assert(map.getNumDims() == n);
    // Skip dense tensor constraints when not requested.
    if (!(mask & SortMask::kIncludeDense) && !enc)
      continue;
    // Each tensor expression and optional dimension ordering (row-major
    // by default) puts an ordering constraint on the loop indices. For
    // example, the tensor expresion A_ijk forces the ordering i < j < k
    // on the loop indices if no explicit dimension ordering is given.
    for (unsigned d = 1, rank = map.getNumResults(); d < rank; d++) {
      AffineExpr f = map.getResult(perm(enc, d - 1));
      AffineExpr t = map.getResult(perm(enc, d));
      addAffineOrderings(adjM, f, t, 0);
    }
    // Push unrelated loops into sparse iteration space, so these
    // will be skipped more often.
    if (mask & SortMask::kIncludeUndef) {
      unsigned tensor = t->getOperandNumber();
      for (unsigned i = 0; i < n; i++)
        if (merger.isDim(tensor, i, Dim::kSparse))
          for (unsigned j = 0; j < n; j++)
            if (merger.isDim(tensor, j, Dim::kUndef))
              adjM[i][j] = true;
    }
  }

  // Topologically sort the iteration graph to determine loop order.
  // Report failure for a cyclic iteration graph.
  topSort.clear();
  topSort.reserve(n);
  std::vector<unsigned> visit(n, 0);
  for (unsigned i = 0; i < n; i++)
    if (visit[i] == 0)
      if (!topSortDFS(i, visit, topSort, adjM))
        return false; // cycle!
  std::reverse(std::begin(topSort), std::end(topSort));
  return true;
}

/// Returns true if tensor has an in-place annotation.
static bool isInPlace(Value val) {
  if (auto arg = val.dyn_cast<BlockArgument>())
    if (auto funcOp = dyn_cast<func::FuncOp>(arg.getOwner()->getParentOp()))
      if (auto attr = funcOp.getArgAttrOfType<BoolAttr>(
              arg.getArgNumber(),
              bufferization::BufferizableOpInterface::kInplaceableAttrName))
        return attr.getValue();
  return false;
}

/// Returns true if tensor materializes uninitialized into the computation.
static bool isMaterializing(Value val) {
  return val.getDefiningOp<linalg::InitTensorOp>() ||
         val.getDefiningOp<InitOp>();
}

/// Returns true when the tensor expression is admissable for codegen.
/// Since all sparse input tensors are admissable, we just need to check
/// whether the out tensor in the tensor expression codegen is admissable.
/// Sets `sparseOut` to the tensor and `outerParNest` to the outer injective
/// nesting depth when a "truly dynamic" sparse tensor output occurs.
static bool isAdmissableTensorExp(Merger &merger, linalg::GenericOp op,
                                  std::vector<unsigned> &topSort, unsigned exp,
                                  OpOperand **sparseOut,
                                  unsigned &outerParNest) {
  OpOperand *lhs = op.getOutputOperand(0);
  unsigned tensor = lhs->getOperandNumber();
  auto enc = getSparseTensorEncoding(lhs->get().getType());
  // An non-annotated output tensor is assumed dense, and becomes a random
  // access n-dim memref. Admissable since insertions cannot occur.
  if (!enc)
    return true;
  // An all-dense annotated "sparse" output tensor becomes a linearized random
  // access 1-dim memref. Also admissable since insertions cannot occur.
  bool allDense = true;
  auto iteratorTypes = op.iterator_types().getValue();
  unsigned numLoops = iteratorTypes.size();
  for (unsigned i = 0; i < numLoops; i++)
    if (merger.isDim(tensor, i, Dim::kSparse)) {
      allDense = false;
      break;
    }
  if (allDense)
    return true;
  // A tensor expression with a sparse output tensor that changes its values
  // but not its nonzero structure, an operation called "simply dynamic" in
  // [Bik96,Ch9], is also admissable without special codegen, provided
  // the tensor's underlying sparse storage scheme can be modified in place.
  if (merger.isSingleCondition(tensor, exp) && isInPlace(lhs->get()))
    return true;
  // Accept "truly dynamic" if the output tensor materializes uninitialized
  // into the computation and insertions occur in lexicographic index order.
  if (isMaterializing(lhs->get())) {
    unsigned nest = 0;
    for (unsigned i = 0; i < numLoops; i++) {
      if (isReductionIterator(iteratorTypes[topSort[i]]))
        break; // terminate at first reduction
      nest++;
    }
    // Determine admissable dynamic insertion situations:
    // (1) fully injective, since there are no reductions,
    // (2) admissable 1-d expansion in innermost dimension.
    if (nest >= op.getRank(lhs) - 1) {
      *sparseOut = lhs;
      outerParNest = nest;
      return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Sparse compiler synthesis methods (reductions).
//===----------------------------------------------------------------------===//

/// Maps reduction kind to vector::CombiningKind.
static vector::CombiningKind getCombiningKind(Reduction kind) {
  switch (kind) {
  case kNoReduc:
    break;
  case kSum:
    return vector::CombiningKind::ADD;
  case kProduct:
    return vector::CombiningKind::MUL;
  case kAnd:
    return vector::CombiningKind::AND;
  case kOr:
    return vector::CombiningKind::OR;
  case kXor:
    return vector::CombiningKind::XOR;
  }
  llvm_unreachable("unknown reduction kind");
}

/// Maps operation to reduction.
static Reduction getReduction(Kind kind) {
  switch (kind) {
  case Kind::kAddF:
  case Kind::kAddI:
  case Kind::kSubF:
  case Kind::kSubI:
    return kSum;
  case Kind::kMulF:
  case Kind::kMulI:
    return kProduct;
  case Kind::kAndI:
    return kAnd;
  case Kind::kOrI:
    return kOr;
  case Kind::kXorI:
    return kXor;
  default:
    llvm_unreachable("unexpected reduction operator");
  }
}

/// Generates an initial value for a vector reduction, following the scheme
/// given in Chapter 5 of "The Software Vectorization Handbook", where the
/// initial scalar value is correctly embedded in the vector reduction value,
/// and a straightforward horizontal reduction will complete the operation.
static Value genVectorReducInit(CodeGen &codegen, PatternRewriter &rewriter,
                                Location loc, VectorType vtp) {
  Value r = codegen.redVal;
  switch (codegen.redKind) {
  case kNoReduc:
    break;
  case kSum:
  case kXor:
    // Initialize reduction vector to: | 0 | .. | 0 | r |
    return rewriter.create<vector::InsertElementOp>(
        loc, r, constantZero(rewriter, loc, vtp),
        constantIndex(rewriter, loc, 0));
  case kProduct:
    // Initialize reduction vector to: | 1 | .. | 1 | r |
    return rewriter.create<vector::InsertElementOp>(
        loc, r, constantOne(rewriter, loc, vtp),
        constantIndex(rewriter, loc, 0));
  case kAnd:
  case kOr:
    // Initialize reduction vector to: | r | .. | r | r |
    return rewriter.create<vector::BroadcastOp>(loc, vtp, r);
  }
  llvm_unreachable("unknown reduction kind");
}

/// Generates final value for a vector reduction.
static Value genVectorReducEnd(CodeGen &codegen, PatternRewriter &rewriter,
                               Location loc, VectorType vtp) {
  vector::CombiningKind kind = getCombiningKind(codegen.redKind);
  return rewriter.create<vector::ReductionOp>(loc, kind, codegen.redVal);
}

/// Updates scalarized reduction value.
static void updateReduc(Merger &merger, CodeGen &codegen, Value reduc) {
  assert(codegen.redKind != kNoReduc);
  codegen.redVal = merger.exp(codegen.redExp).val = reduc;
}

//===----------------------------------------------------------------------===//
// Sparse compiler synthesis methods (statements and expressions).
//===----------------------------------------------------------------------===//

/// Generates buffer for the output tensor. Note that all sparse kernels
/// assume that when all elements are written to (viz. x(i) = y(i) * z(i)),
/// the output buffer is already initialized to all zeroes and only nonzeroes
/// values are computed and written out. For updates (viz. x(i) += y(i) * z(i)),
/// only nonzeroes values are used for the updates and no assumption on the
/// original contents of the output buffer is necessary..
static Value genOutputBuffer(CodeGen &codegen, PatternRewriter &rewriter,
                             linalg::GenericOp op, MemRefType denseTp,
                             ArrayRef<Value> args) {
  Location loc = op.getLoc();
  Value tensor = op.getOutputOperand(0)->get();
  // The output tensor simply could materialize from the buffer that will
  // be generated for the tensor present in the outs() clause. This has
  // the major advantage that the sparse kernel only updates the nonzero
  // positions for the output tensor.
  if (isInPlace(tensor))
    return rewriter.create<bufferization::ToMemrefOp>(loc, denseTp, tensor);
  // By default, a new buffer is allocated which is initialized to the
  // tensor defined in the outs() clause. This is always correct but
  // introduces a dense initialization component that may negatively
  // impact the running complexity of the sparse kernel. If the tensor
  // materializes into the computation, we need to preserve the zero
  // initialization assumption of all sparse output buffers.
  Value alloc = rewriter.create<memref::AllocOp>(loc, denseTp, args);
  if (isMaterializing(tensor)) {
    Value zero = constantZero(rewriter, loc, denseTp.getElementType());
    rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{alloc});
  } else {
    Value init =
        rewriter.create<bufferization::ToMemrefOp>(loc, denseTp, tensor);
    rewriter.create<memref::CopyOp>(loc, init, alloc);
  }
  return alloc;
}

/// Local bufferization of all dense and sparse data structures.
/// This code enables testing the first prototype sparse compiler.
// TODO: replace this with a proliferated bufferization strategy
static void genBuffers(Merger &merger, CodeGen &codegen,
                       PatternRewriter &rewriter, linalg::GenericOp op) {
  Location loc = op.getLoc();
  assert(op.getNumInputsAndOutputs() == op.getNumInputs() + 1);
  // For every tensor, find lower and upper bound on dimensions, set the
  // same bounds on loop indices, and obtain dense or sparse buffer(s).
  SmallVector<Value, 4> args;
  for (OpOperand *t : op.getInputAndOutputOperands()) {
    unsigned tensor = t->getOperandNumber();
    auto shape = op.getShape(t);
    auto map = op.getTiedIndexingMap(t);
    auto enc = getSparseTensorEncoding(t->get().getType());
    // Scan all dimensions of current tensor.
    args.clear();
    for (unsigned d = 0, rank = map.getNumResults(); d < rank; d++) {
      AffineExpr a = map.getResult(perm(enc, d));
      if (a.getKind() != AffineExprKind::DimId)
        continue; // compound
      unsigned idx = a.cast<AffineDimExpr>().getPosition();
      // Handle sparse storage schemes.
      if (merger.isDim(tensor, idx, Dim::kSparse)) {
        auto dynShape = {ShapedType::kDynamicSize};
        auto ptrTp =
            MemRefType::get(dynShape, getPointerOverheadType(rewriter, enc));
        auto indTp =
            MemRefType::get(dynShape, getIndexOverheadType(rewriter, enc));
        Value dim = constantIndex(rewriter, loc, d);
        // Generate sparse primitives to obtains pointer and indices.
        codegen.pointers[tensor][idx] =
            rewriter.create<ToPointersOp>(loc, ptrTp, t->get(), dim);
        codegen.indices[tensor][idx] =
            rewriter.create<ToIndicesOp>(loc, indTp, t->get(), dim);
      }
      // Find upper bound in current dimension.
      unsigned p = perm(enc, d);
      Value up = linalg::createOrFoldDimOp(rewriter, loc, t->get(), p);
      if (ShapedType::isDynamic(shape[p]))
        args.push_back(up);
      assert(codegen.highs[tensor][idx] == nullptr);
      codegen.sizes[idx] = codegen.highs[tensor][idx] = up;
    }
    // Perform the required bufferization. Dense inputs materialize
    // from the input tensors. Dense outputs need special handling.
    // Sparse inputs use sparse primitives to obtain the values.
    // We also accept in-place all-dense annotated "sparse" outputs.
    Type elementType = getElementTypeOrSelf(t->get().getType());
    if (!enc) {
      // Non-annotated dense tensors.
      auto denseTp = MemRefType::get(shape, elementType);
      if (tensor < op.getNumInputs())
        codegen.buffers[tensor] =
            rewriter.create<bufferization::ToMemrefOp>(loc, denseTp, t->get());
      else
        codegen.buffers[tensor] =
            genOutputBuffer(codegen, rewriter, op, denseTp, args);
    } else if (t == codegen.sparseOut) {
      // True sparse output needs a lexIdx array.
      Value rank = constantIndex(rewriter, loc, op.getRank(t));
      auto dynShape = {ShapedType::kDynamicSize};
      auto memTp = MemRefType::get(dynShape, rewriter.getIndexType());
      codegen.lexIdx = rewriter.create<memref::AllocaOp>(loc, memTp, rank);
    } else {
      // Annotated sparse tensors.
      auto dynShape = {ShapedType::kDynamicSize};
      auto sparseTp = MemRefType::get(dynShape, elementType);
      codegen.buffers[tensor] =
          rewriter.create<ToValuesOp>(loc, sparseTp, t->get());
    }
  }
}

/// Constructs vector type.
static VectorType vectorType(CodeGen &codegen, Type etp) {
  unsigned numScalableDims = codegen.options.enableVLAVectorization;
  return VectorType::get(codegen.curVecLength, etp, numScalableDims);
}

/// Constructs vector type from pointer.
static VectorType vectorType(CodeGen &codegen, Value ptr) {
  return vectorType(codegen, ptr.getType().cast<MemRefType>().getElementType());
}

/// Constructs vector iteration mask.
static Value genVectorMask(CodeGen &codegen, PatternRewriter &rewriter,
                           Value iv, Value lo, Value hi, Value step) {
  Location loc = iv.getLoc();
  VectorType mtp = vectorType(codegen, rewriter.getI1Type());
  // Special case if the vector length evenly divides the trip count (for
  // example, "for i = 0, 128, 16"). A constant all-true mask is generated
  // so that all subsequent masked memory operations are immediately folded
  // into unconditional memory operations.
  IntegerAttr loInt, hiInt, stepInt;
  if (matchPattern(lo, m_Constant(&loInt)) &&
      matchPattern(hi, m_Constant(&hiInt)) &&
      matchPattern(step, m_Constant(&stepInt))) {
    if (((hiInt.getInt() - loInt.getInt()) % stepInt.getInt()) == 0)
      return rewriter.create<vector::BroadcastOp>(
          loc, mtp, constantI1(rewriter, loc, true));
  }
  // Otherwise, generate a vector mask that avoids overrunning the upperbound
  // during vector execution. Here we rely on subsequent loop optimizations to
  // avoid executing the mask in all iterations, for example, by splitting the
  // loop into an unconditional vector loop and a scalar cleanup loop.
  auto minMap = AffineMap::get(
      /*dimCount=*/2, /*symbolCount=*/1,
      {rewriter.getAffineSymbolExpr(0),
       rewriter.getAffineDimExpr(0) - rewriter.getAffineDimExpr(1)},
      rewriter.getContext());
  Value end =
      rewriter.createOrFold<AffineMinOp>(loc, minMap, ValueRange{hi, iv, step});
  return rewriter.create<vector::CreateMaskOp>(loc, mtp, end);
}

/// Generates a vectorized load lhs = a[ind[lo:hi]] or lhs = a[lo:hi].
static Value genVectorLoad(CodeGen &codegen, PatternRewriter &rewriter,
                           Value ptr, ArrayRef<Value> args) {
  Location loc = ptr.getLoc();
  VectorType vtp = vectorType(codegen, ptr);
  Value pass = constantZero(rewriter, loc, vtp);
  if (args.back().getType().isa<VectorType>()) {
    SmallVector<Value, 4> scalarArgs(args.begin(), args.end());
    Value indexVec = args.back();
    scalarArgs.back() = constantIndex(rewriter, loc, 0);
    return rewriter.create<vector::GatherOp>(
        loc, vtp, ptr, scalarArgs, indexVec, codegen.curVecMask, pass);
  }
  return rewriter.create<vector::MaskedLoadOp>(loc, vtp, ptr, args,
                                               codegen.curVecMask, pass);
}

/// Generates a vectorized store a[ind[lo:hi]] = rhs or a[lo:hi] = rhs.
static void genVectorStore(CodeGen &codegen, PatternRewriter &rewriter,
                           Value rhs, Value ptr, ArrayRef<Value> args) {
  Location loc = ptr.getLoc();
  if (args.back().getType().isa<VectorType>()) {
    SmallVector<Value, 4> scalarArgs(args.begin(), args.end());
    Value indexVec = args.back();
    scalarArgs.back() = constantIndex(rewriter, loc, 0);
    rewriter.create<vector::ScatterOp>(loc, ptr, scalarArgs, indexVec,
                                       codegen.curVecMask, rhs);
    return;
  }
  rewriter.create<vector::MaskedStoreOp>(loc, ptr, args, codegen.curVecMask,
                                         rhs);
}

/// Generates a vectorized invariant. Here we rely on subsequent loop
/// optimizations to hoist the invariant broadcast out of the vector loop.
static Value genVectorInvariantValue(CodeGen &codegen,
                                     PatternRewriter &rewriter, Value val) {
  VectorType vtp = vectorType(codegen, val.getType());
  return rewriter.create<vector::BroadcastOp>(val.getLoc(), vtp, val);
}

/// Generates an affine expression.
//
// TODO: generalize for sparse tensor subscripts
//
static Value genAffine(CodeGen &codegen, PatternRewriter &rewriter,
                       AffineExpr a, Location loc) {
  switch (a.getKind()) {
  case AffineExprKind::DimId: {
    unsigned idx = a.cast<AffineDimExpr>().getPosition();
    return codegen.loops[idx]; // universal dense index
  }
  case AffineExprKind::Add: {
    auto binOp = a.cast<AffineBinaryOpExpr>();
    return rewriter.create<arith::AddIOp>(
        loc, genAffine(codegen, rewriter, binOp.getLHS(), loc),
        genAffine(codegen, rewriter, binOp.getRHS(), loc));
  }
  case AffineExprKind::Mul: {
    auto binOp = a.cast<AffineBinaryOpExpr>();
    return rewriter.create<arith::MulIOp>(
        loc, genAffine(codegen, rewriter, binOp.getLHS(), loc),
        genAffine(codegen, rewriter, binOp.getRHS(), loc));
  }
  case AffineExprKind::Constant: {
    int64_t c = a.cast<AffineConstantExpr>().getValue();
    return constantIndex(rewriter, loc, c);
  }
  default:
    llvm_unreachable("unexpected affine subscript");
  }
}

/// Generates index for load/store on sparse tensor.
static Value genIndex(CodeGen &codegen, linalg::GenericOp op, OpOperand *t) {
  auto map = op.getTiedIndexingMap(t);
  auto enc = getSparseTensorEncoding(t->get().getType());
  AffineExpr a = map.getResult(perm(enc, map.getNumResults() - 1));
  assert(a.getKind() == AffineExprKind::DimId);
  unsigned idx = a.cast<AffineDimExpr>().getPosition();
  return codegen.loops[idx];
}

/// Generates subscript for load/store on a dense or sparse tensor.
static Value genSubscript(CodeGen &codegen, PatternRewriter &rewriter,
                          linalg::GenericOp op, OpOperand *t,
                          SmallVector<Value, 4> &args) {
  unsigned tensor = t->getOperandNumber();
  auto map = op.getTiedIndexingMap(t);
  auto enc = getSparseTensorEncoding(t->get().getType());
  unsigned rank = map.getNumResults();
  if (enc) {
    // Note that currently, all sparse subscripts are simple.
    // TODO: accept affine too?
    AffineExpr a = map.getResult(perm(enc, rank - 1));
    assert(a.getKind() == AffineExprKind::DimId);
    unsigned idx = a.cast<AffineDimExpr>().getPosition();
    assert(codegen.pidxs[tensor][idx] != nullptr);
    args.push_back(codegen.pidxs[tensor][idx]); // position index
  } else {
    for (unsigned d = 0; d < rank; d++) {
      AffineExpr a = map.getResult(perm(enc, d));
      args.push_back(genAffine(codegen, rewriter, a, op.getLoc()));
    }
  }
  return codegen.buffers[tensor];
}

/// Generates insertion code to implement dynamic tensor load.
static Value genInsertionLoad(CodeGen &codegen, PatternRewriter &rewriter,
                              linalg::GenericOp op, OpOperand *t) {
  Location loc = op.getLoc();
  // Direct lexicographic index order, tensor loads as zero.
  if (!codegen.expValues) {
    Type tp = getElementTypeOrSelf(t->get().getType());
    return constantZero(rewriter, loc, tp);
  }
  // Load from expanded access pattern.
  Value index = genIndex(codegen, op, t);
  return rewriter.create<memref::LoadOp>(loc, codegen.expValues, index);
}

/// Generates insertion code to implement dynamic tensor store.
static void genInsertionStore(CodeGen &codegen, PatternRewriter &rewriter,
                              linalg::GenericOp op, OpOperand *t, Value rhs) {
  Location loc = op.getLoc();
  // Direct insertion in lexicographic index order.
  if (!codegen.expValues) {
    rewriter.create<LexInsertOp>(loc, t->get(), codegen.lexIdx, rhs);
    return;
  }
  // Generates insertion code along expanded access pattern.
  //   if (!expFilled[i]) then
  //     expFilled[i] = true
  //     expAdded[inserts++] = i
  //   endif
  //   values[i] = rhs
  Value index = genIndex(codegen, op, t);
  Value fval = constantI1(rewriter, loc, false);
  Value tval = constantI1(rewriter, loc, true);
  // If statement.
  Value filled = rewriter.create<memref::LoadOp>(loc, codegen.expFilled, index);
  Value cond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                              filled, fval);
  scf::IfOp ifOp = rewriter.create<scf::IfOp>(loc, rewriter.getIndexType(),
                                              cond, /*else=*/true);
  // True branch.
  rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
  rewriter.create<memref::StoreOp>(loc, tval, codegen.expFilled, index);
  rewriter.create<memref::StoreOp>(loc, index, codegen.expAdded,
                                   codegen.expCount);
  Value one = constantIndex(rewriter, loc, 1);
  Value add = rewriter.create<arith::AddIOp>(loc, codegen.expCount, one);
  rewriter.create<scf::YieldOp>(loc, add);
  // False branch.
  rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
  rewriter.create<scf::YieldOp>(loc, codegen.expCount);
  rewriter.setInsertionPointAfter(ifOp);
  // Value assignment.
  codegen.expCount = ifOp.getResult(0);
  rewriter.create<memref::StoreOp>(loc, rhs, codegen.expValues, index);
}

/// Generates a load on a dense or sparse tensor.
static Value genTensorLoad(Merger &merger, CodeGen &codegen,
                           PatternRewriter &rewriter, linalg::GenericOp op,
                           unsigned exp) {
  // Test if the load was hoisted to a higher loop nest.
  Value val = merger.exp(exp).val;
  if (val) {
    if (codegen.curVecLength > 1 && !val.getType().isa<VectorType>())
      return genVectorInvariantValue(codegen, rewriter, val);
    return val;
  }
  // Load during insertion.
  OpOperand *t = op.getInputAndOutputOperands()[merger.exp(exp).tensor];
  if (t == codegen.sparseOut)
    return genInsertionLoad(codegen, rewriter, op, t);
  // Actual load.
  SmallVector<Value, 4> args;
  Value ptr = genSubscript(codegen, rewriter, op, t, args);
  if (codegen.curVecLength > 1)
    return genVectorLoad(codegen, rewriter, ptr, args);
  return rewriter.create<memref::LoadOp>(op.getLoc(), ptr, args);
}

/// Generates a store on a dense or sparse tensor.
static void genTensorStore(Merger &merger, CodeGen &codegen,
                           PatternRewriter &rewriter, linalg::GenericOp op,
                           unsigned exp, Value rhs) {
  Location loc = op.getLoc();
  // Test if this is a scalarized reduction.
  if (codegen.redVal) {
    if (codegen.curVecLength > 1)
      rhs = rewriter.create<arith::SelectOp>(loc, codegen.curVecMask, rhs,
                                             codegen.redVal);
    updateReduc(merger, codegen, rhs);
    return;
  }
  // Store during insertion.
  OpOperand *t = op.getOutputOperand(0);
  if (t == codegen.sparseOut) {
    if (!rhs) {
      // Only unary and binary are allowed to return uninitialized rhs
      // to indicate missing output.
      assert(merger.exp(exp).kind == kUnary || merger.exp(exp).kind == kBinary);
    } else {
      genInsertionStore(codegen, rewriter, op, t, rhs);
    }
    return;
  }
  // Actual store.
  SmallVector<Value, 4> args;
  Value ptr = genSubscript(codegen, rewriter, op, t, args);
  if (codegen.curVecLength > 1)
    genVectorStore(codegen, rewriter, rhs, ptr, args);
  else
    rewriter.create<memref::StoreOp>(loc, rhs, ptr, args);
}

/// Generates a pointer/index load from the sparse storage scheme. Narrower
/// data types need to be zero extended before casting the value into the
/// index type used for looping and indexing.
static Value genLoad(CodeGen &codegen, PatternRewriter &rewriter, Location loc,
                     Value ptr, Value s) {
  // See https://llvm.org/docs/GetElementPtr.html for some background on
  // the complications described below.
  if (codegen.curVecLength > 1) {
    // Since the index vector is used in a subsequent gather/scatter operations,
    // which effectively defines an unsigned pointer + signed index, we must
    // zero extend the vector to an index width. For 8-bit and 16-bit values,
    // an 32-bit index width suffices. For 32-bit values, zero extending the
    // elements into 64-bit loses some performance since the 32-bit indexed
    // gather/scatter is more efficient than the 64-bit index variant (if the
    // negative 32-bit index space is unused, the enableSIMDIndex32 flag can
    // preserve this performance). For 64-bit values, there is no good way
    // to state that the indices are unsigned, with creates the potential of
    // incorrect address calculations in the unlikely case we need such
    // extremely large offsets.
    Type etp = ptr.getType().cast<MemRefType>().getElementType();
    Value vload = genVectorLoad(codegen, rewriter, ptr, {s});
    if (!etp.isa<IndexType>()) {
      if (etp.getIntOrFloatBitWidth() < 32)
        vload = rewriter.create<arith::ExtUIOp>(
            loc, vectorType(codegen, rewriter.getI32Type()), vload);
      else if (etp.getIntOrFloatBitWidth() < 64 &&
               !codegen.options.enableSIMDIndex32)
        vload = rewriter.create<arith::ExtUIOp>(
            loc, vectorType(codegen, rewriter.getI64Type()), vload);
    }
    return vload;
  }
  // For the scalar case, we simply zero extend narrower indices into 64-bit
  // values before casting to index without a performance penalty. Here too,
  // however, indices that already are 64-bit, in theory, cannot express the
  // full range as explained above.
  Value load = rewriter.create<memref::LoadOp>(loc, ptr, s);
  if (!load.getType().isa<IndexType>()) {
    if (load.getType().getIntOrFloatBitWidth() < 64)
      load = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), load);
    load =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), load);
  }
  return load;
}

/// Generates an invariant value.
static Value genInvariantValue(Merger &merger, CodeGen &codegen,
                               PatternRewriter &rewriter, unsigned exp) {
  Value val = merger.exp(exp).val;
  if (codegen.curVecLength > 1)
    return genVectorInvariantValue(codegen, rewriter, val);
  return val;
}

/// Generates an address computation "sz * p + i".
static Value genAddress(CodeGen &codegen, PatternRewriter &rewriter,
                        Location loc, Value size, Value p, Value i) {
  Value mul = rewriter.create<arith::MulIOp>(loc, size, p);
  if (auto vtp = i.getType().dyn_cast<VectorType>()) {
    Value inv =
        rewriter.create<arith::IndexCastOp>(loc, vtp.getElementType(), mul);
    mul = genVectorInvariantValue(codegen, rewriter, inv);
  }
  return rewriter.create<arith::AddIOp>(loc, mul, i);
}

/// Generates an index value.
static Value genIndexValue(Merger &merger, CodeGen &codegen,
                           PatternRewriter &rewriter, unsigned exp,
                           unsigned ldx) {
  unsigned idx = merger.exp(exp).index;
  Value ival = codegen.loops[idx];
  Type itype = ival.getType();
  // During vectorization, we either encounter:
  // (1) indices already in vector form, as in ... = ind[lo:hi], good to go, or
  // (2) single index, as in ... = i, must convert to [i, i+1, ...] for inner i.
  unsigned vl = codegen.curVecLength;
  if (vl > 1 && !itype.isa<VectorType>()) {
    Location loc = ival.getLoc();
    VectorType vtp = vectorType(codegen, itype);
    ival = rewriter.create<vector::BroadcastOp>(loc, vtp, ival);
    if (idx == ldx) {
      Value incr;
      if (vtp.isScalable()) {
        Type stepvty = vectorType(codegen, rewriter.getI64Type());
        Value stepv = rewriter.create<LLVM::StepVectorOp>(loc, stepvty);
        incr = rewriter.create<arith::IndexCastOp>(loc, vtp, stepv);
      } else {
        SmallVector<APInt, 4> integers;
        for (unsigned i = 0; i < vl; i++)
          integers.push_back(APInt(/*width=*/64, i));
        auto values = DenseElementsAttr::get(vtp, integers);
        incr = rewriter.create<arith::ConstantOp>(loc, vtp, values);
      }
      ival = rewriter.create<arith::AddIOp>(loc, ival, incr);
    }
  }
  return ival;
}

/// Recursively generates tensor expression.
static Value genExp(Merger &merger, CodeGen &codegen, PatternRewriter &rewriter,
                    linalg::GenericOp op, unsigned exp, unsigned ldx) {
  Location loc = op.getLoc();
  if (exp == -1u)
    return Value();
  if (merger.exp(exp).kind == Kind::kTensor)
    return genTensorLoad(merger, codegen, rewriter, op, exp);
  if (merger.exp(exp).kind == Kind::kInvariant)
    return genInvariantValue(merger, codegen, rewriter, exp);
  if (merger.exp(exp).kind == Kind::kIndex)
    return genIndexValue(merger, codegen, rewriter, exp, ldx);
  Value v0 =
      genExp(merger, codegen, rewriter, op, merger.exp(exp).children.e0, ldx);
  Value v1 =
      genExp(merger, codegen, rewriter, op, merger.exp(exp).children.e1, ldx);
  return merger.buildExp(rewriter, loc, exp, v0, v1);
}

/// Determines if affine expression is invariant.
static bool isInvariantAffine(const CodeGen &codegen, AffineExpr a,
                              unsigned ldx, bool &atLevel) {
  switch (a.getKind()) {
  case AffineExprKind::DimId: {
    unsigned idx = a.cast<AffineDimExpr>().getPosition();
    if (idx == ldx)
      atLevel = true;
    return codegen.loops[idx] != nullptr; // no longer in play?
  }
  case AffineExprKind::Add:
  case AffineExprKind::Mul: {
    auto binOp = a.cast<AffineBinaryOpExpr>();
    return isInvariantAffine(codegen, binOp.getLHS(), ldx, atLevel) &&
           isInvariantAffine(codegen, binOp.getRHS(), ldx, atLevel);
  }
  default:
    return true;
  }
}

/// Hoists loop invariant tensor loads for which indices have been exhausted.
static void genInvariants(Merger &merger, CodeGen &codegen,
                          PatternRewriter &rewriter, linalg::GenericOp op,
                          unsigned exp, unsigned ldx, bool atStart,
                          Kind last = Kind::kTensor) {
  if (exp == -1u)
    return;
  if (merger.exp(exp).kind == Kind::kTensor) {
    // Inspect tensor indices.
    bool atLevel = ldx == -1u;
    OpOperand *t = op.getInputAndOutputOperands()[merger.exp(exp).tensor];
    auto map = op.getTiedIndexingMap(t);
    auto enc = getSparseTensorEncoding(t->get().getType());
    for (unsigned d = 0, rank = map.getNumResults(); d < rank; d++) {
      AffineExpr a = map.getResult(perm(enc, d));
      if (!isInvariantAffine(codegen, a, ldx, atLevel))
        return; // still in play
    }
    // All exhausted at this level (atLevel denotes exactly at this level).
    if (!atLevel)
      return;
    OpOperand *lhs = op.getOutputOperand(0);
    if (lhs == t) {
      // Start or end a scalarized reduction
      if (atStart) {
        Value load = genTensorLoad(merger, codegen, rewriter, op, exp);
        codegen.redKind = getReduction(last);
        codegen.redExp = exp;
        updateReduc(merger, codegen, load);
      } else {
        Value redVal = codegen.redVal;
        updateReduc(merger, codegen, Value());
        codegen.redExp = -1u;
        codegen.redKind = kNoReduc;
        genTensorStore(merger, codegen, rewriter, op, exp, redVal);
      }
    } else {
      // Start or end loop invariant hoisting of a tensor load.
      merger.exp(exp).val =
          atStart ? genTensorLoad(merger, codegen, rewriter, op, exp) : Value();
    }
  } else if (merger.exp(exp).kind != Kind::kInvariant &&
             merger.exp(exp).kind != Kind::kIndex) {
    // Traverse into the binary operations. Note that we only hoist
    // tensor loads, since subsequent MLIR/LLVM passes know how to
    // deal with all other kinds of derived loop invariants.
    Kind last = merger.exp(exp).kind;
    unsigned e0 = merger.exp(exp).children.e0;
    unsigned e1 = merger.exp(exp).children.e1;
    genInvariants(merger, codegen, rewriter, op, e0, ldx, atStart, last);
    genInvariants(merger, codegen, rewriter, op, e1, ldx, atStart, last);
  }
}

/// Generates an expanded access pattern in innermost dimension.
static void genExpansion(Merger &merger, CodeGen &codegen,
                         PatternRewriter &rewriter, linalg::GenericOp op,
                         unsigned at, bool atStart) {
  OpOperand *lhs = codegen.sparseOut;
  if (!lhs || codegen.outerParNest != op.getRank(lhs) - 1 ||
      at != codegen.outerParNest)
    return; // not needed at this level
  // Generate start or end of an expanded access pattern.
  Value tensor = lhs->get();
  Location loc = op.getLoc();
  if (atStart) {
    auto dynShape = {ShapedType::kDynamicSize};
    Type etp = tensor.getType().cast<ShapedType>().getElementType();
    Type t1 = MemRefType::get(dynShape, etp);
    Type t2 = MemRefType::get(dynShape, rewriter.getI1Type());
    Type t3 = MemRefType::get(dynShape, rewriter.getIndexType());
    Type t4 = rewriter.getIndexType();
    auto res =
        rewriter.create<ExpandOp>(loc, TypeRange({t1, t2, t3, t4}), tensor);
    assert(res.getNumResults() == 4);
    assert(!codegen.expValues);
    codegen.expValues = res.getResult(0);
    codegen.expFilled = res.getResult(1);
    codegen.expAdded = res.getResult(2);
    codegen.expCount = res.getResult(3);
  } else {
    assert(codegen.expValues);
    rewriter.create<CompressOp>(loc, tensor, codegen.lexIdx, codegen.expValues,
                                codegen.expFilled, codegen.expAdded,
                                codegen.expCount);
    codegen.expValues = codegen.expFilled = codegen.expAdded =
        codegen.expCount = Value();
  }
}

/// Generates initialization code for the subsequent loop sequence at
/// current index level. Returns true if the loop sequence needs to
/// maintain the universal index.
static bool genInit(Merger &merger, CodeGen &codegen, PatternRewriter &rewriter,
                    linalg::GenericOp op, std::vector<unsigned> &topSort,
                    unsigned at, BitVector &inits) {
  bool needsUniv = false;
  Location loc = op.getLoc();
  unsigned idx = topSort[at];

  // Initialize sparse positions.
  for (unsigned b = 0, be = inits.size(); b < be; b++) {
    if (inits[b]) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      if (merger.isDim(b, Dim::kSparse)) {
        // Initialize sparse index.
        unsigned pat = at;
        for (; pat != 0; pat--) {
          if (codegen.pidxs[tensor][topSort[pat - 1]])
            break;
        }
        Value ptr = codegen.pointers[tensor][idx];
        Value one = constantIndex(rewriter, loc, 1);
        Value p0 = (pat == 0) ? constantIndex(rewriter, loc, 0)
                              : codegen.pidxs[tensor][topSort[pat - 1]];
        codegen.pidxs[tensor][idx] = genLoad(codegen, rewriter, loc, ptr, p0);
        Value p1 = rewriter.create<arith::AddIOp>(loc, p0, one);
        codegen.highs[tensor][idx] = genLoad(codegen, rewriter, loc, ptr, p1);
      } else {
        // Dense index still in play.
        needsUniv = true;
      }
    }
  }

  // Initialize the universal dense index.
  codegen.loops[idx] = constantIndex(rewriter, loc, 0);
  return needsUniv;
}

/// Returns vectorization strategy. Any implicit inner loop in the Linalg
/// operation is a candidate. Whether it is actually converted to SIMD code
/// depends on the requested strategy.
static bool isVectorFor(CodeGen &codegen, bool isInner, bool isReduction,
                        bool isSparse) {
  // Reject vectorization of sparse output, unless innermost is reduction.
  if (codegen.sparseOut && !isReduction)
    return false;
  // Inspect strategy.
  switch (codegen.options.vectorizationStrategy) {
  case SparseVectorizationStrategy::kNone:
    return false;
  case SparseVectorizationStrategy::kDenseInnerLoop:
    return isInner && !isSparse;
  case SparseVectorizationStrategy::kAnyStorageInnerLoop:
    return isInner;
  }
  llvm_unreachable("unexpected vectorization strategy");
}

/// Returns parallelization strategy. Any implicit loop in the Linalg operation
/// that is marked "parallel" is a candidate. Whether it is actually converted
/// to a parallel operation depends on the requested strategy.
static bool isParallelFor(CodeGen &codegen, bool isOuter, bool isReduction,
                          bool isSparse, bool isVector) {
  // Reject parallelization of sparse output.
  if (codegen.sparseOut)
    return false;
  // Inspect strategy.
  switch (codegen.options.parallelizationStrategy) {
  case SparseParallelizationStrategy::kNone:
    return false;
  case SparseParallelizationStrategy::kDenseOuterLoop:
    return isOuter && !isSparse && !isReduction && !isVector;
  case SparseParallelizationStrategy::kAnyStorageOuterLoop:
    return isOuter && !isReduction && !isVector;
  case SparseParallelizationStrategy::kDenseAnyLoop:
    return !isSparse && !isReduction && !isVector;
  case SparseParallelizationStrategy::kAnyStorageAnyLoop:
    return !isReduction && !isVector;
  }
  llvm_unreachable("unexpected parallelization strategy");
}

/// Checks unit stride for dense tensors. The iteration graph may have ignored
/// dense access patterns in order to avoid cycles (sparse access patterns are
/// always placed innermost), but that means dense access has become strided.
/// This prevents effective vectorization.
static bool denseUnitStrides(Merger &merger, linalg::GenericOp op,
                             unsigned idx) {
  for (OpOperand *t : op.getInputAndOutputOperands()) {
    if (!getSparseTensorEncoding(t->get().getType())) {
      auto map = op.getTiedIndexingMap(t);
      for (unsigned d = 0, rank = map.getNumResults(); d < rank; d++) {
        AffineExpr a = map.getResult(d);
        // Report non-unit stride if innermost index appears at an outer
        // dimension (true non-unit stride) or if the innermost index appears
        // in a compound subscript in the innermost dimension. Even if the
        // latter is unit stride, it does not play well with scatter/gather.
        // TODO: accept unit stride affine innermost like a[i,j+k+1]?
        if (a.isFunctionOfDim(idx) &&
            ((d != rank - 1) || (a.getKind() != AffineExprKind::DimId)))
          return false;
      }
    }
  }
  return true;
}

/// Generates a for-loop on a single index.
static Operation *genFor(Merger &merger, CodeGen &codegen,
                         PatternRewriter &rewriter, linalg::GenericOp op,
                         bool isOuter, bool isInner, unsigned idx,
                         BitVector &indices) {
  unsigned fb = indices.find_first();
  unsigned tensor = merger.tensor(fb);
  assert(idx == merger.index(fb));
  auto iteratorTypes = op.iterator_types().getValue();
  bool isReduction = isReductionIterator(iteratorTypes[idx]);
  bool isSparse = merger.isDim(fb, Dim::kSparse);
  bool isVector = isVectorFor(codegen, isInner, isReduction, isSparse) &&
                  denseUnitStrides(merger, op, idx);
  bool isParallel =
      isParallelFor(codegen, isOuter, isReduction, isSparse, isVector);

  // Prepare vector length.
  if (isVector)
    codegen.curVecLength = codegen.options.vectorLength;

  // Loop bounds and increment.
  Location loc = op.getLoc();
  Value lo = isSparse ? codegen.pidxs[tensor][idx] : codegen.loops[idx];
  Value hi = isSparse ? codegen.highs[tensor][idx] : codegen.sizes[idx];
  Value step = constantIndex(rewriter, loc, codegen.curVecLength);
  if (isVector && codegen.options.enableVLAVectorization) {
    Value vscale = rewriter.create<vector::VectorScaleOp>(
        loc, IndexType::get(rewriter.getContext()));
    step = rewriter.create<arith::MulIOp>(loc, vscale, step);
  }

  // Emit a parallel loop.
  if (isParallel) {
    assert(!isVector);
    scf::ParallelOp parOp = rewriter.create<scf::ParallelOp>(loc, lo, hi, step);
    if (isSparse)
      codegen.pidxs[tensor][idx] = parOp.getInductionVars()[0];
    else
      codegen.loops[idx] = parOp.getInductionVars()[0];
    rewriter.setInsertionPointToStart(parOp.getBody());
    return parOp;
  }

  // Emit a sequential or vector loop.
  SmallVector<Value, 4> operands;
  if (codegen.redVal) {
    // In a vector loop, bring reduction into SIMD form, if not already.
    if (isVector && !codegen.redVal.getType().isa<VectorType>()) {
      VectorType vtp = vectorType(codegen, codegen.redVal.getType());
      Value vred = genVectorReducInit(codegen, rewriter, loc, vtp);
      updateReduc(merger, codegen, vred);
    }
    operands.push_back(codegen.redVal);
  }
  if (codegen.expValues)
    operands.push_back(codegen.expCount);
  scf::ForOp forOp = rewriter.create<scf::ForOp>(loc, lo, hi, step, operands);
  if (codegen.redVal)
    updateReduc(merger, codegen, forOp.getRegionIterArgs().front());
  if (codegen.expValues)
    codegen.expCount = forOp.getRegionIterArgs().back();
  // Assign induction variable to sparse or dense index.
  Value iv = forOp.getInductionVar();
  if (isSparse)
    codegen.pidxs[tensor][idx] = iv;
  else
    codegen.loops[idx] = iv;
  rewriter.setInsertionPointToStart(forOp.getBody());
  // Share vector iteration mask between all subsequent loads/stores.
  if (isVector)
    codegen.curVecMask = genVectorMask(codegen, rewriter, iv, lo, hi, step);
  return forOp;
}

/// Emit a while-loop for co-iteration over multiple indices.
static Operation *genWhile(Merger &merger, CodeGen &codegen,
                           PatternRewriter &rewriter, linalg::GenericOp op,
                           unsigned idx, bool needsUniv, BitVector &indices) {
  SmallVector<Type, 4> types;
  SmallVector<Value, 4> operands;
  // Construct the while-loop with a parameter for each index.
  Type indexType = rewriter.getIndexType();
  for (unsigned b = 0, be = indices.size(); b < be; b++) {
    if (indices[b] && merger.isDim(b, Dim::kSparse)) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      types.push_back(indexType);
      operands.push_back(codegen.pidxs[tensor][idx]);
    }
  }
  if (codegen.redVal) {
    types.push_back(codegen.redVal.getType());
    operands.push_back(codegen.redVal);
  }
  if (codegen.expValues) {
    types.push_back(indexType);
    operands.push_back(codegen.expCount);
  }
  if (needsUniv) {
    types.push_back(indexType);
    operands.push_back(codegen.loops[idx]);
  }
  assert(types.size() == operands.size());
  Location loc = op.getLoc();
  scf::WhileOp whileOp = rewriter.create<scf::WhileOp>(loc, types, operands);

  SmallVector<Location> locs(types.size(), loc);
  Block *before = rewriter.createBlock(&whileOp.getBefore(), {}, types, locs);
  Block *after = rewriter.createBlock(&whileOp.getAfter(), {}, types, locs);

  // Build the "before" region, which effectively consists
  // of a conjunction of "i < upper" tests on all induction.
  rewriter.setInsertionPointToStart(&whileOp.getBefore().front());
  Value cond;
  unsigned o = 0;
  for (unsigned b = 0, be = indices.size(); b < be; b++) {
    if (indices[b] && merger.isDim(b, Dim::kSparse)) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      Value op1 = before->getArgument(o);
      Value op2 = codegen.highs[tensor][idx];
      Value opc = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                                 op1, op2);
      cond = cond ? rewriter.create<arith::AndIOp>(loc, cond, opc) : opc;
      codegen.pidxs[tensor][idx] = after->getArgument(o++);
    }
  }
  if (codegen.redVal)
    updateReduc(merger, codegen, after->getArgument(o++));
  if (codegen.expValues)
    codegen.expCount = after->getArgument(o++);
  if (needsUniv)
    codegen.loops[idx] = after->getArgument(o++);
  assert(o == operands.size());
  rewriter.create<scf::ConditionOp>(loc, cond, before->getArguments());
  rewriter.setInsertionPointToStart(&whileOp.getAfter().front());
  return whileOp;
}

/// Generates a for-loop or a while-loop, depending on whether it implements
/// singleton iteration or co-iteration over the given conjunction.
static Operation *genLoop(Merger &merger, CodeGen &codegen,
                          PatternRewriter &rewriter, linalg::GenericOp op,
                          std::vector<unsigned> &topSort, unsigned at,
                          bool needsUniv, BitVector &indices) {
  unsigned idx = topSort[at];
  if (indices.count() == 1) {
    bool isOuter = at == 0;
    bool isInner = at == topSort.size() - 1;
    return genFor(merger, codegen, rewriter, op, isOuter, isInner, idx,
                  indices);
  }
  return genWhile(merger, codegen, rewriter, op, idx, needsUniv, indices);
}

/// Generates the local variables for this loop, consisting of the sparse
/// indices, restored universal dense index, and dense positions.
static void genLocals(Merger &merger, CodeGen &codegen,
                      PatternRewriter &rewriter, linalg::GenericOp op,
                      std::vector<unsigned> &topSort, unsigned at,
                      bool needsUniv, BitVector &locals) {
  Location loc = op.getLoc();
  unsigned idx = topSort[at];

  // Initialize sparse indices.
  Value min;
  for (unsigned b = 0, be = locals.size(); b < be; b++) {
    if (locals[b] && merger.isDim(b, Dim::kSparse)) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      Value ptr = codegen.indices[tensor][idx];
      Value s = codegen.pidxs[tensor][idx];
      Value load = genLoad(codegen, rewriter, loc, ptr, s);
      codegen.idxs[tensor][idx] = load;
      if (!needsUniv) {
        if (min) {
          Value cmp = rewriter.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::ult, load, min);
          min = rewriter.create<arith::SelectOp>(loc, cmp, load, min);
        } else {
          min = load;
        }
      }
    }
  }

  // Merge dense universal index over minimum.
  if (min) {
    assert(!needsUniv);
    codegen.loops[idx] = min;
  }

  // Initialize dense positions. Note that we generate dense indices of the
  // output tensor unconditionally, since they may not appear in the lattice,
  // but may be needed for linearized codegen.
  for (unsigned b = 0, be = locals.size(); b < be; b++) {
    if ((locals[b] || merger.isOutTensor(b, idx)) &&
        merger.isDim(b, Dim::kDense)) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      unsigned pat = at;
      for (; pat != 0; pat--)
        if (codegen.pidxs[tensor][topSort[pat - 1]])
          break;
      Value p = (pat == 0) ? constantIndex(rewriter, loc, 0)
                           : codegen.pidxs[tensor][topSort[pat - 1]];
      codegen.pidxs[tensor][idx] = genAddress(
          codegen, rewriter, loc, codegen.sizes[idx], p, codegen.loops[idx]);
    }
  }

  // Move the insertion indices in lexicographic index order. During access
  // pattern expansion, we can skip setting the innermost dimension.
  if (codegen.sparseOut && !codegen.expValues) {
    Value pos = constantIndex(rewriter, loc, at);
    rewriter.create<memref::StoreOp>(loc, codegen.loops[idx], codegen.lexIdx,
                                     pos);
  }
}

/// Generates the induction structure for a while-loop.
static void genWhileInduction(Merger &merger, CodeGen &codegen,
                              PatternRewriter &rewriter, linalg::GenericOp op,
                              unsigned idx, bool needsUniv,
                              BitVector &induction, scf::WhileOp whileOp) {
  Location loc = op.getLoc();
  // Finalize each else branch of all if statements.
  if (codegen.redVal || codegen.expValues) {
    while (auto ifOp = dyn_cast_or_null<scf::IfOp>(
               rewriter.getInsertionBlock()->getParentOp())) {
      unsigned y = 0;
      SmallVector<Value, 4> yields;
      if (codegen.redVal) {
        yields.push_back(codegen.redVal);
        updateReduc(merger, codegen, ifOp.getResult(y++));
      }
      if (codegen.expValues) {
        yields.push_back(codegen.expCount);
        codegen.expCount = ifOp->getResult(y++);
      }
      assert(y == yields.size());
      rewriter.create<scf::YieldOp>(loc, yields);
      rewriter.setInsertionPointAfter(ifOp);
    }
  }
  rewriter.setInsertionPointToEnd(&whileOp.getAfter().front());
  // Finalize the induction. Note that the induction could be performed
  // in the individual if-branches to avoid re-evaluating the conditions.
  // However, that would result in a rather elaborate forest of yield
  // instructions during code generation. Moreover, performing the induction
  // after the if-statements more closely resembles code generated by TACO.
  unsigned o = 0;
  SmallVector<Value, 4> operands;
  Value one = constantIndex(rewriter, loc, 1);
  for (unsigned b = 0, be = induction.size(); b < be; b++) {
    if (induction[b] && merger.isDim(b, Dim::kSparse)) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      Value op1 = codegen.idxs[tensor][idx];
      Value op2 = codegen.loops[idx];
      Value op3 = codegen.pidxs[tensor][idx];
      Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                 op1, op2);
      Value add = rewriter.create<arith::AddIOp>(loc, op3, one);
      operands.push_back(rewriter.create<arith::SelectOp>(loc, cmp, add, op3));
      codegen.pidxs[tensor][idx] = whileOp->getResult(o++);
    }
  }
  if (codegen.redVal) {
    operands.push_back(codegen.redVal);
    updateReduc(merger, codegen, whileOp->getResult(o++));
  }
  if (codegen.expValues) {
    operands.push_back(codegen.expCount);
    codegen.expCount = whileOp->getResult(o++);
  }
  if (needsUniv) {
    operands.push_back(
        rewriter.create<arith::AddIOp>(loc, codegen.loops[idx], one));
    codegen.loops[idx] = whileOp->getResult(o++);
  }
  assert(o == operands.size());
  rewriter.create<scf::YieldOp>(loc, operands);
  rewriter.setInsertionPointAfter(whileOp);
}

/// Generates the induction structure for a for-loop.
static void genForInduction(Merger &merger, CodeGen &codegen,
                            PatternRewriter &rewriter, linalg::GenericOp op,
                            Operation *loop) {
  Location loc = op.getLoc();
  unsigned o = 0;
  SmallVector<Value, 4> operands;
  if (codegen.redVal) {
    operands.push_back(codegen.redVal);
    updateReduc(merger, codegen, loop->getResult(o++));
  }
  if (codegen.expValues) {
    operands.push_back(codegen.expCount);
    codegen.expCount = loop->getResult(o++);
  }
  assert(o == operands.size());
  if (o > 0)
    rewriter.create<scf::YieldOp>(loc, operands);
  rewriter.setInsertionPointAfter(loop);
}

/// Generates a single if-statement within a while-loop.
static scf::IfOp genIf(Merger &merger, CodeGen &codegen,
                       PatternRewriter &rewriter, linalg::GenericOp op,
                       unsigned idx, BitVector &conditions) {
  Location loc = op.getLoc();
  SmallVector<Type, 4> types;
  Value cond;
  for (unsigned b = 0, be = conditions.size(); b < be; b++) {
    if (conditions[b]) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      Value clause;
      if (merger.isDim(b, Dim::kSparse)) {
        Value op1 = codegen.idxs[tensor][idx];
        Value op2 = codegen.loops[idx];
        clause = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                op1, op2);
      } else {
        clause = constantI1(rewriter, loc, true);
      }
      cond = cond ? rewriter.create<arith::AndIOp>(loc, cond, clause) : clause;
    }
  }
  if (codegen.redVal)
    types.push_back(codegen.redVal.getType());
  if (codegen.expValues)
    types.push_back(rewriter.getIndexType());
  scf::IfOp ifOp = rewriter.create<scf::IfOp>(loc, types, cond, /*else=*/true);
  rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
  return ifOp;
}

/// Generates end of true branch of if-statement within a while-loop.
static void endIf(Merger &merger, CodeGen &codegen, PatternRewriter &rewriter,
                  linalg::GenericOp op, scf::IfOp ifOp, Operation *loop,
                  Value redInput, Value cntInput) {
  SmallVector<Value, 4> operands;
  if (codegen.redVal) {
    operands.push_back(codegen.redVal);
    updateReduc(merger, codegen, redInput);
  }
  if (codegen.expValues) {
    operands.push_back(codegen.expCount);
    codegen.expCount = cntInput;
  }
  if (!operands.empty())
    rewriter.create<scf::YieldOp>(op.getLoc(), operands);
  rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
}

//===----------------------------------------------------------------------===//
// Sparse compiler synthesis methods (loop sequence).
//===----------------------------------------------------------------------===//

/// Starts a loop sequence at given level. Returns true if
/// the universal loop index must be maintained at this level.
static bool startLoopSeq(Merger &merger, CodeGen &codegen,
                         PatternRewriter &rewriter, linalg::GenericOp op,
                         std::vector<unsigned> &topSort, unsigned exp,
                         unsigned at, unsigned idx, unsigned ldx,
                         unsigned lts) {
  assert(codegen.curVecLength == 1);
  assert(!codegen.loops[idx]);
  // Emit invariants at this loop sequence level.
  genInvariants(merger, codegen, rewriter, op, exp, ldx, /*atStart=*/true);
  // Emit access pattern expansion for sparse tensor output.
  genExpansion(merger, codegen, rewriter, op, at, /*atStart=*/true);
  // Emit further intitialization at this loop sequence level.
  unsigned l0 = merger.set(lts)[0];
  bool needsUniv =
      genInit(merger, codegen, rewriter, op, topSort, at, merger.lat(l0).bits);
  // Maintain the universal index only if it is actually
  // consumed by a subsequent lattice point.
  if (needsUniv) {
    unsigned lsize = merger.set(lts).size();
    for (unsigned i = 1; i < lsize; i++) {
      unsigned li = merger.set(lts)[i];
      if (!merger.hasAnyDimOf(merger.lat(li).simple, Dim::kSparse))
        return true;
    }
  }
  return false;
}

/// Starts a single loop in current sequence.
static Operation *startLoop(Merger &merger, CodeGen &codegen,
                            PatternRewriter &rewriter, linalg::GenericOp op,
                            std::vector<unsigned> &topSort, unsigned at,
                            unsigned li, bool needsUniv) {
  assert(codegen.curVecLength == 1);
  // Emit the for/while-loop control.
  Operation *loop = genLoop(merger, codegen, rewriter, op, topSort, at,
                            needsUniv, merger.lat(li).simple);
  // Emit the locals for this loop.
  genLocals(merger, codegen, rewriter, op, topSort, at, needsUniv,
            merger.lat(li).bits);
  return loop;
}

/// Ends a single loop in current sequence. Returns new values for needsUniv.
static bool endLoop(Merger &merger, CodeGen &codegen, PatternRewriter &rewriter,
                    linalg::GenericOp op, Operation *loop, unsigned idx,
                    unsigned li, bool needsUniv) {
  codegen.curVecLength = 1;
  // End a while-loop.
  if (auto whileOp = dyn_cast<scf::WhileOp>(loop)) {
    genWhileInduction(merger, codegen, rewriter, op, idx, needsUniv,
                      merger.lat(li).bits, whileOp);
    return needsUniv;
  }
  // End a for-loop.
  genForInduction(merger, codegen, rewriter, op, loop);
  return false;
}

/// Ends a loop sequence at given level.
static void endLoopSeq(Merger &merger, CodeGen &codegen,
                       PatternRewriter &rewriter, linalg::GenericOp op,
                       unsigned exp, unsigned at, unsigned idx, unsigned ldx) {
  assert(codegen.curVecLength == 1);
  codegen.loops[idx] = Value();
  // Bring a pending reduction back from SIMD form when sequence ends.
  if (codegen.redVal)
    if (auto vtp = codegen.redVal.getType().dyn_cast<VectorType>())
      updateReduc(merger, codegen,
                  genVectorReducEnd(codegen, rewriter, op.getLoc(), vtp));
  // Unmark bookkeeping of invariants and loop index.
  genInvariants(merger, codegen, rewriter, op, exp, ldx, /*atStart=*/false);
  // Finalize access pattern expansion for sparse tensor output.
  genExpansion(merger, codegen, rewriter, op, at, /*atStart=*/false);
}

/// Recursively generates code while computing iteration lattices in order
/// to manage the complexity of implementing co-iteration over unions
/// and intersections of sparse iterations spaces.
static void genStmt(Merger &merger, CodeGen &codegen, PatternRewriter &rewriter,
                    linalg::GenericOp op, std::vector<unsigned> &topSort,
                    unsigned exp, unsigned at) {
  // At each leaf, assign remaining tensor (sub)expression to output tensor.
  if (at == topSort.size()) {
    unsigned ldx = topSort[at - 1];
    Value rhs = genExp(merger, codegen, rewriter, op, exp, ldx);
    genTensorStore(merger, codegen, rewriter, op, exp, rhs);
    return;
  }

  // Construct iteration lattices for current loop index, with L0 at top.
  unsigned idx = topSort[at];
  unsigned ldx = at == 0 ? -1u : topSort[at - 1];
  unsigned lts = merger.optimizeSet(merger.buildLattices(exp, idx));

  // Start a loop sequence.
  bool needsUniv = startLoopSeq(merger, codegen, rewriter, op, topSort, exp, at,
                                idx, ldx, lts);

  // Emit a loop for every lattice point L0 >= Li in this loop sequence.
  unsigned lsize = merger.set(lts).size();
  for (unsigned i = 0; i < lsize; i++) {
    // Start a loop.
    unsigned li = merger.set(lts)[i];
    Operation *loop =
        startLoop(merger, codegen, rewriter, op, topSort, at, li, needsUniv);

    // Visit all lattices points with Li >= Lj to generate the
    // loop-body, possibly with if statements for coiteration.
    Value redInput = codegen.redVal;
    Value cntInput = codegen.expCount;
    bool isWhile = dyn_cast<scf::WhileOp>(loop) != nullptr;
    for (unsigned j = 0; j < lsize; j++) {
      unsigned lj = merger.set(lts)[j];
      unsigned ej = merger.lat(lj).exp;
      if (li == lj || merger.latGT(li, lj)) {
        // Recurse into body of each branch.
        if (isWhile) {
          scf::IfOp ifOp =
              genIf(merger, codegen, rewriter, op, idx, merger.lat(lj).simple);
          genStmt(merger, codegen, rewriter, op, topSort, ej, at + 1);
          endIf(merger, codegen, rewriter, op, ifOp, loop, redInput, cntInput);
        } else {
          genStmt(merger, codegen, rewriter, op, topSort, ej, at + 1);
        }
      }
    }

    // End a loop.
    needsUniv =
        endLoop(merger, codegen, rewriter, op, loop, idx, li, needsUniv);
  }

  // End a loop sequence.
  endLoopSeq(merger, codegen, rewriter, op, exp, at, idx, ldx);
}

/// Converts the result computed by the sparse kernel into the required form.
static void genResult(Merger &merger, CodeGen &codegen,
                      PatternRewriter &rewriter, linalg::GenericOp op) {
  OpOperand *lhs = op.getOutputOperand(0);
  Type resType = lhs->get().getType();
  if (getSparseTensorEncoding(resType)) {
    // The sparse tensor rematerializes from the original sparse tensor's
    // underlying sparse storage format.
    rewriter.replaceOpWithNewOp<LoadOp>(op, resType, lhs->get(),
                                        codegen.sparseOut == lhs);
  } else {
    // To rematerialize an non-annotated tensor, simply load it
    // from the bufferized value.
    Value val = codegen.buffers.back(); // value array
    rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(op, resType, val);
  }
}

//===----------------------------------------------------------------------===//
// Sparse compiler rewriting methods.
//===----------------------------------------------------------------------===//

namespace {

/// Sparse rewriting rule for generic Lingalg operation.
struct GenericOpSparsifier : public OpRewritePattern<linalg::GenericOp> {
public:
  GenericOpSparsifier(MLIRContext *context, SparsificationOptions o)
      : OpRewritePattern<linalg::GenericOp>(context), options(o) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    // Detects sparse annotations and translate the per-dimension sparsity
    // information for all tensors to loop indices in the kernel.
    assert(op.getNumOutputs() == 1);
    unsigned numTensors = op.getNumInputsAndOutputs();
    unsigned numLoops = op.iterator_types().getValue().size();
    Merger merger(numTensors, numLoops);
    if (!findSparseAnnotations(merger, op))
      return failure();

    // Computes a topologically sorted iteration graph to ensure
    // tensors are visited in natural index order. Fails on cycles.
    // This assumes that higher-level passes have already put the
    // tensors in each tensor expression in a feasible order.
    std::vector<unsigned> topSort;
    if (!computeIterationGraph(merger, op, topSort,
                               SortMask::kIncludeUndef |
                                   SortMask::kIncludeDense) &&
        !computeIterationGraph(merger, op, topSort, SortMask::kIncludeUndef) &&
        !computeIterationGraph(merger, op, topSort, SortMask::kIncludeDense) &&
        !computeIterationGraph(merger, op, topSort, SortMask::kSparseOnly))
      return failure();

    // Builds the tensor expression for the Linalg operation in SSA form.
    Optional<unsigned> optExp = merger.buildTensorExpFromLinalg(op);
    if (!optExp.hasValue())
      return failure();
    unsigned exp = optExp.getValue();

    // Rejects an inadmissable tensor expression.
    OpOperand *sparseOut = nullptr;
    unsigned outerParNest = 0;
    if (!isAdmissableTensorExp(merger, op, topSort, exp, &sparseOut,
                               outerParNest))
      return failure();

    // Recursively generates code.
    merger.setHasSparseOut(sparseOut != nullptr);
    CodeGen codegen(options, numTensors, numLoops, sparseOut, outerParNest);
    genBuffers(merger, codegen, rewriter, op);
    genStmt(merger, codegen, rewriter, op, topSort, exp, 0);
    genResult(merger, codegen, rewriter, op);
    return success();
  }

private:
  /// Options to control sparse code generation.
  SparsificationOptions options;
};

} // namespace

/// Populates the given patterns list with rewriting rules required for
/// the sparsification of linear algebra operations.
void mlir::populateSparsificationPatterns(
    RewritePatternSet &patterns, const SparsificationOptions &options) {
  patterns.add<GenericOpSparsifier>(patterns.getContext(), options);
}
