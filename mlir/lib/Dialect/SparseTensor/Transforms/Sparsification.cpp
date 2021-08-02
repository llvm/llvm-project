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

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/SparseTensor/Utils/Merger.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TensorEncoding.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

namespace {

// Code generation.
struct CodeGen {
  CodeGen(SparsificationOptions o, unsigned numTensors, unsigned numLoops)
      : options(o), loops(numLoops), sizes(numLoops), buffers(numTensors),
        pointers(numTensors, std::vector<Value>(numLoops)),
        indices(numTensors, std::vector<Value>(numLoops)),
        highs(numTensors, std::vector<Value>(numLoops)),
        pidxs(numTensors, std::vector<Value>(numLoops)),
        idxs(numTensors, std::vector<Value>(numLoops)), redExp(-1u), redVal(),
        curVecLength(1), curVecMask() {}
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
  /// reduction are exhausted,  all inner loops can "scalarize" the reduction.
  // TODO: currently only done for (a chain of) innermost for-loops, where it
  // is most effective; we could generalize to more outer and while-loops.
  unsigned redExp;
  Value redVal;
  // Current vector length and mask.
  unsigned curVecLength;
  Value curVecMask;
};

} // namespace

// Helper method to apply dimension ordering permutation.
static unsigned perm(SparseTensorEncodingAttr &enc, unsigned d) {
  if (enc) {
    auto order = enc.getDimOrdering();
    if (order) {
      assert(order.isPermutation());
      return order.getDimPosition(d);
    }
  }
  return d;
}

// Helper method to translate dim level type to internal representation.
static Dim toDim(SparseTensorEncodingAttr &enc, unsigned d) {
  if (enc) {
    SparseTensorEncodingAttr::DimLevelType tp = enc.getDimLevelType()[d];
    if (tp == SparseTensorEncodingAttr::DimLevelType::Compressed)
      return Dim::kSparse;
    if (tp == SparseTensorEncodingAttr::DimLevelType::Singleton)
      return Dim::kSingle;
  }
  return Dim::kDense;
}

/// Helper method to inspect sparse encodings in the tensor types.
/// Fills the per-dimension sparsity information for all tensors.
static bool findSparseAnnotations(Merger &merger, linalg::GenericOp op) {
  bool annotated = false;
  for (OpOperand *t : op.getInputAndOutputOperands()) {
    auto map = op.getTiedIndexingMap(t);
    if (!map.isProjectedPermutation())
      return false;
    auto enc = getSparseTensorEncoding(t->get().getType());
    if (enc)
      annotated = true;
    assert(map.getNumResults() == op.getRank(t));
    for (unsigned d = 0, rank = map.getNumResults(); d < rank; d++) {
      unsigned idx = map.getDimPosition(perm(enc, d));
      merger.setDim(t->getOperandNumber(), idx, toDim(enc, d));
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

/// Computes a topologically sorted iteration graph for the linalg operation.
/// Ensures all tensors are visited in natural index order. This is essential
/// for sparse storage formats since these only support access along fixed
/// dimensions. Even for dense storage formats, however, the natural index
/// order yields innermost unit-stride access with better spatial locality.
static bool computeIterationGraph(Merger &merger, linalg::GenericOp op,
                                  std::vector<unsigned> &topSort,
                                  bool sparseOnly) {
  // Set up an n x n from/to adjacency matrix of the iteration graph
  // for the implicit loop indices i_0 .. i_n-1.
  unsigned n = op.getNumLoops();
  std::vector<std::vector<bool>> adjM(n, std::vector<bool>(n, false));

  // Iterate over the indexing maps of every tensor in the tensor expression.
  for (OpOperand *t : op.getInputAndOutputOperands()) {
    auto map = op.getTiedIndexingMap(t);
    auto enc = getSparseTensorEncoding(t->get().getType());
    assert(map.getNumDims() == n);
    // Skip dense tensor constraints when sparse only is requested.
    if (sparseOnly && !enc)
      continue;
    // Each tensor expression and optional dimension ordering (row-major
    // by default) puts an ordering constraint on the loop indices. For
    // example, the tensor expresion A_ijk forces the ordering i < j < k
    // on the loop indices if no explicit dimension ordering is given.
    for (unsigned d = 1, rank = map.getNumResults(); d < rank; d++) {
      unsigned f = map.getDimPosition(perm(enc, d - 1));
      unsigned t = map.getDimPosition(perm(enc, d));
      adjM[f][t] = true;
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

/// Returns true when the tensor expression is admissable for codegen.
/// Since all sparse input tensors are admissable, we just need to check
/// whether the output tensor in the tensor expression codegen is admissable.
static bool isAdmissableTensorExp(Merger &merger, linalg::GenericOp op,
                                  unsigned exp) {
  OpOperand *lhs = op.getOutputOperand(0);
  unsigned tensor = lhs->getOperandNumber();
  auto enc = getSparseTensorEncoding(lhs->get().getType());
  // An non-annotated output tensor is assumed dense, and becomes a random
  // access n-dim memref. Admissable since inserstions cannot occur.
  if (!enc)
    return true;
  // An all-dense annotated "sparse" output tensor becomes a linearized random
  // access 1-dim memref. Also admissable since insertions cannot occur.
  bool allDense = true;
  unsigned numLoops = op.iterator_types().getValue().size();
  for (unsigned i = 0; i < numLoops; i++)
    if (merger.isDim(tensor, i, Dim::kSparse)) {
      allDense = false;
      break;
    }
  if (allDense)
    return true;
  // A tensor expression with a sparse output tensor that changes its values
  // but not its nonzero structure, an operation called "simply dynamic" in
  // [Bik96,Ch9], is also admissable without special codegen.
  if (merger.isConjunction(tensor, exp))
    return true;
  // Reject for now since this requires changes to the nonzero structure.
  // TODO: implement "workspaces" [Kjolstad2019]
  return false;
}

/// Maps sparse integer option to actual integral storage type.
static Type genIntType(PatternRewriter &rewriter, unsigned width) {
  if (width == 0)
    return rewriter.getIndexType();
  return rewriter.getIntegerType(width);
}

/// Detects in-place annotation on tensor argument.
static bool getInPlace(Value val) {
  if (auto arg = val.dyn_cast<BlockArgument>())
    if (auto funcOp = dyn_cast<FuncOp>(arg.getOwner()->getParentOp()))
      if (auto attr = funcOp.getArgAttrOfType<BoolAttr>(
              arg.getArgNumber(), linalg::LinalgDialect::kInplaceableAttrName))
        return attr.getValue();
  return false;
}

/// Generates buffer for the output tensor.
static Value genOutputBuffer(CodeGen &codegen, PatternRewriter &rewriter,
                             linalg::GenericOp op, MemRefType denseTp,
                             ArrayRef<Value> args) {
  Location loc = op.getLoc();
  Value tensor = op.getOutputOperand(0)->get();
  // The output tensor simply could materialize from the buffer that will
  // be generated for the tensor present in the outs() clause. This has
  // the major advantage that the sparse kernel only updates the nonzero
  // positions for the output tensor.
  if (getInPlace(tensor))
    return rewriter.create<memref::BufferCastOp>(loc, denseTp, tensor);
  // By default, a new buffer is allocated which is initialized to the
  // tensor defined in the outs() clause. This is always correct but
  // introduces a dense initialization component that may negatively
  // impact the running complexity of the sparse kernel.
  Value init = rewriter.create<memref::BufferCastOp>(loc, denseTp, tensor);
  Value alloc = rewriter.create<memref::AllocOp>(loc, denseTp, args);
  rewriter.create<memref::CopyOp>(loc, init, alloc);
  return alloc;
}

/// Local bufferization of all dense and sparse data structures.
/// This code enables testing the first prototype sparse compiler.
// TODO: replace this with a proliferated bufferization strategy
static bool genBuffers(Merger &merger, CodeGen &codegen,
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
      unsigned idx = map.getDimPosition(perm(enc, d));
      // Handle sparse storage schemes.
      if (merger.isDim(tensor, idx, Dim::kSparse)) {
        auto dynShape = {ShapedType::kDynamicSize};
        auto ptrTp = MemRefType::get(
            dynShape, genIntType(rewriter, enc.getPointerBitWidth()));
        auto indTp = MemRefType::get(
            dynShape, genIntType(rewriter, enc.getIndexBitWidth()));
        Value dim = rewriter.create<ConstantIndexOp>(loc, d);
        // Generate sparse primitives to obtains pointer and indices.
        codegen.pointers[tensor][idx] =
            rewriter.create<ToPointersOp>(loc, ptrTp, t->get(), dim);
        codegen.indices[tensor][idx] =
            rewriter.create<ToIndicesOp>(loc, indTp, t->get(), dim);
      }
      // Find lower and upper bound in current dimension.
      Value up;
      if (shape[d] == MemRefType::kDynamicSize) {
        up = rewriter.create<tensor::DimOp>(loc, t->get(), d);
        args.push_back(up);
      } else {
        up = rewriter.create<ConstantIndexOp>(loc, shape[d]);
      }
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
            rewriter.create<memref::BufferCastOp>(loc, denseTp, t->get());
      else
        codegen.buffers[tensor] =
            genOutputBuffer(codegen, rewriter, op, denseTp, args);
    } else {
      // Annotated sparse tensors.
      if (tensor == op.getNumInputs() && !getInPlace(t->get()))
        return false; // reject output if not in-place
      auto dynShape = {ShapedType::kDynamicSize};
      auto sparseTp = MemRefType::get(dynShape, elementType);
      codegen.buffers[tensor] =
          rewriter.create<ToValuesOp>(loc, sparseTp, t->get());
    }
  }
  return true;
}

/// Constructs vector type.
static VectorType vectorType(CodeGen &codegen, Type etp) {
  return VectorType::get(codegen.curVecLength, etp);
}

/// Constructs vector type from pointer.
static VectorType vectorType(CodeGen &codegen, Value ptr) {
  return vectorType(codegen, ptr.getType().cast<MemRefType>().getElementType());
}

/// Constructs vector iteration mask.
static Value genVectorMask(CodeGen &codegen, PatternRewriter &rewriter,
                           Value iv, Value lo, Value hi, Value step) {
  Location loc = iv.getLoc();
  VectorType mtp = vectorType(codegen, rewriter.getIntegerType(1));
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
          loc, mtp, rewriter.create<ConstantIntOp>(loc, 1, 1));
  }
  // Otherwise, generate a vector mask that avoids overrunning the upperbound
  // during vector execution. Here we rely on subsequent loop optimizations to
  // avoid executing the mask in all iterations, for example, by splitting the
  // loop into an unconditional vector loop and a scalar cleanup loop.
  Value end = rewriter.create<SubIOp>(loc, hi, iv);
  return rewriter.create<vector::CreateMaskOp>(loc, mtp, end);
}

/// Generates a vectorized load lhs = a[ind[lo:hi]] or lhs = a[lo:hi].
static Value genVectorLoad(CodeGen &codegen, PatternRewriter &rewriter,
                           Value ptr, ArrayRef<Value> args) {
  Location loc = ptr.getLoc();
  VectorType vtp = vectorType(codegen, ptr);
  Value pass = rewriter.create<ConstantOp>(loc, vtp, rewriter.getZeroAttr(vtp));
  if (args.back().getType().isa<VectorType>()) {
    SmallVector<Value, 4> scalarArgs(args.begin(), args.end());
    Value indexVec = args.back();
    scalarArgs.back() = rewriter.create<ConstantIndexOp>(loc, 0);
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
    scalarArgs.back() = rewriter.create<ConstantIndexOp>(loc, 0);
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
  // Actual load.
  SmallVector<Value, 4> args;
  OpOperand *t = op.getInputAndOutputOperands()[merger.exp(exp).tensor];
  unsigned tensor = t->getOperandNumber();
  auto map = op.getTiedIndexingMap(t);
  auto enc = getSparseTensorEncoding(t->get().getType());
  unsigned rank = map.getNumResults();
  if (enc) {
    unsigned idx = map.getDimPosition(perm(enc, rank - 1));
    assert(codegen.pidxs[tensor][idx] != nullptr);
    args.push_back(codegen.pidxs[tensor][idx]); // position index
  } else {
    for (unsigned d = 0; d < rank; d++) {
      unsigned idx = map.getDimPosition(d);
      args.push_back(codegen.loops[idx]); // universal dense index
    }
  }
  Location loc = op.getLoc();
  Value ptr = codegen.buffers[tensor];
  if (codegen.curVecLength > 1)
    return genVectorLoad(codegen, rewriter, ptr, args);
  return rewriter.create<memref::LoadOp>(loc, ptr, args);
}

/// Generates a store on a dense or sparse tensor.
static void genTensorStore(Merger &merger, CodeGen &codegen,
                           PatternRewriter &rewriter, linalg::GenericOp op,
                           OpOperand *t, Value rhs) {
  Location loc = op.getLoc();
  // Test if this is a scalarized reduction.
  OpOperand *lhs = op.getOutputOperand(0);
  if (lhs == t && codegen.redVal) {
    if (codegen.curVecLength > 1)
      rhs = rewriter.create<SelectOp>(loc, codegen.curVecMask, rhs,
                                      codegen.redVal);
    codegen.redVal = rhs;
    return;
  }
  // Actual store.
  SmallVector<Value, 4> args;
  unsigned tensor = t->getOperandNumber();
  auto map = op.getTiedIndexingMap(t);
  auto enc = getSparseTensorEncoding(t->get().getType());
  unsigned rank = map.getNumResults();
  if (enc) {
    unsigned idx = map.getDimPosition(perm(enc, rank - 1));
    assert(codegen.pidxs[tensor][idx] != nullptr);
    args.push_back(codegen.pidxs[tensor][idx]); // position index
  } else {
    for (unsigned d = 0; d < rank; d++) {
      unsigned idx = map.getDimPosition(d);
      args.push_back(codegen.loops[idx]); // universal dense index
    }
  }
  Value ptr = codegen.buffers[tensor];
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
        vload = rewriter.create<ZeroExtendIOp>(
            loc, vload, vectorType(codegen, rewriter.getIntegerType(32)));
      else if (etp.getIntOrFloatBitWidth() < 64 &&
               !codegen.options.enableSIMDIndex32)
        vload = rewriter.create<ZeroExtendIOp>(
            loc, vload, vectorType(codegen, rewriter.getIntegerType(64)));
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
      load = rewriter.create<ZeroExtendIOp>(loc, load,
                                            rewriter.getIntegerType(64));
    load = rewriter.create<IndexCastOp>(loc, load, rewriter.getIndexType());
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
  Value mul = rewriter.create<MulIOp>(loc, size, p);
  if (auto vtp = i.getType().dyn_cast<VectorType>()) {
    Value inv = rewriter.create<IndexCastOp>(loc, mul, vtp.getElementType());
    mul = genVectorInvariantValue(codegen, rewriter, inv);
  }
  return rewriter.create<AddIOp>(loc, mul, i);
}

/// Generates start of a reduction.
static Value genReductionStart(Merger &merger, CodeGen &codegen,
                               PatternRewriter &rewriter,
                               linalg::GenericOp op) {
  if (codegen.redVal)
    return codegen.redVal; // chained with previous for-loop
  if (codegen.curVecLength > 1) {
    // TODO: assumes + reductions for now
    VectorType vtp = vectorType(codegen, codegen.buffers[codegen.redExp]);
    return rewriter.create<ConstantOp>(op.getLoc(), vtp,
                                       rewriter.getZeroAttr(vtp));
  }
  return genTensorLoad(merger, codegen, rewriter, op, codegen.redExp);
}

/// Generates end of a reduction.
static void genReductionEnd(Merger &merger, CodeGen &codegen,
                            PatternRewriter &rewriter, linalg::GenericOp op) {
  Value red = codegen.redVal;
  if (!red)
    return;
  assert(codegen.curVecLength == 1);
  codegen.redVal = merger.exp(codegen.redExp).val = Value(); // end chain
  OpOperand *lhs = op.getOutputOperand(0);
  if (auto vtp = red.getType().dyn_cast<VectorType>()) {
    // TODO: assumes + reductions for now
    StringAttr kind = rewriter.getStringAttr("add");
    Value ld = genTensorLoad(merger, codegen, rewriter, op, codegen.redExp);
    // Integer reductions don't accept an accumulator.
    if (vtp.getElementType().isa<IntegerType>()) {
      red = rewriter.create<vector::ReductionOp>(op.getLoc(), ld.getType(),
                                                 kind, red, ValueRange{});
      red = rewriter.create<AddIOp>(op.getLoc(), red, ld);
    } else {
      red = rewriter.create<vector::ReductionOp>(op.getLoc(), ld.getType(),
                                                 kind, red, ld);
    }
  }
  genTensorStore(merger, codegen, rewriter, op, lhs, red);
}

/// Recursively generates tensor expression.
static Value genExp(Merger &merger, CodeGen &codegen, PatternRewriter &rewriter,
                    linalg::GenericOp op, unsigned exp) {
  Location loc = op.getLoc();
  if (exp == -1u)
    return Value();
  if (merger.exp(exp).kind == Kind::kTensor)
    return genTensorLoad(merger, codegen, rewriter, op, exp);
  if (merger.exp(exp).kind == Kind::kInvariant)
    return genInvariantValue(merger, codegen, rewriter, exp);
  Value v0 = genExp(merger, codegen, rewriter, op, merger.exp(exp).children.e0);
  Value v1 = genExp(merger, codegen, rewriter, op, merger.exp(exp).children.e1);
  if (merger.exp(exp).kind == Kind::kNegI) {
    // TODO: no negi in std, need to make zero explicit.
    Type tp = op.getOutputTensorTypes()[0].getElementType();
    v1 = v0;
    v0 = rewriter.create<ConstantOp>(loc, tp, rewriter.getZeroAttr(tp));
    if (codegen.curVecLength > 1)
      v0 = genVectorInvariantValue(codegen, rewriter, v0);
  }
  return merger.buildExp(rewriter, loc, exp, v0, v1);
}

/// Hoists loop invariant tensor loads for which indices have been exhausted.
static void genInvariants(Merger &merger, CodeGen &codegen,
                          PatternRewriter &rewriter, linalg::GenericOp op,
                          unsigned exp, unsigned ldx, bool hoist) {
  if (exp == -1u)
    return;
  if (merger.exp(exp).kind == Kind::kTensor) {
    // Inspect tensor indices.
    bool atLevel = ldx == -1u;
    OpOperand *t = op.getInputAndOutputOperands()[merger.exp(exp).tensor];
    auto map = op.getTiedIndexingMap(t);
    auto enc = getSparseTensorEncoding(t->get().getType());
    for (unsigned d = 0, rank = map.getNumResults(); d < rank; d++) {
      unsigned idx = map.getDimPosition(perm(enc, d));
      if (!codegen.loops[idx])
        return; // still in play
      else if (idx == ldx)
        atLevel = true;
    }
    // All exhausted at this level (atLevel denotes exactly at this level).
    OpOperand *lhs = op.getOutputOperand(0);
    if (lhs == t) {
      codegen.redExp = hoist ? exp : -1u;
    } else if (atLevel) {
      merger.exp(exp).val =
          hoist ? genTensorLoad(merger, codegen, rewriter, op, exp) : Value();
    }
  } else if (merger.exp(exp).kind != Kind::kInvariant) {
    // Traverse into the binary operations. Note that we only hoist
    // tensor loads, since subsequent MLIR/LLVM passes know how to
    // deal with all other kinds of derived loop invariants.
    unsigned e0 = merger.exp(exp).children.e0;
    unsigned e1 = merger.exp(exp).children.e1;
    genInvariants(merger, codegen, rewriter, op, e0, ldx, hoist);
    genInvariants(merger, codegen, rewriter, op, e1, ldx, hoist);
  }
}

/// Generates initialization code for the subsequent loop sequence at
/// current index level. Returns true if the loop sequence needs to
/// maintain the universal index.
static bool genInit(Merger &merger, CodeGen &codegen, PatternRewriter &rewriter,
                    linalg::GenericOp op, std::vector<unsigned> &topSort,
                    unsigned at, llvm::BitVector &inits) {
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
        Value one = rewriter.create<ConstantIndexOp>(loc, 1);
        Value p0 = (pat == 0) ? rewriter.create<ConstantIndexOp>(loc, 0)
                              : codegen.pidxs[tensor][topSort[pat - 1]];
        codegen.pidxs[tensor][idx] = genLoad(codegen, rewriter, loc, ptr, p0);
        Value p1 = rewriter.create<AddIOp>(loc, p0, one);
        codegen.highs[tensor][idx] = genLoad(codegen, rewriter, loc, ptr, p1);
      } else {
        // Dense index still in play.
        needsUniv = true;
      }
    }
  }

  // Initialize the universal dense index.
  codegen.loops[idx] = rewriter.create<ConstantIndexOp>(loc, 0);
  return needsUniv;
}

/// Returns vectorization strategy. Any implicit inner loop in the Linalg
/// operation is a candidate. Whether it is actually converted to SIMD code
/// depends on the requested strategy.
static bool isVectorFor(CodeGen &codegen, bool isInner, bool isSparse) {
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

/// Checks unit strides for dense tensors. The iteration graph may have ignored
/// dense access patterns in order to avoid cycles (sparse access patterns are
/// always placed innermost), but that means dense access has become strided.
/// For now, we reject vectorization of such cases.
/// TODO: implement strided load/stores on dense arrays
static bool denseUnitStrides(Merger &merger, linalg::GenericOp op,
                             unsigned idx) {
  for (OpOperand *t : op.getInputAndOutputOperands()) {
    if (!getSparseTensorEncoding(t->get().getType())) {
      auto map = op.getTiedIndexingMap(t);
      for (unsigned d = 0, rank = map.getNumResults(); d < rank; d++) {
        if (map.getDimPosition(d) == idx && d != rank - 1)
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
                         llvm::BitVector &indices) {
  unsigned fb = indices.find_first();
  unsigned tensor = merger.tensor(fb);
  assert(idx == merger.index(fb));
  auto iteratorTypes = op.iterator_types().getValue();
  bool isReduction = linalg::isReductionIteratorType(iteratorTypes[idx]);
  bool isSparse = merger.isDim(fb, Dim::kSparse);
  bool isVector = isVectorFor(codegen, isInner, isSparse) &&
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
  Value step = rewriter.create<ConstantIndexOp>(loc, codegen.curVecLength);

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

  // Emit a sequential loop, potentially with a scalarized reduction.
  bool scalarRed = isInner && codegen.redExp != -1u;
  SmallVector<Value, 4> operands;
  if (scalarRed) {
    Value load = genReductionStart(merger, codegen, rewriter, op);
    operands.push_back(load);
  }
  scf::ForOp forOp = rewriter.create<scf::ForOp>(loc, lo, hi, step, operands);
  if (scalarRed) {
    codegen.redVal = merger.exp(codegen.redExp).val =
        forOp.getRegionIterArgs().front();
  }
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
                           unsigned idx, bool needsUniv,
                           llvm::BitVector &indices) {
  SmallVector<Type, 4> types;
  SmallVector<Value, 4> operands;
  // Construct the while-loop with a parameter for each index.
  Type indexType = rewriter.getIndexType();
  for (unsigned b = 0, be = indices.size(); b < be; b++) {
    if (indices[b] && merger.isDim(b, Dim::kSparse)) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      types.push_back(indexType);
      assert(codegen.pidxs[tensor][idx].getType().isa<IndexType>() &&
             "type mismatch for sparse index");
      operands.push_back(codegen.pidxs[tensor][idx]);
    }
  }
  if (needsUniv) {
    types.push_back(indexType);
    assert(codegen.loops[idx].getType().isa<IndexType>() &&
           "type mismatch for universal index");
    operands.push_back(codegen.loops[idx]);
  }
  Location loc = op.getLoc();
  scf::WhileOp whileOp = rewriter.create<scf::WhileOp>(loc, types, operands);
  Block *before = rewriter.createBlock(&whileOp.before(), {}, types);
  Block *after = rewriter.createBlock(&whileOp.after(), {}, types);

  // Build the "before" region, which effectively consists
  // of a conjunction of "i < upper" tests on all induction.
  rewriter.setInsertionPointToStart(&whileOp.before().front());
  Value cond;
  unsigned o = 0;
  for (unsigned b = 0, be = indices.size(); b < be; b++) {
    if (indices[b] && merger.isDim(b, Dim::kSparse)) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      Value op1 = before->getArgument(o);
      Value op2 = codegen.highs[tensor][idx];
      Value opc = rewriter.create<CmpIOp>(loc, CmpIPredicate::ult, op1, op2);
      cond = cond ? rewriter.create<AndOp>(loc, cond, opc) : opc;
      codegen.pidxs[tensor][idx] = after->getArgument(o++);
    }
  }
  if (needsUniv)
    codegen.loops[idx] = after->getArgument(o++);
  assert(o == operands.size());
  rewriter.create<scf::ConditionOp>(loc, cond, before->getArguments());
  rewriter.setInsertionPointToStart(&whileOp.after().front());
  return whileOp;
}

/// Generates a for-loop or a while-loop, depending on whether it implements
/// singleton iteration or co-iteration over the given conjunction.
static Operation *genLoop(Merger &merger, CodeGen &codegen,
                          PatternRewriter &rewriter, linalg::GenericOp op,
                          std::vector<unsigned> &topSort, unsigned at,
                          bool needsUniv, llvm::BitVector &indices) {
  unsigned idx = topSort[at];
  if (indices.count() == 1) {
    bool isOuter = at == 0;
    bool isInner = at == topSort.size() - 1;
    return genFor(merger, codegen, rewriter, op, isOuter, isInner, idx,
                  indices);
  }
  genReductionEnd(merger, codegen, rewriter, op); // cannot chain
  return genWhile(merger, codegen, rewriter, op, idx, needsUniv, indices);
}

/// Generates the local variables for this loop, consisting of the sparse
/// indices, restored universal dense index, and dense positions.
static void genLocals(Merger &merger, CodeGen &codegen,
                      PatternRewriter &rewriter, linalg::GenericOp op,
                      std::vector<unsigned> &topSort, unsigned at,
                      bool needsUniv, llvm::BitVector &locals) {
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
          Value cmp =
              rewriter.create<CmpIOp>(loc, CmpIPredicate::ult, load, min);
          min = rewriter.create<SelectOp>(loc, cmp, load, min);
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
      Value p = (pat == 0) ? rewriter.create<ConstantIndexOp>(loc, 0)
                           : codegen.pidxs[tensor][topSort[pat - 1]];
      codegen.pidxs[tensor][idx] = genAddress(
          codegen, rewriter, loc, codegen.sizes[idx], p, codegen.loops[idx]);
    }
  }
}

/// Generates the induction structure for a while-loop.
static void genWhileInduction(Merger &merger, CodeGen &codegen,
                              PatternRewriter &rewriter, linalg::GenericOp op,
                              unsigned idx, bool needsUniv,
                              llvm::BitVector &induction, ResultRange results) {
  Location loc = op.getLoc();
  unsigned o = 0;
  SmallVector<Value, 4> operands;
  Value one = rewriter.create<ConstantIndexOp>(loc, 1);
  for (unsigned b = 0, be = induction.size(); b < be; b++) {
    if (induction[b] && merger.isDim(b, Dim::kSparse)) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      Value op1 = codegen.idxs[tensor][idx];
      Value op2 = codegen.loops[idx];
      Value op3 = codegen.pidxs[tensor][idx];
      Value cmp = rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, op1, op2);
      Value add = rewriter.create<AddIOp>(loc, op3, one);
      operands.push_back(rewriter.create<SelectOp>(loc, cmp, add, op3));
      codegen.pidxs[tensor][idx] = results[o++];
    }
  }
  if (needsUniv) {
    operands.push_back(rewriter.create<AddIOp>(loc, codegen.loops[idx], one));
    codegen.loops[idx] = results[o++];
  }
  assert(o == operands.size());
  rewriter.create<scf::YieldOp>(loc, operands);
}

/// Generates a single if-statement within a while-loop.
static scf::IfOp genIf(Merger &merger, CodeGen &codegen,
                       PatternRewriter &rewriter, linalg::GenericOp op,
                       unsigned idx, llvm::BitVector &conditions) {
  Location loc = op.getLoc();
  Value cond;
  for (unsigned b = 0, be = conditions.size(); b < be; b++) {
    if (conditions[b]) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      Value clause;
      if (merger.isDim(b, Dim::kSparse)) {
        Value op1 = codegen.idxs[tensor][idx];
        Value op2 = codegen.loops[idx];
        clause = rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, op1, op2);
      } else {
        clause = rewriter.create<ConstantIntOp>(loc, 1, 1); // true
      }
      cond = cond ? rewriter.create<AndOp>(loc, cond, clause) : clause;
    }
  }
  scf::IfOp ifOp = rewriter.create<scf::IfOp>(loc, cond, /*else*/ true);
  rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());
  return ifOp;
}

/// Recursively generates code while computing iteration lattices in order
/// to manage the complexity of implementing co-iteration over unions
/// and intersections of sparse iterations spaces.
static void genStmt(Merger &merger, CodeGen &codegen, PatternRewriter &rewriter,
                    linalg::GenericOp op, std::vector<unsigned> &topSort,
                    unsigned exp, unsigned at) {
  // At each leaf, assign remaining tensor (sub)expression to output tensor.
  if (at == topSort.size()) {
    OpOperand *lhs = op.getOutputOperand(0);
    Value rhs = genExp(merger, codegen, rewriter, op, exp);
    genTensorStore(merger, codegen, rewriter, op, lhs, rhs);
    return;
  }
  assert(codegen.curVecLength == 1);

  // Construct iteration lattices for current loop index, with L0 at top.
  // Then emit initialization code for the loop sequence at this level.
  // We maintain the universal dense index if dense indices are still
  // in play for a non-singleton loop sequence.
  Location loc = op.getLoc();
  unsigned idx = topSort[at];
  unsigned lts = merger.optimizeSet(merger.buildLattices(exp, idx));
  unsigned lsize = merger.set(lts).size();
  assert(lsize != 0);
  unsigned l0 = merger.set(lts)[0];
  unsigned ldx = at == 0 ? -1u : topSort[at - 1];
  genInvariants(merger, codegen, rewriter, op, exp, ldx, /*hoist=*/true);
  bool needsUniv = false;
  if (genInit(merger, codegen, rewriter, op, topSort, at,
              merger.lat(l0).bits)) {
    // Maintain the universal index only if it is actually
    // consumed by a subsequent lattice point.
    for (unsigned i = 1; i < lsize; i++) {
      unsigned li = merger.set(lts)[i];
      if (!merger.hasAnyDimOf(merger.lat(li).simple, Dim::kSparse)) {
        needsUniv = true;
        break;
      }
    }
  }

  // Emit a loop for every lattice point L0 >= Li.
  for (unsigned i = 0; i < lsize; i++) {
    unsigned li = merger.set(lts)[i];

    // Emit loop.
    codegen.curVecLength = 1;
    llvm::BitVector indices = merger.lat(li).simple;
    Operation *loop =
        genLoop(merger, codegen, rewriter, op, topSort, at, needsUniv, indices);
    genLocals(merger, codegen, rewriter, op, topSort, at, needsUniv,
              merger.lat(li).bits);

    // Visit all lattices points with Li >= Lj to generate the
    // loop-body, possibly with if statements for coiteration.
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
          rewriter.setInsertionPointToStart(&ifOp.elseRegion().front());
        } else {
          genStmt(merger, codegen, rewriter, op, topSort, ej, at + 1);
        }
      }
    }

    // Wrap-up induction and restore insertion point.
    if (isWhile) {
      scf::WhileOp whileOp = cast<scf::WhileOp>(loop);
      rewriter.setInsertionPointToEnd(&whileOp.after().front());
      genWhileInduction(merger, codegen, rewriter, op, idx, needsUniv,
                        merger.lat(li).bits, whileOp.results());
    } else {
      needsUniv = false;
      if (codegen.redVal) {
        rewriter.create<scf::YieldOp>(loc, codegen.redVal);
        codegen.redVal = loop->getResult(0);
      }
    }
    rewriter.setInsertionPointAfter(loop);
  }

  // Wrap-up loop sequence.
  codegen.curVecLength = 1;
  genReductionEnd(merger, codegen, rewriter, op);
  genInvariants(merger, codegen, rewriter, op, exp, ldx, /*hoist=*/false);
  codegen.loops[idx] = Value();
}

/// Converts the result computed by the sparse kernel into the required form.
static void genResult(Merger &merger, CodeGen &codegen,
                      PatternRewriter &rewriter, linalg::GenericOp op) {
  Location loc = op.getLoc();
  OpOperand *lhs = op.getOutputOperand(0);
  Type resType = lhs->get().getType();
  unsigned tensor = lhs->getOperandNumber();
  auto map = op.getTiedIndexingMap(lhs);
  auto enc = getSparseTensorEncoding(resType);
  Value result = codegen.buffers.back(); // value array
  if (enc) {
    // The sparse annotation unambigiously defines the arrays needed
    // to "reconstruct" the sparse tensor from the storage scheme
    // (even though lowering should never need this eventually).
    SmallVector<Value, 4> args;
    for (unsigned d = 0, rank = map.getNumResults(); d < rank; d++) {
      unsigned idx = map.getDimPosition(perm(enc, d));
      if (merger.isDim(tensor, idx, Dim::kSparse)) {
        args.push_back(codegen.pointers[tensor][idx]);
        args.push_back(codegen.indices[tensor][idx]);
      }
    }
    args.push_back(result);
    result = rewriter.create<ToTensorOp>(loc, resType, args);
  } else {
    // To "reconstruct" an non-annotated tensor, sipmly load it
    // from the bufferized value.
    result = rewriter.create<memref::TensorLoadOp>(loc, resType, result);
  }
  rewriter.replaceOp(op, result);
}

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
    if (!computeIterationGraph(merger, op, topSort, /*sparseOnly=*/false) &&
        !computeIterationGraph(merger, op, topSort, /*sparseOnly=*/true))
      return failure();

    // Builds the tensor expression for the Linalg operation in SSA form.
    Optional<unsigned> exp = merger.buildTensorExpFromLinalg(op);
    if (!exp.hasValue())
      return failure();

    // Rejects an inadmissable tensor expression.
    if (!isAdmissableTensorExp(merger, op, exp.getValue()))
      return failure();

    // Recursively generates code.
    CodeGen codegen(options, numTensors, numLoops);
    if (!genBuffers(merger, codegen, rewriter, op))
      return failure(); // could not bufferize
    genStmt(merger, codegen, rewriter, op, topSort, exp.getValue(), 0);
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
