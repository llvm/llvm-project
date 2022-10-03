//===- CodegenUtils.cpp - Utilities for generating MLIR -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CodegenUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

/// Generates a pointer/index load from the sparse storage scheme. Narrower
/// data types need to be zero extended before casting the value into the
/// index type used for looping and indexing.
static Value genIndexLoad(OpBuilder &builder, Location loc, Value ptr,
                          Value s) {
  // For the scalar case, we simply zero extend narrower indices into 64-bit
  // values before casting to index without a performance penalty. Here too,
  // however, indices that already are 64-bit, in theory, cannot express the
  // full range as explained above.
  Value load = builder.create<memref::LoadOp>(loc, ptr, s);
  if (!load.getType().isa<IndexType>()) {
    if (load.getType().getIntOrFloatBitWidth() < 64)
      load = builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), load);
    load =
        builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), load);
  }
  return load;
}

//===----------------------------------------------------------------------===//
// Sparse tensor loop emitter class implementations
//===----------------------------------------------------------------------===//

SparseTensorLoopEmitter::SparseTensorLoopEmitter(ValueRange tensors)
    : tensors(tensors.begin(), tensors.end()), dims(tensors.size()),
      pidxs(tensors.size()), coord(tensors.size()), highs(tensors.size()),
      sizes(tensors.size()), ptrBuffer(tensors.size()),
      idxBuffer(tensors.size()), valBuffer(tensors.size()), loopStack(),
      curLv(tensors.size(), 0) {
  for (size_t i = 0, e = tensors.size(); i < e; i++) {
    auto t = tensors[i];
    auto rtp = t.getType().cast<RankedTensorType>();
    auto rank = static_cast<size_t>(rtp.getRank());
    auto enc = getSparseTensorEncoding(rtp);
    if (enc)
      for (auto dimTp : enc.getDimLevelType())
        dims[i].push_back(dimTp);
    else
      dims[i].assign(rank, SparseTensorEncodingAttr::DimLevelType::Dense);

    // Initialize using empty value.
    pidxs[i].assign(rank, Value());
    coord[i].assign(rank, Value());
    highs[i].assign(rank, Value());
    sizes[i].assign(rank, Value());
    ptrBuffer[i].assign(rank, Value());
    idxBuffer[i].assign(rank, Value());
  }
}

void SparseTensorLoopEmitter::initializeLoopEmit(OpBuilder &builder,
                                                 Location loc) {
  // For every tensor, find lower and upper bound on dimensions, set the
  // same bounds on loop indices, and obtain dense or sparse buffer(s).
  // TODO: Provides ability to generate loop on output buffer (with undef
  // dim level in Merger in GenericOp Sparsification).
  for (size_t t = 0, e = tensors.size(); t < e; t++) {
    auto tensor = tensors[t];
    auto rtp = tensor.getType().cast<RankedTensorType>();
    auto rank = rtp.getRank();
    auto shape = rtp.getShape();
    auto enc = getSparseTensorEncoding(rtp);
    auto dynShape = {ShapedType::kDynamicSize};
    // Scan all dimensions of current tensor.
    for (int64_t d = 0; d < rank; d++) {
      // This should be called only once at beginning.
      assert(!ptrBuffer[t][d] && !idxBuffer[t][d] && !sizes[t][d] &&
             !highs[t][d]);
      // Handle sparse storage schemes.
      if (isCompressedDim(dims[t][d])) {
        auto ptrTp =
            MemRefType::get(dynShape, getPointerOverheadType(builder, enc));
        auto indTp =
            MemRefType::get(dynShape, getIndexOverheadType(builder, enc));
        auto dim = builder.getIndexAttr(d);
        // Generate sparse primitives to obtains pointer and indices.
        ptrBuffer[t][d] = builder.create<ToPointersOp>(loc, ptrTp, tensor, dim);
        idxBuffer[t][d] = builder.create<ToIndicesOp>(loc, indTp, tensor, dim);
      } else if (isSingletonDim(dims[t][d])) {
        llvm_unreachable("TODO: not implemented yet");
      }

      // Find upper bound in current dimension.
      unsigned p = toOrigDim(enc, d);
      Value up = mlir::linalg::createOrFoldDimOp(builder, loc, tensor, p);
      sizes[t][d] = highs[t][d] = up;
    }
    // Perform the required bufferization. Dense inputs materialize
    // from the input tensors. Dense outputs need special handling.
    // Sparse inputs use sparse primitives to obtain the values.
    Type elementType = rtp.getElementType();

    if (!enc) {
      // Non-annotated dense tensors.
      auto denseTp = MemRefType::get(shape, elementType);
      // This is not the output tensor
      valBuffer[t] =
          builder.create<bufferization::ToMemrefOp>(loc, denseTp, tensor);
    } else {
      // Annotated sparse tensors.
      auto dynShape = {ShapedType::kDynamicSize};
      auto sparseTp = MemRefType::get(dynShape, elementType);
      valBuffer[t] = builder.create<ToValuesOp>(loc, sparseTp, tensor);
    }
    // Prepare to enter the first dim for all (input) tensors
    prepareLoopOverTensorAtDim(builder, loc, t, 0);
  }
}

Operation *SparseTensorLoopEmitter::enterLoopOverTensorAtDim(
    OpBuilder &builder, Location loc, size_t tid, size_t dim,
    ArrayRef<Value> reduc) {
  assert(dims[tid].size() > dim);
  // We can not re-enter the same level.
  assert(!coord[tid][dim]);
  Value step = constantIndex(builder, loc, 1);
  bool isCompressed = isCompressedDim(dims[tid][dim]);
  assert(isDenseDim(dims[tid][dim]) || isCompressedDim(dims[tid][dim]));

  Value lo = isCompressed ? pidxs[tid][dim] : constantIndex(builder, loc, 0);
  Value hi = highs[tid][dim];

  // TODO: support reduction.
  if (!reduc.empty())
    llvm_unreachable("TODO: not implemented yet");

  scf::ForOp forOp = builder.create<scf::ForOp>(loc, lo, hi, step, reduc);
  builder.setInsertionPointToStart(forOp.getBody());
  Value iv = forOp.getInductionVar();
  Operation *loop = forOp;

  assert(iv);
  if (isCompressed) {
    pidxs[tid][dim] = iv;
    // Generating a load on the indices array yields the coordinate.
    Value ptr = idxBuffer[tid][dim];
    // TODO: generates load for vector value.
    coord[tid][dim] = genIndexLoad(builder, loc, ptr, iv);
  } else {
    // Dense tensor, the coordinates is the inducation variable.
    coord[tid][dim] = iv;
    // generate pidx for dense dim (pidx = i * sz + j)
    // TODO: handle vector loop.
    Value p = dim == 0 ? constantIndex(builder, loc, 0) : pidxs[tid][dim - 1];
    Value mul = builder.create<arith::MulIOp>(loc, sizes[tid][dim], p);
    Value add = builder.create<arith::AddIOp>(loc, mul, iv);
    pidxs[tid][dim] = add;
  }

  // Prepares for next dim if this is not currently the innermost dimension.
  if (dim != dims[tid].size() - 1)
    prepareLoopOverTensorAtDim(builder, loc, tid, dim + 1);

  loopStack.push_back(LoopLevelInfo({tid}, {dim}, coord[tid][dim]));
  return loop;
}

void SparseTensorLoopEmitter::enterCoiterationOverTensorsAtDims(
    OpBuilder &builder, Location loc, ArrayRef<size_t> ts,
    ArrayRef<size_t> ds) {
  llvm_unreachable("TODO: unimplemented");
}

bool SparseTensorLoopEmitter::prepareLoopOverTensorAtDim(OpBuilder &builder,
                                                         Location loc,
                                                         size_t tid,
                                                         size_t dim) {
  // TODO: generate loop iteration on output tensor based on the shape
  // instead of pointer/indices arrays.
  assert(dims[tid].size() > dim);

  if (isDenseDim(dims[tid][dim]))
    return false;

  // Either the first dimension, or the previous dimension has been set.
  assert(dim == 0 || pidxs[tid][dim - 1]);
  if (isCompressedDim(dims[tid][dim])) {
    Value ptr = ptrBuffer[tid][dim];
    Value c1 = constantIndex(builder, loc, 1);
    Value pLo = dim == 0 ? constantIndex(builder, loc, 0) : pidxs[tid][dim - 1];
    Value pHi = builder.create<arith::AddIOp>(loc, pLo, c1);

    pidxs[tid][dim] = genIndexLoad(builder, loc, ptr, pLo);
    highs[tid][dim] = genIndexLoad(builder, loc, ptr, pHi);

    return true;
  }

  if (isSingletonDim(dims[tid][dim]))
    llvm_unreachable("TODO: not implemented yet");

  llvm_unreachable("Unrecognizable dimesion type!");
}

Value SparseTensorLoopEmitter::emitExtraLocalsForTensorsAtDims(
    OpBuilder &builder, Location loc, size_t tid, size_t dim) {
  llvm_unreachable("TODO: not implemented yet");
}

void SparseTensorLoopEmitter::exitCurrentLoop() {
  // Clean up the values, it would help use to discover potential bug at a
  // earlier stage (instead of silently using a wrong value).
  LoopLevelInfo &loopInfo = loopStack.back();
  assert(loopInfo.tensors.size() == loopInfo.dims.size());
  for (auto info : llvm::zip(loopInfo.tensors, loopInfo.dims)) {
    auto tid = std::get<0>(info);
    auto dim = std::get<1>(info);
    assert(pidxs[tid][dim] && coord[tid][dim] && highs[tid][dim]);
    // Reset to null.
    pidxs[tid][dim] = Value();
    coord[tid][dim] = Value();
    if (!isDenseDim(dims[tid][dim]))
      // Dense dimension, high is fixed.
      highs[tid][dim] = Value();
  }
  loopStack.pop_back();
}

//===----------------------------------------------------------------------===//
// ExecutionEngine/SparseTensorUtils helper functions.
//===----------------------------------------------------------------------===//

OverheadType mlir::sparse_tensor::overheadTypeEncoding(unsigned width) {
  switch (width) {
  case 64:
    return OverheadType::kU64;
  case 32:
    return OverheadType::kU32;
  case 16:
    return OverheadType::kU16;
  case 8:
    return OverheadType::kU8;
  case 0:
    return OverheadType::kIndex;
  }
  llvm_unreachable("Unsupported overhead bitwidth");
}

OverheadType mlir::sparse_tensor::overheadTypeEncoding(Type tp) {
  if (tp.isIndex())
    return OverheadType::kIndex;
  if (auto intTp = tp.dyn_cast<IntegerType>())
    return overheadTypeEncoding(intTp.getWidth());
  llvm_unreachable("Unknown overhead type");
}

Type mlir::sparse_tensor::getOverheadType(Builder &builder, OverheadType ot) {
  switch (ot) {
  case OverheadType::kIndex:
    return builder.getIndexType();
  case OverheadType::kU64:
    return builder.getIntegerType(64);
  case OverheadType::kU32:
    return builder.getIntegerType(32);
  case OverheadType::kU16:
    return builder.getIntegerType(16);
  case OverheadType::kU8:
    return builder.getIntegerType(8);
  }
  llvm_unreachable("Unknown OverheadType");
}

OverheadType mlir::sparse_tensor::pointerOverheadTypeEncoding(
    const SparseTensorEncodingAttr &enc) {
  return overheadTypeEncoding(enc.getPointerBitWidth());
}

OverheadType mlir::sparse_tensor::indexOverheadTypeEncoding(
    const SparseTensorEncodingAttr &enc) {
  return overheadTypeEncoding(enc.getIndexBitWidth());
}

Type mlir::sparse_tensor::getPointerOverheadType(
    Builder &builder, const SparseTensorEncodingAttr &enc) {
  return getOverheadType(builder, pointerOverheadTypeEncoding(enc));
}

Type mlir::sparse_tensor::getIndexOverheadType(
    Builder &builder, const SparseTensorEncodingAttr &enc) {
  return getOverheadType(builder, indexOverheadTypeEncoding(enc));
}

// TODO: Adjust the naming convention for the constructors of
// `OverheadType` so we can use the `MLIR_SPARSETENSOR_FOREVERY_O` x-macro
// here instead of `MLIR_SPARSETENSOR_FOREVERY_FIXED_O`; to further reduce
// the possibility of typo bugs or things getting out of sync.
StringRef mlir::sparse_tensor::overheadTypeFunctionSuffix(OverheadType ot) {
  switch (ot) {
  case OverheadType::kIndex:
    return "0";
#define CASE(ONAME, O)                                                         \
  case OverheadType::kU##ONAME:                                                \
    return #ONAME;
    MLIR_SPARSETENSOR_FOREVERY_FIXED_O(CASE)
#undef CASE
  }
  llvm_unreachable("Unknown OverheadType");
}

StringRef mlir::sparse_tensor::overheadTypeFunctionSuffix(Type tp) {
  return overheadTypeFunctionSuffix(overheadTypeEncoding(tp));
}

PrimaryType mlir::sparse_tensor::primaryTypeEncoding(Type elemTp) {
  if (elemTp.isF64())
    return PrimaryType::kF64;
  if (elemTp.isF32())
    return PrimaryType::kF32;
  if (elemTp.isF16())
    return PrimaryType::kF16;
  if (elemTp.isBF16())
    return PrimaryType::kBF16;
  if (elemTp.isInteger(64))
    return PrimaryType::kI64;
  if (elemTp.isInteger(32))
    return PrimaryType::kI32;
  if (elemTp.isInteger(16))
    return PrimaryType::kI16;
  if (elemTp.isInteger(8))
    return PrimaryType::kI8;
  if (auto complexTp = elemTp.dyn_cast<ComplexType>()) {
    auto complexEltTp = complexTp.getElementType();
    if (complexEltTp.isF64())
      return PrimaryType::kC64;
    if (complexEltTp.isF32())
      return PrimaryType::kC32;
  }
  llvm_unreachable("Unknown primary type");
}

StringRef mlir::sparse_tensor::primaryTypeFunctionSuffix(PrimaryType pt) {
  switch (pt) {
#define CASE(VNAME, V)                                                         \
  case PrimaryType::k##VNAME:                                                  \
    return #VNAME;
    MLIR_SPARSETENSOR_FOREVERY_V(CASE)
#undef CASE
  }
  llvm_unreachable("Unknown PrimaryType");
}

StringRef mlir::sparse_tensor::primaryTypeFunctionSuffix(Type elemTp) {
  return primaryTypeFunctionSuffix(primaryTypeEncoding(elemTp));
}

DimLevelType mlir::sparse_tensor::dimLevelTypeEncoding(
    SparseTensorEncodingAttr::DimLevelType dlt) {
  switch (dlt) {
  case SparseTensorEncodingAttr::DimLevelType::Dense:
    return DimLevelType::kDense;
  case SparseTensorEncodingAttr::DimLevelType::Compressed:
    return DimLevelType::kCompressed;
  case SparseTensorEncodingAttr::DimLevelType::CompressedNu:
    return DimLevelType::kCompressedNu;
  case SparseTensorEncodingAttr::DimLevelType::CompressedNo:
    return DimLevelType::kCompressedNo;
  case SparseTensorEncodingAttr::DimLevelType::CompressedNuNo:
    return DimLevelType::kCompressedNuNo;
  case SparseTensorEncodingAttr::DimLevelType::Singleton:
    return DimLevelType::kSingleton;
  case SparseTensorEncodingAttr::DimLevelType::SingletonNu:
    return DimLevelType::kSingletonNu;
  case SparseTensorEncodingAttr::DimLevelType::SingletonNo:
    return DimLevelType::kSingletonNo;
  case SparseTensorEncodingAttr::DimLevelType::SingletonNuNo:
    return DimLevelType::kSingletonNuNo;
  }
  llvm_unreachable("Unknown SparseTensorEncodingAttr::DimLevelType");
}

//===----------------------------------------------------------------------===//
// Misc code generators.
//===----------------------------------------------------------------------===//

mlir::Attribute mlir::sparse_tensor::getOneAttr(Builder &builder, Type tp) {
  if (tp.isa<FloatType>())
    return builder.getFloatAttr(tp, 1.0);
  if (tp.isa<IndexType>())
    return builder.getIndexAttr(1);
  if (auto intTp = tp.dyn_cast<IntegerType>())
    return builder.getIntegerAttr(tp, APInt(intTp.getWidth(), 1));
  if (tp.isa<RankedTensorType, VectorType>()) {
    auto shapedTp = tp.cast<ShapedType>();
    if (auto one = getOneAttr(builder, shapedTp.getElementType()))
      return DenseElementsAttr::get(shapedTp, one);
  }
  llvm_unreachable("Unsupported attribute type");
}

Value mlir::sparse_tensor::genIsNonzero(OpBuilder &builder, mlir::Location loc,
                                        Value v) {
  Type tp = v.getType();
  Value zero = constantZero(builder, loc, tp);
  if (tp.isa<FloatType>())
    return builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE, v,
                                         zero);
  if (tp.isIntOrIndex())
    return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, v,
                                         zero);
  if (tp.dyn_cast<ComplexType>())
    return builder.create<complex::NotEqualOp>(loc, v, zero);
  llvm_unreachable("Non-numeric type");
}

void mlir::sparse_tensor::genReshapeDstShape(
    Location loc, PatternRewriter &rewriter, SmallVector<Value, 4> &dstShape,
    ArrayRef<Value> srcShape, ArrayRef<int64_t> staticDstShape,
    ArrayRef<ReassociationIndices> reassociation) {
  // Collapse shape.
  if (reassociation.size() < srcShape.size()) {
    unsigned start = 0;
    for (const auto &map : llvm::enumerate(reassociation)) {
      auto dstDim = constantIndex(rewriter, loc, 1);
      for (unsigned i = start; i < start + map.value().size(); i++) {
        dstDim = rewriter.create<arith::MulIOp>(loc, dstDim, srcShape[i]);
      }
      dstShape.push_back(dstDim);
      start = start + map.value().size();
    }
    assert(start == srcShape.size());
    return;
  }

  // Expand shape.
  assert(reassociation.size() == srcShape.size());
  unsigned start = 0;
  // Expand the i-th dimension in srcShape.
  for (unsigned i = 0, size = srcShape.size(); i < size; i++) {
    auto map = reassociation[i];
    auto srcDim = srcShape[i];
    // Iterate through dimensions expanded from the i-th dimension.
    for (unsigned j = start; j < start + map.size(); j++) {
      // There can be only one dynamic sized dimension among dimensions expanded
      // from the i-th dimension in srcShape. For example, if srcDim = 8, then
      // the expanded shape could be <2x?x2>, but not <2x?x?>.
      if (staticDstShape[j] == ShapedType::kDynamicSize) {
        // The expanded dimension has dynamic size. We compute the dimension
        // by dividing srcDim by the product of the static dimensions.
        int64_t product = 1;
        for (unsigned k = start; k < start + map.size(); k++) {
          if (staticDstShape[k] != ShapedType::kDynamicSize) {
            product *= staticDstShape[k];
          }
        }
        // Compute the dynamic dimension size.
        Value productVal = constantIndex(rewriter, loc, product);
        Value dynamicSize =
            rewriter.create<arith::DivUIOp>(loc, srcDim, productVal);
        dstShape.push_back(dynamicSize);
      } else {
        // The expanded dimension is statically known.
        dstShape.push_back(constantIndex(rewriter, loc, staticDstShape[j]));
      }
    }
    start = start + map.size();
  }
  assert(start == staticDstShape.size());
}

void mlir::sparse_tensor::translateIndicesArray(
    OpBuilder &builder, Location loc,
    ArrayRef<ReassociationIndices> reassociation, ValueRange srcIndices,
    ArrayRef<Value> srcShape, ArrayRef<Value> dstShape,
    SmallVectorImpl<Value> &dstIndices) {
  unsigned i = 0;
  unsigned start = 0;
  unsigned dstRank = dstShape.size();
  unsigned srcRank = srcShape.size();
  assert(srcRank == srcIndices.size());
  bool isCollapse = srcRank > dstRank;
  ArrayRef<Value> shape = isCollapse ? srcShape : dstShape;
  // Iterate over reassociation map.
  for (const auto &map : llvm::enumerate(reassociation)) {
    // Prepare strides information in dimension slice.
    Value linear = constantIndex(builder, loc, 1);
    for (unsigned j = start, end = start + map.value().size(); j < end; j++) {
      linear = builder.create<arith::MulIOp>(loc, linear, shape[j]);
    }
    // Start expansion.
    Value val;
    if (!isCollapse)
      val = srcIndices[i];
    // Iterate over dimension slice.
    for (unsigned j = start, end = start + map.value().size(); j < end; j++) {
      linear = builder.create<arith::DivUIOp>(loc, linear, shape[j]);
      if (isCollapse) {
        Value old = srcIndices[j];
        Value mul = builder.create<arith::MulIOp>(loc, old, linear);
        val = val ? builder.create<arith::AddIOp>(loc, val, mul) : mul;
      } else {
        Value old = val;
        val = builder.create<arith::DivUIOp>(loc, val, linear);
        assert(dstIndices.size() == j);
        dstIndices.push_back(val);
        val = builder.create<arith::RemUIOp>(loc, old, linear);
      }
    }
    // Finalize collapse.
    if (isCollapse) {
      assert(dstIndices.size() == i);
      dstIndices.push_back(val);
    }
    start += map.value().size();
    i++;
  }
  assert(dstIndices.size() == dstRank);
}
