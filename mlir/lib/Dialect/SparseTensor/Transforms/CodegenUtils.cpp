//===- CodegenUtils.cpp - Utilities for generating MLIR -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CodegenUtils.h"
#include "SparseTensorStorageLayout.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include <optional>

using namespace mlir;
using namespace mlir::sparse_tensor;

/// If the tensor is a sparse constant, generates and returns the pair of
/// the constants for the indices and the values.
static std::optional<std::pair<Value, Value>>
genSplitSparseConstant(OpBuilder &builder, Location loc, Value tensor) {
  if (auto constOp = tensor.getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = constOp.getValue().dyn_cast<SparseElementsAttr>()) {
      DenseElementsAttr indicesAttr = attr.getIndices();
      Value indices = builder.create<arith::ConstantOp>(loc, indicesAttr);
      DenseElementsAttr valuesAttr = attr.getValues();
      Value values = builder.create<arith::ConstantOp>(loc, valuesAttr);
      return std::make_pair(indices, values);
    }
  }
  return {};
}

/// Generates the code to copy the index at indices[ivs] to ind, and return
/// the value at value[ivs].
static Value genIndexAndValueForSparse(OpBuilder &builder, Location loc,
                                       Value indices, Value values,
                                       SmallVectorImpl<Value> &indicesArray,
                                       ValueRange ivs, unsigned rank) {
  for (unsigned i = 0; i < rank; i++) {
    Value idx = constantIndex(builder, loc, i);
    Value val = builder.create<tensor::ExtractOp>(loc, indices,
                                                  ValueRange{ivs[0], idx});
    val = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), val);
    // builder.create<memref::StoreOp>(loc, val, ind, idx);
    indicesArray.push_back(val);
  }
  return builder.create<tensor::ExtractOp>(loc, values, ivs[0]);
}

/// Generates the code to read the value from tensor[ivs], and conditionally
/// stores the indices ivs to the memory in ind. The generated code looks like
/// the following and the insertion point after this routine is inside the
/// if-then branch behind the assignment to ind. This is to ensure that the
/// code that uses the ind, such as an addEltX call generated after, is inside
/// the if-then branch.
///    if (tensor[ivs] != 0)
///      ind = ivs
static Value genIndexAndValueForDense(OpBuilder &builder, Location loc,
                                      Value tensor,
                                      SmallVectorImpl<Value> &indicesArray,
                                      ValueRange ivs) {
  Value val = genValueForDense(builder, loc, tensor, ivs);
  indicesArray.append(ivs.begin(), ivs.end());
  return val;
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

OverheadType
mlir::sparse_tensor::pointerOverheadTypeEncoding(SparseTensorEncodingAttr enc) {
  return overheadTypeEncoding(enc.getPointerBitWidth());
}

OverheadType
mlir::sparse_tensor::indexOverheadTypeEncoding(SparseTensorEncodingAttr enc) {
  return overheadTypeEncoding(enc.getIndexBitWidth());
}

Type mlir::sparse_tensor::getPointerOverheadType(Builder &builder,
                                                 SparseTensorEncodingAttr enc) {
  return getOverheadType(builder, pointerOverheadTypeEncoding(enc));
}

Type mlir::sparse_tensor::getIndexOverheadType(Builder &builder,
                                               SparseTensorEncodingAttr enc) {
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
    Location loc, PatternRewriter &rewriter, SmallVectorImpl<Value> &dstShape,
    ArrayRef<Value> srcShape, ArrayRef<StaticSize> staticDstShape,
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
    const auto &map = reassociation[i];
    auto srcDim = srcShape[i];
    // Iterate through dimensions expanded from the i-th dimension.
    for (unsigned j = start; j < start + map.size(); j++) {
      // There can be only one dynamic sized dimension among dimensions
      // expanded from the i-th dimension in srcShape.
      // For example, if srcDim = 8, then the expanded shape could be <2x?x2>,
      // but not <2x?x?>.
      if (staticDstShape[j] == ShapedType::kDynamic) {
        // The expanded dimension has dynamic size. We compute the dimension
        // by dividing srcDim by the product of the static dimensions.
        StaticSize product = 1;
        for (unsigned k = start; k < start + map.size(); k++) {
          if (staticDstShape[k] != ShapedType::kDynamic) {
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

FlatSymbolRefAttr mlir::sparse_tensor::getFunc(ModuleOp module, StringRef name,
                                               TypeRange resultType,
                                               ValueRange operands,
                                               EmitCInterface emitCInterface) {
  MLIRContext *context = module.getContext();
  auto result = SymbolRefAttr::get(context, name);
  auto func = module.lookupSymbol<func::FuncOp>(result.getAttr());
  if (!func) {
    OpBuilder moduleBuilder(module.getBodyRegion());
    func = moduleBuilder.create<func::FuncOp>(
        module.getLoc(), name,
        FunctionType::get(context, operands.getTypes(), resultType));
    func.setPrivate();
    if (static_cast<bool>(emitCInterface))
      func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(context));
  }
  return result;
}

func::CallOp mlir::sparse_tensor::createFuncCall(
    OpBuilder &builder, Location loc, StringRef name, TypeRange resultType,
    ValueRange operands, EmitCInterface emitCInterface) {
  auto module = builder.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
  FlatSymbolRefAttr fn =
      getFunc(module, name, resultType, operands, emitCInterface);
  return builder.create<func::CallOp>(loc, resultType, fn, operands);
}

Type mlir::sparse_tensor::getOpaquePointerType(OpBuilder &builder) {
  return LLVM::LLVMPointerType::get(builder.getI8Type());
}

Value mlir::sparse_tensor::genAlloca(OpBuilder &builder, Location loc,
                                     unsigned sz, Type tp, bool staticShape) {
  if (staticShape) {
    auto memTp = MemRefType::get({sz}, tp);
    return builder.create<memref::AllocaOp>(loc, memTp);
  }
  return genAlloca(builder, loc, constantIndex(builder, loc, sz), tp);
}

Value mlir::sparse_tensor::genAlloca(OpBuilder &builder, Location loc, Value sz,
                                     Type tp) {
  auto memTp = MemRefType::get({ShapedType::kDynamic}, tp);
  return builder.create<memref::AllocaOp>(loc, memTp, ValueRange{sz});
}

Value mlir::sparse_tensor::genAllocaScalar(OpBuilder &builder, Location loc,
                                           Type tp) {
  return builder.create<memref::AllocaOp>(loc, MemRefType::get({}, tp));
}

Value mlir::sparse_tensor::allocaBuffer(OpBuilder &builder, Location loc,
                                        ValueRange values) {
  const unsigned sz = values.size();
  assert(sz >= 1);
  Value buffer = genAlloca(builder, loc, sz, values[0].getType());
  for (unsigned i = 0; i < sz; i++) {
    Value idx = constantIndex(builder, loc, i);
    builder.create<memref::StoreOp>(loc, values[i], buffer, idx);
  }
  return buffer;
}

Value mlir::sparse_tensor::allocDenseTensor(OpBuilder &builder, Location loc,
                                            RankedTensorType tensorTp,
                                            ValueRange sizes) {
  Type elemTp = tensorTp.getElementType();
  auto shape = tensorTp.getShape();
  auto memTp = MemRefType::get(shape, elemTp);
  SmallVector<Value> dynamicSizes;
  for (unsigned i = 0, rank = tensorTp.getRank(); i < rank; i++) {
    if (shape[i] == ShapedType::kDynamic)
      dynamicSizes.push_back(sizes[i]);
  }
  Value mem = builder.create<memref::AllocOp>(loc, memTp, dynamicSizes);
  Value zero = constantZero(builder, loc, elemTp);
  builder.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{mem});
  return mem;
}

void mlir::sparse_tensor::deallocDenseTensor(OpBuilder &builder, Location loc,
                                             Value buffer) {
  builder.create<memref::DeallocOp>(loc, buffer);
}

Value mlir::sparse_tensor::genValueForDense(OpBuilder &builder, Location loc,
                                            Value tensor, ValueRange ivs) {
  Value val = builder.create<tensor::ExtractOp>(loc, tensor, ivs);
  Value cond = genIsNonzero(builder, loc, val);
  scf::IfOp ifOp = builder.create<scf::IfOp>(loc, cond, /*else*/ false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  return val;
}

// FIXME:
// 1. Dense tensors loop should be generated by loop emitter.
// 2. Support reduction variables to propagate SSA chains properly.
void mlir::sparse_tensor::genDenseTensorOrSparseConstantIterLoop(
    OpBuilder &builder, Location loc, Value src, unsigned rank,
    function_ref<void(OpBuilder &, Location, Value, ValueRange)> bodyBuilder) {
  SmallVector<Value> indicesArray;
  SmallVector<Value> lo;
  SmallVector<Value> hi;
  SmallVector<Value> st;
  Value zero = constantIndex(builder, loc, 0);
  Value one = constantIndex(builder, loc, 1);
  auto indicesValues = genSplitSparseConstant(builder, loc, src);
  bool isCOOConstant = indicesValues.has_value();
  Value indices;
  Value values;
  if (isCOOConstant) {
    indices = indicesValues->first;
    values = indicesValues->second;
    lo.push_back(zero);
    hi.push_back(linalg::createOrFoldDimOp(builder, loc, values, 0));
    st.push_back(one);
  } else {
    for (unsigned i = 0; i < rank; i++) {
      lo.push_back(zero);
      hi.push_back(linalg::createOrFoldDimOp(builder, loc, src, i));
      st.push_back(one);
    }
  }

  scf::buildLoopNest(
      builder, loc, lo, hi, st, {},
      [&](OpBuilder &builder, Location loc, ValueRange ivs,
          ValueRange args) -> scf::ValueVector {
        Value val;
        if (isCOOConstant)
          val = genIndexAndValueForSparse(builder, loc, indices, values,
                                          indicesArray, ivs, rank);
        else
          val = genIndexAndValueForDense(builder, loc, src, indicesArray, ivs);
        bodyBuilder(builder, loc, val, indicesArray);
        return {};
      });
}

void mlir::sparse_tensor::sizesFromSrc(OpBuilder &builder,
                                       SmallVectorImpl<Value> &sizes,
                                       Location loc, Value src) {
  const Dimension dimRank = getSparseTensorType(src).getDimRank();
  for (Dimension d = 0; d < dimRank; d++)
    sizes.push_back(linalg::createOrFoldDimOp(builder, loc, src, d));
}

Operation *mlir::sparse_tensor::getTop(Operation *op) {
  for (; isa<scf::ForOp>(op->getParentOp()) ||
         isa<scf::WhileOp>(op->getParentOp()) ||
         isa<scf::ParallelOp>(op->getParentOp()) ||
         isa<scf::IfOp>(op->getParentOp());
       op = op->getParentOp())
    ;
  return op;
}

void sparse_tensor::foreachInSparseConstant(
    Location loc, RewriterBase &rewriter, SparseElementsAttr attr,
    AffineMap order, function_ref<void(ArrayRef<Value>, Value)> callback) {
  Dimension dimRank = getSparseTensorType(attr).getDimRank();
  // Foreach on constant.
  DenseElementsAttr indicesAttr = attr.getIndices();
  DenseElementsAttr valuesAttr = attr.getValues();

  using CooValuePair = std::pair<SmallVector<IntegerAttr>, Attribute>;
  SmallVector<CooValuePair> cooV;
  for (size_t i = 0, nse = valuesAttr.size(); i < nse; i++) {
    cooV.emplace_back();
    for (Dimension j = 0; j < dimRank; j++) {
      auto coordAttr = indicesAttr.getValues<IntegerAttr>()[i * dimRank + j];
      cooV.back().first.push_back(coordAttr);
    }
    auto valAttr = valuesAttr.getValues<Attribute>()[i];
    cooV.back().second = valAttr;
  }

  // Sorts the sparse element attribute based on coordinates.
  std::sort(cooV.begin(), cooV.end(),
            [order](const CooValuePair &lhs, const CooValuePair &rhs) {
              const SmallVectorImpl<IntegerAttr> &lc = lhs.first;
              const SmallVectorImpl<IntegerAttr> &rc = rhs.first;
              for (size_t i = 0, e = lc.size(); i < e; i++) {
                auto l =
                    order
                        ? order.getResult(i).cast<AffineDimExpr>().getPosition()
                        : i;
                if (lc[l].getInt() == rc[l].getInt())
                  continue;
                return lc[l].getInt() < rc[l].getInt();
              }
              llvm_unreachable("no equal coordinate in sparse element attr");
            });

  SmallVector<Value> coords;
  for (size_t i = 0, nse = valuesAttr.size(); i < nse; i++) {
    coords.clear();
    for (Dimension j = 0; j < dimRank; j++) {
      auto coordAttr = cooV[i].first[j];
      auto coord =
          rewriter.create<arith::ConstantIndexOp>(loc, coordAttr.getInt());
      // Remaps coordinates.
      coords.push_back(coord);
    }
    Value val;
    if (attr.getElementType().isa<ComplexType>()) {
      auto valAttr = cooV[i].second.cast<ArrayAttr>();
      val = rewriter.create<complex::ConstantOp>(loc, attr.getElementType(),
                                                 valAttr);
    } else {
      auto valAttr = cooV[i].second.cast<TypedAttr>();
      // Remaps value.
      val = rewriter.create<arith::ConstantOp>(loc, valAttr);
    }
    assert(val);
    callback(coords, val);
  }
}

void sparse_tensor::storeIndices(OpBuilder &builder, Location loc,
                                 unsigned size, Value ind, ValueRange ivs,
                                 unsigned offsetDim, Value offset) {
#ifndef NDEBUG
  const auto memTp = ind.getType().cast<MemRefType>();
  (void)memTp;
  assert(memTp.getRank() == 1);
  const DynSize memSh = memTp.getDimSize(0);
  (void)memSh;
  assert(ShapedType::isDynamic(memSh) || memSh == static_cast<DynSize>(size));
  assert(ivs.size() == static_cast<size_t>(size));
  assert(offsetDim < size);
#endif // NDEBUG

  for (unsigned i = 0; i < size; i++) {
    Value idx = ivs[i];
    if (offsetDim == i && offset)
      idx = builder.create<arith::AddIOp>(loc, idx, offset);
    builder.create<memref::StoreOp>(loc, idx, ind,
                                    constantIndex(builder, loc, i));
  }
}

Value sparse_tensor::reshapeValuesToLevels(OpBuilder &builder, Location loc,
                                           SparseTensorEncodingAttr enc,
                                           ValueRange dimSizes,
                                           Value valuesBuffer,
                                           Value idxBuffer) {
  // Use the `idxBuffer` to store the level sizes.
  const Level lvlRank = enc.getLvlRank();
  SmallVector<Value> lvlSizes;
  lvlSizes.reserve(lvlRank);
  for (Level l = 0; l < lvlRank; l++)
    // FIXME: `toOrigDim` is deprecated.
    lvlSizes.push_back(dimSizes[toOrigDim(enc, l)]);
  storeIndices(builder, loc, lvlRank, idxBuffer, lvlSizes);
  // The memref ReshapeOp requires the sizes buffer to have a static
  // shape.
  const auto iTp = builder.getIndexType();
  const SmallVector<DynSize, 1> idxBufferShape{static_cast<DynSize>(lvlRank)};
  const auto idxBufferTp = MemRefType::get(idxBufferShape, iTp);
  idxBuffer = builder.create<memref::CastOp>(loc, idxBufferTp, idxBuffer);
  const SmallVector<DynSize> resShape(lvlRank, ShapedType::kDynamic);
  const Type elemTp = getMemRefType(valuesBuffer).getElementType();
  const auto resTp = MemRefType::get(resShape, elemTp);
  return builder.create<memref::ReshapeOp>(loc, resTp, valuesBuffer, idxBuffer);
}

Value sparse_tensor::genToPointers(OpBuilder &builder, Location loc,
                                   Value tensor, Level lvl) {
  const auto srcTp = getSparseTensorType(tensor);
  const Type ptrTp = getPointerOverheadType(builder, srcTp.getEncoding());
  const Type memTp = get1DMemRefType(ptrTp, /*withLayout=*/false);
  return builder.create<ToPointersOp>(loc, memTp, tensor,
                                      builder.getIndexAttr(lvl));
}

Value sparse_tensor::genToIndices(OpBuilder &builder, Location loc,
                                  Value tensor, Level lvl, Level cooStart) {
  const auto srcTp = getSparseTensorType(tensor);
  const Type idxTp = getIndexOverheadType(builder, srcTp.getEncoding());
  const Type memTp = get1DMemRefType(idxTp, /*withLayout=*/lvl >= cooStart);
  return builder.create<ToIndicesOp>(loc, memTp, tensor,
                                     builder.getIndexAttr(lvl));
}

Value sparse_tensor::genToValues(OpBuilder &builder, Location loc,
                                 Value tensor) {
  RankedTensorType srcTp = getRankedTensorType(tensor);
  Type valTp = get1DMemRefType(srcTp.getElementType(),
                               /*withLayout=*/false);
  return builder.create<ToValuesOp>(loc, valTp, tensor);
}

Value sparse_tensor::genValMemSize(OpBuilder &builder, Location loc,
                                   Value tensor) {
  return getDescriptorFromTensorTuple(tensor).getValMemSize(builder, loc);
}
