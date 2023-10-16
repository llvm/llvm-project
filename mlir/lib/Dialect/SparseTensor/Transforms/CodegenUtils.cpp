//===- CodegenUtils.cpp - Utilities for generating MLIR -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CodegenUtils.h"
#include "SparseTensorDescriptor.h"

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
  if (auto intTp = dyn_cast<IntegerType>(tp))
    return overheadTypeEncoding(intTp.getWidth());
  llvm_unreachable("Unknown overhead type");
}

// TODO: should offer an overload of this that takes a `MLIRContext*`
// instead of the builder, similar to `detail::getIntegerOrIndexType`.
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
mlir::sparse_tensor::posTypeEncoding(SparseTensorEncodingAttr enc) {
  return overheadTypeEncoding(enc.getPosWidth());
}

OverheadType
mlir::sparse_tensor::crdTypeEncoding(SparseTensorEncodingAttr enc) {
  return overheadTypeEncoding(enc.getCrdWidth());
}

// TODO: we ought to add some `static_assert` tests to ensure that the
// `STEA::get{Pos,Crd}Type` methods agree with `getOverheadType(builder,
// {pos,crd}OverheadTypeEncoding(enc))`

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
  if (auto complexTp = dyn_cast<ComplexType>(elemTp)) {
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

Value sparse_tensor::genCast(OpBuilder &builder, Location loc, Value value,
                             Type dstTp) {
  const Type srcTp = value.getType();
  if (srcTp == dstTp)
    return value;

  // int <=> index
  if (isa<IndexType>(srcTp) || isa<IndexType>(dstTp))
    return builder.create<arith::IndexCastOp>(loc, dstTp, value);

  const auto srcIntTp = dyn_cast_or_null<IntegerType>(srcTp);
  const bool isUnsignedCast = srcIntTp ? srcIntTp.isUnsigned() : false;
  return mlir::convertScalarToDtype(builder, loc, value, dstTp, isUnsignedCast);
}

Value sparse_tensor::genIndexLoad(OpBuilder &builder, Location loc, Value mem,
                                  Value s) {
  Value load = builder.create<memref::LoadOp>(loc, mem, s);
  if (!isa<IndexType>(load.getType())) {
    if (load.getType().getIntOrFloatBitWidth() < 64)
      load = builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), load);
    load =
        builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), load);
  }
  return load;
}

mlir::TypedAttr mlir::sparse_tensor::getOneAttr(Builder &builder, Type tp) {
  if (isa<FloatType>(tp))
    return builder.getFloatAttr(tp, 1.0);
  if (isa<IndexType>(tp))
    return builder.getIndexAttr(1);
  if (auto intTp = dyn_cast<IntegerType>(tp))
    return builder.getIntegerAttr(tp, APInt(intTp.getWidth(), 1));
  if (isa<RankedTensorType, VectorType>(tp)) {
    auto shapedTp = cast<ShapedType>(tp);
    if (auto one = getOneAttr(builder, shapedTp.getElementType()))
      return DenseElementsAttr::get(shapedTp, one);
  }
  llvm_unreachable("Unsupported attribute type");
}

Value mlir::sparse_tensor::genIsNonzero(OpBuilder &builder, mlir::Location loc,
                                        Value v) {
  Type tp = v.getType();
  Value zero = constantZero(builder, loc, tp);
  if (isa<FloatType>(tp))
    return builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE, v,
                                         zero);
  if (tp.isIntOrIndex())
    return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, v,
                                         zero);
  if (dyn_cast<ComplexType>(tp))
    return builder.create<complex::NotEqualOp>(loc, v, zero);
  llvm_unreachable("Non-numeric type");
}

void mlir::sparse_tensor::genReshapeDstShape(
    OpBuilder &builder, Location loc, SmallVectorImpl<Value> &dstShape,
    ArrayRef<Value> srcShape, ArrayRef<StaticSize> staticDstShape,
    ArrayRef<ReassociationIndices> reassociation) {
  // Collapse shape.
  if (reassociation.size() < srcShape.size()) {
    unsigned start = 0;
    for (const auto &map : llvm::enumerate(reassociation)) {
      auto dstDim = constantIndex(builder, loc, 1);
      for (unsigned i = start; i < start + map.value().size(); i++) {
        dstDim = builder.create<arith::MulIOp>(loc, dstDim, srcShape[i]);
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
        Value productVal = constantIndex(builder, loc, product);
        Value dynamicSize =
            builder.create<arith::DivUIOp>(loc, srcDim, productVal);
        dstShape.push_back(dynamicSize);
      } else {
        // The expanded dimension is statically known.
        dstShape.push_back(constantIndex(builder, loc, staticDstShape[j]));
      }
    }
    start = start + map.size();
  }
  assert(start == staticDstShape.size());
}

void mlir::sparse_tensor::reshapeCvs(
    OpBuilder &builder, Location loc,
    ArrayRef<ReassociationIndices> reassociation, // NOLINT
    ValueRange srcSizes, ValueRange srcCvs,       // NOLINT
    ValueRange dstSizes, SmallVectorImpl<Value> &dstCvs) {
  const unsigned srcRank = srcSizes.size();
  const unsigned dstRank = dstSizes.size();
  assert(srcRank == srcCvs.size() && "Source rank mismatch");
  const bool isCollapse = srcRank > dstRank;
  const ValueRange sizes = isCollapse ? srcSizes : dstSizes;
  // Iterate over reassociation map.
  unsigned i = 0;
  unsigned start = 0;
  for (const auto &map : llvm::enumerate(reassociation)) {
    // Prepare strides information in dimension slice.
    Value linear = constantIndex(builder, loc, 1);
    for (unsigned j = start, end = start + map.value().size(); j < end; j++) {
      linear = builder.create<arith::MulIOp>(loc, linear, sizes[j]);
    }
    // Start expansion.
    Value val;
    if (!isCollapse)
      val = srcCvs[i];
    // Iterate over dimension slice.
    for (unsigned j = start, end = start + map.value().size(); j < end; j++) {
      linear = builder.create<arith::DivUIOp>(loc, linear, sizes[j]);
      if (isCollapse) {
        const Value mul = builder.create<arith::MulIOp>(loc, srcCvs[j], linear);
        val = val ? builder.create<arith::AddIOp>(loc, val, mul) : mul;
      } else {
        const Value old = val;
        val = builder.create<arith::DivUIOp>(loc, val, linear);
        assert(dstCvs.size() == j);
        dstCvs.push_back(val);
        val = builder.create<arith::RemUIOp>(loc, old, linear);
      }
    }
    // Finalize collapse.
    if (isCollapse) {
      assert(dstCvs.size() == i);
      dstCvs.push_back(val);
    }
    start += map.value().size();
    i++;
  }
  assert(dstCvs.size() == dstRank);
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

Type mlir::sparse_tensor::getOpaquePointerType(MLIRContext *ctx) {
  return LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
}

Type mlir::sparse_tensor::getOpaquePointerType(Builder &builder) {
  return getOpaquePointerType(builder.getContext());
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
    OpBuilder &builder, Location loc, SparseElementsAttr attr, AffineMap order,
    function_ref<void(ArrayRef<Value>, Value)> callback) {
  const Dimension dimRank = getSparseTensorType(attr).getDimRank();
  const auto coordinates = attr.getIndices().getValues<IntegerAttr>();
  const auto values = attr.getValues().getValues<Attribute>();

  // This is like the `Element<V>` class in the runtime library, but for
  // MLIR attributes.  In the future we may want to move this out into
  // a proper class definition to help improve code legibility (e.g.,
  // `first` -> `coords`, `second` -> `value`) as well as being able
  // to factor out analogues of `ElementLT<V>` for the sort below, etc.
  using ElementAttr = std::pair<SmallVector<IntegerAttr>, Attribute>;

  // Construct the COO from the SparseElementsAttr.
  SmallVector<ElementAttr> elems;
  for (size_t i = 0, nse = values.size(); i < nse; i++) {
    elems.emplace_back();
    elems.back().second = values[i];
    auto &coords = elems.back().first;
    coords.reserve(dimRank);
    for (Dimension d = 0; d < dimRank; d++)
      coords.push_back(coordinates[i * dimRank + d]);
  }

  // Sorts the sparse element attribute based on coordinates.
  std::sort(elems.begin(), elems.end(),
            [order, dimRank](const ElementAttr &lhs, const ElementAttr &rhs) {
              const auto &lhsCoords = lhs.first;
              const auto &rhsCoords = rhs.first;
              for (Dimension d = 0; d < dimRank; d++) {
                // FIXME: This only makes sense for permutations.
                // And since we don't check that `order` is a permutation,
                // it can also cause OOB errors when we use `l`.
                const Level l = order ? order.getDimPosition(d) : d;
                if (lhsCoords[l].getInt() == rhsCoords[l].getInt())
                  continue;
                return lhsCoords[l].getInt() < rhsCoords[l].getInt();
              }
              if (std::addressof(lhs) == std::addressof(rhs))
                return false;
              llvm_unreachable("no equal coordinate in sparse element attr");
            });

  SmallVector<Value> cvs;
  cvs.reserve(dimRank);
  for (size_t i = 0, nse = values.size(); i < nse; i++) {
    // Remap coordinates.
    cvs.clear();
    for (Dimension d = 0; d < dimRank; d++) {
      auto crd = elems[i].first[d].getInt();
      cvs.push_back(builder.create<arith::ConstantIndexOp>(loc, crd));
    }
    // Remap value.
    Value val;
    if (isa<ComplexType>(attr.getElementType())) {
      auto valAttr = cast<ArrayAttr>(elems[i].second);
      val = builder.create<complex::ConstantOp>(loc, attr.getElementType(),
                                                valAttr);
    } else {
      auto valAttr = cast<TypedAttr>(elems[i].second);
      val = builder.create<arith::ConstantOp>(loc, valAttr);
    }
    assert(val);
    callback(cvs, val);
  }
}

SmallVector<Value> sparse_tensor::loadAll(OpBuilder &builder, Location loc,
                                          size_t size, Value mem,
                                          size_t offsetIdx, Value offsetVal) {
#ifndef NDEBUG
  const auto memTp = cast<MemRefType>(mem.getType());
  assert(memTp.getRank() == 1);
  const DynSize memSh = memTp.getDimSize(0);
  assert(ShapedType::isDynamic(memSh) || memSh >= static_cast<DynSize>(size));
  assert(offsetIdx == 0 || offsetIdx < size);
#endif // NDEBUG
  SmallVector<Value> vs;
  vs.reserve(size);
  for (unsigned i = 0; i < size; i++) {
    Value v = builder.create<memref::LoadOp>(loc, mem,
                                             constantIndex(builder, loc, i));
    if (i == offsetIdx && offsetVal)
      v = builder.create<arith::AddIOp>(loc, v, offsetVal);
    vs.push_back(v);
  }
  return vs;
}

void sparse_tensor::storeAll(OpBuilder &builder, Location loc, Value mem,
                             ValueRange vs, size_t offsetIdx, Value offsetVal) {
#ifndef NDEBUG
  const size_t vsize = vs.size();
  const auto memTp = cast<MemRefType>(mem.getType());
  assert(memTp.getRank() == 1);
  const DynSize memSh = memTp.getDimSize(0);
  assert(ShapedType::isDynamic(memSh) || memSh >= static_cast<DynSize>(vsize));
  assert(offsetIdx == 0 || offsetIdx < vsize);
#endif // NDEBUG
  for (const auto &v : llvm::enumerate(vs)) {
    const Value w =
        (offsetIdx == v.index() && offsetVal)
            ? builder.create<arith::AddIOp>(loc, v.value(), offsetVal)
            : v.value();
    builder.create<memref::StoreOp>(loc, w, mem,
                                    constantIndex(builder, loc, v.index()));
  }
}

Value sparse_tensor::reshapeValuesToLevels(OpBuilder &builder, Location loc,
                                           SparseTensorEncodingAttr enc,
                                           ValueRange dimSizes,
                                           Value valuesBuffer,
                                           Value lvlCoords) {
  // Reuse the `lvlCoords` buffer to store the level-sizes.
  const Level lvlRank = enc.getLvlRank();
  SmallVector<Value> lvlSizes;
  lvlSizes.reserve(lvlRank);
  for (Level l = 0; l < lvlRank; l++)
    // FIXME: `toOrigDim` is deprecated.
    lvlSizes.push_back(dimSizes[toOrigDim(enc, l)]);
  storeAll(builder, loc, lvlCoords, lvlSizes);
  // The memref ReshapeOp requires the sizes buffer to have a static
  // shape.
  const auto iTp = builder.getIndexType();
  const SmallVector<DynSize, 1> lvlSizesShape{static_cast<DynSize>(lvlRank)};
  const auto lvlSizesTp = MemRefType::get(lvlSizesShape, iTp);
  lvlCoords = builder.create<memref::CastOp>(loc, lvlSizesTp, lvlCoords);
  // Finally, create the ReshapeOp.
  const SmallVector<DynSize> resShape(lvlRank, ShapedType::kDynamic);
  const Type elemTp = getMemRefType(valuesBuffer).getElementType();
  const auto resTp = MemRefType::get(resShape, elemTp);
  return builder.create<memref::ReshapeOp>(loc, resTp, valuesBuffer, lvlCoords);
}

TypedValue<BaseMemRefType>
sparse_tensor::genToMemref(OpBuilder &builder, Location loc, Value tensor) {
  auto tTp = llvm::cast<TensorType>(tensor.getType());
  auto mTp = MemRefType::get(tTp.getShape(), tTp.getElementType());
  return builder.create<bufferization::ToMemrefOp>(loc, mTp, tensor)
      .getResult();
}

Value sparse_tensor::genToPositions(OpBuilder &builder, Location loc,
                                    Value tensor, Level lvl) {
  const auto srcTp = getSparseTensorType(tensor);
  const Type posTp = srcTp.getPosType();
  const Type memTp = get1DMemRefType(posTp, /*withLayout=*/false);
  return builder.create<ToPositionsOp>(loc, memTp, tensor,
                                       builder.getIndexAttr(lvl));
}

Value sparse_tensor::genToCoordinates(OpBuilder &builder, Location loc,
                                      Value tensor, Level lvl, Level cooStart) {
  const auto srcTp = getSparseTensorType(tensor);
  const Type crdTp = srcTp.getCrdType();
  const Type memTp = get1DMemRefType(crdTp, /*withLayout=*/lvl >= cooStart);
  return builder.create<ToCoordinatesOp>(loc, memTp, tensor,
                                         builder.getIndexAttr(lvl));
}

Value sparse_tensor::genToCoordinatesBuffer(OpBuilder &builder, Location loc,
                                            Value tensor) {
  const auto srcTp = getSparseTensorType(tensor);
  const Type crdTp = srcTp.getCrdType();
  const Type memTp = get1DMemRefType(crdTp, /*withLayout=*/false);
  return builder.create<ToCoordinatesBufferOp>(loc, memTp, tensor);
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

Value sparse_tensor::createOrFoldSliceOffsetOp(OpBuilder &builder, Location loc,
                                               Value tensor, Dimension dim) {
  auto enc = getSparseTensorEncoding(tensor.getType());
  assert(enc && enc.isSlice());
  std::optional<unsigned> offset = enc.getStaticDimSliceOffset(dim);
  if (offset.has_value())
    return constantIndex(builder, loc, *offset);
  return builder.create<ToSliceOffsetOp>(loc, tensor, APInt(64, dim));
}

Value sparse_tensor::createOrFoldSliceStrideOp(OpBuilder &builder, Location loc,
                                               Value tensor, Dimension dim) {
  auto enc = getSparseTensorEncoding(tensor.getType());
  assert(enc && enc.isSlice());
  std::optional<unsigned> stride = enc.getStaticDimSliceStride(dim);
  if (stride.has_value())
    return constantIndex(builder, loc, *stride);
  return builder.create<ToSliceStrideOp>(loc, tensor, APInt(64, dim));
}

void sparse_tensor::fillDimShape(OpBuilder &builder, Location loc,
                                 SparseTensorType stt,
                                 SmallVectorImpl<Value> &out) {
  out.clear();
  out.reserve(stt.getDimRank());
  for (const DynSize sh : stt.getDimShape()) {
    const auto s = ShapedType::isDynamic(sh) ? 0 : sh;
    out.push_back(constantIndex(builder, loc, s));
  }
}

Value sparse_tensor::genReader(OpBuilder &builder, Location loc,
                               SparseTensorType stt, Value tensor,
                               /*out*/ SmallVectorImpl<Value> &dimShapesValues,
                               /*out*/ Value &dimSizesBuffer) {
  // Construct the dimShapes buffer. The buffer contains the static size
  // per dimension, or otherwise a zero for a dynamic size.
  fillDimShape(builder, loc, stt, dimShapesValues);
  Value dimShapesBuffer = allocaBuffer(builder, loc, dimShapesValues);
  // Create the `CheckedSparseTensorReader`. This reader performs a
  // consistency check on the static sizes, but accepts any size
  // of each dimension with a dynamic size.
  Type opaqueTp = getOpaquePointerType(builder);
  Type eltTp = stt.getElementType();
  Value valTp = constantPrimaryTypeEncoding(builder, loc, eltTp);
  Value reader =
      createFuncCall(builder, loc, "createCheckedSparseTensorReader", opaqueTp,
                     {tensor, dimShapesBuffer, valTp}, EmitCInterface::On)
          .getResult(0);
  // For static shapes, the shape buffer can be used right away. For dynamic
  // shapes, use the information from the reader to construct a buffer that
  // supplies the actual size for each dynamic dimension.
  dimSizesBuffer = dimShapesBuffer;
  if (stt.hasDynamicDimShape()) {
    Type indexTp = builder.getIndexType();
    auto memTp = MemRefType::get({ShapedType::kDynamic}, indexTp);
    dimSizesBuffer =
        createFuncCall(builder, loc, "getSparseTensorReaderDimSizes", memTp,
                       reader, EmitCInterface::On)
            .getResult(0);
  }
  return reader;
}

Value sparse_tensor::genMapBuffers(OpBuilder &builder, Location loc,
                                   SparseTensorType stt,
                                   ArrayRef<Value> dimShapesValues,
                                   Value dimSizesBuffer,
                                   /*out*/ Value &dim2lvlBuffer,
                                   /*out*/ Value &lvl2dimBuffer) {
  const Dimension dimRank = stt.getDimRank();
  const Level lvlRank = stt.getLvlRank();
  // For an identity mapping, the dim2lvl and lvl2dim mappings are
  // identical as are dimSizes and lvlSizes, so buffers are reused
  // as much as possible.
  if (stt.isIdentity()) {
    assert(dimRank == lvlRank);
    SmallVector<Value> iotaValues;
    iotaValues.reserve(lvlRank);
    for (Level l = 0; l < lvlRank; l++)
      iotaValues.push_back(constantIndex(builder, loc, l));
    dim2lvlBuffer = lvl2dimBuffer = allocaBuffer(builder, loc, iotaValues);
    return dimSizesBuffer;
  }
  // Otherwise, some code needs to be generated to set up the buffers.
  // TODO: use the lvl2dim once available and deal with non-permutations!
  const auto dimToLvl = stt.getDimToLvl();
  assert(dimToLvl.isPermutation());
  SmallVector<Value> dim2lvlValues(dimRank);
  SmallVector<Value> lvl2dimValues(lvlRank);
  SmallVector<Value> lvlSizesValues(lvlRank);
  for (Level l = 0; l < lvlRank; l++) {
    // The `d`th source variable occurs in the `l`th result position.
    Dimension d = dimToLvl.getDimPosition(l);
    Value lvl = constantIndex(builder, loc, l);
    Value dim = constantIndex(builder, loc, d);
    dim2lvlValues[d] = lvl;
    lvl2dimValues[l] = dim;
    if (stt.isDynamicDim(d))
      lvlSizesValues[l] =
          builder.create<memref::LoadOp>(loc, dimSizesBuffer, dim);
    else
      lvlSizesValues[l] = dimShapesValues[d];
  }
  dim2lvlBuffer = allocaBuffer(builder, loc, dim2lvlValues);
  lvl2dimBuffer = allocaBuffer(builder, loc, lvl2dimValues);
  return allocaBuffer(builder, loc, lvlSizesValues);
}
