//===- SparseTensorStorageLayout.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines utilities for the sparse memory layout.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_SPARSETENSORBUILDER_H_
#define MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_SPARSETENSORBUILDER_H_

#include "mlir/Conversion/LLVMCommon/StructBuilder.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// SparseTensorDescriptor and helpers that manage the sparse tensor memory
// layout scheme during "direct code generation" (i.e. when sparsification
// generates the buffers as part of actual IR, in constrast with the library
// approach where data structures are hidden behind opaque pointers).
//
// The sparse tensor storage scheme for a rank-dimensional tensor is organized
// as a single compound type with the following fields. Note that every memref
// with ? size actually behaves as a "vector", i.e. the stored size is the
// capacity and the used size resides in the storage_specifier struct.
//
// struct {
//   ; per-dimension d:
//   ;  if dense:
//        <nothing>
//   ;  if compresed:
//        memref<? x ptr>  pointers-d  ; pointers for sparse dim d
//        memref<? x idx>  indices-d   ; indices for sparse dim d
//   ;  if singleton:
//        memref<? x idx>  indices-d   ; indices for singleton dim d
//
//   memref<? x eltType> values        ; values
//
//   struct sparse_tensor.storage_specifier {
//     array<rank x int> dimSizes    ; sizes for each dimension
//     array<n x int> memSizes;      ; sizes for each data memref
//   }
// };
//
// In addition, for a "trailing COO region", defined as a compressed
// dimension followed by one ore more singleton dimensions, the default
// SOA storage that is inherent to the TACO format is optimized into an
// AOS storage where all indices of a stored element appear consecutively.
// In such cases, a special operation (sparse_tensor.indices_buffer) must
// be used to access the AOS index array. In the code below, the method
// `getCOOStart` is used to find the start of the "trailing COO region".
//
// Examples.
//
// #CSR storage of 2-dim matrix yields
//   memref<?xindex>                           ; pointers-1
//   memref<?xindex>                           ; indices-1
//   memref<?xf64>                             ; values
//   struct<(array<2 x i64>, array<3 x i64>)>) ; dim0, dim1, 3xsizes
//
// #COO storage of 2-dim matrix yields
//   memref<?xindex>,                          ; pointers-0, essentially [0,sz]
//   memref<?xindex>                           ; AOS index storage
//   memref<?xf64>                             ; values
//   struct<(array<2 x i64>, array<3 x i64>)>) ; dim0, dim1, 3xsizes
//
//===----------------------------------------------------------------------===//

enum class SparseTensorFieldKind : uint32_t {
  StorageSpec = 0,
  PtrMemRef = 1,
  IdxMemRef = 2,
  ValMemRef = 3
};

static_assert(static_cast<uint32_t>(SparseTensorFieldKind::PtrMemRef) ==
              static_cast<uint32_t>(StorageSpecifierKind::PtrMemSize));
static_assert(static_cast<uint32_t>(SparseTensorFieldKind::IdxMemRef) ==
              static_cast<uint32_t>(StorageSpecifierKind::IdxMemSize));
static_assert(static_cast<uint32_t>(SparseTensorFieldKind::ValMemRef) ==
              static_cast<uint32_t>(StorageSpecifierKind::ValMemSize));

/// For each field that will be allocated for the given sparse tensor encoding,
/// calls the callback with the corresponding field index, field kind, dimension
/// (for sparse tensor level memrefs) and dimlevelType.
/// The field index always starts with zero and increments by one between two
/// callback invocations.
/// Ideally, all other methods should rely on this function to query a sparse
/// tensor fields instead of relying on ad-hoc index computation.
void foreachFieldInSparseTensor(
    SparseTensorEncodingAttr,
    llvm::function_ref<bool(unsigned /*fieldIdx*/,
                            SparseTensorFieldKind /*fieldKind*/,
                            unsigned /*dim (if applicable)*/,
                            DimLevelType /*DLT (if applicable)*/)>);

/// Same as above, except that it also builds the Type for the corresponding
/// field.
void foreachFieldAndTypeInSparseTensor(
    RankedTensorType,
    llvm::function_ref<bool(Type /*fieldType*/, unsigned /*fieldIdx*/,
                            SparseTensorFieldKind /*fieldKind*/,
                            unsigned /*dim (if applicable)*/,
                            DimLevelType /*DLT (if applicable)*/)>);

/// Gets the total number of fields for the given sparse tensor encoding.
unsigned getNumFieldsFromEncoding(SparseTensorEncodingAttr enc);

/// Gets the total number of data fields (index arrays, pointer arrays, and a
/// value array) for the given sparse tensor encoding.
unsigned getNumDataFieldsFromEncoding(SparseTensorEncodingAttr enc);

inline StorageSpecifierKind toSpecifierKind(SparseTensorFieldKind kind) {
  assert(kind != SparseTensorFieldKind::StorageSpec);
  return static_cast<StorageSpecifierKind>(kind);
}

inline SparseTensorFieldKind toFieldKind(StorageSpecifierKind kind) {
  assert(kind != StorageSpecifierKind::DimSize);
  return static_cast<SparseTensorFieldKind>(kind);
}

/// Provides methods to access fields of a sparse tensor with the given
/// encoding.
class StorageLayout {
public:
  explicit StorageLayout(SparseTensorEncodingAttr enc) : enc(enc) {}

  ///
  /// Getters: get the field index for required field.
  ///

  unsigned getMemRefFieldIndex(SparseTensorFieldKind kind,
                               std::optional<unsigned> dim) const {
    return getFieldIndexAndStride(kind, dim).first;
  }

  unsigned getMemRefFieldIndex(StorageSpecifierKind kind,
                               std::optional<unsigned> dim) const {
    return getMemRefFieldIndex(toFieldKind(kind), dim);
  }

  static unsigned getNumFieldsFromEncoding(SparseTensorEncodingAttr enc) {
    return sparse_tensor::getNumFieldsFromEncoding(enc);
  }

  static void foreachFieldInSparseTensor(
      const SparseTensorEncodingAttr enc,
      llvm::function_ref<bool(unsigned, SparseTensorFieldKind, unsigned,
                              DimLevelType)>
          callback) {
    return sparse_tensor::foreachFieldInSparseTensor(enc, callback);
  }

  std::pair<unsigned, unsigned>
  getFieldIndexAndStride(SparseTensorFieldKind kind,
                         std::optional<unsigned> dim) const {
    unsigned fieldIdx = -1u;
    unsigned stride = 1;
    if (kind == SparseTensorFieldKind::IdxMemRef) {
      assert(dim.has_value());
      unsigned cooStart = getCOOStart(enc);
      unsigned rank = enc.getDimLevelType().size();
      if (dim.value() >= cooStart && dim.value() < rank) {
        dim = cooStart;
        stride = rank - cooStart;
      }
    }
    foreachFieldInSparseTensor(
        enc,
        [dim, kind, &fieldIdx](unsigned fIdx, SparseTensorFieldKind fKind,
                               unsigned fDim, DimLevelType dlt) -> bool {
          if ((dim && fDim == dim.value() && kind == fKind) ||
              (kind == fKind && fKind == SparseTensorFieldKind::ValMemRef)) {
            fieldIdx = fIdx;
            // Returns false to break the iteration.
            return false;
          }
          return true;
        });
    assert(fieldIdx != -1u);
    return std::pair<unsigned, unsigned>(fieldIdx, stride);
  }

private:
  SparseTensorEncodingAttr enc;
};

class SparseTensorSpecifier {
public:
  explicit SparseTensorSpecifier(Value specifier)
      : specifier(cast<TypedValue<StorageSpecifierType>>(specifier)) {}

  // Undef value for dimension sizes, all zero value for memory sizes.
  static Value getInitValue(OpBuilder &builder, Location loc,
                            RankedTensorType rtp);

  /*implicit*/ operator Value() { return specifier; }

  Value getSpecifierField(OpBuilder &builder, Location loc,
                          StorageSpecifierKind kind,
                          std::optional<unsigned> dim);

  void setSpecifierField(OpBuilder &builder, Location loc, Value v,
                         StorageSpecifierKind kind,
                         std::optional<unsigned> dim);

  Type getFieldType(StorageSpecifierKind kind, std::optional<unsigned> dim) {
    return specifier.getType().getFieldType(kind, dim);
  }

private:
  TypedValue<StorageSpecifierType> specifier;
};

/// A helper class around an array of values that corresponds to a sparse
/// tensor. This class provides a set of meaningful APIs to query and update
/// a particular field in a consistent way. Users should not make assumptions
/// on how a sparse tensor is laid out but instead rely on this class to access
/// the right value for the right field.
template <typename ValueArrayRef>
class SparseTensorDescriptorImpl {
protected:
  SparseTensorDescriptorImpl(Type tp, ValueArrayRef fields)
      : rType(tp.cast<RankedTensorType>()), fields(fields) {
    assert(getSparseTensorEncoding(tp) &&
           getNumFieldsFromEncoding(getSparseTensorEncoding(tp)) ==
               fields.size());
    // We should make sure the class is trivially copyable (and should be small
    // enough) such that we can pass it by value.
    static_assert(std::is_trivially_copyable_v<
                  SparseTensorDescriptorImpl<ValueArrayRef>>);
  }

public:
  unsigned getMemRefFieldIndex(SparseTensorFieldKind kind,
                               std::optional<unsigned> dim) const {
    // Delegates to storage layout.
    StorageLayout layout(getSparseTensorEncoding(rType));
    return layout.getMemRefFieldIndex(kind, dim);
  }

  unsigned getNumFields() const { return fields.size(); }

  ///
  /// Getters: get the value for required field.
  ///

  Value getSpecifierField(OpBuilder &builder, Location loc,
                          StorageSpecifierKind kind,
                          std::optional<unsigned> dim) const {
    SparseTensorSpecifier md(fields.back());
    return md.getSpecifierField(builder, loc, kind, dim);
  }

  Value getDimSize(OpBuilder &builder, Location loc, unsigned dim) const {
    return getSpecifierField(builder, loc, StorageSpecifierKind::DimSize, dim);
  }

  Value getPtrMemRef(unsigned ptrDim) const {
    return getMemRefField(SparseTensorFieldKind::PtrMemRef, ptrDim);
  }

  Value getValMemRef() const {
    return getMemRefField(SparseTensorFieldKind::ValMemRef, std::nullopt);
  }

  Value getMemRefField(SparseTensorFieldKind kind,
                       std::optional<unsigned> dim) const {
    return getField(getMemRefFieldIndex(kind, dim));
  }

  Value getMemRefField(unsigned fidx) const {
    assert(fidx < fields.size() - 1);
    return getField(fidx);
  }

  Value getPtrMemSize(OpBuilder &builder, Location loc, unsigned dim) const {
    return getSpecifierField(builder, loc, StorageSpecifierKind::PtrMemSize,
                             dim);
  }

  Value getIdxMemSize(OpBuilder &builder, Location loc, unsigned dim) const {
    return getSpecifierField(builder, loc, StorageSpecifierKind::IdxMemSize,
                             dim);
  }

  Value getValMemSize(OpBuilder &builder, Location loc) const {
    return getSpecifierField(builder, loc, StorageSpecifierKind::ValMemSize,
                             std::nullopt);
  }

  Type getMemRefElementType(SparseTensorFieldKind kind,
                            std::optional<unsigned> dim) const {
    return getMemRefField(kind, dim)
        .getType()
        .template cast<MemRefType>()
        .getElementType();
  }

  Value getField(unsigned fidx) const {
    assert(fidx < fields.size());
    return fields[fidx];
  }

  ValueRange getMemRefFields() const {
    ValueRange ret = fields;
    // Drop the last metadata fields.
    return ret.slice(0, fields.size() - 1);
  }

  std::pair<unsigned, unsigned>
  getIdxMemRefIndexAndStride(unsigned idxDim) const {
    StorageLayout layout(getSparseTensorEncoding(rType));
    return layout.getFieldIndexAndStride(SparseTensorFieldKind::IdxMemRef,
                                         idxDim);
  }

  Value getAOSMemRef() const {
    auto enc = getSparseTensorEncoding(rType);
    unsigned cooStart = getCOOStart(enc);
    assert(cooStart < enc.getDimLevelType().size());
    return getMemRefField(SparseTensorFieldKind::IdxMemRef, cooStart);
  }

  RankedTensorType getTensorType() const { return rType; }
  ValueArrayRef getFields() const { return fields; }

protected:
  RankedTensorType rType;
  ValueArrayRef fields;
};

/// Uses ValueRange for immutable descriptors.
class SparseTensorDescriptor : public SparseTensorDescriptorImpl<ValueRange> {
public:
  SparseTensorDescriptor(Type tp, ValueRange buffers)
      : SparseTensorDescriptorImpl<ValueRange>(tp, buffers) {}

  Value getIdxMemRefOrView(OpBuilder &builder, Location loc,
                           unsigned idxDim) const;
};

/// Uses SmallVectorImpl<Value> & for mutable descriptors.
/// Using SmallVector for mutable descriptor allows users to reuse it as a
/// tmp buffers to append value for some special cases, though users should
/// be responsible to restore the buffer to legal states after their use. It
/// is probably not a clean way, but it is the most efficient way to avoid
/// copying the fields into another SmallVector. If a more clear way is
/// wanted, we should change it to MutableArrayRef instead.
class MutSparseTensorDescriptor
    : public SparseTensorDescriptorImpl<SmallVectorImpl<Value> &> {
public:
  MutSparseTensorDescriptor(Type tp, SmallVectorImpl<Value> &buffers)
      : SparseTensorDescriptorImpl<SmallVectorImpl<Value> &>(tp, buffers) {}

  // Allow implicit type conversion from mutable descriptors to immutable ones
  // (but not vice versa).
  /*implicit*/ operator SparseTensorDescriptor() const {
    return SparseTensorDescriptor(rType, fields);
  }

  ///
  /// Adds additional setters for mutable descriptor, update the value for
  /// required field.
  ///

  void setMemRefField(SparseTensorFieldKind kind, std::optional<unsigned> dim,
                      Value v) {
    fields[getMemRefFieldIndex(kind, dim)] = v;
  }

  void setMemRefField(unsigned fidx, Value v) {
    assert(fidx < fields.size() - 1);
    fields[fidx] = v;
  }

  void setField(unsigned fidx, Value v) {
    assert(fidx < fields.size());
    fields[fidx] = v;
  }

  void setSpecifierField(OpBuilder &builder, Location loc,
                         StorageSpecifierKind kind, std::optional<unsigned> dim,
                         Value v) {
    SparseTensorSpecifier md(fields.back());
    md.setSpecifierField(builder, loc, v, kind, dim);
    fields.back() = md;
  }

  void setDimSize(OpBuilder &builder, Location loc, unsigned dim, Value v) {
    setSpecifierField(builder, loc, StorageSpecifierKind::DimSize, dim, v);
  }
};

/// Returns the "tuple" value of the adapted tensor.
inline UnrealizedConversionCastOp getTuple(Value tensor) {
  return llvm::cast<UnrealizedConversionCastOp>(tensor.getDefiningOp());
}

/// Packs the given values as a "tuple" value.
inline Value genTuple(OpBuilder &builder, Location loc, Type tp,
                      ValueRange values) {
  return builder.create<UnrealizedConversionCastOp>(loc, TypeRange(tp), values)
      .getResult(0);
}

inline Value genTuple(OpBuilder &builder, Location loc,
                      SparseTensorDescriptor desc) {
  return genTuple(builder, loc, desc.getTensorType(), desc.getFields());
}

inline SparseTensorDescriptor getDescriptorFromTensorTuple(Value tensor) {
  auto tuple = getTuple(tensor);
  return SparseTensorDescriptor(tuple.getResultTypes()[0], tuple.getInputs());
}

inline MutSparseTensorDescriptor
getMutDescriptorFromTensorTuple(Value tensor, SmallVectorImpl<Value> &fields) {
  auto tuple = getTuple(tensor);
  fields.assign(tuple.getInputs().begin(), tuple.getInputs().end());
  return MutSparseTensorDescriptor(tuple.getResultTypes()[0], fields);
}

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_SPARSETENSORBUILDER_H_
