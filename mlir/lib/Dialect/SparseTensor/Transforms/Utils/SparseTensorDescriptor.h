//===- SparseTensorDescriptor.h ---------------------------------*- C++ -*-===//
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

#ifndef MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_UTILS_SPARSETENSORDESCRIPTOR_H_
#define MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_UTILS_SPARSETENSORDESCRIPTOR_H_

#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorStorageLayout.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorType.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"

namespace mlir {
namespace sparse_tensor {

class SparseTensorSpecifier {
public:
  explicit SparseTensorSpecifier(Value specifier)
      : specifier(cast<TypedValue<StorageSpecifierType>>(specifier)) {}

  // Undef value for level-sizes, all zero values for memory-sizes.
  static Value getInitValue(OpBuilder &builder, Location loc,
                            SparseTensorType stt);

  /*implicit*/ operator Value() { return specifier; }

  Value getSpecifierField(OpBuilder &builder, Location loc,
                          StorageSpecifierKind kind, std::optional<Level> lvl);

  void setSpecifierField(OpBuilder &builder, Location loc, Value v,
                         StorageSpecifierKind kind, std::optional<Level> lvl);

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
  SparseTensorDescriptorImpl(SparseTensorType stt, ValueArrayRef fields)
      : rType(stt), fields(fields), layout(stt) {
    assert(layout.getNumFields() == getNumFields());
    // We should make sure the class is trivially copyable (and should be small
    // enough) such that we can pass it by value.
    static_assert(std::is_trivially_copyable_v<
                  SparseTensorDescriptorImpl<ValueArrayRef>>);
  }

public:
  FieldIndex getMemRefFieldIndex(SparseTensorFieldKind kind,
                                 std::optional<Level> lvl) const {
    // Delegates to storage layout.
    return layout.getMemRefFieldIndex(kind, lvl);
  }

  unsigned getNumFields() const { return fields.size(); }

  ///
  /// Getters: get the value for required field.
  ///

  Value getSpecifier() const { return fields.back(); }

  Value getSpecifierField(OpBuilder &builder, Location loc,
                          StorageSpecifierKind kind,
                          std::optional<Level> lvl) const {
    SparseTensorSpecifier md(fields.back());
    return md.getSpecifierField(builder, loc, kind, lvl);
  }

  Value getLvlSize(OpBuilder &builder, Location loc, Level lvl) const {
    return getSpecifierField(builder, loc, StorageSpecifierKind::LvlSize, lvl);
  }

  Value getPosMemRef(Level lvl) const {
    return getMemRefField(SparseTensorFieldKind::PosMemRef, lvl);
  }

  Value getValMemRef() const {
    return getMemRefField(SparseTensorFieldKind::ValMemRef, std::nullopt);
  }

  Value getMemRefField(SparseTensorFieldKind kind,
                       std::optional<Level> lvl) const {
    return getField(getMemRefFieldIndex(kind, lvl));
  }

  Value getMemRefField(FieldIndex fidx) const {
    assert(fidx < fields.size() - 1);
    return getField(fidx);
  }

  Value getPosMemSize(OpBuilder &builder, Location loc, Level lvl) const {
    return getSpecifierField(builder, loc, StorageSpecifierKind::PosMemSize,
                             lvl);
  }

  Value getCrdMemSize(OpBuilder &builder, Location loc, Level lvl) const {
    return getSpecifierField(builder, loc, StorageSpecifierKind::CrdMemSize,
                             lvl);
  }

  Value getValMemSize(OpBuilder &builder, Location loc) const {
    return getSpecifierField(builder, loc, StorageSpecifierKind::ValMemSize,
                             std::nullopt);
  }

  Type getMemRefElementType(SparseTensorFieldKind kind,
                            std::optional<Level> lvl) const {
    return getMemRefType(getMemRefField(kind, lvl)).getElementType();
  }

  Value getField(FieldIndex fidx) const {
    assert(fidx < fields.size());
    return fields[fidx];
  }

  ValueRange getMemRefFields() const {
    return fields.drop_back(); // drop the last metadata fields
  }

  std::pair<FieldIndex, unsigned> getCrdMemRefIndexAndStride(Level lvl) const {
    return layout.getFieldIndexAndStride(SparseTensorFieldKind::CrdMemRef, lvl);
  }

  Value getAOSMemRef() const {
    const Level cooStart = rType.getAoSCOOStart();
    assert(cooStart < rType.getLvlRank());
    return getMemRefField(SparseTensorFieldKind::CrdMemRef, cooStart);
  }

  RankedTensorType getRankedTensorType() const { return rType; }
  ValueArrayRef getFields() const { return fields; }
  StorageLayout getLayout() const { return layout; }

protected:
  SparseTensorType rType;
  ValueArrayRef fields;
  StorageLayout layout;
};

/// Uses ValueRange for immutable descriptors.
class SparseTensorDescriptor : public SparseTensorDescriptorImpl<ValueRange> {
public:
  SparseTensorDescriptor(SparseTensorType stt, ValueRange buffers)
      : SparseTensorDescriptorImpl<ValueRange>(stt, buffers) {}

  Value getCrdMemRefOrView(OpBuilder &builder, Location loc, Level lvl) const;
};

/// Using SmallVector for mutable descriptor allows users to reuse it as a
/// tmp buffers to append value for some special cases, though users should
/// be responsible to restore the buffer to legal states after their use. It
/// is probably not a clean way, but it is the most efficient way to avoid
/// copying the fields into another SmallVector. If a more clear way is
/// wanted, we should change it to MutableArrayRef instead.
class MutSparseTensorDescriptor
    : public SparseTensorDescriptorImpl<SmallVectorImpl<Value> &> {
public:
  MutSparseTensorDescriptor(SparseTensorType stt,
                            SmallVectorImpl<Value> &buffers)
      : SparseTensorDescriptorImpl<SmallVectorImpl<Value> &>(stt, buffers) {}

  // Allow implicit type conversion from mutable descriptors to immutable ones
  // (but not vice versa).
  /*implicit*/ operator SparseTensorDescriptor() const {
    return SparseTensorDescriptor(rType, fields);
  }

  ///
  /// Adds additional setters for mutable descriptor, update the value for
  /// required field.
  ///

  void setMemRefField(SparseTensorFieldKind kind, std::optional<Level> lvl,
                      Value v) {
    fields[getMemRefFieldIndex(kind, lvl)] = v;
  }

  void setMemRefField(FieldIndex fidx, Value v) {
    assert(fidx < fields.size() - 1);
    fields[fidx] = v;
  }

  void setField(FieldIndex fidx, Value v) {
    assert(fidx < fields.size());
    fields[fidx] = v;
  }

  void setSpecifier(Value newSpec) { fields.back() = newSpec; }

  void setSpecifierField(OpBuilder &builder, Location loc,
                         StorageSpecifierKind kind, std::optional<Level> lvl,
                         Value v) {
    SparseTensorSpecifier md(fields.back());
    md.setSpecifierField(builder, loc, v, kind, lvl);
    fields.back() = md;
  }

  void setValMemSize(OpBuilder &builder, Location loc, Value v) {
    setSpecifierField(builder, loc, StorageSpecifierKind::ValMemSize,
                      std::nullopt, v);
  }

  void setCrdMemSize(OpBuilder &builder, Location loc, Level lvl, Value v) {
    setSpecifierField(builder, loc, StorageSpecifierKind::CrdMemSize, lvl, v);
  }

  void setPosMemSize(OpBuilder &builder, Location loc, Level lvl, Value v) {
    setSpecifierField(builder, loc, StorageSpecifierKind::PosMemSize, lvl, v);
  }

  void setLvlSize(OpBuilder &builder, Location loc, Level lvl, Value v) {
    setSpecifierField(builder, loc, StorageSpecifierKind::LvlSize, lvl, v);
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
  return genTuple(builder, loc, desc.getRankedTensorType(), desc.getFields());
}

inline SparseTensorDescriptor getDescriptorFromTensorTuple(Value tensor) {
  auto tuple = getTuple(tensor);
  SparseTensorType stt(cast<RankedTensorType>(tuple.getResultTypes()[0]));
  return SparseTensorDescriptor(stt, tuple.getInputs());
}

inline MutSparseTensorDescriptor
getMutDescriptorFromTensorTuple(Value tensor, SmallVectorImpl<Value> &fields) {
  auto tuple = getTuple(tensor);
  fields.assign(tuple.getInputs().begin(), tuple.getInputs().end());
  SparseTensorType stt(cast<RankedTensorType>(tuple.getResultTypes()[0]));
  return MutSparseTensorDescriptor(stt, fields);
}

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_UTILS_SPARSETENSODESCRIPTOR_H_
