//===- TypeDetail.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIB_MLIR_TOOLS_PDLL_AST_TYPEDETAIL_H_
#define LIB_MLIR_TOOLS_PDLL_AST_TYPEDETAIL_H_

#include "mlir/Tools/PDLL/AST/Types.h"

namespace mlir {
namespace pdll {
namespace ast {
//===----------------------------------------------------------------------===//
// Type
//===----------------------------------------------------------------------===//

struct Type::Storage : public StorageUniquer::BaseStorage {
  Storage(TypeID typeID) : typeID(typeID) {}

  /// The type identifier for the derived type class.
  TypeID typeID;
};

namespace detail {

/// A utility CRTP base class that defines many of the necessary utilities for
/// defining a PDLL AST Type.
template <typename ConcreteT, typename KeyT = void>
struct TypeStorageBase : public Type::Storage {
  using KeyTy = KeyT;
  using Base = TypeStorageBase<ConcreteT, KeyT>;
  TypeStorageBase(KeyTy key)
      : Type::Storage(TypeID::get<ConcreteT>()), key(key) {}

  /// Construct an instance with the given storage allocator.
  static ConcreteT *construct(StorageUniquer::StorageAllocator &alloc,
                              const KeyTy &key) {
    return new (alloc.allocate<ConcreteT>()) ConcreteT(key);
  }

  /// Utility methods required by the storage allocator.
  bool operator==(const KeyTy &key) const { return this->key == key; }

  /// Return the key value of this storage class.
  const KeyTy &getValue() const { return key; }

protected:
  KeyTy key;
};
/// A specialization of the storage base for singleton types.
template <typename ConcreteT>
struct TypeStorageBase<ConcreteT, void> : public Type::Storage {
  using Base = TypeStorageBase<ConcreteT, void>;
  TypeStorageBase() : Type::Storage(TypeID::get<ConcreteT>()) {}
};

//===----------------------------------------------------------------------===//
// AttributeType
//===----------------------------------------------------------------------===//

struct AttributeTypeStorage : public TypeStorageBase<AttributeTypeStorage> {};

//===----------------------------------------------------------------------===//
// ConstraintType
//===----------------------------------------------------------------------===//

struct ConstraintTypeStorage : public TypeStorageBase<ConstraintTypeStorage> {};

//===----------------------------------------------------------------------===//
// OperationType
//===----------------------------------------------------------------------===//

struct OperationTypeStorage
    : public TypeStorageBase<OperationTypeStorage,
                             std::pair<StringRef, const ods::Operation *>> {
  using Base::Base;

  static OperationTypeStorage *
  construct(StorageUniquer::StorageAllocator &alloc,
            const std::pair<StringRef, const ods::Operation *> &key) {
    return new (alloc.allocate<OperationTypeStorage>()) OperationTypeStorage(
        std::make_pair(alloc.copyInto(key.first), key.second));
  }
};

//===----------------------------------------------------------------------===//
// RangeType
//===----------------------------------------------------------------------===//

struct RangeTypeStorage : public TypeStorageBase<RangeTypeStorage, Type> {
  using Base::Base;
};

//===----------------------------------------------------------------------===//
// RewriteType
//===----------------------------------------------------------------------===//

struct RewriteTypeStorage : public TypeStorageBase<RewriteTypeStorage> {};

//===----------------------------------------------------------------------===//
// TupleType
//===----------------------------------------------------------------------===//

struct TupleTypeStorage
    : public TypeStorageBase<TupleTypeStorage,
                             std::pair<ArrayRef<Type>, ArrayRef<StringRef>>> {
  using Base::Base;

  static TupleTypeStorage *
  construct(StorageUniquer::StorageAllocator &alloc,
            std::pair<ArrayRef<Type>, ArrayRef<StringRef>> key) {
    SmallVector<StringRef> names = llvm::to_vector(llvm::map_range(
        key.second, [&](StringRef name) { return alloc.copyInto(name); }));
    return new (alloc.allocate<TupleTypeStorage>())
        TupleTypeStorage(std::make_pair(alloc.copyInto(key.first),
                                        alloc.copyInto(llvm::ArrayRef(names))));
  }
};

//===----------------------------------------------------------------------===//
// TypeType
//===----------------------------------------------------------------------===//

struct TypeTypeStorage : public TypeStorageBase<TypeTypeStorage> {};

//===----------------------------------------------------------------------===//
// ValueType
//===----------------------------------------------------------------------===//

struct ValueTypeStorage : public TypeStorageBase<ValueTypeStorage> {};

} // namespace detail
} // namespace ast
} // namespace pdll
} // namespace mlir

#endif // LIB_MLIR_TOOLS_PDLL_AST_TYPEDETAIL_H_
