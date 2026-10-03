//===- Types.h --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_PDLL_AST_TYPES_H_
#define MLIR_TOOLS_PDLL_AST_TYPES_H_

#include "mlir/Support/LLVM.h"
#include "mlir/Support/StorageUniquer.h"
#include <optional>

namespace mlir {
namespace pdll {
namespace ods {
class Operation;
} // namespace ods

namespace ast {
class Context;

//===----------------------------------------------------------------------===//
// Type
//===----------------------------------------------------------------------===//

class Type {
public:
  /// This class represents the internal storage of the Type class.
  struct Storage;

  /// This class provides several utilities when defining derived type classes.
  template <typename ImplT, typename BaseT = Type>
  class TypeBase : public BaseT {
  public:
    using Base = TypeBase<ImplT, BaseT>;
    using ImplTy = ImplT;
    using BaseT::BaseT;

    /// Provide type casting support.
    static bool classof(Type type) {
      return type.getTypeID() == TypeID::get<ImplTy>();
    }
  };

  Type(Storage *impl = nullptr) : impl(impl) {}

  bool operator==(const Type &other) const { return impl == other.impl; }
  bool operator!=(const Type &other) const { return !(*this == other); }
  explicit operator bool() const { return impl; }

  /// Return the internal storage instance of this type.
  Storage *getImpl() const { return impl; }

  /// Return the TypeID instance of this type.
  TypeID getTypeID() const;

  /// Print this type to the given stream.
  void print(raw_ostream &os) const;

  /// Try to refine this type with the one provided. Given two compatible types,
  /// this will return a merged type contains as much detail from the two types.
  /// For example, if refining two operation types and one contains a name,
  /// while the other doesn't, the refined type contains the name. If the two
  /// types are incompatible, null is returned.
  Type refineWith(Type other) const;

protected:
  /// Return the internal storage instance of this type reinterpreted as the
  /// given derived storage type.
  template <typename T>
  const T *getImplAs() const {
    return static_cast<const T *>(impl);
  }

private:
  Storage *impl;
};

inline llvm::hash_code hash_value(Type type) {
  return DenseMapInfo<Type::Storage *>::getHashValue(type.getImpl());
}

inline raw_ostream &operator<<(raw_ostream &os, Type type) {
  type.print(os);
  return os;
}

//===----------------------------------------------------------------------===//
// Type::Storage
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
// AttributeTypeStorage
//===----------------------------------------------------------------------===//

struct AttributeTypeStorage : public TypeStorageBase<AttributeTypeStorage> {};

//===----------------------------------------------------------------------===//
// ConstraintTypeStorage
//===----------------------------------------------------------------------===//

struct ConstraintTypeStorage : public TypeStorageBase<ConstraintTypeStorage> {};

//===----------------------------------------------------------------------===//
// OperationTypeStorage
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
// RangeTypeStorage
//===----------------------------------------------------------------------===//

struct RangeTypeStorage : public TypeStorageBase<RangeTypeStorage, Type> {
  using Base::Base;
};

//===----------------------------------------------------------------------===//
// RewriteTypeStorage
//===----------------------------------------------------------------------===//

struct RewriteTypeStorage : public TypeStorageBase<RewriteTypeStorage> {};

//===----------------------------------------------------------------------===//
// TupleTypeStorage
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
// TypeTypeStorage
//===----------------------------------------------------------------------===//

struct TypeTypeStorage : public TypeStorageBase<TypeTypeStorage> {};

//===----------------------------------------------------------------------===//
// ValueTypeStorage
//===----------------------------------------------------------------------===//

struct ValueTypeStorage : public TypeStorageBase<ValueTypeStorage> {};

} // namespace detail

//===----------------------------------------------------------------------===//
// AttributeType
//===----------------------------------------------------------------------===//

/// This class represents a PDLL type that corresponds to an mlir::Attribute.
class AttributeType : public Type::TypeBase<detail::AttributeTypeStorage> {
public:
  using Base::Base;

  /// Return an instance of the Attribute type.
  static AttributeType get(Context &context);
};

//===----------------------------------------------------------------------===//
// ConstraintType
//===----------------------------------------------------------------------===//

/// This class represents a PDLL type that corresponds to a constraint. This
/// type has no MLIR C++ API correspondance.
class ConstraintType : public Type::TypeBase<detail::ConstraintTypeStorage> {
public:
  using Base::Base;

  /// Return an instance of the Constraint type.
  static ConstraintType get(Context &context);
};

//===----------------------------------------------------------------------===//
// OperationType
//===----------------------------------------------------------------------===//

/// This class represents a PDLL type that corresponds to an mlir::Operation.
class OperationType : public Type::TypeBase<detail::OperationTypeStorage> {
public:
  using Base::Base;

  /// Return an instance of the Operation type with an optional operation name.
  /// If no name is provided, this type may refer to any operation.
  static OperationType get(Context &context,
                           std::optional<StringRef> name = std::nullopt,
                           const ods::Operation *odsOp = nullptr);

  /// Return the name of this operation type, or std::nullopt if it doesn't have
  /// on.
  std::optional<StringRef> getName() const;

  /// Return the ODS operation that this type refers to, or nullptr if the ODS
  /// operation is unknown.
  const ods::Operation *getODSOperation() const;
};

//===----------------------------------------------------------------------===//
// RangeType
//===----------------------------------------------------------------------===//

/// This class represents a PDLL type that corresponds to a range of elements
/// with a given element type.
class RangeType : public Type::TypeBase<detail::RangeTypeStorage> {
public:
  using Base::Base;

  /// Return an instance of the Range type with the given element type.
  static RangeType get(Context &context, Type elementType);

  /// Return the element type of this range.
  Type getElementType() const;
};

//===----------------------------------------------------------------------===//
// TypeRangeType
//===----------------------------------------------------------------------===//

/// This class represents a PDLL type that corresponds to an mlir::TypeRange.
class TypeRangeType : public RangeType {
public:
  using RangeType::RangeType;

  /// Provide type casting support.
  static bool classof(Type type);

  /// Return an instance of the TypeRange type.
  static TypeRangeType get(Context &context);
};

//===----------------------------------------------------------------------===//
// ValueRangeType
//===----------------------------------------------------------------------===//

/// This class represents a PDLL type that corresponds to an mlir::ValueRange.
class ValueRangeType : public RangeType {
public:
  using RangeType::RangeType;

  /// Provide type casting support.
  static bool classof(Type type);

  /// Return an instance of the ValueRange type.
  static ValueRangeType get(Context &context);
};

//===----------------------------------------------------------------------===//
// RewriteType
//===----------------------------------------------------------------------===//

/// This class represents a PDLL type that corresponds to a rewrite reference.
/// This type has no MLIR C++ API correspondance.
class RewriteType : public Type::TypeBase<detail::RewriteTypeStorage> {
public:
  using Base::Base;

  /// Return an instance of the Rewrite type.
  static RewriteType get(Context &context);
};

//===----------------------------------------------------------------------===//
// TupleType
//===----------------------------------------------------------------------===//

/// This class represents a PDLL tuple type, i.e. an ordered set of element
/// types with optional names.
class TupleType : public Type::TypeBase<detail::TupleTypeStorage> {
public:
  using Base::Base;

  /// Return an instance of the Tuple type.
  static TupleType get(Context &context, ArrayRef<Type> elementTypes,
                       ArrayRef<StringRef> elementNames);
  static TupleType get(Context &context, ArrayRef<Type> elementTypes = {});

  /// Return the element types of this tuple.
  ArrayRef<Type> getElementTypes() const;

  /// Return the element names of this tuple.
  ArrayRef<StringRef> getElementNames() const;

  /// Return the number of elements within this tuple.
  size_t size() const { return getElementTypes().size(); }

  /// Return if the tuple has no elements.
  bool empty() const { return size() == 0; }
};

//===----------------------------------------------------------------------===//
// TypeType
//===----------------------------------------------------------------------===//

/// This class represents a PDLL type that corresponds to an mlir::Type.
class TypeType : public Type::TypeBase<detail::TypeTypeStorage> {
public:
  using Base::Base;

  /// Return an instance of the Type type.
  static TypeType get(Context &context);
};

//===----------------------------------------------------------------------===//
// ValueType
//===----------------------------------------------------------------------===//

/// This class represents a PDLL type that corresponds to an mlir::Value.
class ValueType : public Type::TypeBase<detail::ValueTypeStorage> {
public:
  using Base::Base;

  /// Return an instance of the Value type.
  static ValueType get(Context &context);
};

} // namespace ast
} // namespace pdll
} // namespace mlir

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::pdll::ast::detail::AttributeTypeStorage)
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::pdll::ast::detail::ConstraintTypeStorage)
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::pdll::ast::detail::OperationTypeStorage)
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::pdll::ast::detail::RangeTypeStorage)
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::pdll::ast::detail::RewriteTypeStorage)
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::pdll::ast::detail::TupleTypeStorage)
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::pdll::ast::detail::TypeTypeStorage)
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::pdll::ast::detail::ValueTypeStorage)

namespace llvm {
template <>
struct DenseMapInfo<mlir::pdll::ast::Type> {
  static mlir::pdll::ast::Type getEmptyKey() {
    void *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::pdll::ast::Type(
        static_cast<mlir::pdll::ast::Type::Storage *>(pointer));
  }
  static mlir::pdll::ast::Type getTombstoneKey() {
    void *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::pdll::ast::Type(
        static_cast<mlir::pdll::ast::Type::Storage *>(pointer));
  }
  static unsigned getHashValue(mlir::pdll::ast::Type val) {
    return llvm::hash_value(val.getImpl());
  }
  static bool isEqual(mlir::pdll::ast::Type lhs, mlir::pdll::ast::Type rhs) {
    return lhs == rhs;
  }
};

/// Add support for llvm style casts.
/// We provide a cast between To and From if From is mlir::pdll::ast::Type or
/// derives from it
template <typename To, typename From>
struct CastInfo<
    To, From,
    std::enable_if_t<
        std::is_same_v<mlir::pdll::ast::Type, std::remove_const_t<From>> ||
        std::is_base_of_v<mlir::pdll::ast::Type, From>>>
    : NullableValueCastFailed<To>,
      DefaultDoCastIfPossible<To, From, CastInfo<To, From>> {
  static inline bool isPossible(mlir::pdll::ast::Type ty) {
    /// Return a constant true instead of a dynamic true when casting to self or
    /// up the hierarchy.
    if constexpr (std::is_base_of_v<To, From>) {
      return true;
    } else {
      return To::classof(ty);
    };
  }
  static inline To doCast(mlir::pdll::ast::Type ty) { return To(ty.getImpl()); }
};
} // namespace llvm

#endif // MLIR_TOOLS_PDLL_AST_TYPES_H_
