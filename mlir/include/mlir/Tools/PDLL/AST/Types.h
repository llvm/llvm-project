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

namespace mlir {
namespace pdll {
namespace ods {
class Operation;
} // namespace ods

namespace ast {
class Context;

namespace detail {
struct AttributeTypeStorage;
struct ConstraintTypeStorage;
struct OperationTypeStorage;
struct RangeTypeStorage;
struct RewriteTypeStorage;
struct TupleTypeStorage;
struct TypeTypeStorage;
struct ValueTypeStorage;
} // namespace detail

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

  /// Provide type casting support.
  template <typename U>
  bool isa() const {
    assert(impl && "isa<> used on a null type.");
    return U::classof(*this);
  }
  template <typename U, typename V, typename... Others>
  bool isa() const {
    return isa<U>() || isa<V, Others...>();
  }
  template <typename U>
  U dyn_cast() const {
    return isa<U>() ? U(impl) : U(nullptr);
  }
  template <typename U>
  U dyn_cast_or_null() const {
    return (impl && isa<U>()) ? U(impl) : U(nullptr);
  }
  template <typename U>
  U cast() const {
    assert(isa<U>());
    return U(impl);
  }

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
                           Optional<StringRef> name = std::nullopt,
                           const ods::Operation *odsOp = nullptr);

  /// Return the name of this operation type, or None if it doesn't have on.
  Optional<StringRef> getName() const;

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
  static TupleType get(Context &context,
                       ArrayRef<Type> elementTypes = std::nullopt);

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
} // namespace llvm

#endif // MLIR_TOOLS_PDLL_AST_TYPES_H_
