//===- Predicate.h - Pattern predicates -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for "predicates" used when converting PDL into
// a matcher tree. Predicates are composed of three different parts:
//
//  * Positions
//    - A position refers to a specific location on the input DAG, i.e. an
//      existing MLIR entity being matched. These can be attributes, operands,
//      operations, results, and types. Each position also defines a relation to
//      its parent. For example, the operand `[0] -> 1` has a parent operation
//      position `[0]`. The attribute `[0, 1] -> "myAttr"` has parent operation
//      position of `[0, 1]`. The operation `[0, 1]` has a parent operand edge
//      `[0] -> 1` (i.e. it is the defining op of operand 1). The only position
//      without a parent is `[0]`, which refers to the root operation.
//  * Questions
//    - A question refers to a query on a specific positional value. For
//    example, an operation name question checks the name of an operation
//    position.
//  * Answers
//    - An answer is the expected result of a question. For example, when
//    matching an operation with the name "foo.op". The question would be an
//    operation name question, with an expected answer of "foo.op".
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_CONVERSION_PDLTOPDLINTERP_PREDICATE_H_
#define MLIR_LIB_CONVERSION_PDLTOPDLINTERP_PREDICATE_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace pdl_to_pdl_interp {
namespace Predicates {
/// An enumeration of the kinds of predicates.
enum Kind : unsigned {
  /// Positions, ordered by decreasing priority.
  OperationPos,
  OperandPos,
  AttributePos,
  ResultPos,
  TypePos,

  // Questions, ordered by dependency and decreasing priority.
  IsNotNullQuestion,
  OperationNameQuestion,
  TypeQuestion,
  AttributeQuestion,
  OperandCountQuestion,
  ResultCountQuestion,
  EqualToQuestion,
  ConstraintQuestion,

  // Answers.
  AttributeAnswer,
  TrueAnswer,
  OperationNameAnswer,
  TypeAnswer,
  UnsignedAnswer,
};
} // end namespace Predicates

/// Base class for all predicates, used to allow efficient pointer comparison.
template <typename ConcreteT, typename BaseT, typename Key,
          Predicates::Kind Kind>
class PredicateBase : public BaseT {
public:
  using KeyTy = Key;
  using Base = PredicateBase<ConcreteT, BaseT, Key, Kind>;

  template <typename KeyT>
  explicit PredicateBase(KeyT &&key)
      : BaseT(Kind), key(std::forward<KeyT>(key)) {}

  /// Get an instance of this position.
  template <typename... Args>
  static ConcreteT *get(StorageUniquer &uniquer, Args &&...args) {
    return uniquer.get<ConcreteT>(/*initFn=*/{}, std::forward<Args>(args)...);
  }

  /// Construct an instance with the given storage allocator.
  template <typename KeyT>
  static ConcreteT *construct(StorageUniquer::StorageAllocator &alloc,
                              KeyT &&key) {
    return new (alloc.allocate<ConcreteT>()) ConcreteT(std::forward<KeyT>(key));
  }

  /// Utility methods required by the storage allocator.
  bool operator==(const KeyTy &key) const { return this->key == key; }
  static bool classof(const BaseT *pred) { return pred->getKind() == Kind; }

  /// Return the key value of this predicate.
  const KeyTy &getValue() const { return key; }

protected:
  KeyTy key;
};

/// Base storage for simple predicates that only unique with the kind.
template <typename ConcreteT, typename BaseT, Predicates::Kind Kind>
class PredicateBase<ConcreteT, BaseT, void, Kind> : public BaseT {
public:
  using Base = PredicateBase<ConcreteT, BaseT, void, Kind>;

  explicit PredicateBase() : BaseT(Kind) {}

  static ConcreteT *get(StorageUniquer &uniquer) {
    return uniquer.get<ConcreteT>();
  }
  static bool classof(const BaseT *pred) { return pred->getKind() == Kind; }
};

//===----------------------------------------------------------------------===//
// Positions
//===----------------------------------------------------------------------===//

struct OperationPosition;

/// A position describes a value on the input IR on which a predicate may be
/// applied, such as an operation or attribute. This enables re-use between
/// predicates, and assists generating bytecode and memory management.
///
/// Operation positions form the base of other positions, which are formed
/// relative to a parent operation, e.g. OperandPosition<[0] -> 1>. Operations
/// are indexed by child index: [0, 1, 2] refers to the 3rd child of the 2nd
/// child of the root operation.
///
/// Positions are linked to their parent position, which describes how to obtain
/// a positional value. As a concrete example, getting OperationPosition<[0, 1]>
/// would be `root->getOperand(1)->getDefiningOp()`, so its parent is
/// OperandPosition<[0] -> 1>, whose parent is OperationPosition<[0]>.
class Position : public StorageUniquer::BaseStorage {
public:
  explicit Position(Predicates::Kind kind) : kind(kind) {}
  virtual ~Position();

  /// Returns the base node position. This is an array of indices.
  virtual ArrayRef<unsigned> getIndex() const = 0;

  /// Returns the parent position. The root operation position has no parent.
  Position *getParent() const { return parent; }

  /// Returns the kind of this position.
  Predicates::Kind getKind() const { return kind; }

protected:
  /// Link to the parent position.
  Position *parent = nullptr;

private:
  /// The kind of this position.
  Predicates::Kind kind;
};

//===----------------------------------------------------------------------===//
// AttributePosition

/// A position describing an attribute of an operation.
struct AttributePosition
    : public PredicateBase<AttributePosition, Position,
                           std::pair<OperationPosition *, Identifier>,
                           Predicates::AttributePos> {
  explicit AttributePosition(const KeyTy &key);

  /// Returns the index of this position.
  ArrayRef<unsigned> getIndex() const final { return parent->getIndex(); }

  /// Returns the attribute name of this position.
  Identifier getName() const { return key.second; }
};

//===----------------------------------------------------------------------===//
// OperandPosition

/// A position describing an operand of an operation.
struct OperandPosition
    : public PredicateBase<OperandPosition, Position,
                           std::pair<OperationPosition *, unsigned>,
                           Predicates::OperandPos> {
  explicit OperandPosition(const KeyTy &key);

  /// Returns the index of this position.
  ArrayRef<unsigned> getIndex() const final { return parent->getIndex(); }

  /// Returns the operand number of this position.
  unsigned getOperandNumber() const { return key.second; }
};

//===----------------------------------------------------------------------===//
// OperationPosition

/// An operation position describes an operation node in the IR. Other position
/// kinds are formed with respect to an operation position.
struct OperationPosition
    : public PredicateBase<OperationPosition, Position, ArrayRef<unsigned>,
                           Predicates::OperationPos> {
  using Base::Base;

  /// Gets the root position, which is always [0].
  static OperationPosition *getRoot(StorageUniquer &uniquer) {
    return get(uniquer, ArrayRef<unsigned>(0));
  }
  /// Gets a node position for the given index.
  static OperationPosition *get(StorageUniquer &uniquer,
                                ArrayRef<unsigned> index);

  /// Constructs an instance with the given storage allocator.
  static OperationPosition *construct(StorageUniquer::StorageAllocator &alloc,
                                      ArrayRef<unsigned> key) {
    return Base::construct(alloc, alloc.copyInto(key));
  }

  /// Returns the index of this position.
  ArrayRef<unsigned> getIndex() const final { return key; }

  /// Returns if this operation position corresponds to the root.
  bool isRoot() const { return key.size() == 1 && key[0] == 0; }
};

//===----------------------------------------------------------------------===//
// ResultPosition

/// A position describing a result of an operation.
struct ResultPosition
    : public PredicateBase<ResultPosition, Position,
                           std::pair<OperationPosition *, unsigned>,
                           Predicates::ResultPos> {
  explicit ResultPosition(const KeyTy &key) : Base(key) { parent = key.first; }

  /// Returns the index of this position.
  ArrayRef<unsigned> getIndex() const final { return key.first->getIndex(); }

  /// Returns the result number of this position.
  unsigned getResultNumber() const { return key.second; }
};

//===----------------------------------------------------------------------===//
// TypePosition

/// A position describing the result type of an entity, i.e. an Attribute,
/// Operand, Result, etc.
struct TypePosition : public PredicateBase<TypePosition, Position, Position *,
                                           Predicates::TypePos> {
  explicit TypePosition(const KeyTy &key) : Base(key) {
    assert((isa<AttributePosition>(key) || isa<OperandPosition>(key) ||
            isa<ResultPosition>(key)) &&
           "expected parent to be an attribute, operand, or result");
    parent = key;
  }

  /// Returns the index of this position.
  ArrayRef<unsigned> getIndex() const final { return key->getIndex(); }
};

//===----------------------------------------------------------------------===//
// Qualifiers
//===----------------------------------------------------------------------===//

/// An ordinal predicate consists of a "Question" and a set of acceptable
/// "Answers" (later converted to ordinal values). A predicate will query some
/// property of a positional value and decide what to do based on the result.
///
/// This makes top-level predicate representations ordinal (SwitchOp). Later,
/// predicates that end up with only one acceptable answer (including all
/// boolean kinds) will be converted to boolean predicates (PredicateOp) in the
/// matcher.
///
/// For simplicity, both are represented as "qualifiers", with a base kind and
/// perhaps additional properties. For example, all OperationName predicates ask
/// the same question, but GenericConstraint predicates may ask different ones.
class Qualifier : public StorageUniquer::BaseStorage {
public:
  explicit Qualifier(Predicates::Kind kind) : kind(kind) {}

  /// Returns the kind of this qualifier.
  Predicates::Kind getKind() const { return kind; }

private:
  /// The kind of this position.
  Predicates::Kind kind;
};

//===----------------------------------------------------------------------===//
// Answers

/// An Answer representing an `Attribute` value.
struct AttributeAnswer
    : public PredicateBase<AttributeAnswer, Qualifier, Attribute,
                           Predicates::AttributeAnswer> {
  using Base::Base;
};

/// An Answer representing an `OperationName` value.
struct OperationNameAnswer
    : public PredicateBase<OperationNameAnswer, Qualifier, OperationName,
                           Predicates::OperationNameAnswer> {
  using Base::Base;
};

/// An Answer representing a boolean `true` value.
struct TrueAnswer
    : PredicateBase<TrueAnswer, Qualifier, void, Predicates::TrueAnswer> {
  using Base::Base;
};

/// An Answer representing a `Type` value.
struct TypeAnswer : public PredicateBase<TypeAnswer, Qualifier, Type,
                                         Predicates::TypeAnswer> {
  using Base::Base;
};

/// An Answer representing an unsigned value.
struct UnsignedAnswer
    : public PredicateBase<UnsignedAnswer, Qualifier, unsigned,
                           Predicates::UnsignedAnswer> {
  using Base::Base;
};

//===----------------------------------------------------------------------===//
// Questions

/// Compare an `Attribute` to a constant value.
struct AttributeQuestion
    : public PredicateBase<AttributeQuestion, Qualifier, void,
                           Predicates::AttributeQuestion> {};

/// Apply a parameterized constraint to multiple position values.
struct ConstraintQuestion
    : public PredicateBase<
          ConstraintQuestion, Qualifier,
          std::tuple<StringRef, ArrayRef<Position *>, Attribute>,
          Predicates::ConstraintQuestion> {
  using Base::Base;

  /// Construct an instance with the given storage allocator.
  static ConstraintQuestion *construct(StorageUniquer::StorageAllocator &alloc,
                                       KeyTy key) {
    return Base::construct(alloc, KeyTy{alloc.copyInto(std::get<0>(key)),
                                        alloc.copyInto(std::get<1>(key)),
                                        std::get<2>(key)});
  }
};

/// Compare the equality of two values.
struct EqualToQuestion
    : public PredicateBase<EqualToQuestion, Qualifier, Position *,
                           Predicates::EqualToQuestion> {
  using Base::Base;
};

/// Compare a positional value with null, i.e. check if it exists.
struct IsNotNullQuestion
    : public PredicateBase<IsNotNullQuestion, Qualifier, void,
                           Predicates::IsNotNullQuestion> {};

/// Compare the number of operands of an operation with a known value.
struct OperandCountQuestion
    : public PredicateBase<OperandCountQuestion, Qualifier, void,
                           Predicates::OperandCountQuestion> {};

/// Compare the name of an operation with a known value.
struct OperationNameQuestion
    : public PredicateBase<OperationNameQuestion, Qualifier, void,
                           Predicates::OperationNameQuestion> {};

/// Compare the number of results of an operation with a known value.
struct ResultCountQuestion
    : public PredicateBase<ResultCountQuestion, Qualifier, void,
                           Predicates::ResultCountQuestion> {};

/// Compare the type of an attribute or value with a known type.
struct TypeQuestion : public PredicateBase<TypeQuestion, Qualifier, void,
                                           Predicates::TypeQuestion> {};

//===----------------------------------------------------------------------===//
// PredicateUniquer
//===----------------------------------------------------------------------===//

/// This class provides a storage uniquer that is used to allocate predicate
/// instances.
class PredicateUniquer : public StorageUniquer {
public:
  PredicateUniquer() {
    // Register the types of Positions with the uniquer.
    registerParametricStorageType<AttributePosition>();
    registerParametricStorageType<OperandPosition>();
    registerParametricStorageType<OperationPosition>();
    registerParametricStorageType<ResultPosition>();
    registerParametricStorageType<TypePosition>();

    // Register the types of Questions with the uniquer.
    registerParametricStorageType<AttributeAnswer>();
    registerParametricStorageType<OperationNameAnswer>();
    registerParametricStorageType<TypeAnswer>();
    registerParametricStorageType<UnsignedAnswer>();
    registerSingletonStorageType<TrueAnswer>();

    // Register the types of Answers with the uniquer.
    registerParametricStorageType<ConstraintQuestion>();
    registerParametricStorageType<EqualToQuestion>();
    registerSingletonStorageType<AttributeQuestion>();
    registerSingletonStorageType<IsNotNullQuestion>();
    registerSingletonStorageType<OperandCountQuestion>();
    registerSingletonStorageType<OperationNameQuestion>();
    registerSingletonStorageType<ResultCountQuestion>();
    registerSingletonStorageType<TypeQuestion>();
  }
};

//===----------------------------------------------------------------------===//
// PredicateBuilder
//===----------------------------------------------------------------------===//

/// This class provides utilities for constructing predicates.
class PredicateBuilder {
public:
  PredicateBuilder(PredicateUniquer &uniquer, MLIRContext *ctx)
      : uniquer(uniquer), ctx(ctx) {}

  //===--------------------------------------------------------------------===//
  // Positions
  //===--------------------------------------------------------------------===//

  /// Returns the root operation position.
  Position *getRoot() { return OperationPosition::getRoot(uniquer); }

  /// Returns the parent position defining the value held by the given operand.
  OperationPosition *getParent(OperandPosition *p) {
    std::vector<unsigned> index = p->getIndex();
    index.push_back(p->getOperandNumber());
    return OperationPosition::get(uniquer, index);
  }

  /// Returns an attribute position for an attribute of the given operation.
  Position *getAttribute(OperationPosition *p, StringRef name) {
    return AttributePosition::get(uniquer, p, Identifier::get(name, ctx));
  }

  /// Returns an operand position for an operand of the given operation.
  Position *getOperand(OperationPosition *p, unsigned operand) {
    return OperandPosition::get(uniquer, p, operand);
  }

  /// Returns a result position for a result of the given operation.
  Position *getResult(OperationPosition *p, unsigned result) {
    return ResultPosition::get(uniquer, p, result);
  }

  /// Returns a type position for the given entity.
  Position *getType(Position *p) { return TypePosition::get(uniquer, p); }

  //===--------------------------------------------------------------------===//
  // Qualifiers
  //===--------------------------------------------------------------------===//

  /// An ordinal predicate consists of a "Question" and a set of acceptable
  /// "Answers" (later converted to ordinal values). A predicate will query some
  /// property of a positional value and decide what to do based on the result.
  using Predicate = std::pair<Qualifier *, Qualifier *>;

  /// Create a predicate comparing an attribute to a known value.
  Predicate getAttributeConstraint(Attribute attr) {
    return {AttributeQuestion::get(uniquer),
            AttributeAnswer::get(uniquer, attr)};
  }

  /// Create a predicate comparing two values.
  Predicate getEqualTo(Position *pos) {
    return {EqualToQuestion::get(uniquer, pos), TrueAnswer::get(uniquer)};
  }

  /// Create a predicate that applies a generic constraint.
  Predicate getConstraint(StringRef name, ArrayRef<Position *> pos,
                          Attribute params) {
    return {
        ConstraintQuestion::get(uniquer, std::make_tuple(name, pos, params)),
        TrueAnswer::get(uniquer)};
  }

  /// Create a predicate comparing a value with null.
  Predicate getIsNotNull() {
    return {IsNotNullQuestion::get(uniquer), TrueAnswer::get(uniquer)};
  }

  /// Create a predicate comparing the number of operands of an operation to a
  /// known value.
  Predicate getOperandCount(unsigned count) {
    return {OperandCountQuestion::get(uniquer),
            UnsignedAnswer::get(uniquer, count)};
  }

  /// Create a predicate comparing the name of an operation to a known value.
  Predicate getOperationName(StringRef name) {
    return {OperationNameQuestion::get(uniquer),
            OperationNameAnswer::get(uniquer, OperationName(name, ctx))};
  }

  /// Create a predicate comparing the number of results of an operation to a
  /// known value.
  Predicate getResultCount(unsigned count) {
    return {ResultCountQuestion::get(uniquer),
            UnsignedAnswer::get(uniquer, count)};
  }

  /// Create a predicate comparing the type of an attribute or value to a known
  /// type.
  Predicate getTypeConstraint(Type type) {
    return {TypeQuestion::get(uniquer), TypeAnswer::get(uniquer, type)};
  }

private:
  /// The uniquer used when allocating predicate nodes.
  PredicateUniquer &uniquer;

  /// The current MLIR context.
  MLIRContext *ctx;
};

} // end namespace pdl_to_pdl_interp
} // end namespace mlir

#endif // MLIR_CONVERSION_PDLTOPDLINTERP_PREDICATE_H_
