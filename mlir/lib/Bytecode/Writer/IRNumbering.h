//===- IRNumbering.h - MLIR bytecode IR numbering ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains various utilities that number IR structures in preparation
// for bytecode emission.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_MLIR_BYTECODE_WRITER_IRNUMBERING_H
#define LIB_MLIR_BYTECODE_WRITER_IRNUMBERING_H

#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/MapVector.h"

namespace mlir {
class BytecodeWriterConfig;

namespace bytecode {
namespace detail {
struct DialectNumbering;

//===----------------------------------------------------------------------===//
// Attribute and Type Numbering
//===----------------------------------------------------------------------===//

/// This class represents a numbering entry for an Attribute or Type.
struct AttrTypeNumbering {
  AttrTypeNumbering(PointerUnion<Attribute, Type> value) : value(value) {}

  /// The concrete value.
  PointerUnion<Attribute, Type> value;

  /// The number assigned to this value.
  unsigned number = 0;

  /// The number of references to this value.
  unsigned refCount = 1;

  /// The dialect of this value.
  DialectNumbering *dialect = nullptr;
};
struct AttributeNumbering : public AttrTypeNumbering {
  AttributeNumbering(Attribute value) : AttrTypeNumbering(value) {}
  Attribute getValue() const { return value.get<Attribute>(); }
};
struct TypeNumbering : public AttrTypeNumbering {
  TypeNumbering(Type value) : AttrTypeNumbering(value) {}
  Type getValue() const { return value.get<Type>(); }
};

//===----------------------------------------------------------------------===//
// OpName Numbering
//===----------------------------------------------------------------------===//

/// This class represents the numbering entry of an operation name.
struct OpNameNumbering {
  OpNameNumbering(DialectNumbering *dialect, OperationName name)
      : dialect(dialect), name(name) {}

  /// The dialect of this value.
  DialectNumbering *dialect;

  /// The concrete name.
  OperationName name;

  /// The number assigned to this name.
  unsigned number = 0;

  /// The number of references to this name.
  unsigned refCount = 1;
};

//===----------------------------------------------------------------------===//
// Dialect Numbering
//===----------------------------------------------------------------------===//

/// This class represents a numbering entry for an Dialect.
struct DialectNumbering {
  DialectNumbering(StringRef name, unsigned number)
      : name(name), number(number) {}

  /// The namespace of the dialect.
  StringRef name;

  /// The number assigned to the dialect.
  unsigned number;

  /// The loaded dialect, or nullptr if the dialect isn't loaded.
  Dialect *dialect = nullptr;
};

//===----------------------------------------------------------------------===//
// IRNumberingState
//===----------------------------------------------------------------------===//

/// This class manages numbering IR entities in preparation of bytecode
/// emission.
class IRNumberingState {
public:
  IRNumberingState(Operation *op);

  /// Return the numbered dialects.
  auto getDialects() {
    return llvm::make_pointee_range(llvm::make_second_range(dialects));
  }
  auto getAttributes() { return llvm::make_pointee_range(orderedAttrs); }
  auto getOpNames() { return llvm::make_pointee_range(orderedOpNames); }
  auto getTypes() { return llvm::make_pointee_range(orderedTypes); }

  /// Return the number for the given IR unit.
  unsigned getNumber(Attribute attr) {
    assert(attrs.count(attr) && "attribute not numbered");
    return attrs[attr]->number;
  }
  unsigned getNumber(Block *block) {
    assert(blockIDs.count(block) && "block not numbered");
    return blockIDs[block];
  }
  unsigned getNumber(OperationName opName) {
    assert(opNames.count(opName) && "opName not numbered");
    return opNames[opName]->number;
  }
  unsigned getNumber(Type type) {
    assert(types.count(type) && "type not numbered");
    return types[type]->number;
  }
  unsigned getNumber(Value value) {
    assert(valueIDs.count(value) && "value not numbered");
    return valueIDs[value];
  }

  /// Return the block and value counts of the given region.
  std::pair<unsigned, unsigned> getBlockValueCount(Region *region) {
    assert(regionBlockValueCounts.count(region) && "value not numbered");
    return regionBlockValueCounts[region];
  }

  /// Return the number of operations in the given block.
  unsigned getOperationCount(Block *block) {
    assert(blockOperationCounts.count(block) && "block not numbered");
    return blockOperationCounts[block];
  }

private:
  /// Number the given IR unit for bytecode emission.
  void number(Attribute attr);
  void number(Block &block);
  DialectNumbering &numberDialect(Dialect *dialect);
  DialectNumbering &numberDialect(StringRef dialect);
  void number(Operation &op);
  void number(OperationName opName);
  void number(Region &region);
  void number(Type type);

  /// Mapping from IR to the respective numbering entries.
  DenseMap<Attribute, AttributeNumbering *> attrs;
  DenseMap<OperationName, OpNameNumbering *> opNames;
  DenseMap<Type, TypeNumbering *> types;
  DenseMap<Dialect *, DialectNumbering *> registeredDialects;
  llvm::MapVector<StringRef, DialectNumbering *> dialects;
  std::vector<AttributeNumbering *> orderedAttrs;
  std::vector<OpNameNumbering *> orderedOpNames;
  std::vector<TypeNumbering *> orderedTypes;

  /// Allocators used for the various numbering entries.
  llvm::SpecificBumpPtrAllocator<AttributeNumbering> attrAllocator;
  llvm::SpecificBumpPtrAllocator<DialectNumbering> dialectAllocator;
  llvm::SpecificBumpPtrAllocator<OpNameNumbering> opNameAllocator;
  llvm::SpecificBumpPtrAllocator<TypeNumbering> typeAllocator;

  /// The value ID for each Block and Value.
  DenseMap<Block *, unsigned> blockIDs;
  DenseMap<Value, unsigned> valueIDs;

  /// The number of operations in each block.
  DenseMap<Block *, unsigned> blockOperationCounts;

  /// A map from region to the number of blocks and values within that region.
  DenseMap<Region *, std::pair<unsigned, unsigned>> regionBlockValueCounts;

  /// The next value ID to assign when numbering.
  unsigned nextValueID = 0;
};
} // namespace detail
} // namespace bytecode
} // namespace mlir

#endif
