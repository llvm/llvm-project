//===- TestTypes.h - MLIR Test Dialect Types --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains types defined by the TestDialect for testing various
// features of MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TESTTYPES_H
#define MLIR_TESTTYPES_H

#include <tuple>

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace test {

/// FieldInfo represents a field in the StructType data type. It is used as a
/// parameter in TestTypeDefs.td.
struct FieldInfo {
  StringRef name;
  Type type;

  // Custom allocation called from generated constructor code
  FieldInfo allocateInto(TypeStorageAllocator &alloc) const {
    return FieldInfo{alloc.copyInto(name), type};
  }
};

} // namespace test
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "TestTypeDefs.h.inc"

namespace mlir {
namespace test {

#include "TestTypeInterfaces.h.inc"

/// This class is a simple test type that uses a generated interface.
struct TestType : public Type::TypeBase<TestType, Type, TypeStorage,
                                        TestTypeInterface::Trait> {
  using Base::Base;

  /// Provide a definition for the necessary interface methods.
  void printTypeC(Location loc) const {
    emitRemark(loc) << *this << " - TestC";
  }
};

/// Storage for simple named recursive types, where the type is identified by
/// its name and can "contain" another type, including itself.
struct TestRecursiveTypeStorage : public TypeStorage {
  using KeyTy = StringRef;

  explicit TestRecursiveTypeStorage(StringRef key) : name(key), body(Type()) {}

  bool operator==(const KeyTy &other) const { return name == other; }

  static TestRecursiveTypeStorage *construct(TypeStorageAllocator &allocator,
                                             const KeyTy &key) {
    return new (allocator.allocate<TestRecursiveTypeStorage>())
        TestRecursiveTypeStorage(allocator.copyInto(key));
  }

  LogicalResult mutate(TypeStorageAllocator &allocator, Type newBody) {
    // Cannot set a different body than before.
    if (body && body != newBody)
      return failure();

    body = newBody;
    return success();
  }

  StringRef name;
  Type body;
};

/// Simple recursive type identified by its name and pointing to another named
/// type, potentially itself. This requires the body to be mutated separately
/// from type creation.
class TestRecursiveType
    : public Type::TypeBase<TestRecursiveType, Type, TestRecursiveTypeStorage> {
public:
  using Base::Base;

  static TestRecursiveType get(MLIRContext *ctx, StringRef name) {
    return Base::get(ctx, name);
  }

  /// Body getter and setter.
  LogicalResult setBody(Type body) { return Base::mutate(body); }
  Type getBody() { return getImpl()->body; }

  /// Name/key getter.
  StringRef getName() { return getImpl()->name; }
};

} // namespace test
} // namespace mlir

#endif // MLIR_TESTTYPES_H
