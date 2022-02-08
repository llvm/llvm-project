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

#include "TestTraits.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

namespace test {
class TestAttrWithFormatAttr;

/// FieldInfo represents a field in the StructType data type. It is used as a
/// parameter in TestTypeDefs.td.
struct FieldInfo {
  ::llvm::StringRef name;
  ::mlir::Type type;

  // Custom allocation called from generated constructor code
  FieldInfo allocateInto(::mlir::TypeStorageAllocator &alloc) const {
    return FieldInfo{alloc.copyInto(name), type};
  }
};

/// A custom type for a test type parameter.
struct CustomParam {
  int value;

  bool operator==(const CustomParam &other) const {
    return other.value == value;
  }
};

inline llvm::hash_code hash_value(const test::CustomParam &param) {
  return llvm::hash_value(param.value);
}

} // namespace test

namespace mlir {
template <>
struct FieldParser<test::CustomParam> {
  static FailureOr<test::CustomParam> parse(AsmParser &parser) {
    auto value = FieldParser<int>::parse(parser);
    if (failed(value))
      return failure();
    return test::CustomParam{value.getValue()};
  }
};

inline mlir::AsmPrinter &operator<<(mlir::AsmPrinter &printer,
                                    test::CustomParam param) {
  return printer << param.value;
}

/// Overload the attribute parameter parser for optional integers.
template <>
struct FieldParser<Optional<int>> {
  static FailureOr<Optional<int>> parse(AsmParser &parser) {
    Optional<int> value;
    value.emplace();
    OptionalParseResult result = parser.parseOptionalInteger(*value);
    if (result.hasValue()) {
      if (succeeded(*result))
        return value;
      return failure();
    }
    value.reset();
    return value;
  }
};
} // namespace mlir

#include "TestTypeInterfaces.h.inc"

#define GET_TYPEDEF_CLASSES
#include "TestTypeDefs.h.inc"

namespace test {

/// Storage for simple named recursive types, where the type is identified by
/// its name and can "contain" another type, including itself.
struct TestRecursiveTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = ::llvm::StringRef;

  explicit TestRecursiveTypeStorage(::llvm::StringRef key)
      : name(key), body(::mlir::Type()) {}

  bool operator==(const KeyTy &other) const { return name == other; }

  static TestRecursiveTypeStorage *
  construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<TestRecursiveTypeStorage>())
        TestRecursiveTypeStorage(allocator.copyInto(key));
  }

  ::mlir::LogicalResult mutate(::mlir::TypeStorageAllocator &allocator,
                               ::mlir::Type newBody) {
    // Cannot set a different body than before.
    if (body && body != newBody)
      return ::mlir::failure();

    body = newBody;
    return ::mlir::success();
  }

  ::llvm::StringRef name;
  ::mlir::Type body;
};

/// Simple recursive type identified by its name and pointing to another named
/// type, potentially itself. This requires the body to be mutated separately
/// from type creation.
class TestRecursiveType
    : public ::mlir::Type::TypeBase<TestRecursiveType, ::mlir::Type,
                                    TestRecursiveTypeStorage> {
public:
  using Base::Base;

  static TestRecursiveType get(::mlir::MLIRContext *ctx,
                               ::llvm::StringRef name) {
    return Base::get(ctx, name);
  }

  /// Body getter and setter.
  ::mlir::LogicalResult setBody(Type body) { return Base::mutate(body); }
  ::mlir::Type getBody() { return getImpl()->body; }

  /// Name/key getter.
  ::llvm::StringRef getName() { return getImpl()->name; }
};

} // namespace test

#endif // MLIR_TESTTYPES_H
