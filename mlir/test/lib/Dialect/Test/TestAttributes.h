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

#ifndef MLIR_TESTATTRIBUTES_H
#define MLIR_TESTATTRIBUTES_H

#include <tuple>

#include "TestTraits.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"

#include "TestAttrInterfaces.h.inc"
#include "TestOpEnums.h.inc"
#include "mlir/IR/DialectResourceBlobManager.h"

namespace test {
class TestDialect;

/// A handle used to reference external elements instances.
using TestDialectResourceBlobHandle =
    mlir::DialectResourceBlobHandle<TestDialect>;

/// Storage for simple named recursive attribute, where the attribute is
/// identified by its name and can "contain" another attribute, including
/// itself.
struct TestRecursiveAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = ::llvm::StringRef;

  explicit TestRecursiveAttrStorage(::llvm::StringRef key) : name(key) {}

  bool operator==(const KeyTy &other) const { return name == other; }

  static TestRecursiveAttrStorage *
  construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<TestRecursiveAttrStorage>())
        TestRecursiveAttrStorage(allocator.copyInto(key));
  }

  ::mlir::LogicalResult mutate(::mlir::AttributeStorageAllocator &allocator,
                               ::mlir::Attribute newBody) {
    // Cannot set a different body than before.
    if (body && body != newBody)
      return ::mlir::failure();

    body = newBody;
    return ::mlir::success();
  }

  ::llvm::StringRef name;
  ::mlir::Attribute body;
};

} // namespace test

#define GET_ATTRDEF_CLASSES
#include "TestAttrDefs.h.inc"

#endif // MLIR_TESTATTRIBUTES_H
