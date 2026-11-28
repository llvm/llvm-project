//===- TestOpsInterfaces.h - Test dialect ops in `Interfaces` ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TEST_DIALECT_TEST_TESTOPSINTERFACES_H
#define MLIR_TEST_DIALECT_TEST_TESTOPSINTERFACES_H

#include "TestDialect.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/LLVMIR/NVVMRequiresSMTraits.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace test {
//===----------------------------------------------------------------------===//
// TestResource
//===----------------------------------------------------------------------===//

/// A test resource for side effects (under DefaultResource).
struct TestResource : public mlir::SideEffects::Resource::Base<
                          TestResource, mlir::SideEffects::DefaultResource> {
  llvm::StringRef getName() const final { return "<Test>"; }
  mlir::SideEffects::Resource *getParent() const override {
    return mlir::SideEffects::DefaultResource::get();
  }
};

/// A test resource that is a root (disjoint from DefaultResource).
struct TestNonAddressableResource
    : public mlir::SideEffects::Resource::Base<TestNonAddressableResource> {
  llvm::StringRef getName() const final { return "<TestNonAddressable>"; }
  bool isAddressable() const override { return false; }
};

/// Two disjoint sub-resources (roots) for testing sibling disjointness.
struct TestNonAddressableSubResourceA
    : public mlir::SideEffects::Resource::Base<TestNonAddressableSubResourceA> {
  TestNonAddressableSubResourceA() = default;
  llvm::StringRef getName() const override {
    return "TestNonAddressableSubResourceA";
  }
  bool isAddressable() const override { return false; }

protected:
  TestNonAddressableSubResourceA(mlir::TypeID id) : Base(id) {}
};

struct TestNonAddressableSubResourceB
    : public mlir::SideEffects::Resource::Base<TestNonAddressableSubResourceB> {
  TestNonAddressableSubResourceB() = default;
  llvm::StringRef getName() const override {
    return "TestNonAddressableSubResourceB";
  }
  bool isAddressable() const override { return false; }

protected:
  TestNonAddressableSubResourceB(mlir::TypeID id) : Base(id) {}
};

struct TestNonAddressableResourceA
    : public mlir::SideEffects::Resource::Base<TestNonAddressableResourceA,
                                               TestNonAddressableSubResourceA> {
  llvm::StringRef getName() const final { return "<TestNonAddressableA>"; }
  bool isAddressable() const override { return false; }
  mlir::SideEffects::Resource *getParent() const override {
    return TestNonAddressableSubResourceA::get();
  }
};

struct TestNonAddressableResourceB
    : public mlir::SideEffects::Resource::Base<TestNonAddressableResourceB,
                                               TestNonAddressableSubResourceB> {
  llvm::StringRef getName() const final { return "<TestNonAddressableB>"; }
  bool isAddressable() const override { return false; }
  mlir::SideEffects::Resource *getParent() const override {
    return TestNonAddressableSubResourceB::get();
  }
};
} // namespace test

#define GET_OP_CLASSES
#include "TestOpsInterfaces.h.inc"

#endif // MLIR_TEST_DIALECT_TEST_TESTOPSINTERFACES_H
