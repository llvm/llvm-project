//===- TestTraits.h - MLIR Test Traits --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains traits defined by the TestDialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TESTTRAITS_H
#define MLIR_TESTTRAITS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace TypeTrait {

/// A trait defined on types for testing purposes.
template <typename ConcreteType>
class TestTypeTrait : public TypeTrait::TraitBase<ConcreteType, TestTypeTrait> {
};

} // namespace TypeTrait

namespace AttributeTrait {

/// A trait defined on attributes for testing purposes.
template <typename ConcreteType>
class TestAttrTrait
    : public AttributeTrait::TraitBase<ConcreteType, TestAttrTrait> {};

} // namespace AttributeTrait
} // namespace mlir

#endif // MLIR_TESTTRAITS_H
