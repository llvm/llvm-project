//===- OpenACCVariableInfo.h - OpenACC Variable Info Attr -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the VariableInfoAttr base class and the IsVariableInfo
// trait.
//
// Any dialect can define Language-specific variable metadata attribute by
// manually attaching the IsVariableInfo trait and using VariableInfoAttr as
// the baseCppClass.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENACC_OPENACCVARIABLEINFO_H
#define MLIR_DIALECT_OPENACC_OPENACCVARIABLEINFO_H

#include "mlir/IR/Attributes.h"

namespace mlir {
namespace acc {
namespace AttributeTrait {
/// Trait attached to attributes that are OpenACC variable info attributes.
template <typename ConcreteType>
struct IsVariableInfo
    : public mlir::AttributeTrait::TraitBase<ConcreteType, IsVariableInfo> {};
} // namespace AttributeTrait

/// Base attribute class for language-specific variable information carried
/// through the OpenACC type interface helpers.
class VariableInfoAttr : public mlir::Attribute {
public:
  using Attribute::Attribute;

  static bool classof(mlir::Attribute attr) {
    return attr.hasTrait<::mlir::acc::AttributeTrait::IsVariableInfo>();
  }
};

} // namespace acc
} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_OPENACCVARIABLEINFO_H
