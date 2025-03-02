//===- IRTypes.h - Type Interfaces ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BINDINGS_PYTHON_IRTYPES_H
#define MLIR_BINDINGS_PYTHON_IRTYPES_H

#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace mlir {

/// Shaped Type Interface - ShapedType
class PyShapedType : public python::PyConcreteType<PyShapedType> {
public:
  static const IsAFunctionTy isaFunction;
  static constexpr const char *pyClassName = "ShapedType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);

private:
  void requireHasRank();
};

} // namespace mlir

#endif // MLIR_BINDINGS_PYTHON_IRTYPES_H
