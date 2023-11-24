//===- SideEffects.h - Side Effects classes ---------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Wrapper around side effect related classes defined in TableGen.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_SIDEEFFECTS_H_
#define MLIR_TABLEGEN_SIDEEFFECTS_H_

#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/Operator.h"

namespace mlir {
namespace tblgen {

// This class represents a specific instance of an effect that is being
// exhibited.
class SideEffect : public Operator::VariableDecorator {
public:
  // Return the name of the C++ effect.
  StringRef getName() const;

  // Return the name of the base C++ effect.
  StringRef getBaseEffectName() const;

  // Return the name of the Interface that the effect belongs to.
  std::string getInterfaceTrait() const;

  // Return the name of the resource class.
  StringRef getResource() const;

  // Return the stage of the effect happen.
  int64_t getStage() const;

  // Return if this side effect act on every single value of resource.
  bool getEffectOnfullRegion() const;

  static bool classof(const Operator::VariableDecorator *var);
};

// This class represents an instance of a side effect interface applied to an
// operation. This is a wrapper around an OpInterfaceTrait that also includes
// the effects that are applied.
class SideEffectTrait : public InterfaceTrait {
public:
  // Return the effects that are attached to the side effect interface.
  Operator::var_decorator_range getEffects() const;

  // Return the name of the base C++ effect.
  StringRef getBaseEffectName() const;

  static bool classof(const Trait *t);
};

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_SIDEEFFECTS_H_
