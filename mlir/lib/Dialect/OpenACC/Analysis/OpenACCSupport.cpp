//===- OpenACCSupport.cpp - OpenACCSupport Implementation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the OpenACCSupport analysis interface.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACCUtils.h"

namespace mlir {
namespace acc {

std::string OpenACCSupport::getVariableName(Value v) {
  if (impl)
    return impl->getVariableName(v);
  return acc::getVariableName(v);
}

std::string OpenACCSupport::getRecipeName(RecipeKind kind, Type type,
                                          Value var) {
  if (impl)
    return impl->getRecipeName(kind, type, var);
  // The default implementation assumes that only type matters
  // and the actual instance of variable is not relevant.
  auto recipeName = acc::getRecipeName(kind, type);
  if (recipeName.empty())
    emitNYI(var ? var.getLoc() : UnknownLoc::get(type.getContext()),
            "variable privatization (incomplete recipe name handling)");
  return recipeName;
}

InFlightDiagnostic OpenACCSupport::emitNYI(Location loc, const Twine &message) {
  if (impl)
    return impl->emitNYI(loc, message);
  return mlir::emitError(loc, "not yet implemented: " + message);
}

} // namespace acc
} // namespace mlir
