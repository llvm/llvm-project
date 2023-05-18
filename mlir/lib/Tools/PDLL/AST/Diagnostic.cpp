//===- Diagnostic.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/PDLL/AST/Diagnostic.h"

using namespace mlir;
using namespace mlir::pdll::ast;

//===----------------------------------------------------------------------===//
// InFlightDiagnostic
//===----------------------------------------------------------------------===//

void InFlightDiagnostic::report() {
  // If this diagnostic is still inflight and it hasn't been abandoned, then
  // report it.
  if (isInFlight()) {
    owner->report(std::move(*impl));
    owner = nullptr;
  }
  impl.reset();
}
