//===- DiagnosedSilenceableFailure.cpp - Tri-state result -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the DiagnosedSilenceableFailure class allowing to store
// a tri-state result (definite failure, recoverable failure, success) with an
// optional associated list of diagnostics.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/Utils/DiagnosedSilenceableFailure.h"

using namespace mlir;

LogicalResult mlir::DiagnosedSilenceableFailure::checkAndReport() {
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  assert(!reported && "attempting to report a diagnostic more than once");
  reported = true;
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
  if (!diagnostics.empty()) {
    for (auto &&diagnostic : diagnostics) {
      diagnostic.getLocation().getContext()->getDiagEngine().emit(
          std::move(diagnostic));
    }
    diagnostics.clear();
    result = ::mlir::failure();
  }
  return result;
}
