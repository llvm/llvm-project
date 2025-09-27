//===- OpenMPOffloadPrivatizationPrepare.h -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENMP_TRANSFORMS_PREPAREFOROMPOFFLOADPRIVATIZATIONPASS_H
#define MLIR_DIALECT_OPENMP_TRANSFORMS_PREPAREFOROMPOFFLOADPRIVATIZATIONPASS_H

#include <memory>

namespace mlir {
class Pass;
namespace omp {
#define GEN_PASS_DECL_PREPAREFOROMPOFFLOADPRIVATIZATIONPASS
#include "mlir/Dialect/OpenMP/Transforms/Passes.h.inc"
} // namespace omp
} // namespace mlir

#endif // MLIR_DIALECT_OPENMP_TRANSFORMS_PREPAREFOROMPOFFLOADPRIVATIZATIONPASS_H
