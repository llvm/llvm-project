//===- MeshToMPI.h - Convert Mesh to MPI dialect ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_MESHTOMPI_MESHTOMPI_H
#define MLIR_CONVERSION_MESHTOMPI_MESHTOMPI_H

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class Pass;

#define GEN_PASS_DECL_CONVERTMESHTOMPIPASS
#include "mlir/Conversion/Passes.h.inc"

/// Lowers Mesh communication operations (updateHalo, AllGater, ...)
/// to MPI primitives.
std::unique_ptr<::mlir::Pass> createConvertMeshToMPIPass();

} // namespace mlir

#endif // MLIR_CONVERSION_MESHTOMPI_MESHTOMPI_H
