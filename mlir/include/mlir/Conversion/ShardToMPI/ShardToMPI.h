//===- ShardToMPI.h - Convert Shard to MPI dialect --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SHARDTOMPI_SHARDTOMPI_H
#define MLIR_CONVERSION_SHARDTOMPI_SHARDTOMPI_H

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class Pass;

#define GEN_PASS_DECL_CONVERTSHARDTOMPIPASS
#include "mlir/Conversion/Passes.h.inc"

} // namespace mlir

#endif // MLIR_CONVERSION_SHARDTOMPI_SHARDTOMPI_H
