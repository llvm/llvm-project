//===-- LinalgAllOps.td - Combined Linalg dialect operations -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LINALG_ALL_OPS
#define LINALG_ALL_OPS

// Generate the core, structured and relayout operations by op partition from
// one file, so that they share a single dialect registration hook.
include "mlir/Dialect/Linalg/IR/LinalgOps.td"
include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.td"
include "mlir/Dialect/Linalg/IR/LinalgRelayoutOps.td"

#endif // LINALG_ALL_OPS
