//===-- Passes.td - Part tensor pass definition file -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PARTTENSOR_TRANSFORMS_PASSES
#define MLIR_DIALECT_PARTTENSOR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def PartTensorConversionPass : Pass<"part-tensor-conversion", "ModuleOp"> {
  let summary = "Convert part tensors and primitives to library calls";
  let description = [{
    A pass that converts part tensor primitives into calls into a runtime
    support library. Part tensor types are converted into opaque pointers
    to the underlying part storage schemes.
  }];
  let constructor = "mlir::createPartTensorConversionPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "bufferization::BufferizationDialect",
    "LLVM::LLVMDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "part_tensor::PartTensorDialect",
  ];
  let options = [];
}

#endif // MLIR_DIALECT_PARTTENSOR_TRANSFORMS_PASSES
