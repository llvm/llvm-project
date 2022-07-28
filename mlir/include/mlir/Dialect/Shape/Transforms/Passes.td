//===-- Passes.td - ShapeOps pass definition file ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SHAPE_TRANSFORMS_PASSES
#define MLIR_DIALECT_SHAPE_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def InsertDimensionSymbols : Pass<"insert-dimension-symbols", "ModuleOp"> {
  let summary = "Insert dimension symbols on the entry function.";
  let constructor = "mlir::createInsertDimensionSymbolsPass()";
  let options = [
    Option<"entryFunc", "entry-func", "std::string",
            /*default=*/"\"main\"",
            "Used to specify the entry-function.">,
  ];
  let dependentDialects = ["shape::ShapeDialect"];
}

def OutlineShapeComputation : Pass<"outline-shape-computation", "ModuleOp"> {
  let summary = "Outline shape computation";
  let constructor = "mlir::createOutlineShapeComputationPass()";
}

def RemoveShapeConstraints : Pass<"remove-shape-constraints", "func::FuncOp"> {
  let summary = "Replace all cstr_ ops with a true witness";
  let constructor = "mlir::createRemoveShapeConstraintsPass()";
}

def ShapeToShapeLowering : Pass<"shape-to-shape-lowering", "func::FuncOp"> {
  let summary = "Legalize Shape dialect to be convertible to Arithmetic";
  let constructor = "mlir::createShapeToShapeLowering()";
}

// TODO: Generalize this to allow any type conversions desired.
def ShapeBufferize : Pass<"shape-bufferize", "func::FuncOp"> {
  let summary = "Bufferize the shape dialect.";
  let constructor = "mlir::createShapeBufferizePass()";
  let dependentDialects = ["bufferization::BufferizationDialect",
                           "memref::MemRefDialect"];
}
#endif // MLIR_DIALECT_SHAPE_TRANSFORMS_PASSES
