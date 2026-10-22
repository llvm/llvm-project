//===-- Passes.td - pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MLPROGRAM_TRANSFORMS_PASSES
#define MLIR_DIALECT_MLPROGRAM_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def MLProgramPipelineGlobalsPass
    : Pass<"mlprogram-pipeline-globals", "ModuleOp"> {
  let summary = "Optimize `ml_program` global operations for read and store";
  let description = [{
    `ml_program`'s load and store operations can be optimized for
    write-write or write-read sets of operations. This allows known
    tensors to not be re-read when the value is already known in IR.

    The pass is designed to handle both nested regions and function calls
    safely.
  }];
}

def MLProgramExternalizeConstantsPass
    : Pass<"mlprogram-externalize-constants", "ModuleOp"> {
  let summary = "Externalize large constants to a separate weights module";
  let options = [
    Option<"minSize", "min-size", "int64_t", /*default=*/"1024",
           "Minimum size of ElementsAttr to externalize">,
    Option<"outputFilename", "output-filename", "std::string", /*default=*/"",
           "Output filename for the weights module">,
    Option<"dropLocations", "drop-locations", "bool", /*default=*/"false",
           "Drop locations (use UnknownLoc) for the externalized constants">,
  ];
  let dependentDialects = ["mlir::ml_program::MLProgramDialect"];
}

def MLProgramRepopulateConstantsPass
    : Pass<"mlprogram-repopulate-constants", "ModuleOp"> {
  let summary = "Repopulate externalized constants from a weights module";
  let options = [
    Option<"inputFilename", "input-filename", "std::string", /*default=*/"",
           "Input filename for the weights module">,
  ];
  let dependentDialects = ["mlir::ml_program::MLProgramDialect"];
}

#endif // MLIR_DIALECT_MLPROGRAM_TRANSFORMS_PASSES
