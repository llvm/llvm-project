//===- MlirOptMain.h - MLIR Optimizer Driver main ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <memory>

namespace llvm {
class raw_ostream;
class MemoryBuffer;
} // end namespace llvm

namespace mlir {
class PassPipelineCLParser;

/// Perform the core processing behind `mlir-opt`:
/// - outputStream is the stream where the resulting IR is printed.
/// - buffer is the in-memory file to parser and process.
/// - passPipeline is the specification of the pipeline that will be applied.
/// - splitInputFile will look for a "-----" marker in the input file, and load
/// each chunk in an individual ModuleOp processed separately.
/// - verifyDiagnostics enables a verification mode where comments starting with
/// "expected-(error|note|remark|warning)" are parsed in the input and matched
/// against emitted diagnostics.
/// - verifyPasses enables the IR verifier in-between each pass in the pipeline.
/// - allowUnregisteredDialects allows to parse and create operation without
/// registering the Dialect in the MLIRContext.
LogicalResult MlirOptMain(llvm::raw_ostream &outputStream,
                          std::unique_ptr<llvm::MemoryBuffer> buffer,
                          const PassPipelineCLParser &passPipeline,
                          bool splitInputFile, bool verifyDiagnostics,
                          bool verifyPasses, bool allowUnregisteredDialects);

/// Implementation for tools like `mlir-opt`.
LogicalResult MlirOptMain(int argc, char **argv, llvm::StringRef toolName);

} // end namespace mlir
