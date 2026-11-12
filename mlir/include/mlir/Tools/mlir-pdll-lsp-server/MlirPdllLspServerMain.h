//===- MlirPdllLspServerMain.h - MLIR PDLL Language Server main -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-pdll-lsp-server for when built as standalone
// binary.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIR_PDLL_LSP_SERVER_MLIRPDLLLSPSERVERMAIN_H
#define MLIR_TOOLS_MLIR_PDLL_LSP_SERVER_MLIRPDLLLSPSERVERMAIN_H

namespace llvm {
struct LogicalResult;
} // namespace llvm

namespace mlir {
/// Implementation for tools like `mlir-pdll-lsp-server`.
llvm::LogicalResult MlirPdllLspServerMain(int argc, char **argv);

} // namespace mlir

#endif // MLIR_TOOLS_MLIR_PDLL_LSP_SERVER_MLIRPDLLLSPSERVERMAIN_H
