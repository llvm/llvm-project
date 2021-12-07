//===- MlirLspServerMain.h - MLIR Language Server main ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-lsp-server for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRLSPSERVER_MLIRLSPSERVERMAIN_H
#define MLIR_TOOLS_MLIRLSPSERVER_MLIRLSPSERVERMAIN_H

namespace mlir {
class DialectRegistry;
struct LogicalResult;

/// Implementation for tools like `mlir-lsp-server`.
/// - registry should contain all the dialects that can be parsed in source IR
/// passed to the server.
LogicalResult MlirLspServerMain(int argc, char **argv,
                                DialectRegistry &registry);

} // namespace mlir

#endif // MLIR_TOOLS_MLIRLSPSERVER_MLIRLSPSERVERMAIN_H
