//===- MlirTblgenMain.h - MLIR Tablegen Driver main -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-tblgen for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIR_TBLGEN_MLIRTBLGENMAIN_H
#define MLIR_TOOLS_MLIR_TBLGEN_MLIRTBLGENMAIN_H

namespace mlir {
/// Main Program for tools like 'mlir-tblgen' with custom backends. To add
/// a new backend, simply create a new 'mlir::GenRegistration' global variable.
/// See its documentation for more info.
///
/// The 'argc' and 'argv' arguments are simply forwarded from a main function.
/// The return value is the exit code from llvm::TableGenMain.
int MlirTblgenMain(int argc, char **argv);
} // namespace mlir

#endif // MLIR_TOOLS_MLIR_TBLGEN_MLIRTBLGENMAIN_H
