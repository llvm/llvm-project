//===- MLIRGen.h - MLIR PDLL Code Generation --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_PDLL_CODEGEN_MLIRGEN_H_
#define MLIR_TOOLS_PDLL_CODEGEN_MLIRGEN_H_

#include <memory>

namespace llvm {
class SourceMgr;
} // namespace llvm

namespace mlir {
class MLIRContext;
class ModuleOp;
template <typename OpT>
class OwningOpRef;

namespace pdll {
namespace ast {
class Context;
class Module;
} // namespace ast

/// Given a PDLL module, generate an MLIR PDL pattern module within the given
/// MLIR context.
OwningOpRef<ModuleOp> codegenPDLLToMLIR(MLIRContext *mlirContext,
                                        const ast::Context &context,
                                        const llvm::SourceMgr &sourceMgr,
                                        const ast::Module &module);
} // namespace pdll
} // namespace mlir

#endif // MLIR_TOOLS_PDLL_CODEGEN_MLIRGEN_H_
