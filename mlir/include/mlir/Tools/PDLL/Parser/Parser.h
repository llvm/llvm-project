//===- Parser.h - MLIR PDLL Frontend Parser ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_PDLL_PARSER_PARSER_H_
#define MLIR_TOOLS_PDLL_PARSER_PARSER_H_

#include <memory>

#include "mlir/Support/LogicalResult.h"

namespace llvm {
class SourceMgr;
} // namespace llvm

namespace mlir {
namespace pdll {
namespace ast {
class Context;
class Module;
} // namespace ast

/// Parse an AST module from the main file of the given source manager.
FailureOr<ast::Module *> parsePDLAST(ast::Context &ctx,
                                     llvm::SourceMgr &sourceMgr);
} // namespace pdll
} // namespace mlir

#endif // MLIR_TOOLS_PDLL_PARSER_PARSER_H_
