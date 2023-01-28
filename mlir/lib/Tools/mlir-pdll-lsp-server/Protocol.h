//===--- Protocol.h - Language Server Protocol Implementation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains structs for LSP commands that are specific to the PDLL
// server.
//
// Each struct has a toJSON and fromJSON function, that converts between
// the struct and a JSON representation. (See JSON.h)
//
// Some structs also have operator<< serialization. This is for debugging and
// tests, and is not generally machine-readable.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_MLIR_TOOLS_MLIRPDLLLSPSERVER_PROTOCOL_H_
#define LIB_MLIR_TOOLS_MLIRPDLLLSPSERVER_PROTOCOL_H_

#include "mlir/Tools/lsp-server-support/Protocol.h"

namespace mlir {
namespace lsp {
//===----------------------------------------------------------------------===//
// PDLLViewOutputParams
//===----------------------------------------------------------------------===//

/// The type of output to view from PDLL.
enum class PDLLViewOutputKind {
  AST,
  MLIR,
  CPP,
};

/// Represents the parameters used when viewing the output of a PDLL file.
struct PDLLViewOutputParams {
  /// The URI of the document to view the output of.
  URIForFile uri;

  /// The kind of output to generate.
  PDLLViewOutputKind kind;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, PDLLViewOutputKind &result,
              llvm::json::Path path);
bool fromJSON(const llvm::json::Value &value, PDLLViewOutputParams &result,
              llvm::json::Path path);

//===----------------------------------------------------------------------===//
// PDLLViewOutputResult
//===----------------------------------------------------------------------===//

/// Represents the result of viewing the output of a PDLL file.
struct PDLLViewOutputResult {
  /// The string representation of the output.
  std::string output;
};

/// Add support for JSON serialization.
llvm::json::Value toJSON(const PDLLViewOutputResult &value);

} // namespace lsp
} // namespace mlir

#endif
