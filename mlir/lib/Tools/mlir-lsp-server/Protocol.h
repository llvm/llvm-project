//===--- Protocol.h - Language Server Protocol Implementation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains structs for LSP commands that are specific to the MLIR
// server.
//
// Each struct has a toJSON and fromJSON function, that converts between
// the struct and a JSON representation. (See JSON.h)
//
// Some structs also have operator<< serialization. This is for debugging and
// tests, and is not generally machine-readable.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_MLIR_TOOLS_MLIRLSPSERVER_PROTOCOL_H_
#define LIB_MLIR_TOOLS_MLIRLSPSERVER_PROTOCOL_H_

#include "llvm/Support/LSP/Protocol.h"

namespace llvm {
namespace lsp {
//===----------------------------------------------------------------------===//
// MLIRConvertBytecodeParams
//===----------------------------------------------------------------------===//

/// This class represents the parameters used when converting between MLIR's
/// bytecode and textual format.
struct MLIRConvertBytecodeParams {
  /// The input file containing the bytecode or textual format.
  URIForFile uri;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, MLIRConvertBytecodeParams &result,
              llvm::json::Path path);

//===----------------------------------------------------------------------===//
// MLIRConvertBytecodeResult
//===----------------------------------------------------------------------===//

/// This class represents the result of converting between MLIR's bytecode and
/// textual format.
struct MLIRConvertBytecodeResult {
  /// The resultant output of the conversion.
  std::string output;
};

/// Add support for JSON serialization.
llvm::json::Value toJSON(const MLIRConvertBytecodeResult &value);

} // namespace lsp
} // namespace llvm

#endif
