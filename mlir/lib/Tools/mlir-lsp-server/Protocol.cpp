//===--- Protocol.cpp - Language Server Protocol Implementation -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the serialization code for the MLIR specific LSP structs.
//
//===----------------------------------------------------------------------===//

#include "Protocol.h"
#include "llvm/Support/JSON.h"

using namespace mlir;
using namespace mlir::lsp;

//===----------------------------------------------------------------------===//
// MLIRConvertBytecodeParams
//===----------------------------------------------------------------------===//

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         MLIRConvertBytecodeParams &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("uri", result.uri);
}

//===----------------------------------------------------------------------===//
// MLIRConvertBytecodeResult
//===----------------------------------------------------------------------===//

llvm::json::Value mlir::lsp::toJSON(const MLIRConvertBytecodeResult &value) {
  return llvm::json::Object{{"output", value.output}};
}
