//===--- Protocol.h - Language Server Protocol Implementation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMLSP_PROTOCOL_H
#define LLVM_TOOLS_LLVMLSP_PROTOCOL_H

#include "llvm/Support/LSP/Protocol.h"

// This file is using the LSP syntax for identifier names which is different
// from the LLVM coding standard. To avoid the clang-tidy warnings, we're
// disabling one check here.
// NOLINTBEGIN(readability-identifier-naming)

namespace llvm {
namespace lsp {
struct GetCfgParams {
  URIForFile uri;
  Position position;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &Value, GetCfgParams &Result,
              llvm::json::Path Path);

struct CFG {
  URIForFile uri;
  std::string node_id;
  std::string function;
};

llvm::json::Value toJSON(const CFG &Value);

struct BbLocationParams {
  /// The URI of the SVG file containing the CFG.
  URIForFile uri;
  /// The ID of the node representing the basic block.
  std::string node_id;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &Value, BbLocationParams &Result,
              llvm::json::Path Path);

struct BbLocation {
  /// The URI of the `.ll` file containing the basic block.
  URIForFile uri;
  /// The range of the basic block corresponding to the node ID.
  Range range;
};

llvm::json::Value toJSON(const BbLocation &Value);

struct GetPassListParams {
  /// The URI of the `.ll` file for which the pass list is requested.
  URIForFile uri;
  /// The optimization pipeline string, in the format passed to the `opt` tool.
  std::string pipeline;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &Value, GetPassListParams &Result,
              llvm::json::Path Path);

struct PassList {
  /// A list of passes in the pipeline, formatted as `<number>-<name>`.
  std::vector<std::string> list;
  /// A list of descriptions corresponding to each pass.
  std::vector<std::string> descriptions;
};

llvm::json::Value toJSON(const PassList &Value);

struct GetIRAfterPassParams {
  /// The URI of the `.ll` file for which the intermediate IR is requested.
  URIForFile uri;
  /// The optimization pipeline string, in the format passed to the `opt` tool.
  std::string pipeline;
  /// The number of the pass in the pipeline after which to return the IR.
  unsigned passnumber;
};

bool fromJSON(const llvm::json::Value &Value, GetIRAfterPassParams &Result,
              llvm::json::Path Path);

struct IR {
  /// The URI of the `.ll` file containing the generated intermediate IR.
  URIForFile uri;
};

llvm::json::Value toJSON(const IR &Value);

} // namespace lsp
} // namespace llvm

// NOLINTEND(readability-identifier-naming)

#endif // LLVM_TOOLS_LLVMLSP_PROTOCOL_H
