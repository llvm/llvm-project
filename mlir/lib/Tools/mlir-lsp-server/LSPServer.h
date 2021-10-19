//===- LSPServer.h - MLIR LSP Server ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIB_MLIR_TOOLS_MLIRLSPSERVER_LSPSERVER_H
#define LIB_MLIR_TOOLS_MLIRLSPSERVER_LSPSERVER_H

#include <memory>

namespace mlir {
struct LogicalResult;

namespace lsp {
class JSONTransport;
class MLIRServer;

/// This class represents the main LSP server, and handles communication with
/// the LSP client.
class LSPServer {
public:
  /// Construct a new language server with the given MLIR server.
  LSPServer(MLIRServer &server, JSONTransport &transport);
  ~LSPServer();

  /// Run the main loop of the server.
  LogicalResult run();

private:
  struct Impl;

  std::unique_ptr<Impl> impl;
};
} // namespace lsp
} // namespace mlir

#endif // LIB_MLIR_TOOLS_MLIRLSPSERVER_LSPSERVER_H
