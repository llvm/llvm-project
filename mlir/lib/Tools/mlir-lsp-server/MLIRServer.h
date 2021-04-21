//===- MLIRServer.h - MLIR General Language Server --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIB_MLIR_TOOLS_MLIRLSPSERVER_SERVER_H_
#define LIB_MLIR_TOOLS_MLIRLSPSERVER_SERVER_H_

#include "mlir/Support/LLVM.h"
#include <memory>

namespace mlir {
class DialectRegistry;

namespace lsp {
struct Location;
struct Position;
class URIForFile;

/// This class implements all of the MLIR related functionality necessary for a
/// language server. This class allows for keeping the MLIR specific logic
/// separate from the logic that involves LSP server/client communication.
class MLIRServer {
public:
  /// Construct a new server with the given dialect regitstry.
  MLIRServer(DialectRegistry &registry);
  ~MLIRServer();

  /// Add or update the document at the given URI.
  void addOrUpdateDocument(const URIForFile &uri, StringRef contents);

  /// Remove the document with the given uri.
  void removeDocument(const URIForFile &uri);

  /// Return the locations of the object pointed at by the given position.
  void getLocationsOf(const URIForFile &uri, const Position &defPos,
                      std::vector<Location> &locations);

  /// Find all references of the object pointed at by the given position.
  void findReferencesOf(const URIForFile &uri, const Position &pos,
                        std::vector<Location> &references);

private:
  struct Impl;

  std::unique_ptr<Impl> impl;
};

} // namespace lsp
} // namespace mlir

#endif // LIB_MLIR_TOOLS_MLIRLSPSERVER_SERVER_H_
