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
#include "llvm/Support/Error.h"
#include <memory>

namespace mlir {
class DialectRegistry;

namespace lsp {
struct CodeAction;
struct CodeActionContext;
struct CompletionList;
struct Diagnostic;
struct DocumentSymbol;
struct Hover;
struct Location;
struct MLIRConvertBytecodeResult;
struct Position;
struct Range;
class URIForFile;

/// This class implements all of the MLIR related functionality necessary for a
/// language server. This class allows for keeping the MLIR specific logic
/// separate from the logic that involves LSP server/client communication.
class MLIRServer {
public:
  /// Construct a new server with the given dialect regitstry.
  MLIRServer(DialectRegistry &registry);
  ~MLIRServer();

  /// Add or update the document, with the provided `version`, at the given URI.
  /// Any diagnostics emitted for this document should be added to
  /// `diagnostics`.
  void addOrUpdateDocument(const URIForFile &uri, StringRef contents,
                           int64_t version,
                           std::vector<Diagnostic> &diagnostics);

  /// Remove the document with the given uri. Returns the version of the removed
  /// document, or None if the uri did not have a corresponding document within
  /// the server.
  Optional<int64_t> removeDocument(const URIForFile &uri);

  /// Return the locations of the object pointed at by the given position.
  void getLocationsOf(const URIForFile &uri, const Position &defPos,
                      std::vector<Location> &locations);

  /// Find all references of the object pointed at by the given position.
  void findReferencesOf(const URIForFile &uri, const Position &pos,
                        std::vector<Location> &references);

  /// Find a hover description for the given hover position, or None if one
  /// couldn't be found.
  Optional<Hover> findHover(const URIForFile &uri, const Position &hoverPos);

  /// Find all of the document symbols within the given file.
  void findDocumentSymbols(const URIForFile &uri,
                           std::vector<DocumentSymbol> &symbols);

  /// Get the code completion list for the position within the given file.
  CompletionList getCodeCompletion(const URIForFile &uri,
                                   const Position &completePos);

  /// Get the set of code actions within the file.
  void getCodeActions(const URIForFile &uri, const Range &pos,
                      const CodeActionContext &context,
                      std::vector<CodeAction> &actions);

  /// Convert the given bytecode file to the textual format.
  llvm::Expected<MLIRConvertBytecodeResult>
  convertFromBytecode(const URIForFile &uri);

  /// Convert the given textual file to the bytecode format.
  llvm::Expected<MLIRConvertBytecodeResult>
  convertToBytecode(const URIForFile &uri);

private:
  struct Impl;

  std::unique_ptr<Impl> impl;
};

} // namespace lsp
} // namespace mlir

#endif // LIB_MLIR_TOOLS_MLIRLSPSERVER_SERVER_H_
