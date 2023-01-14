//===- TableGenServer.h - TableGen Language Server --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIB_MLIR_TOOLS_TBLGENLSPSERVER_TABLEGENSERVER_H_
#define LIB_MLIR_TOOLS_TBLGENLSPSERVER_TABLEGENSERVER_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>
#include <optional>

namespace mlir {
namespace lsp {
struct Diagnostic;
struct DocumentLink;
struct Hover;
struct Location;
struct Position;
struct TextDocumentContentChangeEvent;
class URIForFile;

/// This class implements all of the TableGen related functionality necessary
/// for a language server. This class allows for keeping the TableGen specific
/// logic separate from the logic that involves LSP server/client communication.
class TableGenServer {
public:
  struct Options {
    Options(const std::vector<std::string> &compilationDatabases,
            const std::vector<std::string> &extraDirs)
        : compilationDatabases(compilationDatabases), extraDirs(extraDirs) {}

    /// The filenames for databases containing compilation commands for TableGen
    /// files passed to the server.
    const std::vector<std::string> &compilationDatabases;

    /// Additional list of include directories to search.
    const std::vector<std::string> &extraDirs;
  };

  TableGenServer(const Options &options);
  ~TableGenServer();

  /// Add the document, with the provided `version`, at the given URI. Any
  /// diagnostics emitted for this document should be added to `diagnostics`.
  void addDocument(const URIForFile &uri, StringRef contents, int64_t version,
                   std::vector<Diagnostic> &diagnostics);

  /// Update the document, with the provided `version`, at the given URI. Any
  /// diagnostics emitted for this document should be added to `diagnostics`.
  void updateDocument(const URIForFile &uri,
                      ArrayRef<TextDocumentContentChangeEvent> changes,
                      int64_t version, std::vector<Diagnostic> &diagnostics);

  /// Remove the document with the given uri. Returns the version of the removed
  /// document, or std::nullopt if the uri did not have a corresponding document
  /// within the server.
  Optional<int64_t> removeDocument(const URIForFile &uri);

  /// Return the locations of the object pointed at by the given position.
  void getLocationsOf(const URIForFile &uri, const Position &defPos,
                      std::vector<Location> &locations);

  /// Find all references of the object pointed at by the given position.
  void findReferencesOf(const URIForFile &uri, const Position &pos,
                        std::vector<Location> &references);

  /// Return the document links referenced by the given file.
  void getDocumentLinks(const URIForFile &uri,
                        std::vector<DocumentLink> &documentLinks);

  /// Find a hover description for the given hover position, or std::nullopt if
  /// one couldn't be found.
  std::optional<Hover> findHover(const URIForFile &uri,
                                 const Position &hoverPos);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace lsp
} // namespace mlir

#endif // LIB_MLIR_TOOLS_TBLGENLSPSERVER_TABLEGENSERVER_H_
