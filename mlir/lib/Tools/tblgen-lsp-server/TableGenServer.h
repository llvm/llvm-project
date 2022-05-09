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

namespace mlir {
namespace lsp {
struct Diagnostic;
class URIForFile;

/// This class implements all of the TableGen related functionality necessary
/// for a language server. This class allows for keeping the TableGen specific
/// logic separate from the logic that involves LSP server/client communication.
class TableGenServer {
public:
  TableGenServer();
  ~TableGenServer();

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

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace lsp
} // namespace mlir

#endif // LIB_MLIR_TOOLS_TBLGENLSPSERVER_TABLEGENSERVER_H_
