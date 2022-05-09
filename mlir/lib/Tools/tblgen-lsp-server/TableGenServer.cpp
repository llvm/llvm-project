//===- TableGenServer.cpp - TableGen Language Server ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TableGenServer.h"

#include "../lsp-server-support/CompilationDatabase.h"
#include "../lsp-server-support/Logging.h"
#include "../lsp-server-support/Protocol.h"
#include "../lsp-server-support/SourceMgrUtils.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/TableGen/Parser.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;

/// Returns a language server uri for the given source location. `mainFileURI`
/// corresponds to the uri for the main file of the source manager.
static lsp::URIForFile getURIFromLoc(const llvm::SourceMgr &mgr, SMLoc loc,
                                     const lsp::URIForFile &mainFileURI) {
  int bufferId = mgr.FindBufferContainingLoc(loc);
  if (bufferId == 0 || bufferId == static_cast<int>(mgr.getMainFileID()))
    return mainFileURI;
  llvm::Expected<lsp::URIForFile> fileForLoc = lsp::URIForFile::fromFile(
      mgr.getBufferInfo(bufferId).Buffer->getBufferIdentifier());
  if (fileForLoc)
    return *fileForLoc;
  lsp::Logger::error("Failed to create URI for include file: {0}",
                     llvm::toString(fileForLoc.takeError()));
  return mainFileURI;
}

/// Returns a language server location from the given source range.
static lsp::Location getLocationFromLoc(llvm::SourceMgr &mgr, SMLoc loc,
                                        const lsp::URIForFile &uri) {
  return lsp::Location(getURIFromLoc(mgr, loc, uri),
                       lsp::Range(mgr, lsp::convertTokenLocToRange(loc)));
}

/// Convert the given TableGen diagnostic to the LSP form.
static Optional<lsp::Diagnostic>
getLspDiagnoticFromDiag(const llvm::SMDiagnostic &diag,
                        const lsp::URIForFile &uri) {
  auto *sourceMgr = const_cast<llvm::SourceMgr *>(diag.getSourceMgr());
  if (!sourceMgr || !diag.getLoc().isValid())
    return llvm::None;

  lsp::Diagnostic lspDiag;
  lspDiag.source = "tablegen";
  lspDiag.category = "Parse Error";

  // Try to grab a file location for this diagnostic.
  lsp::Location loc = getLocationFromLoc(*sourceMgr, diag.getLoc(), uri);
  lspDiag.range = loc.range;

  // Skip diagnostics that weren't emitted within the main file.
  if (loc.uri != uri)
    return llvm::None;

  // Convert the severity for the diagnostic.
  switch (diag.getKind()) {
  case llvm::SourceMgr::DK_Warning:
    lspDiag.severity = lsp::DiagnosticSeverity::Warning;
    break;
  case llvm::SourceMgr::DK_Error:
    lspDiag.severity = lsp::DiagnosticSeverity::Error;
    break;
  case llvm::SourceMgr::DK_Note:
    // Notes are emitted separately from the main diagnostic, so we just treat
    // them as remarks given that we can't determine the diagnostic to relate
    // them to.
  case llvm::SourceMgr::DK_Remark:
    lspDiag.severity = lsp::DiagnosticSeverity::Information;
    break;
  }
  lspDiag.message = diag.getMessage().str();

  return lspDiag;
}

//===----------------------------------------------------------------------===//
// TableGenTextFile
//===----------------------------------------------------------------------===//

namespace {
/// This class represents a text file containing one or more TableGen documents.
class TableGenTextFile {
public:
  TableGenTextFile(const lsp::URIForFile &uri, StringRef fileContents,
                   int64_t version,
                   const std::vector<std::string> &extraIncludeDirs,
                   std::vector<lsp::Diagnostic> &diagnostics);

  /// Return the current version of this text file.
  int64_t getVersion() const { return version; }

  //===--------------------------------------------------------------------===//
  // Document Links
  //===--------------------------------------------------------------------===//

  void getDocumentLinks(const lsp::URIForFile &uri,
                        std::vector<lsp::DocumentLink> &links);

  //===--------------------------------------------------------------------===//
  // Hover
  //===--------------------------------------------------------------------===//

  Optional<lsp::Hover> findHover(const lsp::URIForFile &uri,
                                 const lsp::Position &hoverPos);

private:
  /// The full string contents of the file.
  std::string contents;

  /// The version of this file.
  int64_t version;

  /// The include directories for this file.
  std::vector<std::string> includeDirs;

  /// The source manager containing the contents of the input file.
  llvm::SourceMgr sourceMgr;

  /// The record keeper containing the parsed tablegen constructs.
  llvm::RecordKeeper recordKeeper;

  /// The set of includes of the parsed file.
  SmallVector<lsp::SourceMgrInclude> parsedIncludes;
};
} // namespace

TableGenTextFile::TableGenTextFile(
    const lsp::URIForFile &uri, StringRef fileContents, int64_t version,
    const std::vector<std::string> &extraIncludeDirs,
    std::vector<lsp::Diagnostic> &diagnostics)
    : contents(fileContents.str()), version(version) {
  auto memBuffer = llvm::MemoryBuffer::getMemBufferCopy(contents, uri.file());
  if (!memBuffer) {
    lsp::Logger::error("Failed to create memory buffer for file", uri.file());
    return;
  }

  // Build the set of include directories for this file.
  llvm::SmallString<32> uriDirectory(uri.file());
  llvm::sys::path::remove_filename(uriDirectory);
  includeDirs.push_back(uriDirectory.str().str());
  includeDirs.insert(includeDirs.end(), extraIncludeDirs.begin(),
                     extraIncludeDirs.end());

  sourceMgr.setIncludeDirs(includeDirs);
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), SMLoc());

  // This class provides a context argument for the llvm::SourceMgr diagnostic
  // handler.
  struct DiagHandlerContext {
    std::vector<lsp::Diagnostic> &diagnostics;
    const lsp::URIForFile &uri;
  } handlerContext{diagnostics, uri};

  // Set the diagnostic handler for the tablegen source manager.
  sourceMgr.setDiagHandler(
      [](const llvm::SMDiagnostic &diag, void *rawHandlerContext) {
        auto *ctx = reinterpret_cast<DiagHandlerContext *>(rawHandlerContext);
        if (auto lspDiag = getLspDiagnoticFromDiag(diag, ctx->uri))
          ctx->diagnostics.push_back(*lspDiag);
      },
      &handlerContext);
  bool failedToParse = llvm::TableGenParseFile(sourceMgr, recordKeeper);

  // Process all of the include files.
  lsp::gatherIncludeFiles(sourceMgr, parsedIncludes);
  if (failedToParse)
    return;
}

//===--------------------------------------------------------------------===//
// TableGenTextFile: Document Links
//===--------------------------------------------------------------------===//

void TableGenTextFile::getDocumentLinks(const lsp::URIForFile &uri,
                                        std::vector<lsp::DocumentLink> &links) {
  for (const lsp::SourceMgrInclude &include : parsedIncludes)
    links.emplace_back(include.range, include.uri);
}

//===----------------------------------------------------------------------===//
// TableGenTextFile: Hover
//===----------------------------------------------------------------------===//

Optional<lsp::Hover>
TableGenTextFile::findHover(const lsp::URIForFile &uri,
                            const lsp::Position &hoverPos) {
  // Check for a reference to an include.
  for (const lsp::SourceMgrInclude &include : parsedIncludes)
    if (include.range.contains(hoverPos))
      return include.buildHover();
  return llvm::None;
}

//===----------------------------------------------------------------------===//
// TableGenServer::Impl
//===----------------------------------------------------------------------===//

struct lsp::TableGenServer::Impl {
  explicit Impl(const Options &options)
      : options(options), compilationDatabase(options.compilationDatabases) {}

  /// TableGen LSP options.
  const Options &options;

  /// The compilation database containing additional information for files
  /// passed to the server.
  lsp::CompilationDatabase compilationDatabase;

  /// The files held by the server, mapped by their URI file name.
  llvm::StringMap<std::unique_ptr<TableGenTextFile>> files;
};

//===----------------------------------------------------------------------===//
// TableGenServer
//===----------------------------------------------------------------------===//

lsp::TableGenServer::TableGenServer(const Options &options)
    : impl(std::make_unique<Impl>(options)) {}
lsp::TableGenServer::~TableGenServer() = default;

void lsp::TableGenServer::addOrUpdateDocument(
    const URIForFile &uri, StringRef contents, int64_t version,
    std::vector<Diagnostic> &diagnostics) {
  // Build the set of additional include directories.
  std::vector<std::string> additionalIncludeDirs = impl->options.extraDirs;
  const auto &fileInfo = impl->compilationDatabase.getFileInfo(uri.file());
  llvm::append_range(additionalIncludeDirs, fileInfo.includeDirs);

  impl->files[uri.file()] = std::make_unique<TableGenTextFile>(
      uri, contents, version, additionalIncludeDirs, diagnostics);
}

Optional<int64_t> lsp::TableGenServer::removeDocument(const URIForFile &uri) {
  auto it = impl->files.find(uri.file());
  if (it == impl->files.end())
    return llvm::None;

  int64_t version = it->second->getVersion();
  impl->files.erase(it);
  return version;
}

void lsp::TableGenServer::getDocumentLinks(
    const URIForFile &uri, std::vector<DocumentLink> &documentLinks) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    return fileIt->second->getDocumentLinks(uri, documentLinks);
}

Optional<lsp::Hover> lsp::TableGenServer::findHover(const URIForFile &uri,
                                                    const Position &hoverPos) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    return fileIt->second->findHover(uri, hoverPos);
  return llvm::None;
}
