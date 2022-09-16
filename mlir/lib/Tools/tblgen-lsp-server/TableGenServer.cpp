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
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/PointerUnion.h"
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
static lsp::Location getLocationFromLoc(llvm::SourceMgr &mgr, SMRange loc,
                                        const lsp::URIForFile &uri) {
  return lsp::Location(getURIFromLoc(mgr, loc.Start, uri),
                       lsp::Range(mgr, loc));
}
static lsp::Location getLocationFromLoc(llvm::SourceMgr &mgr, SMLoc loc,
                                        const lsp::URIForFile &uri) {
  return getLocationFromLoc(mgr, lsp::convertTokenLocToRange(loc), uri);
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
// TableGenIndex
//===----------------------------------------------------------------------===//

namespace {
/// This class represents a single symbol definition within a TableGen index. It
/// contains the definition of the symbol, the location of the symbol, and any
/// recorded references.
struct TableGenIndexSymbol {
  TableGenIndexSymbol(const llvm::Record *record)
      : definition(record),
        defLoc(lsp::convertTokenLocToRange(record->getLoc().front())) {}
  TableGenIndexSymbol(const llvm::RecordVal *value)
      : definition(value),
        defLoc(lsp::convertTokenLocToRange(value->getLoc())) {}

  /// The main definition of the symbol.
  PointerUnion<const llvm::Record *, const llvm::RecordVal *> definition;

  /// The source location of the definition.
  SMRange defLoc;

  /// The source location of the references of the definition.
  SmallVector<SMRange> references;
};

/// This class provides an index for definitions/uses within a TableGen
/// document. It provides efficient lookup of a definition given an input source
/// range.
class TableGenIndex {
public:
  TableGenIndex() : intervalMap(allocator) {}

  /// Initialize the index with the given RecordKeeper.
  void initialize(const llvm::RecordKeeper &records);

  /// Lookup a symbol for the given location. Returns nullptr if no symbol could
  /// be found. If provided, `overlappedRange` is set to the range that the
  /// provided `loc` overlapped with.
  const TableGenIndexSymbol *lookup(SMLoc loc,
                                    SMRange *overlappedRange = nullptr) const;

private:
  /// The type of interval map used to store source references. SMRange is
  /// half-open, so we also need to use a half-open interval map.
  using MapT = llvm::IntervalMap<
      const char *, const TableGenIndexSymbol *,
      llvm::IntervalMapImpl::NodeSizer<const char *,
                                       const TableGenIndexSymbol *>::LeafSize,
      llvm::IntervalMapHalfOpenInfo<const char *>>;

  /// An allocator for the interval map.
  MapT::Allocator allocator;

  /// An interval map containing a corresponding definition mapped to a source
  /// interval.
  MapT intervalMap;

  /// A mapping between definitions and their corresponding symbol.
  DenseMap<const void *, std::unique_ptr<TableGenIndexSymbol>> defToSymbol;
};
} // namespace

void TableGenIndex::initialize(const llvm::RecordKeeper &records) {
  auto getOrInsertDef = [&](const auto *def) -> TableGenIndexSymbol * {
    auto it = defToSymbol.try_emplace(def, nullptr);
    if (it.second)
      it.first->second = std::make_unique<TableGenIndexSymbol>(def);
    return &*it.first->second;
  };
  auto insertRef = [&](TableGenIndexSymbol *sym, SMRange refLoc,
                       bool isDef = false) {
    const char *startLoc = refLoc.Start.getPointer();
    const char *endLoc = refLoc.End.getPointer();

    // If the location we got was empty, try to lex a token from the start
    // location.
    if (startLoc == endLoc) {
      refLoc = lsp::convertTokenLocToRange(SMLoc::getFromPointer(startLoc));
      startLoc = refLoc.Start.getPointer();
      endLoc = refLoc.End.getPointer();

      // If the location is still empty, bail on trying to use this reference
      // location.
      if (startLoc == endLoc)
        return;
    }

    // Check to see if a symbol is already attached to this location.
    // IntervalMap doesn't allow overlapping inserts, and we don't really
    // want multiple symbols attached to a source location anyways. This
    // shouldn't really happen in practice, but we should handle it gracefully.
    if (!intervalMap.overlaps(startLoc, endLoc))
      intervalMap.insert(startLoc, endLoc, sym);

    if (!isDef)
      sym->references.push_back(refLoc);
  };
  auto classes =
      llvm::make_pointee_range(llvm::make_second_range(records.getClasses()));
  auto defs =
      llvm::make_pointee_range(llvm::make_second_range(records.getDefs()));
  for (const llvm::Record &def : llvm::concat<llvm::Record>(classes, defs)) {
    auto *sym = getOrInsertDef(&def);
    insertRef(sym, sym->defLoc, /*isDef=*/true);

    // Add references to the definition.
    for (SMLoc loc : def.getLoc().drop_front())
      insertRef(sym, lsp::convertTokenLocToRange(loc));
    for (SMRange loc : def.getReferenceLocs())
      insertRef(sym, loc);

    // Add definitions for any values.
    for (const llvm::RecordVal &value : def.getValues()) {
      auto *sym = getOrInsertDef(&value);
      insertRef(sym, sym->defLoc, /*isDef=*/true);
      for (SMRange refLoc : value.getReferenceLocs())
        insertRef(sym, refLoc);
    }
  }
}

const TableGenIndexSymbol *
TableGenIndex::lookup(SMLoc loc, SMRange *overlappedRange) const {
  auto it = intervalMap.find(loc.getPointer());
  if (!it.valid() || loc.getPointer() < it.start())
    return nullptr;

  if (overlappedRange) {
    *overlappedRange = SMRange(SMLoc::getFromPointer(it.start()),
                               SMLoc::getFromPointer(it.stop()));
  }
  return it.value();
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

  /// Update the file to the new version using the provided set of content
  /// changes. Returns failure if the update was unsuccessful.
  LogicalResult update(const lsp::URIForFile &uri, int64_t newVersion,
                       ArrayRef<lsp::TextDocumentContentChangeEvent> changes,
                       std::vector<lsp::Diagnostic> &diagnostics);

  //===--------------------------------------------------------------------===//
  // Definitions and References
  //===--------------------------------------------------------------------===//

  void getLocationsOf(const lsp::URIForFile &uri, const lsp::Position &defPos,
                      std::vector<lsp::Location> &locations);
  void findReferencesOf(const lsp::URIForFile &uri, const lsp::Position &pos,
                        std::vector<lsp::Location> &references);

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
  /// Initialize the text file from the given file contents.
  void initialize(const lsp::URIForFile &uri, int64_t newVersion,
                  std::vector<lsp::Diagnostic> &diagnostics);

  /// The full string contents of the file.
  std::string contents;

  /// The version of this file.
  int64_t version;

  /// The include directories for this file.
  std::vector<std::string> includeDirs;

  /// The source manager containing the contents of the input file.
  llvm::SourceMgr sourceMgr;

  /// The record keeper containing the parsed tablegen constructs.
  std::unique_ptr<llvm::RecordKeeper> recordKeeper;

  /// The index of the parsed file.
  TableGenIndex index;

  /// The set of includes of the parsed file.
  SmallVector<lsp::SourceMgrInclude> parsedIncludes;
};
} // namespace

TableGenTextFile::TableGenTextFile(
    const lsp::URIForFile &uri, StringRef fileContents, int64_t version,
    const std::vector<std::string> &extraIncludeDirs,
    std::vector<lsp::Diagnostic> &diagnostics)
    : contents(fileContents.str()), version(version) {
  // Build the set of include directories for this file.
  llvm::SmallString<32> uriDirectory(uri.file());
  llvm::sys::path::remove_filename(uriDirectory);
  includeDirs.push_back(uriDirectory.str().str());
  includeDirs.insert(includeDirs.end(), extraIncludeDirs.begin(),
                     extraIncludeDirs.end());

  // Initialize the file.
  initialize(uri, version, diagnostics);
}

LogicalResult
TableGenTextFile::update(const lsp::URIForFile &uri, int64_t newVersion,
                         ArrayRef<lsp::TextDocumentContentChangeEvent> changes,
                         std::vector<lsp::Diagnostic> &diagnostics) {
  if (failed(lsp::TextDocumentContentChangeEvent::applyTo(changes, contents))) {
    lsp::Logger::error("Failed to update contents of {0}", uri.file());
    return failure();
  }

  // If the file contents were properly changed, reinitialize the text file.
  initialize(uri, newVersion, diagnostics);
  return success();
}

void TableGenTextFile::initialize(const lsp::URIForFile &uri,
                                  int64_t newVersion,
                                  std::vector<lsp::Diagnostic> &diagnostics) {
  version = newVersion;
  sourceMgr = llvm::SourceMgr();
  recordKeeper = std::make_unique<llvm::RecordKeeper>();

  // Build a buffer for this file.
  auto memBuffer = llvm::MemoryBuffer::getMemBuffer(contents, uri.file());
  if (!memBuffer) {
    lsp::Logger::error("Failed to create memory buffer for file", uri.file());
    return;
  }
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
  bool failedToParse = llvm::TableGenParseFile(sourceMgr, *recordKeeper);

  // Process all of the include files.
  lsp::gatherIncludeFiles(sourceMgr, parsedIncludes);
  if (failedToParse)
    return;

  // If we successfully parsed the file, we can now build the index.
  index.initialize(*recordKeeper);
}

//===----------------------------------------------------------------------===//
// TableGenTextFile: Definitions and References
//===----------------------------------------------------------------------===//

void TableGenTextFile::getLocationsOf(const lsp::URIForFile &uri,
                                      const lsp::Position &defPos,
                                      std::vector<lsp::Location> &locations) {
  SMLoc posLoc = defPos.getAsSMLoc(sourceMgr);
  const TableGenIndexSymbol *symbol = index.lookup(posLoc);
  if (!symbol)
    return;

  locations.push_back(getLocationFromLoc(sourceMgr, symbol->defLoc, uri));
}

void TableGenTextFile::findReferencesOf(
    const lsp::URIForFile &uri, const lsp::Position &pos,
    std::vector<lsp::Location> &references) {
  SMLoc posLoc = pos.getAsSMLoc(sourceMgr);
  const TableGenIndexSymbol *symbol = index.lookup(posLoc);
  if (!symbol)
    return;

  references.push_back(getLocationFromLoc(sourceMgr, symbol->defLoc, uri));
  for (SMRange refLoc : symbol->references)
    references.push_back(getLocationFromLoc(sourceMgr, refLoc, uri));
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

void lsp::TableGenServer::addDocument(const URIForFile &uri, StringRef contents,
                                      int64_t version,
                                      std::vector<Diagnostic> &diagnostics) {
  // Build the set of additional include directories.
  std::vector<std::string> additionalIncludeDirs = impl->options.extraDirs;
  const auto &fileInfo = impl->compilationDatabase.getFileInfo(uri.file());
  llvm::append_range(additionalIncludeDirs, fileInfo.includeDirs);

  impl->files[uri.file()] = std::make_unique<TableGenTextFile>(
      uri, contents, version, additionalIncludeDirs, diagnostics);
}

void lsp::TableGenServer::updateDocument(
    const URIForFile &uri, ArrayRef<TextDocumentContentChangeEvent> changes,
    int64_t version, std::vector<Diagnostic> &diagnostics) {
  // Check that we actually have a document for this uri.
  auto it = impl->files.find(uri.file());
  if (it == impl->files.end())
    return;

  // Try to update the document. If we fail, erase the file from the server. A
  // failed updated generally means we've fallen out of sync somewhere.
  if (failed(it->second->update(uri, version, changes, diagnostics)))
    impl->files.erase(it);
}

Optional<int64_t> lsp::TableGenServer::removeDocument(const URIForFile &uri) {
  auto it = impl->files.find(uri.file());
  if (it == impl->files.end())
    return llvm::None;

  int64_t version = it->second->getVersion();
  impl->files.erase(it);
  return version;
}

void lsp::TableGenServer::getLocationsOf(const URIForFile &uri,
                                         const Position &defPos,
                                         std::vector<Location> &locations) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    fileIt->second->getLocationsOf(uri, defPos, locations);
}

void lsp::TableGenServer::findReferencesOf(const URIForFile &uri,
                                           const Position &pos,
                                           std::vector<Location> &references) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    fileIt->second->findReferencesOf(uri, pos, references);
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
