//===- TableGenServer.cpp - TableGen Language Server ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TableGenServer.h"

#include "mlir/Support/IndentedOstream.h"
#include "mlir/Tools/lsp-server-support/CompilationDatabase.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Protocol.h"
#include "mlir/Tools/lsp-server-support/SourceMgrUtils.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Path.h"
#include "llvm/TableGen/Parser.h"
#include "llvm/TableGen/Record.h"
#include <optional>

using namespace mlir;
using llvm::Record;
using llvm::RecordKeeper;
using llvm::RecordVal;
using llvm::SourceMgr;

/// Returns the range of a lexical token given a SMLoc corresponding to the
/// start of an token location. The range is computed heuristically, and
/// supports identifier-like tokens, strings, etc.
static SMRange convertTokenLocToRange(SMLoc loc) {
  return lsp::convertTokenLocToRange(loc, "$");
}

/// Returns a language server uri for the given source location. `mainFileURI`
/// corresponds to the uri for the main file of the source manager.
static lsp::URIForFile getURIFromLoc(const SourceMgr &mgr, SMLoc loc,
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
static lsp::Location getLocationFromLoc(SourceMgr &mgr, SMRange loc,
                                        const lsp::URIForFile &uri) {
  return lsp::Location(getURIFromLoc(mgr, loc.Start, uri),
                       lsp::Range(mgr, loc));
}
static lsp::Location getLocationFromLoc(SourceMgr &mgr, SMLoc loc,
                                        const lsp::URIForFile &uri) {
  return getLocationFromLoc(mgr, convertTokenLocToRange(loc), uri);
}

/// Convert the given TableGen diagnostic to the LSP form.
static std::optional<lsp::Diagnostic>
getLspDiagnoticFromDiag(const llvm::SMDiagnostic &diag,
                        const lsp::URIForFile &uri) {
  auto *sourceMgr = const_cast<SourceMgr *>(diag.getSourceMgr());
  if (!sourceMgr || !diag.getLoc().isValid())
    return std::nullopt;

  lsp::Diagnostic lspDiag;
  lspDiag.source = "tablegen";
  lspDiag.category = "Parse Error";

  // Try to grab a file location for this diagnostic.
  lsp::Location loc = getLocationFromLoc(*sourceMgr, diag.getLoc(), uri);
  lspDiag.range = loc.range;

  // Skip diagnostics that weren't emitted within the main file.
  if (loc.uri != uri)
    return std::nullopt;

  // Convert the severity for the diagnostic.
  switch (diag.getKind()) {
  case SourceMgr::DK_Warning:
    lspDiag.severity = lsp::DiagnosticSeverity::Warning;
    break;
  case SourceMgr::DK_Error:
    lspDiag.severity = lsp::DiagnosticSeverity::Error;
    break;
  case SourceMgr::DK_Note:
    // Notes are emitted separately from the main diagnostic, so we just treat
    // them as remarks given that we can't determine the diagnostic to relate
    // them to.
  case SourceMgr::DK_Remark:
    lspDiag.severity = lsp::DiagnosticSeverity::Information;
    break;
  }
  lspDiag.message = diag.getMessage().str();

  return lspDiag;
}

/// Get the base definition of the given record value, or nullptr if one
/// couldn't be found.
static std::pair<const Record *, const RecordVal *>
getBaseValue(const Record *record, const RecordVal *value) {
  if (value->isTemplateArg())
    return {nullptr, nullptr};

  // Find a base value for the field in the super classes of the given record.
  // On success, `record` is updated to the new parent record.
  StringRef valueName = value->getName();
  auto findValueInSupers = [&](const Record *&record) -> const RecordVal * {
    for (const Record *parentRecord : record->getSuperClasses()) {
      if (auto *newBase = parentRecord->getValue(valueName)) {
        record = parentRecord;
        return newBase;
      }
    }
    return nullptr;
  };

  // Try to find the lowest definition of the record value.
  std::pair<const Record *, const RecordVal *> baseValue = {};
  while (const RecordVal *newBase = findValueInSupers(record))
    baseValue = {record, newBase};

  // Check that the base isn't the same as the current value (e.g. if the value
  // wasn't overridden).
  if (!baseValue.second || baseValue.second->getLoc() == value->getLoc())
    return {nullptr, nullptr};
  return baseValue;
}

//===----------------------------------------------------------------------===//
// TableGenIndex
//===----------------------------------------------------------------------===//

namespace {
/// This class represents a single symbol definition within a TableGen index. It
/// contains the definition of the symbol, the location of the symbol, and any
/// recorded references.
struct TableGenIndexSymbol {
  TableGenIndexSymbol(const Record *record)
      : definition(record),
        defLoc(convertTokenLocToRange(record->getLoc().front())) {}
  TableGenIndexSymbol(const RecordVal *value)
      : definition(value), defLoc(convertTokenLocToRange(value->getLoc())) {}
  virtual ~TableGenIndexSymbol() = default;

  // The main definition of the symbol.
  PointerUnion<const Record *, const RecordVal *> definition;

  /// The source location of the definition.
  SMRange defLoc;

  /// The source location of the references of the definition.
  SmallVector<SMRange> references;
};
/// This class represents a single record symbol.
struct TableGenRecordSymbol : public TableGenIndexSymbol {
  TableGenRecordSymbol(const Record *record) : TableGenIndexSymbol(record) {}
  ~TableGenRecordSymbol() override = default;

  static bool classof(const TableGenIndexSymbol *symbol) {
    return isa<const Record *>(symbol->definition);
  }

  /// Return the value of this symbol.
  const Record *getValue() const { return cast<const Record *>(definition); }
};
/// This class represents a single record value symbol.
struct TableGenRecordValSymbol : public TableGenIndexSymbol {
  TableGenRecordValSymbol(const Record *record, const RecordVal *value)
      : TableGenIndexSymbol(value), record(record) {}
  ~TableGenRecordValSymbol() override = default;

  static bool classof(const TableGenIndexSymbol *symbol) {
    return isa<const RecordVal *>(symbol->definition);
  }

  /// Return the value of this symbol.
  const RecordVal *getValue() const {
    return cast<const RecordVal *>(definition);
  }

  /// The parent record of this symbol.
  const Record *record;
};

/// This class provides an index for definitions/uses within a TableGen
/// document. It provides efficient lookup of a definition given an input source
/// range.
class TableGenIndex {
public:
  TableGenIndex() : intervalMap(allocator) {}

  /// Initialize the index with the given RecordKeeper.
  void initialize(const RecordKeeper &records);

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

  /// Get or insert a symbol for the given record.
  TableGenIndexSymbol *getOrInsertDef(const Record *record) {
    auto it = defToSymbol.try_emplace(record, nullptr);
    if (it.second)
      it.first->second = std::make_unique<TableGenRecordSymbol>(record);
    return &*it.first->second;
  }
  /// Get or insert a symbol for the given record value.
  TableGenIndexSymbol *getOrInsertDef(const Record *record,
                                      const RecordVal *value) {
    auto it = defToSymbol.try_emplace(value, nullptr);
    if (it.second) {
      it.first->second =
          std::make_unique<TableGenRecordValSymbol>(record, value);
    }
    return &*it.first->second;
  }

  /// An allocator for the interval map.
  MapT::Allocator allocator;

  /// An interval map containing a corresponding definition mapped to a source
  /// interval.
  MapT intervalMap;

  /// A mapping between definitions and their corresponding symbol.
  DenseMap<const void *, std::unique_ptr<TableGenIndexSymbol>> defToSymbol;
};
} // namespace

void TableGenIndex::initialize(const RecordKeeper &records) {
  intervalMap.clear();
  defToSymbol.clear();

  auto insertRef = [&](TableGenIndexSymbol *sym, SMRange refLoc,
                       bool isDef = false) {
    const char *startLoc = refLoc.Start.getPointer();
    const char *endLoc = refLoc.End.getPointer();

    // If the location we got was empty, try to lex a token from the start
    // location.
    if (startLoc == endLoc) {
      refLoc = convertTokenLocToRange(SMLoc::getFromPointer(startLoc));
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
  for (const Record &def : llvm::concat<Record>(classes, defs)) {
    auto *sym = getOrInsertDef(&def);
    insertRef(sym, sym->defLoc, /*isDef=*/true);

    // Add references to the definition.
    for (SMLoc loc : def.getLoc().drop_front())
      insertRef(sym, convertTokenLocToRange(loc));
    for (SMRange loc : def.getReferenceLocs())
      insertRef(sym, loc);

    // Add definitions for any values.
    for (const RecordVal &value : def.getValues()) {
      auto *sym = getOrInsertDef(&def, &value);
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

  std::optional<lsp::Hover> findHover(const lsp::URIForFile &uri,
                                      const lsp::Position &hoverPos);
  lsp::Hover buildHoverForRecord(const Record *record,
                                 const SMRange &hoverRange);
  lsp::Hover buildHoverForTemplateArg(const Record *record,
                                      const RecordVal *value,
                                      const SMRange &hoverRange);
  lsp::Hover buildHoverForField(const Record *record, const RecordVal *value,
                                const SMRange &hoverRange);

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
  SourceMgr sourceMgr;

  /// The record keeper containing the parsed tablegen constructs.
  std::unique_ptr<RecordKeeper> recordKeeper;

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
  llvm::append_range(includeDirs, extraIncludeDirs);

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
  sourceMgr = SourceMgr();
  recordKeeper = std::make_unique<RecordKeeper>();

  // Build a buffer for this file.
  auto memBuffer = llvm::MemoryBuffer::getMemBuffer(contents, uri.file());
  if (!memBuffer) {
    lsp::Logger::error("Failed to create memory buffer for file", uri.file());
    return;
  }
  sourceMgr.setIncludeDirs(includeDirs);
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), SMLoc());

  // This class provides a context argument for the SourceMgr diagnostic
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

  // If this symbol is a record value and the def position is already the def of
  // the symbol, check to see if the value has a base definition. This allows
  // for a "go-to-def" on a "let" to resolve the definition in the base class.
  auto *valSym = dyn_cast<TableGenRecordValSymbol>(symbol);
  if (valSym && lsp::contains(valSym->defLoc, posLoc)) {
    if (auto *val = getBaseValue(valSym->record, valSym->getValue()).second) {
      locations.push_back(getLocationFromLoc(sourceMgr, val->getLoc(), uri));
      return;
    }
  }

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

std::optional<lsp::Hover>
TableGenTextFile::findHover(const lsp::URIForFile &uri,
                            const lsp::Position &hoverPos) {
  // Check for a reference to an include.
  for (const lsp::SourceMgrInclude &include : parsedIncludes)
    if (include.range.contains(hoverPos))
      return include.buildHover();

  // Find the symbol at the given location.
  SMRange hoverRange;
  SMLoc posLoc = hoverPos.getAsSMLoc(sourceMgr);
  const TableGenIndexSymbol *symbol = index.lookup(posLoc, &hoverRange);
  if (!symbol)
    return std::nullopt;

  // Build hover for a Record.
  if (auto *record = dyn_cast<TableGenRecordSymbol>(symbol))
    return buildHoverForRecord(record->getValue(), hoverRange);

  // Build hover for a RecordVal, which is either a template argument or a
  // field.
  auto *recordVal = cast<TableGenRecordValSymbol>(symbol);
  const RecordVal *value = recordVal->getValue();
  if (value->isTemplateArg())
    return buildHoverForTemplateArg(recordVal->record, value, hoverRange);
  return buildHoverForField(recordVal->record, value, hoverRange);
}

lsp::Hover TableGenTextFile::buildHoverForRecord(const Record *record,
                                                 const SMRange &hoverRange) {
  lsp::Hover hover(lsp::Range(sourceMgr, hoverRange));
  {
    llvm::raw_string_ostream hoverOS(hover.contents.value);

    // Format the type of record this is.
    if (record->isClass()) {
      hoverOS << "**class** `" << record->getName() << "`";
    } else if (record->isAnonymous()) {
      hoverOS << "**anonymous class**";
    } else {
      hoverOS << "**def** `" << record->getName() << "`";
    }
    hoverOS << "\n***\n";

    // Check if this record has summary/description fields. These are often used
    // to hold documentation for the record.
    auto printAndFormatField = [&](StringRef fieldName) {
      // Check that the record actually has the given field, and that it's a
      // string.
      const RecordVal *value = record->getValue(fieldName);
      if (!value || !value->getValue())
        return;
      auto *stringValue = dyn_cast<llvm::StringInit>(value->getValue());
      if (!stringValue)
        return;

      raw_indented_ostream ros(hoverOS);
      ros.printReindented(stringValue->getValue().rtrim(" \t"));
      hoverOS << "\n***\n";
    };
    printAndFormatField("summary");
    printAndFormatField("description");

    // Check for documentation in the source file.
    if (std::optional<std::string> doc =
            lsp::extractSourceDocComment(sourceMgr, record->getLoc().front())) {
      hoverOS << "\n" << *doc << "\n";
    }
  }
  return hover;
}

lsp::Hover TableGenTextFile::buildHoverForTemplateArg(
    const Record *record, const RecordVal *value, const SMRange &hoverRange) {
  lsp::Hover hover(lsp::Range(sourceMgr, hoverRange));
  {
    llvm::raw_string_ostream hoverOS(hover.contents.value);
    StringRef name = value->getName().rsplit(':').second;

    hoverOS << "**template arg** `" << name << "`\n***\nType: `";
    value->getType()->print(hoverOS);
    hoverOS << "`\n";
  }
  return hover;
}

lsp::Hover TableGenTextFile::buildHoverForField(const Record *record,
                                                const RecordVal *value,
                                                const SMRange &hoverRange) {
  lsp::Hover hover(lsp::Range(sourceMgr, hoverRange));
  {
    llvm::raw_string_ostream hoverOS(hover.contents.value);
    hoverOS << "**field** `" << value->getName() << "`\n***\nType: `";
    value->getType()->print(hoverOS);
    hoverOS << "`\n***\n";

    // Check for documentation in the source file.
    if (std::optional<std::string> doc =
            lsp::extractSourceDocComment(sourceMgr, value->getLoc())) {
      hoverOS << "\n" << *doc << "\n";
      hoverOS << "\n***\n";
    }

    // Check to see if there is a base value that we can use for
    // documentation.
    auto [baseRecord, baseValue] = getBaseValue(record, value);
    if (baseValue) {
      if (std::optional<std::string> doc =
              lsp::extractSourceDocComment(sourceMgr, baseValue->getLoc())) {
        hoverOS << "\n *From `" << baseRecord->getName() << "`*:\n\n"
                << *doc << "\n";
      }
    }
  }
  return hover;
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

std::optional<int64_t>
lsp::TableGenServer::removeDocument(const URIForFile &uri) {
  auto it = impl->files.find(uri.file());
  if (it == impl->files.end())
    return std::nullopt;

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

std::optional<lsp::Hover>
lsp::TableGenServer::findHover(const URIForFile &uri,
                               const Position &hoverPos) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    return fileIt->second->findHover(uri, hoverPos);
  return std::nullopt;
}
