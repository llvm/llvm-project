//===- MLIRServer.cpp - MLIR Generic Language Server ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MLIRServer.h"
#include "lsp/Logging.h"
#include "lsp/Protocol.h"
#include "mlir/IR/Operation.h"
#include "mlir/Parser.h"
#include "mlir/Parser/AsmParserState.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;

/// Returns a language server position for the given source location.
static lsp::Position getPosFromLoc(llvm::SourceMgr &mgr, llvm::SMLoc loc) {
  std::pair<unsigned, unsigned> lineAndCol = mgr.getLineAndColumn(loc);
  lsp::Position pos;
  pos.line = lineAndCol.first - 1;
  pos.character = lineAndCol.second;
  return pos;
}

/// Returns a source location from the given language server position.
static llvm::SMLoc getPosFromLoc(llvm::SourceMgr &mgr, lsp::Position pos) {
  return mgr.FindLocForLineAndColumn(mgr.getMainFileID(), pos.line + 1,
                                     pos.character);
}

/// Returns a language server range for the given source range.
static lsp::Range getRangeFromLoc(llvm::SourceMgr &mgr, llvm::SMRange range) {
  // lsp::Range is an inclusive range, SMRange is half-open.
  llvm::SMLoc inclusiveEnd =
      llvm::SMLoc::getFromPointer(range.End.getPointer() - 1);
  return {getPosFromLoc(mgr, range.Start), getPosFromLoc(mgr, inclusiveEnd)};
}

/// Returns a language server location from the given source range.
static lsp::Location getLocationFromLoc(llvm::SourceMgr &mgr,
                                        llvm::SMRange range,
                                        const lsp::URIForFile &uri) {
  return lsp::Location{uri, getRangeFromLoc(mgr, range)};
}

/// Returns a language server location from the given MLIR file location.
static Optional<lsp::Location> getLocationFromLoc(FileLineColLoc loc) {
  llvm::Expected<lsp::URIForFile> sourceURI =
      lsp::URIForFile::fromFile(loc.getFilename());
  if (!sourceURI) {
    lsp::Logger::error("Failed to create URI for file `{0}`: {1}",
                       loc.getFilename(),
                       llvm::toString(sourceURI.takeError()));
    return llvm::None;
  }

  lsp::Position position;
  position.line = loc.getLine() - 1;
  position.character = loc.getColumn();
  return lsp::Location{*sourceURI, lsp::Range{position, position}};
}

/// Collect all of the locations from the given MLIR location that are not
/// contained within the given URI.
static void collectLocationsFromLoc(Location loc,
                                    std::vector<lsp::Location> &locations,
                                    const lsp::URIForFile &uri) {
  SetVector<Location> visitedLocs;
  loc->walk([&](Location nestedLoc) {
    FileLineColLoc fileLoc = nestedLoc.dyn_cast<FileLineColLoc>();
    if (!fileLoc || !visitedLocs.insert(nestedLoc))
      return WalkResult::advance();

    Optional<lsp::Location> sourceLoc = getLocationFromLoc(fileLoc);
    if (sourceLoc && sourceLoc->uri != uri)
      locations.push_back(*sourceLoc);
    return WalkResult::advance();
  });
}

/// Returns true if the given range contains the given source location. Note
/// that this has slightly different behavior than SMRange because it is
/// inclusive of the end location.
static bool contains(llvm::SMRange range, llvm::SMLoc loc) {
  return range.Start.getPointer() <= loc.getPointer() &&
         loc.getPointer() <= range.End.getPointer();
}

/// Returns true if the given location is contained by the definition or one of
/// the uses of the given SMDefinition.
static bool isDefOrUse(const AsmParserState::SMDefinition &def,
                       llvm::SMLoc loc) {
  auto isUseFn = [&](const llvm::SMRange &range) {
    return contains(range, loc);
  };
  return contains(def.loc, loc) || llvm::any_of(def.uses, isUseFn);
}

//===----------------------------------------------------------------------===//
// MLIRDocument
//===----------------------------------------------------------------------===//

namespace {
/// This class represents all of the information pertaining to a specific MLIR
/// document.
struct MLIRDocument {
  MLIRDocument(const lsp::URIForFile &uri, StringRef contents,
               DialectRegistry &registry);

  void getLocationsOf(const lsp::URIForFile &uri, const lsp::Position &defPos,
                      std::vector<lsp::Location> &locations);
  void findReferencesOf(const lsp::URIForFile &uri, const lsp::Position &pos,
                        std::vector<lsp::Location> &references);

  /// The context used to hold the state contained by the parsed document.
  MLIRContext context;

  /// The high level parser state used to find definitions and references within
  /// the source file.
  AsmParserState asmState;

  /// The container for the IR parsed from the input file.
  Block parsedIR;

  /// The source manager containing the contents of the input file.
  llvm::SourceMgr sourceMgr;
};
} // namespace

MLIRDocument::MLIRDocument(const lsp::URIForFile &uri, StringRef contents,
                           DialectRegistry &registry)
    : context(registry) {
  context.allowUnregisteredDialects();
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diag) {
    // TODO: What should we do with these diagnostics?
    //  * Cache and show to the user?
    //  * Ignore?
    lsp::Logger::error("Error when parsing MLIR document `{0}`: `{1}`",
                       uri.file(), diag.str());
  });

  // Try to parsed the given IR string.
  auto memBuffer = llvm::MemoryBuffer::getMemBufferCopy(contents, uri.file());
  if (!memBuffer) {
    lsp::Logger::error("Failed to create memory buffer for file", uri.file());
    return;
  }

  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), llvm::SMLoc());
  if (failed(
          parseSourceFile(sourceMgr, &parsedIR, &context, nullptr, &asmState)))
    return;
}

void MLIRDocument::getLocationsOf(const lsp::URIForFile &uri,
                                  const lsp::Position &defPos,
                                  std::vector<lsp::Location> &locations) {
  llvm::SMLoc posLoc = getPosFromLoc(sourceMgr, defPos);

  // Functor used to check if an SM definition contains the position.
  auto containsPosition = [&](const AsmParserState::SMDefinition &def) {
    if (!isDefOrUse(def, posLoc))
      return false;
    locations.push_back(getLocationFromLoc(sourceMgr, def.loc, uri));
    return true;
  };

  // Check all definitions related to operations.
  for (const AsmParserState::OperationDefinition &op : asmState.getOpDefs()) {
    if (contains(op.loc, posLoc))
      return collectLocationsFromLoc(op.op->getLoc(), locations, uri);
    for (const auto &result : op.resultGroups)
      if (containsPosition(result.second))
        return collectLocationsFromLoc(op.op->getLoc(), locations, uri);
  }

  // Check all definitions related to blocks.
  for (const AsmParserState::BlockDefinition &block : asmState.getBlockDefs()) {
    if (containsPosition(block.definition))
      return;
    for (const AsmParserState::SMDefinition &arg : block.arguments)
      if (containsPosition(arg))
        return;
  }
}

void MLIRDocument::findReferencesOf(const lsp::URIForFile &uri,
                                    const lsp::Position &pos,
                                    std::vector<lsp::Location> &references) {
  // Functor used to append all of the definitions/uses of the given SM
  // definition to the reference list.
  auto appendSMDef = [&](const AsmParserState::SMDefinition &def) {
    references.push_back(getLocationFromLoc(sourceMgr, def.loc, uri));
    for (const llvm::SMRange &use : def.uses)
      references.push_back(getLocationFromLoc(sourceMgr, use, uri));
  };

  llvm::SMLoc posLoc = getPosFromLoc(sourceMgr, pos);

  // Check all definitions related to operations.
  for (const AsmParserState::OperationDefinition &op : asmState.getOpDefs()) {
    if (contains(op.loc, posLoc)) {
      for (const auto &result : op.resultGroups)
        appendSMDef(result.second);
      return;
    }
    for (const auto &result : op.resultGroups)
      if (isDefOrUse(result.second, posLoc))
        return appendSMDef(result.second);
  }

  // Check all definitions related to blocks.
  for (const AsmParserState::BlockDefinition &block : asmState.getBlockDefs()) {
    if (isDefOrUse(block.definition, posLoc))
      return appendSMDef(block.definition);

    for (const AsmParserState::SMDefinition &arg : block.arguments)
      if (isDefOrUse(arg, posLoc))
        return appendSMDef(arg);
  }
}

//===----------------------------------------------------------------------===//
// MLIRServer::Impl
//===----------------------------------------------------------------------===//

struct lsp::MLIRServer::Impl {
  Impl(DialectRegistry &registry) : registry(registry) {}

  /// The registry containing dialects that can be recognized in parsed .mlir
  /// files.
  DialectRegistry &registry;

  /// The documents held by the server, mapped by their URI file name.
  llvm::StringMap<std::unique_ptr<MLIRDocument>> documents;
};

//===----------------------------------------------------------------------===//
// MLIRServer
//===----------------------------------------------------------------------===//

lsp::MLIRServer::MLIRServer(DialectRegistry &registry)
    : impl(std::make_unique<Impl>(registry)) {}
lsp::MLIRServer::~MLIRServer() {}

void lsp::MLIRServer::addOrUpdateDocument(const URIForFile &uri,
                                          StringRef contents) {
  impl->documents[uri.file()] =
      std::make_unique<MLIRDocument>(uri, contents, impl->registry);
}

void lsp::MLIRServer::removeDocument(const URIForFile &uri) {
  impl->documents.erase(uri.file());
}

void lsp::MLIRServer::getLocationsOf(const URIForFile &uri,
                                     const Position &defPos,
                                     std::vector<Location> &locations) {
  auto fileIt = impl->documents.find(uri.file());
  if (fileIt != impl->documents.end())
    fileIt->second->getLocationsOf(uri, defPos, locations);
}

void lsp::MLIRServer::findReferencesOf(const URIForFile &uri,
                                       const Position &pos,
                                       std::vector<Location> &references) {
  auto fileIt = impl->documents.find(uri.file());
  if (fileIt != impl->documents.end())
    fileIt->second->findReferencesOf(uri, pos, references);
}
