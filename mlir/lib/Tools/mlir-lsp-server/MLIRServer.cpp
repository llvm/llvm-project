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
  pos.character = lineAndCol.second - 1;
  return pos;
}

/// Returns a source location from the given language server position.
static llvm::SMLoc getPosFromLoc(llvm::SourceMgr &mgr, lsp::Position pos) {
  return mgr.FindLocForLineAndColumn(mgr.getMainFileID(), pos.line + 1,
                                     pos.character);
}

/// Returns a language server range for the given source range.
static lsp::Range getRangeFromLoc(llvm::SourceMgr &mgr, llvm::SMRange range) {
  return {getPosFromLoc(mgr, range.Start), getPosFromLoc(mgr, range.End)};
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
  return lsp::Location{*sourceURI, lsp::Range(position)};
}

/// Returns a language server location from the given MLIR location, or None if
/// one couldn't be created. `uri` is an optional additional filter that, when
/// present, is used to filter sub locations that do not share the same uri.
static Optional<lsp::Location>
getLocationFromLoc(llvm::SourceMgr &sourceMgr, Location loc,
                   const lsp::URIForFile *uri = nullptr) {
  Optional<lsp::Location> location;
  loc->walk([&](Location nestedLoc) {
    FileLineColLoc fileLoc = nestedLoc.dyn_cast<FileLineColLoc>();
    if (!fileLoc)
      return WalkResult::advance();

    Optional<lsp::Location> sourceLoc = getLocationFromLoc(fileLoc);
    if (sourceLoc && (!uri || sourceLoc->uri == *uri)) {
      location = *sourceLoc;
      llvm::SMLoc loc = sourceMgr.FindLocForLineAndColumn(
          sourceMgr.getMainFileID(), fileLoc.getLine(), fileLoc.getColumn());

      // Use range of potential identifier starting at location, else length 1
      // range.
      location->range.end.character += 1;
      if (Optional<llvm::SMRange> range =
              AsmParserState::convertIdLocToRange(loc)) {
        auto lineCol = sourceMgr.getLineAndColumn(range->End);
        location->range.end.character =
            std::max(fileLoc.getColumn() + 1, lineCol.second - 1);
      }
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return location;
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
/// the uses of the given SMDefinition. If provided, `overlappedRange` is set to
/// the range within `def` that the provided `loc` overlapped with.
static bool isDefOrUse(const AsmParserState::SMDefinition &def, llvm::SMLoc loc,
                       llvm::SMRange *overlappedRange = nullptr) {
  // Check the main definition.
  if (contains(def.loc, loc)) {
    if (overlappedRange)
      *overlappedRange = def.loc;
    return true;
  }

  // Check the uses.
  auto useIt = llvm::find_if(def.uses, [&](const llvm::SMRange &range) {
    return contains(range, loc);
  });
  if (useIt != def.uses.end()) {
    if (overlappedRange)
      *overlappedRange = *useIt;
    return true;
  }
  return false;
}

/// Given a location pointing to a result, return the result number it refers
/// to or None if it refers to all of the results.
static Optional<unsigned> getResultNumberFromLoc(llvm::SMLoc loc) {
  // Skip all of the identifier characters.
  auto isIdentifierChar = [](char c) {
    return isalnum(c) || c == '%' || c == '$' || c == '.' || c == '_' ||
           c == '-';
  };
  const char *curPtr = loc.getPointer();
  while (isIdentifierChar(*curPtr))
    ++curPtr;

  // Check to see if this location indexes into the result group, via `#`. If it
  // doesn't, we can't extract a sub result number.
  if (*curPtr != '#')
    return llvm::None;

  // Compute the sub result number from the remaining portion of the string.
  const char *numberStart = ++curPtr;
  while (llvm::isDigit(*curPtr))
    ++curPtr;
  StringRef numberStr(numberStart, curPtr - numberStart);
  unsigned resultNumber = 0;
  return numberStr.consumeInteger(10, resultNumber) ? Optional<unsigned>()
                                                    : resultNumber;
}

/// Given a source location range, return the text covered by the given range.
/// If the range is invalid, returns None.
static Optional<StringRef> getTextFromRange(llvm::SMRange range) {
  if (!range.isValid())
    return None;
  const char *startPtr = range.Start.getPointer();
  return StringRef(startPtr, range.End.getPointer() - startPtr);
}

/// Given a block, return its position in its parent region.
static unsigned getBlockNumber(Block *block) {
  return std::distance(block->getParent()->begin(), block->getIterator());
}

/// Given a block and source location, print the source name of the block to the
/// given output stream.
static void printDefBlockName(raw_ostream &os, Block *block,
                              llvm::SMRange loc = {}) {
  // Try to extract a name from the source location.
  Optional<StringRef> text = getTextFromRange(loc);
  if (text && text->startswith("^")) {
    os << *text;
    return;
  }

  // Otherwise, we don't have a name so print the block number.
  os << "<Block #" << getBlockNumber(block) << ">";
}
static void printDefBlockName(raw_ostream &os,
                              const AsmParserState::BlockDefinition &def) {
  printDefBlockName(os, def.block, def.definition.loc);
}

/// Convert the given MLIR diagnostic to the LSP form.
static lsp::Diagnostic getLspDiagnoticFromDiag(llvm::SourceMgr &sourceMgr,
                                               Diagnostic &diag,
                                               const lsp::URIForFile &uri) {
  lsp::Diagnostic lspDiag;
  lspDiag.source = "mlir";

  // Note: Right now all of the diagnostics are treated as parser issues, but
  // some are parser and some are verifier.
  lspDiag.category = "Parse Error";

  // Try to grab a file location for this diagnostic.
  // TODO: For simplicity, we just grab the first one. It may be likely that we
  // will need a more interesting heuristic here.'
  Optional<lsp::Location> lspLocation =
      getLocationFromLoc(sourceMgr, diag.getLocation(), &uri);
  if (lspLocation)
    lspDiag.range = lspLocation->range;

  // Convert the severity for the diagnostic.
  switch (diag.getSeverity()) {
  case DiagnosticSeverity::Note:
    llvm_unreachable("expected notes to be handled separately");
  case DiagnosticSeverity::Warning:
    lspDiag.severity = lsp::DiagnosticSeverity::Warning;
    break;
  case DiagnosticSeverity::Error:
    lspDiag.severity = lsp::DiagnosticSeverity::Error;
    break;
  case DiagnosticSeverity::Remark:
    lspDiag.severity = lsp::DiagnosticSeverity::Information;
    break;
  }
  lspDiag.message = diag.str();

  // Attach any notes to the main diagnostic as related information.
  std::vector<lsp::DiagnosticRelatedInformation> relatedDiags;
  for (Diagnostic &note : diag.getNotes()) {
    lsp::Location noteLoc;
    if (Optional<lsp::Location> loc =
            getLocationFromLoc(sourceMgr, note.getLocation()))
      noteLoc = *loc;
    else
      noteLoc.uri = uri;
    relatedDiags.emplace_back(noteLoc, note.str());
  }
  if (!relatedDiags.empty())
    lspDiag.relatedInformation = std::move(relatedDiags);

  return lspDiag;
}

//===----------------------------------------------------------------------===//
// MLIRDocument
//===----------------------------------------------------------------------===//

namespace {
/// This class represents all of the information pertaining to a specific MLIR
/// document.
struct MLIRDocument {
  MLIRDocument(MLIRContext &context, const lsp::URIForFile &uri,
               StringRef contents, std::vector<lsp::Diagnostic> &diagnostics);
  MLIRDocument(const MLIRDocument &) = delete;
  MLIRDocument &operator=(const MLIRDocument &) = delete;

  //===--------------------------------------------------------------------===//
  // Definitions and References
  //===--------------------------------------------------------------------===//

  void getLocationsOf(const lsp::URIForFile &uri, const lsp::Position &defPos,
                      std::vector<lsp::Location> &locations);
  void findReferencesOf(const lsp::URIForFile &uri, const lsp::Position &pos,
                        std::vector<lsp::Location> &references);

  //===--------------------------------------------------------------------===//
  // Hover
  //===--------------------------------------------------------------------===//

  Optional<lsp::Hover> findHover(const lsp::URIForFile &uri,
                                 const lsp::Position &hoverPos);
  Optional<lsp::Hover>
  buildHoverForOperation(llvm::SMRange hoverRange,
                         const AsmParserState::OperationDefinition &op);
  lsp::Hover buildHoverForOperationResult(llvm::SMRange hoverRange,
                                          Operation *op, unsigned resultStart,
                                          unsigned resultEnd,
                                          llvm::SMLoc posLoc);
  lsp::Hover buildHoverForBlock(llvm::SMRange hoverRange,
                                const AsmParserState::BlockDefinition &block);
  lsp::Hover
  buildHoverForBlockArgument(llvm::SMRange hoverRange, BlockArgument arg,
                             const AsmParserState::BlockDefinition &block);

  //===--------------------------------------------------------------------===//
  // Document Symbols
  //===--------------------------------------------------------------------===//

  void findDocumentSymbols(std::vector<lsp::DocumentSymbol> &symbols);
  void findDocumentSymbols(Operation *op,
                           std::vector<lsp::DocumentSymbol> &symbols);

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  /// The high level parser state used to find definitions and references within
  /// the source file.
  AsmParserState asmState;

  /// The container for the IR parsed from the input file.
  Block parsedIR;

  /// The source manager containing the contents of the input file.
  llvm::SourceMgr sourceMgr;
};
} // namespace

MLIRDocument::MLIRDocument(MLIRContext &context, const lsp::URIForFile &uri,
                           StringRef contents,
                           std::vector<lsp::Diagnostic> &diagnostics) {
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diag) {
    diagnostics.push_back(getLspDiagnoticFromDiag(sourceMgr, diag, uri));
  });

  // Try to parsed the given IR string.
  auto memBuffer = llvm::MemoryBuffer::getMemBufferCopy(contents, uri.file());
  if (!memBuffer) {
    lsp::Logger::error("Failed to create memory buffer for file", uri.file());
    return;
  }

  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), llvm::SMLoc());
  if (failed(parseSourceFile(sourceMgr, &parsedIR, &context, nullptr,
                             &asmState))) {
    // If parsing failed, clear out any of the current state.
    parsedIR.clear();
    asmState = AsmParserState();
    return;
  }
}

//===----------------------------------------------------------------------===//
// MLIRDocument: Definitions and References
//===----------------------------------------------------------------------===//

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
    for (const auto &symUse : op.symbolUses) {
      if (contains(symUse, posLoc)) {
        locations.push_back(getLocationFromLoc(sourceMgr, op.loc, uri));
        return collectLocationsFromLoc(op.op->getLoc(), locations, uri);
      }
    }
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
      for (const auto &symUse : op.symbolUses)
        if (contains(symUse, posLoc))
          references.push_back(getLocationFromLoc(sourceMgr, symUse, uri));
      return;
    }
    for (const auto &result : op.resultGroups)
      if (isDefOrUse(result.second, posLoc))
        return appendSMDef(result.second);
    for (const auto &symUse : op.symbolUses) {
      if (!contains(symUse, posLoc))
        continue;
      for (const auto &symUse : op.symbolUses)
        references.push_back(getLocationFromLoc(sourceMgr, symUse, uri));
      return;
    }
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
// MLIRDocument: Hover
//===----------------------------------------------------------------------===//

Optional<lsp::Hover> MLIRDocument::findHover(const lsp::URIForFile &uri,
                                             const lsp::Position &hoverPos) {
  llvm::SMLoc posLoc = getPosFromLoc(sourceMgr, hoverPos);
  llvm::SMRange hoverRange;

  // Check for Hovers on operations and results.
  for (const AsmParserState::OperationDefinition &op : asmState.getOpDefs()) {
    // Check if the position points at this operation.
    if (contains(op.loc, posLoc))
      return buildHoverForOperation(op.loc, op);

    // Check if the position points at the symbol name.
    for (auto &use : op.symbolUses)
      if (contains(use, posLoc))
        return buildHoverForOperation(use, op);

    // Check if the position points at a result group.
    for (unsigned i = 0, e = op.resultGroups.size(); i < e; ++i) {
      const auto &result = op.resultGroups[i];
      if (!isDefOrUse(result.second, posLoc, &hoverRange))
        continue;

      // Get the range of results covered by the over position.
      unsigned resultStart = result.first;
      unsigned resultEnd =
          (i == e - 1) ? op.op->getNumResults() : op.resultGroups[i + 1].first;
      return buildHoverForOperationResult(hoverRange, op.op, resultStart,
                                          resultEnd, posLoc);
    }
  }

  // Check to see if the hover is over a block argument.
  for (const AsmParserState::BlockDefinition &block : asmState.getBlockDefs()) {
    if (isDefOrUse(block.definition, posLoc, &hoverRange))
      return buildHoverForBlock(hoverRange, block);

    for (const auto &arg : llvm::enumerate(block.arguments)) {
      if (!isDefOrUse(arg.value(), posLoc, &hoverRange))
        continue;

      return buildHoverForBlockArgument(
          hoverRange, block.block->getArgument(arg.index()), block);
    }
  }
  return llvm::None;
}

Optional<lsp::Hover> MLIRDocument::buildHoverForOperation(
    llvm::SMRange hoverRange, const AsmParserState::OperationDefinition &op) {
  lsp::Hover hover(getRangeFromLoc(sourceMgr, hoverRange));
  llvm::raw_string_ostream os(hover.contents.value);

  // Add the operation name to the hover.
  os << "\"" << op.op->getName() << "\"";
  if (SymbolOpInterface symbol = dyn_cast<SymbolOpInterface>(op.op))
    os << " : " << symbol.getVisibility() << " @" << symbol.getName() << "";
  os << "\n\n";

  os << "Generic Form:\n\n```mlir\n";

  // Temporary drop the regions of this operation so that they don't get
  // printed in the output. This helps keeps the size of the output hover
  // small.
  SmallVector<std::unique_ptr<Region>> regions;
  for (Region &region : op.op->getRegions()) {
    regions.emplace_back(std::make_unique<Region>());
    regions.back()->takeBody(region);
  }

  op.op->print(
      os, OpPrintingFlags().printGenericOpForm().elideLargeElementsAttrs());
  os << "\n```\n";

  // Move the regions back to the current operation.
  for (Region &region : op.op->getRegions())
    region.takeBody(*regions.back());

  return hover;
}

lsp::Hover MLIRDocument::buildHoverForOperationResult(llvm::SMRange hoverRange,
                                                      Operation *op,
                                                      unsigned resultStart,
                                                      unsigned resultEnd,
                                                      llvm::SMLoc posLoc) {
  lsp::Hover hover(getRangeFromLoc(sourceMgr, hoverRange));
  llvm::raw_string_ostream os(hover.contents.value);

  // Add the parent operation name to the hover.
  os << "Operation: \"" << op->getName() << "\"\n\n";

  // Check to see if the location points to a specific result within the
  // group.
  if (Optional<unsigned> resultNumber = getResultNumberFromLoc(posLoc)) {
    if ((resultStart + *resultNumber) < resultEnd) {
      resultStart += *resultNumber;
      resultEnd = resultStart + 1;
    }
  }

  // Add the range of results and their types to the hover info.
  if ((resultStart + 1) == resultEnd) {
    os << "Result #" << resultStart << "\n\n"
       << "Type: `" << op->getResult(resultStart).getType() << "`\n\n";
  } else {
    os << "Result #[" << resultStart << ", " << (resultEnd - 1) << "]\n\n"
       << "Types: ";
    llvm::interleaveComma(
        op->getResults().slice(resultStart, resultEnd), os,
        [&](Value result) { os << "`" << result.getType() << "`"; });
  }

  return hover;
}

lsp::Hover
MLIRDocument::buildHoverForBlock(llvm::SMRange hoverRange,
                                 const AsmParserState::BlockDefinition &block) {
  lsp::Hover hover(getRangeFromLoc(sourceMgr, hoverRange));
  llvm::raw_string_ostream os(hover.contents.value);

  // Print the given block to the hover output stream.
  auto printBlockToHover = [&](Block *newBlock) {
    if (const auto *def = asmState.getBlockDef(newBlock))
      printDefBlockName(os, *def);
    else
      printDefBlockName(os, newBlock);
  };

  // Display the parent operation, block number, predecessors, and successors.
  os << "Operation: \"" << block.block->getParentOp()->getName() << "\"\n\n"
     << "Block #" << getBlockNumber(block.block) << "\n\n";
  if (!block.block->hasNoPredecessors()) {
    os << "Predecessors: ";
    llvm::interleaveComma(block.block->getPredecessors(), os,
                          printBlockToHover);
    os << "\n\n";
  }
  if (!block.block->hasNoSuccessors()) {
    os << "Successors: ";
    llvm::interleaveComma(block.block->getSuccessors(), os, printBlockToHover);
    os << "\n\n";
  }

  return hover;
}

lsp::Hover MLIRDocument::buildHoverForBlockArgument(
    llvm::SMRange hoverRange, BlockArgument arg,
    const AsmParserState::BlockDefinition &block) {
  lsp::Hover hover(getRangeFromLoc(sourceMgr, hoverRange));
  llvm::raw_string_ostream os(hover.contents.value);

  // Display the parent operation, block, the argument number, and the type.
  os << "Operation: \"" << block.block->getParentOp()->getName() << "\"\n\n"
     << "Block: ";
  printDefBlockName(os, block);
  os << "\n\nArgument #" << arg.getArgNumber() << "\n\n"
     << "Type: `" << arg.getType() << "`\n\n";

  return hover;
}

//===----------------------------------------------------------------------===//
// MLIRDocument: Document Symbols
//===----------------------------------------------------------------------===//

void MLIRDocument::findDocumentSymbols(
    std::vector<lsp::DocumentSymbol> &symbols) {
  for (Operation &op : parsedIR)
    findDocumentSymbols(&op, symbols);
}

void MLIRDocument::findDocumentSymbols(
    Operation *op, std::vector<lsp::DocumentSymbol> &symbols) {
  std::vector<lsp::DocumentSymbol> *childSymbols = &symbols;

  // Check for the source information of this operation.
  if (const AsmParserState::OperationDefinition *def = asmState.getOpDef(op)) {
    // If this operation defines a symbol, record it.
    if (SymbolOpInterface symbol = dyn_cast<SymbolOpInterface>(op)) {
      symbols.emplace_back(symbol.getName(),
                           op->hasTrait<OpTrait::FunctionLike>()
                               ? lsp::SymbolKind::Function
                               : lsp::SymbolKind::Class,
                           getRangeFromLoc(sourceMgr, def->scopeLoc),
                           getRangeFromLoc(sourceMgr, def->loc));
      childSymbols = &symbols.back().children;

    } else if (op->hasTrait<OpTrait::SymbolTable>()) {
      // Otherwise, if this is a symbol table push an anonymous document symbol.
      symbols.emplace_back("<" + op->getName().getStringRef() + ">",
                           lsp::SymbolKind::Namespace,
                           getRangeFromLoc(sourceMgr, def->scopeLoc),
                           getRangeFromLoc(sourceMgr, def->loc));
      childSymbols = &symbols.back().children;
    }
  }

  // Recurse into the regions of this operation.
  if (!op->getNumRegions())
    return;
  for (Region &region : op->getRegions())
    for (Operation &childOp : region.getOps())
      findDocumentSymbols(&childOp, *childSymbols);
}

//===----------------------------------------------------------------------===//
// MLIRTextFileChunk
//===----------------------------------------------------------------------===//

namespace {
/// This class represents a single chunk of an MLIR text file.
struct MLIRTextFileChunk {
  MLIRTextFileChunk(MLIRContext &context, uint64_t lineOffset,
                    const lsp::URIForFile &uri, StringRef contents,
                    std::vector<lsp::Diagnostic> &diagnostics)
      : lineOffset(lineOffset), document(context, uri, contents, diagnostics) {}

  /// Adjust the line number of the given range to anchor at the beginning of
  /// the file, instead of the beginning of this chunk.
  void adjustLocForChunkOffset(lsp::Range &range) {
    adjustLocForChunkOffset(range.start);
    adjustLocForChunkOffset(range.end);
  }
  /// Adjust the line number of the given position to anchor at the beginning of
  /// the file, instead of the beginning of this chunk.
  void adjustLocForChunkOffset(lsp::Position &pos) { pos.line += lineOffset; }

  /// The line offset of this chunk from the beginning of the file.
  uint64_t lineOffset;
  /// The document referred to by this chunk.
  MLIRDocument document;
};
} // namespace

//===----------------------------------------------------------------------===//
// MLIRTextFile
//===----------------------------------------------------------------------===//

namespace {
/// This class represents a text file containing one or more MLIR documents.
class MLIRTextFile {
public:
  MLIRTextFile(const lsp::URIForFile &uri, StringRef fileContents,
               int64_t version, DialectRegistry &registry,
               std::vector<lsp::Diagnostic> &diagnostics);

  /// Return the current version of this text file.
  int64_t getVersion() const { return version; }

  //===--------------------------------------------------------------------===//
  // LSP Queries
  //===--------------------------------------------------------------------===//

  void getLocationsOf(const lsp::URIForFile &uri, lsp::Position defPos,
                      std::vector<lsp::Location> &locations);
  void findReferencesOf(const lsp::URIForFile &uri, lsp::Position pos,
                        std::vector<lsp::Location> &references);
  Optional<lsp::Hover> findHover(const lsp::URIForFile &uri,
                                 lsp::Position hoverPos);
  void findDocumentSymbols(std::vector<lsp::DocumentSymbol> &symbols);

private:
  /// Find the MLIR document that contains the given position, and update the
  /// position to be anchored at the start of the found chunk instead of the
  /// beginning of the file.
  MLIRTextFileChunk &getChunkFor(lsp::Position &pos);

  /// The context used to hold the state contained by the parsed document.
  MLIRContext context;

  /// The full string contents of the file.
  std::string contents;

  /// The version of this file.
  int64_t version;

  /// The number of lines in the file.
  int64_t totalNumLines;

  /// The chunks of this file. The order of these chunks is the order in which
  /// they appear in the text file.
  std::vector<std::unique_ptr<MLIRTextFileChunk>> chunks;
};
} // namespace

MLIRTextFile::MLIRTextFile(const lsp::URIForFile &uri, StringRef fileContents,
                           int64_t version, DialectRegistry &registry,
                           std::vector<lsp::Diagnostic> &diagnostics)
    : context(registry, MLIRContext::Threading::DISABLED),
      contents(fileContents.str()), version(version), totalNumLines(0) {
  context.allowUnregisteredDialects();

  // Split the file into separate MLIR documents.
  // TODO: Find a way to share the split file marker with other tools. We don't
  // want to use `splitAndProcessBuffer` here, but we do want to make sure this
  // marker doesn't go out of sync.
  SmallVector<StringRef, 8> subContents;
  StringRef(contents).split(subContents, "// -----");
  chunks.emplace_back(std::make_unique<MLIRTextFileChunk>(
      context, /*lineOffset=*/0, uri, subContents.front(), diagnostics));

  uint64_t lineOffset = subContents.front().count('\n');
  for (StringRef docContents : llvm::drop_begin(subContents)) {
    unsigned currentNumDiags = diagnostics.size();
    auto chunk = std::make_unique<MLIRTextFileChunk>(context, lineOffset, uri,
                                                     docContents, diagnostics);
    lineOffset += docContents.count('\n');

    // Adjust locations used in diagnostics to account for the offset from the
    // beginning of the file.
    for (lsp::Diagnostic &diag :
         llvm::drop_begin(diagnostics, currentNumDiags)) {
      chunk->adjustLocForChunkOffset(diag.range);

      if (!diag.relatedInformation)
        continue;
      for (auto &it : *diag.relatedInformation)
        if (it.location.uri == uri)
          chunk->adjustLocForChunkOffset(it.location.range);
    }
    chunks.emplace_back(std::move(chunk));
  }
  totalNumLines = lineOffset;
}

void MLIRTextFile::getLocationsOf(const lsp::URIForFile &uri,
                                  lsp::Position defPos,
                                  std::vector<lsp::Location> &locations) {
  MLIRTextFileChunk &chunk = getChunkFor(defPos);
  chunk.document.getLocationsOf(uri, defPos, locations);

  // Adjust any locations within this file for the offset of this chunk.
  if (chunk.lineOffset == 0)
    return;
  for (lsp::Location &loc : locations)
    if (loc.uri == uri)
      chunk.adjustLocForChunkOffset(loc.range);
}

void MLIRTextFile::findReferencesOf(const lsp::URIForFile &uri,
                                    lsp::Position pos,
                                    std::vector<lsp::Location> &references) {
  MLIRTextFileChunk &chunk = getChunkFor(pos);
  chunk.document.findReferencesOf(uri, pos, references);

  // Adjust any locations within this file for the offset of this chunk.
  if (chunk.lineOffset == 0)
    return;
  for (lsp::Location &loc : references)
    if (loc.uri == uri)
      chunk.adjustLocForChunkOffset(loc.range);
}

Optional<lsp::Hover> MLIRTextFile::findHover(const lsp::URIForFile &uri,
                                             lsp::Position hoverPos) {
  MLIRTextFileChunk &chunk = getChunkFor(hoverPos);
  Optional<lsp::Hover> hoverInfo = chunk.document.findHover(uri, hoverPos);

  // Adjust any locations within this file for the offset of this chunk.
  if (chunk.lineOffset != 0 && hoverInfo && hoverInfo->range)
    chunk.adjustLocForChunkOffset(*hoverInfo->range);
  return hoverInfo;
}

void MLIRTextFile::findDocumentSymbols(
    std::vector<lsp::DocumentSymbol> &symbols) {
  if (chunks.size() == 1)
    return chunks.front()->document.findDocumentSymbols(symbols);

  // If there are multiple chunks in this file, we create top-level symbols for
  // each chunk.
  for (unsigned i = 0, e = chunks.size(); i < e; ++i) {
    MLIRTextFileChunk &chunk = *chunks[i];
    lsp::Position startPos(chunk.lineOffset);
    lsp::Position endPos((i == e - 1) ? totalNumLines - 1
                                      : chunks[i + 1]->lineOffset);
    lsp::DocumentSymbol symbol("<file-split-" + Twine(i) + ">",
                               lsp::SymbolKind::Namespace,
                               /*range=*/lsp::Range(startPos, endPos),
                               /*selectionRange=*/lsp::Range(startPos));
    chunk.document.findDocumentSymbols(symbol.children);

    // Fixup the locations of document symbols within this chunk.
    if (i != 0) {
      SmallVector<lsp::DocumentSymbol *> symbolsToFix;
      for (lsp::DocumentSymbol &childSymbol : symbol.children)
        symbolsToFix.push_back(&childSymbol);

      while (!symbolsToFix.empty()) {
        lsp::DocumentSymbol *symbol = symbolsToFix.pop_back_val();
        chunk.adjustLocForChunkOffset(symbol->range);
        chunk.adjustLocForChunkOffset(symbol->selectionRange);

        for (lsp::DocumentSymbol &childSymbol : symbol->children)
          symbolsToFix.push_back(&childSymbol);
      }
    }

    // Push the symbol for this chunk.
    symbols.emplace_back(std::move(symbol));
  }
}

MLIRTextFileChunk &MLIRTextFile::getChunkFor(lsp::Position &pos) {
  if (chunks.size() == 1)
    return *chunks.front();

  // Search for the first chunk with a greater line offset, the previous chunk
  // is the one that contains `pos`.
  auto it = llvm::upper_bound(
      chunks, pos, [](const lsp::Position &pos, const auto &chunk) {
        return static_cast<uint64_t>(pos.line) < chunk->lineOffset;
      });
  MLIRTextFileChunk &chunk = it == chunks.end() ? *chunks.back() : **(--it);
  pos.line -= chunk.lineOffset;
  return chunk;
}

//===----------------------------------------------------------------------===//
// MLIRServer::Impl
//===----------------------------------------------------------------------===//

struct lsp::MLIRServer::Impl {
  Impl(DialectRegistry &registry) : registry(registry) {}

  /// The registry containing dialects that can be recognized in parsed .mlir
  /// files.
  DialectRegistry &registry;

  /// The files held by the server, mapped by their URI file name.
  llvm::StringMap<std::unique_ptr<MLIRTextFile>> files;
};

//===----------------------------------------------------------------------===//
// MLIRServer
//===----------------------------------------------------------------------===//

lsp::MLIRServer::MLIRServer(DialectRegistry &registry)
    : impl(std::make_unique<Impl>(registry)) {}
lsp::MLIRServer::~MLIRServer() {}

void lsp::MLIRServer::addOrUpdateDocument(
    const URIForFile &uri, StringRef contents, int64_t version,
    std::vector<Diagnostic> &diagnostics) {
  impl->files[uri.file()] = std::make_unique<MLIRTextFile>(
      uri, contents, version, impl->registry, diagnostics);
}

Optional<int64_t> lsp::MLIRServer::removeDocument(const URIForFile &uri) {
  auto it = impl->files.find(uri.file());
  if (it == impl->files.end())
    return llvm::None;

  int64_t version = it->second->getVersion();
  impl->files.erase(it);
  return version;
}

void lsp::MLIRServer::getLocationsOf(const URIForFile &uri,
                                     const Position &defPos,
                                     std::vector<Location> &locations) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    fileIt->second->getLocationsOf(uri, defPos, locations);
}

void lsp::MLIRServer::findReferencesOf(const URIForFile &uri,
                                       const Position &pos,
                                       std::vector<Location> &references) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    fileIt->second->findReferencesOf(uri, pos, references);
}

Optional<lsp::Hover> lsp::MLIRServer::findHover(const URIForFile &uri,
                                                const Position &hoverPos) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    return fileIt->second->findHover(uri, hoverPos);
  return llvm::None;
}

void lsp::MLIRServer::findDocumentSymbols(
    const URIForFile &uri, std::vector<DocumentSymbol> &symbols) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    fileIt->second->findDocumentSymbols(symbols);
}
