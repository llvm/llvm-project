//===- MLIRServer.cpp - MLIR Generic Language Server ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MLIRServer.h"
#include "Protocol.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/AsmParser/AsmParserState.h"
#include "mlir/AsmParser/CodeComplete.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/SourceMgrUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/SourceMgr.h"
#include <optional>

using namespace mlir;

/// Returns the range of a lexical token given a SMLoc corresponding to the
/// start of an token location. The range is computed heuristically, and
/// supports identifier-like tokens, strings, etc.
static SMRange convertTokenLocToRange(SMLoc loc) {
  return lsp::convertTokenLocToRange(loc, "$-.");
}

/// Returns a language server location from the given MLIR file location.
/// `uriScheme` is the scheme to use when building new uris.
static std::optional<lsp::Location> getLocationFromLoc(StringRef uriScheme,
                                                       FileLineColLoc loc) {
  llvm::Expected<lsp::URIForFile> sourceURI =
      lsp::URIForFile::fromFile(loc.getFilename(), uriScheme);
  if (!sourceURI) {
    lsp::Logger::error("Failed to create URI for file `{0}`: {1}",
                       loc.getFilename(),
                       llvm::toString(sourceURI.takeError()));
    return std::nullopt;
  }

  lsp::Position position;
  position.line = loc.getLine() - 1;
  position.character = loc.getColumn() ? loc.getColumn() - 1 : 0;
  return lsp::Location{*sourceURI, lsp::Range(position)};
}

/// Returns a language server location from the given MLIR location, or
/// std::nullopt if one couldn't be created. `uriScheme` is the scheme to use
/// when building new uris. `uri` is an optional additional filter that, when
/// present, is used to filter sub locations that do not share the same uri.
static std::optional<lsp::Location>
getLocationFromLoc(llvm::SourceMgr &sourceMgr, Location loc,
                   StringRef uriScheme, const lsp::URIForFile *uri = nullptr) {
  std::optional<lsp::Location> location;
  loc->walk([&](Location nestedLoc) {
    FileLineColLoc fileLoc = dyn_cast<FileLineColLoc>(nestedLoc);
    if (!fileLoc)
      return WalkResult::advance();

    std::optional<lsp::Location> sourceLoc =
        getLocationFromLoc(uriScheme, fileLoc);
    if (sourceLoc && (!uri || sourceLoc->uri == *uri)) {
      location = *sourceLoc;
      SMLoc loc = sourceMgr.FindLocForLineAndColumn(
          sourceMgr.getMainFileID(), fileLoc.getLine(), fileLoc.getColumn());

      // Use range of potential identifier starting at location, else length 1
      // range.
      location->range.end.character += 1;
      if (std::optional<SMRange> range = convertTokenLocToRange(loc)) {
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
    FileLineColLoc fileLoc = dyn_cast<FileLineColLoc>(nestedLoc);
    if (!fileLoc || !visitedLocs.insert(nestedLoc))
      return WalkResult::advance();

    std::optional<lsp::Location> sourceLoc =
        getLocationFromLoc(uri.scheme(), fileLoc);
    if (sourceLoc && sourceLoc->uri != uri)
      locations.push_back(*sourceLoc);
    return WalkResult::advance();
  });
}

/// Returns true if the given range contains the given source location. Note
/// that this has slightly different behavior than SMRange because it is
/// inclusive of the end location.
static bool contains(SMRange range, SMLoc loc) {
  return range.Start.getPointer() <= loc.getPointer() &&
         loc.getPointer() <= range.End.getPointer();
}

/// Returns true if the given location is contained by the definition or one of
/// the uses of the given SMDefinition. If provided, `overlappedRange` is set to
/// the range within `def` that the provided `loc` overlapped with.
static bool isDefOrUse(const AsmParserState::SMDefinition &def, SMLoc loc,
                       SMRange *overlappedRange = nullptr) {
  // Check the main definition.
  if (contains(def.loc, loc)) {
    if (overlappedRange)
      *overlappedRange = def.loc;
    return true;
  }

  // Check the uses.
  const auto *useIt = llvm::find_if(
      def.uses, [&](const SMRange &range) { return contains(range, loc); });
  if (useIt != def.uses.end()) {
    if (overlappedRange)
      *overlappedRange = *useIt;
    return true;
  }
  return false;
}

/// Given a location pointing to a result, return the result number it refers
/// to or std::nullopt if it refers to all of the results.
static std::optional<unsigned> getResultNumberFromLoc(SMLoc loc) {
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
    return std::nullopt;

  // Compute the sub result number from the remaining portion of the string.
  const char *numberStart = ++curPtr;
  while (llvm::isDigit(*curPtr))
    ++curPtr;
  StringRef numberStr(numberStart, curPtr - numberStart);
  unsigned resultNumber = 0;
  return numberStr.consumeInteger(10, resultNumber) ? std::optional<unsigned>()
                                                    : resultNumber;
}

/// Given a source location range, return the text covered by the given range.
/// If the range is invalid, returns std::nullopt.
static std::optional<StringRef> getTextFromRange(SMRange range) {
  if (!range.isValid())
    return std::nullopt;
  const char *startPtr = range.Start.getPointer();
  return StringRef(startPtr, range.End.getPointer() - startPtr);
}

/// Given a block, return its position in its parent region.
static unsigned getBlockNumber(Block *block) {
  return std::distance(block->getParent()->begin(), block->getIterator());
}

/// Given a block and source location, print the source name of the block to the
/// given output stream.
static void printDefBlockName(raw_ostream &os, Block *block, SMRange loc = {}) {
  // Try to extract a name from the source location.
  std::optional<StringRef> text = getTextFromRange(loc);
  if (text && text->starts_with("^")) {
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
  StringRef uriScheme = uri.scheme();
  std::optional<lsp::Location> lspLocation =
      getLocationFromLoc(sourceMgr, diag.getLocation(), uriScheme, &uri);
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
    if (std::optional<lsp::Location> loc =
            getLocationFromLoc(sourceMgr, note.getLocation(), uriScheme))
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

  std::optional<lsp::Hover> findHover(const lsp::URIForFile &uri,
                                      const lsp::Position &hoverPos);
  std::optional<lsp::Hover>
  buildHoverForOperation(SMRange hoverRange,
                         const AsmParserState::OperationDefinition &op);
  lsp::Hover buildHoverForOperationResult(SMRange hoverRange, Operation *op,
                                          unsigned resultStart,
                                          unsigned resultEnd, SMLoc posLoc);
  lsp::Hover buildHoverForBlock(SMRange hoverRange,
                                const AsmParserState::BlockDefinition &block);
  lsp::Hover
  buildHoverForBlockArgument(SMRange hoverRange, BlockArgument arg,
                             const AsmParserState::BlockDefinition &block);

  lsp::Hover buildHoverForAttributeAlias(
      SMRange hoverRange, const AsmParserState::AttributeAliasDefinition &attr);
  lsp::Hover
  buildHoverForTypeAlias(SMRange hoverRange,
                         const AsmParserState::TypeAliasDefinition &type);

  //===--------------------------------------------------------------------===//
  // Document Symbols
  //===--------------------------------------------------------------------===//

  void findDocumentSymbols(std::vector<lsp::DocumentSymbol> &symbols);
  void findDocumentSymbols(Operation *op,
                           std::vector<lsp::DocumentSymbol> &symbols);

  //===--------------------------------------------------------------------===//
  // Code Completion
  //===--------------------------------------------------------------------===//

  lsp::CompletionList getCodeCompletion(const lsp::URIForFile &uri,
                                        const lsp::Position &completePos,
                                        const DialectRegistry &registry);

  //===--------------------------------------------------------------------===//
  // Code Action
  //===--------------------------------------------------------------------===//

  void getCodeActionForDiagnostic(const lsp::URIForFile &uri,
                                  lsp::Position &pos, StringRef severity,
                                  StringRef message,
                                  std::vector<lsp::TextEdit> &edits);

  //===--------------------------------------------------------------------===//
  // Bytecode
  //===--------------------------------------------------------------------===//

  llvm::Expected<lsp::MLIRConvertBytecodeResult> convertToBytecode();

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  /// The high level parser state used to find definitions and references within
  /// the source file.
  AsmParserState asmState;

  /// The container for the IR parsed from the input file.
  Block parsedIR;

  /// A collection of external resources, which we want to propagate up to the
  /// user.
  FallbackAsmResourceMap fallbackResourceMap;

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

  ParserConfig config(&context, /*verifyAfterParse=*/true,
                      &fallbackResourceMap);
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), SMLoc());
  if (failed(parseAsmSourceFile(sourceMgr, &parsedIR, config, &asmState))) {
    // If parsing failed, clear out any of the current state.
    parsedIR.clear();
    asmState = AsmParserState();
    fallbackResourceMap = FallbackAsmResourceMap();
    return;
  }
}

//===----------------------------------------------------------------------===//
// MLIRDocument: Definitions and References
//===----------------------------------------------------------------------===//

void MLIRDocument::getLocationsOf(const lsp::URIForFile &uri,
                                  const lsp::Position &defPos,
                                  std::vector<lsp::Location> &locations) {
  SMLoc posLoc = defPos.getAsSMLoc(sourceMgr);

  // Functor used to check if an SM definition contains the position.
  auto containsPosition = [&](const AsmParserState::SMDefinition &def) {
    if (!isDefOrUse(def, posLoc))
      return false;
    locations.emplace_back(uri, sourceMgr, def.loc);
    return true;
  };

  // Check all definitions related to operations.
  for (const AsmParserState::OperationDefinition &op : asmState.getOpDefs()) {
    if (contains(op.loc, posLoc))
      return collectLocationsFromLoc(op.op->getLoc(), locations, uri);
    for (const auto &result : op.resultGroups)
      if (containsPosition(result.definition))
        return collectLocationsFromLoc(op.op->getLoc(), locations, uri);
    for (const auto &symUse : op.symbolUses) {
      if (contains(symUse, posLoc)) {
        locations.emplace_back(uri, sourceMgr, op.loc);
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

  // Check all alias definitions.
  for (const AsmParserState::AttributeAliasDefinition &attr :
       asmState.getAttributeAliasDefs()) {
    if (containsPosition(attr.definition))
      return;
  }
  for (const AsmParserState::TypeAliasDefinition &type :
       asmState.getTypeAliasDefs()) {
    if (containsPosition(type.definition))
      return;
  }
}

void MLIRDocument::findReferencesOf(const lsp::URIForFile &uri,
                                    const lsp::Position &pos,
                                    std::vector<lsp::Location> &references) {
  // Functor used to append all of the definitions/uses of the given SM
  // definition to the reference list.
  auto appendSMDef = [&](const AsmParserState::SMDefinition &def) {
    references.emplace_back(uri, sourceMgr, def.loc);
    for (const SMRange &use : def.uses)
      references.emplace_back(uri, sourceMgr, use);
  };

  SMLoc posLoc = pos.getAsSMLoc(sourceMgr);

  // Check all definitions related to operations.
  for (const AsmParserState::OperationDefinition &op : asmState.getOpDefs()) {
    if (contains(op.loc, posLoc)) {
      for (const auto &result : op.resultGroups)
        appendSMDef(result.definition);
      for (const auto &symUse : op.symbolUses)
        if (contains(symUse, posLoc))
          references.emplace_back(uri, sourceMgr, symUse);
      return;
    }
    for (const auto &result : op.resultGroups)
      if (isDefOrUse(result.definition, posLoc))
        return appendSMDef(result.definition);
    for (const auto &symUse : op.symbolUses) {
      if (!contains(symUse, posLoc))
        continue;
      for (const auto &symUse : op.symbolUses)
        references.emplace_back(uri, sourceMgr, symUse);
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

  // Check all alias definitions.
  for (const AsmParserState::AttributeAliasDefinition &attr :
       asmState.getAttributeAliasDefs()) {
    if (isDefOrUse(attr.definition, posLoc))
      return appendSMDef(attr.definition);
  }
  for (const AsmParserState::TypeAliasDefinition &type :
       asmState.getTypeAliasDefs()) {
    if (isDefOrUse(type.definition, posLoc))
      return appendSMDef(type.definition);
  }
}

//===----------------------------------------------------------------------===//
// MLIRDocument: Hover
//===----------------------------------------------------------------------===//

std::optional<lsp::Hover>
MLIRDocument::findHover(const lsp::URIForFile &uri,
                        const lsp::Position &hoverPos) {
  SMLoc posLoc = hoverPos.getAsSMLoc(sourceMgr);
  SMRange hoverRange;

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
      if (!isDefOrUse(result.definition, posLoc, &hoverRange))
        continue;

      // Get the range of results covered by the over position.
      unsigned resultStart = result.startIndex;
      unsigned resultEnd = (i == e - 1) ? op.op->getNumResults()
                                        : op.resultGroups[i + 1].startIndex;
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

  // Check to see if the hover is over an alias.
  for (const AsmParserState::AttributeAliasDefinition &attr :
       asmState.getAttributeAliasDefs()) {
    if (isDefOrUse(attr.definition, posLoc, &hoverRange))
      return buildHoverForAttributeAlias(hoverRange, attr);
  }
  for (const AsmParserState::TypeAliasDefinition &type :
       asmState.getTypeAliasDefs()) {
    if (isDefOrUse(type.definition, posLoc, &hoverRange))
      return buildHoverForTypeAlias(hoverRange, type);
  }

  return std::nullopt;
}

std::optional<lsp::Hover> MLIRDocument::buildHoverForOperation(
    SMRange hoverRange, const AsmParserState::OperationDefinition &op) {
  lsp::Hover hover(lsp::Range(sourceMgr, hoverRange));
  llvm::raw_string_ostream os(hover.contents.value);

  // Add the operation name to the hover.
  os << "\"" << op.op->getName() << "\"";
  if (SymbolOpInterface symbol = dyn_cast<SymbolOpInterface>(op.op))
    os << " : " << symbol.getVisibility() << " @" << symbol.getName() << "";
  os << "\n\n";

  os << "Generic Form:\n\n```mlir\n";

  op.op->print(os, OpPrintingFlags()
                       .printGenericOpForm()
                       .elideLargeElementsAttrs()
                       .skipRegions());
  os << "\n```\n";

  return hover;
}

lsp::Hover MLIRDocument::buildHoverForOperationResult(SMRange hoverRange,
                                                      Operation *op,
                                                      unsigned resultStart,
                                                      unsigned resultEnd,
                                                      SMLoc posLoc) {
  lsp::Hover hover(lsp::Range(sourceMgr, hoverRange));
  llvm::raw_string_ostream os(hover.contents.value);

  // Add the parent operation name to the hover.
  os << "Operation: \"" << op->getName() << "\"\n\n";

  // Check to see if the location points to a specific result within the
  // group.
  if (std::optional<unsigned> resultNumber = getResultNumberFromLoc(posLoc)) {
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
MLIRDocument::buildHoverForBlock(SMRange hoverRange,
                                 const AsmParserState::BlockDefinition &block) {
  lsp::Hover hover(lsp::Range(sourceMgr, hoverRange));
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
    SMRange hoverRange, BlockArgument arg,
    const AsmParserState::BlockDefinition &block) {
  lsp::Hover hover(lsp::Range(sourceMgr, hoverRange));
  llvm::raw_string_ostream os(hover.contents.value);

  // Display the parent operation, block, the argument number, and the type.
  os << "Operation: \"" << block.block->getParentOp()->getName() << "\"\n\n"
     << "Block: ";
  printDefBlockName(os, block);
  os << "\n\nArgument #" << arg.getArgNumber() << "\n\n"
     << "Type: `" << arg.getType() << "`\n\n";

  return hover;
}

lsp::Hover MLIRDocument::buildHoverForAttributeAlias(
    SMRange hoverRange, const AsmParserState::AttributeAliasDefinition &attr) {
  lsp::Hover hover(lsp::Range(sourceMgr, hoverRange));
  llvm::raw_string_ostream os(hover.contents.value);

  os << "Attribute Alias: \"" << attr.name << "\n\n";
  os << "Value: ```mlir\n" << attr.value << "\n```\n\n";

  return hover;
}

lsp::Hover MLIRDocument::buildHoverForTypeAlias(
    SMRange hoverRange, const AsmParserState::TypeAliasDefinition &type) {
  lsp::Hover hover(lsp::Range(sourceMgr, hoverRange));
  llvm::raw_string_ostream os(hover.contents.value);

  os << "Type Alias: \"" << type.name << "\n\n";
  os << "Value: ```mlir\n" << type.value << "\n```\n\n";

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
                           isa<FunctionOpInterface>(op)
                               ? lsp::SymbolKind::Function
                               : lsp::SymbolKind::Class,
                           lsp::Range(sourceMgr, def->scopeLoc),
                           lsp::Range(sourceMgr, def->loc));
      childSymbols = &symbols.back().children;

    } else if (op->hasTrait<OpTrait::SymbolTable>()) {
      // Otherwise, if this is a symbol table push an anonymous document symbol.
      symbols.emplace_back("<" + op->getName().getStringRef() + ">",
                           lsp::SymbolKind::Namespace,
                           lsp::Range(sourceMgr, def->scopeLoc),
                           lsp::Range(sourceMgr, def->loc));
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
// MLIRDocument: Code Completion
//===----------------------------------------------------------------------===//

namespace {
class LSPCodeCompleteContext : public AsmParserCodeCompleteContext {
public:
  LSPCodeCompleteContext(SMLoc completeLoc, lsp::CompletionList &completionList,
                         MLIRContext *ctx)
      : AsmParserCodeCompleteContext(completeLoc),
        completionList(completionList), ctx(ctx) {}

  /// Signal code completion for a dialect name, with an optional prefix.
  void completeDialectName(StringRef prefix) final {
    for (StringRef dialect : ctx->getAvailableDialects()) {
      lsp::CompletionItem item(prefix + dialect,
                               lsp::CompletionItemKind::Module,
                               /*sortText=*/"3");
      item.detail = "dialect";
      completionList.items.emplace_back(item);
    }
  }
  using AsmParserCodeCompleteContext::completeDialectName;

  /// Signal code completion for an operation name within the given dialect.
  void completeOperationName(StringRef dialectName) final {
    Dialect *dialect = ctx->getOrLoadDialect(dialectName);
    if (!dialect)
      return;

    for (const auto &op : ctx->getRegisteredOperations()) {
      if (&op.getDialect() != dialect)
        continue;

      lsp::CompletionItem item(
          op.getStringRef().drop_front(dialectName.size() + 1),
          lsp::CompletionItemKind::Field,
          /*sortText=*/"1");
      item.detail = "operation";
      completionList.items.emplace_back(item);
    }
  }

  /// Append the given SSA value as a code completion result for SSA value
  /// completions.
  void appendSSAValueCompletion(StringRef name, std::string typeData) final {
    // Check if we need to insert the `%` or not.
    bool stripPrefix = getCodeCompleteLoc().getPointer()[-1] == '%';

    lsp::CompletionItem item(name, lsp::CompletionItemKind::Variable);
    if (stripPrefix)
      item.insertText = name.drop_front(1).str();
    item.detail = std::move(typeData);
    completionList.items.emplace_back(item);
  }

  /// Append the given block as a code completion result for block name
  /// completions.
  void appendBlockCompletion(StringRef name) final {
    // Check if we need to insert the `^` or not.
    bool stripPrefix = getCodeCompleteLoc().getPointer()[-1] == '^';

    lsp::CompletionItem item(name, lsp::CompletionItemKind::Field);
    if (stripPrefix)
      item.insertText = name.drop_front(1).str();
    completionList.items.emplace_back(item);
  }

  /// Signal a completion for the given expected token.
  void completeExpectedTokens(ArrayRef<StringRef> tokens, bool optional) final {
    for (StringRef token : tokens) {
      lsp::CompletionItem item(token, lsp::CompletionItemKind::Keyword,
                               /*sortText=*/"0");
      item.detail = optional ? "optional" : "";
      completionList.items.emplace_back(item);
    }
  }

  /// Signal a completion for an attribute.
  void completeAttribute(const llvm::StringMap<Attribute> &aliases) override {
    appendSimpleCompletions({"affine_set", "affine_map", "dense",
                             "dense_resource", "false", "loc", "sparse", "true",
                             "unit"},
                            lsp::CompletionItemKind::Field,
                            /*sortText=*/"1");

    completeDialectName("#");
    completeAliases(aliases, "#");
  }
  void completeDialectAttributeOrAlias(
      const llvm::StringMap<Attribute> &aliases) override {
    completeDialectName();
    completeAliases(aliases);
  }

  /// Signal a completion for a type.
  void completeType(const llvm::StringMap<Type> &aliases) override {
    // Handle the various builtin types.
    appendSimpleCompletions({"memref", "tensor", "complex", "tuple", "vector",
                             "bf16", "f16", "f32", "f64", "f80", "f128",
                             "index", "none"},
                            lsp::CompletionItemKind::Field,
                            /*sortText=*/"1");

    // Handle the builtin integer types.
    for (StringRef type : {"i", "si", "ui"}) {
      lsp::CompletionItem item(type + "<N>", lsp::CompletionItemKind::Field,
                               /*sortText=*/"1");
      item.insertText = type.str();
      completionList.items.emplace_back(item);
    }

    // Insert completions for dialect types and aliases.
    completeDialectName("!");
    completeAliases(aliases, "!");
  }
  void
  completeDialectTypeOrAlias(const llvm::StringMap<Type> &aliases) override {
    completeDialectName();
    completeAliases(aliases);
  }

  /// Add completion results for the given set of aliases.
  template <typename T>
  void completeAliases(const llvm::StringMap<T> &aliases,
                       StringRef prefix = "") {
    for (const auto &alias : aliases) {
      lsp::CompletionItem item(prefix + alias.getKey(),
                               lsp::CompletionItemKind::Field,
                               /*sortText=*/"2");
      llvm::raw_string_ostream(item.detail) << "alias: " << alias.getValue();
      completionList.items.emplace_back(item);
    }
  }

  /// Add a set of simple completions that all have the same kind.
  void appendSimpleCompletions(ArrayRef<StringRef> completions,
                               lsp::CompletionItemKind kind,
                               StringRef sortText = "") {
    for (StringRef completion : completions)
      completionList.items.emplace_back(completion, kind, sortText);
  }

private:
  lsp::CompletionList &completionList;
  MLIRContext *ctx;
};
} // namespace

lsp::CompletionList
MLIRDocument::getCodeCompletion(const lsp::URIForFile &uri,
                                const lsp::Position &completePos,
                                const DialectRegistry &registry) {
  SMLoc posLoc = completePos.getAsSMLoc(sourceMgr);
  if (!posLoc.isValid())
    return lsp::CompletionList();

  // To perform code completion, we run another parse of the module with the
  // code completion context provided.
  MLIRContext tmpContext(registry, MLIRContext::Threading::DISABLED);
  tmpContext.allowUnregisteredDialects();
  lsp::CompletionList completionList;
  LSPCodeCompleteContext lspCompleteContext(posLoc, completionList,
                                            &tmpContext);

  Block tmpIR;
  AsmParserState tmpState;
  (void)parseAsmSourceFile(sourceMgr, &tmpIR, &tmpContext, &tmpState,
                           &lspCompleteContext);
  return completionList;
}

//===----------------------------------------------------------------------===//
// MLIRDocument: Code Action
//===----------------------------------------------------------------------===//

void MLIRDocument::getCodeActionForDiagnostic(
    const lsp::URIForFile &uri, lsp::Position &pos, StringRef severity,
    StringRef message, std::vector<lsp::TextEdit> &edits) {
  // Ignore diagnostics that print the current operation. These are always
  // enabled for the language server, but not generally during normal
  // parsing/verification.
  if (message.starts_with("see current operation: "))
    return;

  // Get the start of the line containing the diagnostic.
  const auto &buffer = sourceMgr.getBufferInfo(sourceMgr.getMainFileID());
  const char *lineStart = buffer.getPointerForLineNumber(pos.line + 1);
  if (!lineStart)
    return;
  StringRef line(lineStart, pos.character);

  // Add a text edit for adding an expected-* diagnostic check for this
  // diagnostic.
  lsp::TextEdit edit;
  edit.range = lsp::Range(lsp::Position(pos.line, 0));

  // Use the indent of the current line for the expected-* diagnostic.
  size_t indent = line.find_first_not_of(' ');
  if (indent == StringRef::npos)
    indent = line.size();

  edit.newText.append(indent, ' ');
  llvm::raw_string_ostream(edit.newText)
      << "// expected-" << severity << " @below {{" << message << "}}\n";
  edits.emplace_back(std::move(edit));
}

//===----------------------------------------------------------------------===//
// MLIRDocument: Bytecode
//===----------------------------------------------------------------------===//

llvm::Expected<lsp::MLIRConvertBytecodeResult>
MLIRDocument::convertToBytecode() {
  // TODO: We currently require a single top-level operation, but this could
  // conceptually be relaxed.
  if (!llvm::hasSingleElement(parsedIR)) {
    if (parsedIR.empty()) {
      return llvm::make_error<lsp::LSPError>(
          "expected a single and valid top-level operation, please ensure "
          "there are no errors",
          lsp::ErrorCode::RequestFailed);
    }
    return llvm::make_error<lsp::LSPError>(
        "expected a single top-level operation", lsp::ErrorCode::RequestFailed);
  }

  lsp::MLIRConvertBytecodeResult result;
  {
    BytecodeWriterConfig writerConfig(fallbackResourceMap);

    std::string rawBytecodeBuffer;
    llvm::raw_string_ostream os(rawBytecodeBuffer);
    // No desired bytecode version set, so no need to check for error.
    (void)writeBytecodeToFile(&parsedIR.front(), os, writerConfig);
    result.output = llvm::encodeBase64(rawBytecodeBuffer);
  }
  return result;
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
               int64_t version, lsp::DialectRegistryFn registry_fn,
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
  std::optional<lsp::Hover> findHover(const lsp::URIForFile &uri,
                                      lsp::Position hoverPos);
  void findDocumentSymbols(std::vector<lsp::DocumentSymbol> &symbols);
  lsp::CompletionList getCodeCompletion(const lsp::URIForFile &uri,
                                        lsp::Position completePos);
  void getCodeActions(const lsp::URIForFile &uri, const lsp::Range &pos,
                      const lsp::CodeActionContext &context,
                      std::vector<lsp::CodeAction> &actions);
  llvm::Expected<lsp::MLIRConvertBytecodeResult> convertToBytecode();

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
  int64_t totalNumLines = 0;

  /// The chunks of this file. The order of these chunks is the order in which
  /// they appear in the text file.
  std::vector<std::unique_ptr<MLIRTextFileChunk>> chunks;
};
} // namespace

MLIRTextFile::MLIRTextFile(const lsp::URIForFile &uri, StringRef fileContents,
                           int64_t version, lsp::DialectRegistryFn registry_fn,
                           std::vector<lsp::Diagnostic> &diagnostics)
    : context(registry_fn(uri), MLIRContext::Threading::DISABLED),
      contents(fileContents.str()), version(version) {
  context.allowUnregisteredDialects();

  // Split the file into separate MLIR documents.
  SmallVector<StringRef, 8> subContents;
  StringRef(contents).split(subContents, kDefaultSplitMarker);
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

std::optional<lsp::Hover> MLIRTextFile::findHover(const lsp::URIForFile &uri,
                                                  lsp::Position hoverPos) {
  MLIRTextFileChunk &chunk = getChunkFor(hoverPos);
  std::optional<lsp::Hover> hoverInfo = chunk.document.findHover(uri, hoverPos);

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

lsp::CompletionList MLIRTextFile::getCodeCompletion(const lsp::URIForFile &uri,
                                                    lsp::Position completePos) {
  MLIRTextFileChunk &chunk = getChunkFor(completePos);
  lsp::CompletionList completionList = chunk.document.getCodeCompletion(
      uri, completePos, context.getDialectRegistry());

  // Adjust any completion locations.
  for (lsp::CompletionItem &item : completionList.items) {
    if (item.textEdit)
      chunk.adjustLocForChunkOffset(item.textEdit->range);
    for (lsp::TextEdit &edit : item.additionalTextEdits)
      chunk.adjustLocForChunkOffset(edit.range);
  }
  return completionList;
}

void MLIRTextFile::getCodeActions(const lsp::URIForFile &uri,
                                  const lsp::Range &pos,
                                  const lsp::CodeActionContext &context,
                                  std::vector<lsp::CodeAction> &actions) {
  // Create actions for any diagnostics in this file.
  for (auto &diag : context.diagnostics) {
    if (diag.source != "mlir")
      continue;
    lsp::Position diagPos = diag.range.start;
    MLIRTextFileChunk &chunk = getChunkFor(diagPos);

    // Add a new code action that inserts a "expected" diagnostic check.
    lsp::CodeAction action;
    action.title = "Add expected-* diagnostic checks";
    action.kind = lsp::CodeAction::kQuickFix.str();

    StringRef severity;
    switch (diag.severity) {
    case lsp::DiagnosticSeverity::Error:
      severity = "error";
      break;
    case lsp::DiagnosticSeverity::Warning:
      severity = "warning";
      break;
    default:
      continue;
    }

    // Get edits for the diagnostic.
    std::vector<lsp::TextEdit> edits;
    chunk.document.getCodeActionForDiagnostic(uri, diagPos, severity,
                                              diag.message, edits);

    // Walk the related diagnostics, this is how we encode notes.
    if (diag.relatedInformation) {
      for (auto &noteDiag : *diag.relatedInformation) {
        if (noteDiag.location.uri != uri)
          continue;
        diagPos = noteDiag.location.range.start;
        diagPos.line -= chunk.lineOffset;
        chunk.document.getCodeActionForDiagnostic(uri, diagPos, "note",
                                                  noteDiag.message, edits);
      }
    }
    // Fixup the locations for any edits.
    for (lsp::TextEdit &edit : edits)
      chunk.adjustLocForChunkOffset(edit.range);

    action.edit.emplace();
    action.edit->changes[uri.uri().str()] = std::move(edits);
    action.diagnostics = {diag};

    actions.emplace_back(std::move(action));
  }
}

llvm::Expected<lsp::MLIRConvertBytecodeResult>
MLIRTextFile::convertToBytecode() {
  // Bail out if there is more than one chunk, bytecode wants a single module.
  if (chunks.size() != 1) {
    return llvm::make_error<lsp::LSPError>(
        "unexpected split file, please remove all `// -----`",
        lsp::ErrorCode::RequestFailed);
  }
  return chunks.front()->document.convertToBytecode();
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
  Impl(lsp::DialectRegistryFn registry_fn) : registry_fn(registry_fn) {}

  /// The registry factory for containing dialects that can be recognized in
  /// parsed .mlir files.
  lsp::DialectRegistryFn registry_fn;

  /// The files held by the server, mapped by their URI file name.
  llvm::StringMap<std::unique_ptr<MLIRTextFile>> files;
};

//===----------------------------------------------------------------------===//
// MLIRServer
//===----------------------------------------------------------------------===//

lsp::MLIRServer::MLIRServer(lsp::DialectRegistryFn registry_fn)
    : impl(std::make_unique<Impl>(registry_fn)) {}
lsp::MLIRServer::~MLIRServer() = default;

void lsp::MLIRServer::addOrUpdateDocument(
    const URIForFile &uri, StringRef contents, int64_t version,
    std::vector<Diagnostic> &diagnostics) {
  impl->files[uri.file()] = std::make_unique<MLIRTextFile>(
      uri, contents, version, impl->registry_fn, diagnostics);
}

std::optional<int64_t> lsp::MLIRServer::removeDocument(const URIForFile &uri) {
  auto it = impl->files.find(uri.file());
  if (it == impl->files.end())
    return std::nullopt;

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

std::optional<lsp::Hover> lsp::MLIRServer::findHover(const URIForFile &uri,
                                                     const Position &hoverPos) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    return fileIt->second->findHover(uri, hoverPos);
  return std::nullopt;
}

void lsp::MLIRServer::findDocumentSymbols(
    const URIForFile &uri, std::vector<DocumentSymbol> &symbols) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    fileIt->second->findDocumentSymbols(symbols);
}

lsp::CompletionList
lsp::MLIRServer::getCodeCompletion(const URIForFile &uri,
                                   const Position &completePos) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    return fileIt->second->getCodeCompletion(uri, completePos);
  return CompletionList();
}

void lsp::MLIRServer::getCodeActions(const URIForFile &uri, const Range &pos,
                                     const CodeActionContext &context,
                                     std::vector<CodeAction> &actions) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    fileIt->second->getCodeActions(uri, pos, context, actions);
}

llvm::Expected<lsp::MLIRConvertBytecodeResult>
lsp::MLIRServer::convertFromBytecode(const URIForFile &uri) {
  MLIRContext tempContext(impl->registry_fn(uri));
  tempContext.allowUnregisteredDialects();

  // Collect any errors during parsing.
  std::string errorMsg;
  ScopedDiagnosticHandler diagHandler(
      &tempContext,
      [&](mlir::Diagnostic &diag) { errorMsg += diag.str() + "\n"; });

  // Handling for external resources, which we want to propagate up to the user.
  FallbackAsmResourceMap fallbackResourceMap;

  // Setup the parser config.
  ParserConfig parserConfig(&tempContext, /*verifyAfterParse=*/true,
                            &fallbackResourceMap);

  // Try to parse the given source file.
  Block parsedBlock;
  if (failed(parseSourceFile(uri.file(), &parsedBlock, parserConfig))) {
    return llvm::make_error<lsp::LSPError>(
        "failed to parse bytecode source file: " + errorMsg,
        lsp::ErrorCode::RequestFailed);
  }

  // TODO: We currently expect a single top-level operation, but this could
  // conceptually be relaxed.
  if (!llvm::hasSingleElement(parsedBlock)) {
    return llvm::make_error<lsp::LSPError>(
        "expected bytecode to contain a single top-level operation",
        lsp::ErrorCode::RequestFailed);
  }

  // Print the module to a buffer.
  lsp::MLIRConvertBytecodeResult result;
  {
    // Extract the top-level op so that aliases get printed.
    // FIXME: We should be able to enable aliases without having to do this!
    OwningOpRef<Operation *> topOp = &parsedBlock.front();
    topOp->remove();

    AsmState state(*topOp, OpPrintingFlags().enableDebugInfo().assumeVerified(),
                   /*locationMap=*/nullptr, &fallbackResourceMap);

    llvm::raw_string_ostream os(result.output);
    topOp->print(os, state);
  }
  return std::move(result);
}

llvm::Expected<lsp::MLIRConvertBytecodeResult>
lsp::MLIRServer::convertToBytecode(const URIForFile &uri) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt == impl->files.end()) {
    return llvm::make_error<lsp::LSPError>(
        "language server does not contain an entry for this source file",
        lsp::ErrorCode::RequestFailed);
  }
  return fileIt->second->convertToBytecode();
}
