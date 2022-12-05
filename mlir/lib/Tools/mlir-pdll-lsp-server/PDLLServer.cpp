//===- PDLLServer.cpp - PDLL Language Server ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PDLLServer.h"

#include "../lsp-server-support/CompilationDatabase.h"
#include "../lsp-server-support/Logging.h"
#include "../lsp-server-support/SourceMgrUtils.h"
#include "Protocol.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Tools/PDLL/AST/Context.h"
#include "mlir/Tools/PDLL/AST/Nodes.h"
#include "mlir/Tools/PDLL/AST/Types.h"
#include "mlir/Tools/PDLL/CodeGen/CPPGen.h"
#include "mlir/Tools/PDLL/CodeGen/MLIRGen.h"
#include "mlir/Tools/PDLL/ODS/Constraint.h"
#include "mlir/Tools/PDLL/ODS/Context.h"
#include "mlir/Tools/PDLL/ODS/Dialect.h"
#include "mlir/Tools/PDLL/ODS/Operation.h"
#include "mlir/Tools/PDLL/Parser/CodeComplete.h"
#include "mlir/Tools/PDLL/Parser/Parser.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace mlir;
using namespace mlir::pdll;

/// Returns a language server uri for the given source location. `mainFileURI`
/// corresponds to the uri for the main file of the source manager.
static lsp::URIForFile getURIFromLoc(llvm::SourceMgr &mgr, SMRange loc,
                                     const lsp::URIForFile &mainFileURI) {
  int bufferId = mgr.FindBufferContainingLoc(loc.Start);
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

/// Returns true if the given location is in the main file of the source
/// manager.
static bool isMainFileLoc(llvm::SourceMgr &mgr, SMRange loc) {
  return mgr.FindBufferContainingLoc(loc.Start) == mgr.getMainFileID();
}

/// Returns a language server location from the given source range.
static lsp::Location getLocationFromLoc(llvm::SourceMgr &mgr, SMRange range,
                                        const lsp::URIForFile &uri) {
  return lsp::Location(getURIFromLoc(mgr, range, uri), lsp::Range(mgr, range));
}

/// Convert the given MLIR diagnostic to the LSP form.
static Optional<lsp::Diagnostic>
getLspDiagnoticFromDiag(llvm::SourceMgr &sourceMgr, const ast::Diagnostic &diag,
                        const lsp::URIForFile &uri) {
  lsp::Diagnostic lspDiag;
  lspDiag.source = "pdll";

  // FIXME: Right now all of the diagnostics are treated as parser issues, but
  // some are parser and some are verifier.
  lspDiag.category = "Parse Error";

  // Try to grab a file location for this diagnostic.
  lsp::Location loc = getLocationFromLoc(sourceMgr, diag.getLocation(), uri);
  lspDiag.range = loc.range;

  // Skip diagnostics that weren't emitted within the main file.
  if (loc.uri != uri)
    return std::nullopt;

  // Convert the severity for the diagnostic.
  switch (diag.getSeverity()) {
  case ast::Diagnostic::Severity::DK_Note:
    llvm_unreachable("expected notes to be handled separately");
  case ast::Diagnostic::Severity::DK_Warning:
    lspDiag.severity = lsp::DiagnosticSeverity::Warning;
    break;
  case ast::Diagnostic::Severity::DK_Error:
    lspDiag.severity = lsp::DiagnosticSeverity::Error;
    break;
  case ast::Diagnostic::Severity::DK_Remark:
    lspDiag.severity = lsp::DiagnosticSeverity::Information;
    break;
  }
  lspDiag.message = diag.getMessage().str();

  // Attach any notes to the main diagnostic as related information.
  std::vector<lsp::DiagnosticRelatedInformation> relatedDiags;
  for (const ast::Diagnostic &note : diag.getNotes()) {
    relatedDiags.emplace_back(
        getLocationFromLoc(sourceMgr, note.getLocation(), uri),
        note.getMessage().str());
  }
  if (!relatedDiags.empty())
    lspDiag.relatedInformation = std::move(relatedDiags);

  return lspDiag;
}

/// Get or extract the documentation for the given decl.
static Optional<std::string> getDocumentationFor(llvm::SourceMgr &sourceMgr,
                                                 const ast::Decl *decl) {
  // If the decl already had documentation set, use it.
  if (Optional<StringRef> doc = decl->getDocComment())
    return doc->str();

  // If the decl doesn't yet have documentation, try to extract it from the
  // source file.
  return lsp::extractSourceDocComment(sourceMgr, decl->getLoc().Start);
}

//===----------------------------------------------------------------------===//
// PDLIndex
//===----------------------------------------------------------------------===//

namespace {
struct PDLIndexSymbol {
  explicit PDLIndexSymbol(const ast::Decl *definition)
      : definition(definition) {}
  explicit PDLIndexSymbol(const ods::Operation *definition)
      : definition(definition) {}

  /// Return the location of the definition of this symbol.
  SMRange getDefLoc() const {
    if (const ast::Decl *decl = definition.dyn_cast<const ast::Decl *>()) {
      const ast::Name *declName = decl->getName();
      return declName ? declName->getLoc() : decl->getLoc();
    }
    return definition.get<const ods::Operation *>()->getLoc();
  }

  /// The main definition of the symbol.
  PointerUnion<const ast::Decl *, const ods::Operation *> definition;
  /// The set of references to the symbol.
  std::vector<SMRange> references;
};

/// This class provides an index for definitions/uses within a PDL document.
/// It provides efficient lookup of a definition given an input source range.
class PDLIndex {
public:
  PDLIndex() : intervalMap(allocator) {}

  /// Initialize the index with the given ast::Module.
  void initialize(const ast::Module &module, const ods::Context &odsContext);

  /// Lookup a symbol for the given location. Returns nullptr if no symbol could
  /// be found. If provided, `overlappedRange` is set to the range that the
  /// provided `loc` overlapped with.
  const PDLIndexSymbol *lookup(SMLoc loc,
                               SMRange *overlappedRange = nullptr) const;

private:
  /// The type of interval map used to store source references. SMRange is
  /// half-open, so we also need to use a half-open interval map.
  using MapT =
      llvm::IntervalMap<const char *, const PDLIndexSymbol *,
                        llvm::IntervalMapImpl::NodeSizer<
                            const char *, const PDLIndexSymbol *>::LeafSize,
                        llvm::IntervalMapHalfOpenInfo<const char *>>;

  /// An allocator for the interval map.
  MapT::Allocator allocator;

  /// An interval map containing a corresponding definition mapped to a source
  /// interval.
  MapT intervalMap;

  /// A mapping between definitions and their corresponding symbol.
  DenseMap<const void *, std::unique_ptr<PDLIndexSymbol>> defToSymbol;
};
} // namespace

void PDLIndex::initialize(const ast::Module &module,
                          const ods::Context &odsContext) {
  auto getOrInsertDef = [&](const auto *def) -> PDLIndexSymbol * {
    auto it = defToSymbol.try_emplace(def, nullptr);
    if (it.second)
      it.first->second = std::make_unique<PDLIndexSymbol>(def);
    return &*it.first->second;
  };
  auto insertDeclRef = [&](PDLIndexSymbol *sym, SMRange refLoc,
                           bool isDef = false) {
    const char *startLoc = refLoc.Start.getPointer();
    const char *endLoc = refLoc.End.getPointer();
    if (!intervalMap.overlaps(startLoc, endLoc)) {
      intervalMap.insert(startLoc, endLoc, sym);
      if (!isDef)
        sym->references.push_back(refLoc);
    }
  };
  auto insertODSOpRef = [&](StringRef opName, SMRange refLoc) {
    const ods::Operation *odsOp = odsContext.lookupOperation(opName);
    if (!odsOp)
      return;

    PDLIndexSymbol *symbol = getOrInsertDef(odsOp);
    insertDeclRef(symbol, odsOp->getLoc(), /*isDef=*/true);
    insertDeclRef(symbol, refLoc);
  };

  module.walk([&](const ast::Node *node) {
    // Handle references to PDL decls.
    if (const auto *decl = dyn_cast<ast::OpNameDecl>(node)) {
      if (Optional<StringRef> name = decl->getName())
        insertODSOpRef(*name, decl->getLoc());
    } else if (const ast::Decl *decl = dyn_cast<ast::Decl>(node)) {
      const ast::Name *name = decl->getName();
      if (!name)
        return;
      PDLIndexSymbol *declSym = getOrInsertDef(decl);
      insertDeclRef(declSym, name->getLoc(), /*isDef=*/true);

      if (const auto *varDecl = dyn_cast<ast::VariableDecl>(decl)) {
        // Record references to any constraints.
        for (const auto &it : varDecl->getConstraints())
          insertDeclRef(getOrInsertDef(it.constraint), it.referenceLoc);
      }
    } else if (const auto *expr = dyn_cast<ast::DeclRefExpr>(node)) {
      insertDeclRef(getOrInsertDef(expr->getDecl()), expr->getLoc());
    }
  });
}

const PDLIndexSymbol *PDLIndex::lookup(SMLoc loc,
                                       SMRange *overlappedRange) const {
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
// PDLDocument
//===----------------------------------------------------------------------===//

namespace {
/// This class represents all of the information pertaining to a specific PDL
/// document.
struct PDLDocument {
  PDLDocument(const lsp::URIForFile &uri, StringRef contents,
              const std::vector<std::string> &extraDirs,
              std::vector<lsp::Diagnostic> &diagnostics);
  PDLDocument(const PDLDocument &) = delete;
  PDLDocument &operator=(const PDLDocument &) = delete;

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
  Optional<lsp::Hover> findHover(const ast::Decl *decl,
                                 const SMRange &hoverRange);
  lsp::Hover buildHoverForOpName(const ods::Operation *op,
                                 const SMRange &hoverRange);
  lsp::Hover buildHoverForVariable(const ast::VariableDecl *varDecl,
                                   const SMRange &hoverRange);
  lsp::Hover buildHoverForPattern(const ast::PatternDecl *decl,
                                  const SMRange &hoverRange);
  lsp::Hover buildHoverForCoreConstraint(const ast::CoreConstraintDecl *decl,
                                         const SMRange &hoverRange);
  template <typename T>
  lsp::Hover buildHoverForUserConstraintOrRewrite(StringRef typeName,
                                                  const T *decl,
                                                  const SMRange &hoverRange);

  //===--------------------------------------------------------------------===//
  // Document Symbols
  //===--------------------------------------------------------------------===//

  void findDocumentSymbols(std::vector<lsp::DocumentSymbol> &symbols);

  //===--------------------------------------------------------------------===//
  // Code Completion
  //===--------------------------------------------------------------------===//

  lsp::CompletionList getCodeCompletion(const lsp::URIForFile &uri,
                                        const lsp::Position &completePos);

  //===--------------------------------------------------------------------===//
  // Signature Help
  //===--------------------------------------------------------------------===//

  lsp::SignatureHelp getSignatureHelp(const lsp::URIForFile &uri,
                                      const lsp::Position &helpPos);

  //===--------------------------------------------------------------------===//
  // Inlay Hints
  //===--------------------------------------------------------------------===//

  void getInlayHints(const lsp::URIForFile &uri, const lsp::Range &range,
                     std::vector<lsp::InlayHint> &inlayHints);
  void getInlayHintsFor(const ast::VariableDecl *decl,
                        const lsp::URIForFile &uri,
                        std::vector<lsp::InlayHint> &inlayHints);
  void getInlayHintsFor(const ast::CallExpr *expr, const lsp::URIForFile &uri,
                        std::vector<lsp::InlayHint> &inlayHints);
  void getInlayHintsFor(const ast::OperationExpr *expr,
                        const lsp::URIForFile &uri,
                        std::vector<lsp::InlayHint> &inlayHints);

  /// Add a parameter hint for the given expression using `label`.
  void addParameterHintFor(std::vector<lsp::InlayHint> &inlayHints,
                           const ast::Expr *expr, StringRef label);

  //===--------------------------------------------------------------------===//
  // PDLL ViewOutput
  //===--------------------------------------------------------------------===//

  void getPDLLViewOutput(raw_ostream &os, lsp::PDLLViewOutputKind kind);

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  /// The include directories for this file.
  std::vector<std::string> includeDirs;

  /// The source manager containing the contents of the input file.
  llvm::SourceMgr sourceMgr;

  /// The ODS and AST contexts.
  ods::Context odsContext;
  ast::Context astContext;

  /// The parsed AST module, or failure if the file wasn't valid.
  FailureOr<ast::Module *> astModule;

  /// The index of the parsed module.
  PDLIndex index;

  /// The set of includes of the parsed module.
  SmallVector<lsp::SourceMgrInclude> parsedIncludes;
};
} // namespace

PDLDocument::PDLDocument(const lsp::URIForFile &uri, StringRef contents,
                         const std::vector<std::string> &extraDirs,
                         std::vector<lsp::Diagnostic> &diagnostics)
    : astContext(odsContext) {
  auto memBuffer = llvm::MemoryBuffer::getMemBufferCopy(contents, uri.file());
  if (!memBuffer) {
    lsp::Logger::error("Failed to create memory buffer for file", uri.file());
    return;
  }

  // Build the set of include directories for this file.
  llvm::SmallString<32> uriDirectory(uri.file());
  llvm::sys::path::remove_filename(uriDirectory);
  includeDirs.push_back(uriDirectory.str().str());
  includeDirs.insert(includeDirs.end(), extraDirs.begin(), extraDirs.end());

  sourceMgr.setIncludeDirs(includeDirs);
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), SMLoc());

  astContext.getDiagEngine().setHandlerFn([&](const ast::Diagnostic &diag) {
    if (auto lspDiag = getLspDiagnoticFromDiag(sourceMgr, diag, uri))
      diagnostics.push_back(std::move(*lspDiag));
  });
  astModule = parsePDLLAST(astContext, sourceMgr, /*enableDocumentation=*/true);

  // Initialize the set of parsed includes.
  lsp::gatherIncludeFiles(sourceMgr, parsedIncludes);

  // If we failed to parse the module, there is nothing left to initialize.
  if (failed(astModule))
    return;

  // Prepare the AST index with the parsed module.
  index.initialize(**astModule, odsContext);
}

//===----------------------------------------------------------------------===//
// PDLDocument: Definitions and References
//===----------------------------------------------------------------------===//

void PDLDocument::getLocationsOf(const lsp::URIForFile &uri,
                                 const lsp::Position &defPos,
                                 std::vector<lsp::Location> &locations) {
  SMLoc posLoc = defPos.getAsSMLoc(sourceMgr);
  const PDLIndexSymbol *symbol = index.lookup(posLoc);
  if (!symbol)
    return;

  locations.push_back(getLocationFromLoc(sourceMgr, symbol->getDefLoc(), uri));
}

void PDLDocument::findReferencesOf(const lsp::URIForFile &uri,
                                   const lsp::Position &pos,
                                   std::vector<lsp::Location> &references) {
  SMLoc posLoc = pos.getAsSMLoc(sourceMgr);
  const PDLIndexSymbol *symbol = index.lookup(posLoc);
  if (!symbol)
    return;

  references.push_back(getLocationFromLoc(sourceMgr, symbol->getDefLoc(), uri));
  for (SMRange refLoc : symbol->references)
    references.push_back(getLocationFromLoc(sourceMgr, refLoc, uri));
}

//===--------------------------------------------------------------------===//
// PDLDocument: Document Links
//===--------------------------------------------------------------------===//

void PDLDocument::getDocumentLinks(const lsp::URIForFile &uri,
                                   std::vector<lsp::DocumentLink> &links) {
  for (const lsp::SourceMgrInclude &include : parsedIncludes)
    links.emplace_back(include.range, include.uri);
}

//===----------------------------------------------------------------------===//
// PDLDocument: Hover
//===----------------------------------------------------------------------===//

Optional<lsp::Hover> PDLDocument::findHover(const lsp::URIForFile &uri,
                                            const lsp::Position &hoverPos) {
  SMLoc posLoc = hoverPos.getAsSMLoc(sourceMgr);

  // Check for a reference to an include.
  for (const lsp::SourceMgrInclude &include : parsedIncludes)
    if (include.range.contains(hoverPos))
      return include.buildHover();

  // Find the symbol at the given location.
  SMRange hoverRange;
  const PDLIndexSymbol *symbol = index.lookup(posLoc, &hoverRange);
  if (!symbol)
    return std::nullopt;

  // Add hover for operation names.
  if (const auto *op = symbol->definition.dyn_cast<const ods::Operation *>())
    return buildHoverForOpName(op, hoverRange);
  const auto *decl = symbol->definition.get<const ast::Decl *>();
  return findHover(decl, hoverRange);
}

Optional<lsp::Hover> PDLDocument::findHover(const ast::Decl *decl,
                                            const SMRange &hoverRange) {
  // Add hover for variables.
  if (const auto *varDecl = dyn_cast<ast::VariableDecl>(decl))
    return buildHoverForVariable(varDecl, hoverRange);

  // Add hover for patterns.
  if (const auto *patternDecl = dyn_cast<ast::PatternDecl>(decl))
    return buildHoverForPattern(patternDecl, hoverRange);

  // Add hover for core constraints.
  if (const auto *cst = dyn_cast<ast::CoreConstraintDecl>(decl))
    return buildHoverForCoreConstraint(cst, hoverRange);

  // Add hover for user constraints.
  if (const auto *cst = dyn_cast<ast::UserConstraintDecl>(decl))
    return buildHoverForUserConstraintOrRewrite("Constraint", cst, hoverRange);

  // Add hover for user rewrites.
  if (const auto *rewrite = dyn_cast<ast::UserRewriteDecl>(decl))
    return buildHoverForUserConstraintOrRewrite("Rewrite", rewrite, hoverRange);

  return std::nullopt;
}

lsp::Hover PDLDocument::buildHoverForOpName(const ods::Operation *op,
                                            const SMRange &hoverRange) {
  lsp::Hover hover(lsp::Range(sourceMgr, hoverRange));
  {
    llvm::raw_string_ostream hoverOS(hover.contents.value);
    hoverOS << "**OpName**: `" << op->getName() << "`\n***\n"
            << op->getSummary() << "\n***\n"
            << op->getDescription();
  }
  return hover;
}

lsp::Hover PDLDocument::buildHoverForVariable(const ast::VariableDecl *varDecl,
                                              const SMRange &hoverRange) {
  lsp::Hover hover(lsp::Range(sourceMgr, hoverRange));
  {
    llvm::raw_string_ostream hoverOS(hover.contents.value);
    hoverOS << "**Variable**: `" << varDecl->getName().getName() << "`\n***\n"
            << "Type: `" << varDecl->getType() << "`\n";
  }
  return hover;
}

lsp::Hover PDLDocument::buildHoverForPattern(const ast::PatternDecl *decl,
                                             const SMRange &hoverRange) {
  lsp::Hover hover(lsp::Range(sourceMgr, hoverRange));
  {
    llvm::raw_string_ostream hoverOS(hover.contents.value);
    hoverOS << "**Pattern**";
    if (const ast::Name *name = decl->getName())
      hoverOS << ": `" << name->getName() << "`";
    hoverOS << "\n***\n";
    if (Optional<uint16_t> benefit = decl->getBenefit())
      hoverOS << "Benefit: " << *benefit << "\n";
    if (decl->hasBoundedRewriteRecursion())
      hoverOS << "HasBoundedRewriteRecursion\n";
    hoverOS << "RootOp: `"
            << decl->getRootRewriteStmt()->getRootOpExpr()->getType() << "`\n";

    // Format the documentation for the decl.
    if (Optional<std::string> doc = getDocumentationFor(sourceMgr, decl))
      hoverOS << "\n" << *doc << "\n";
  }
  return hover;
}

lsp::Hover
PDLDocument::buildHoverForCoreConstraint(const ast::CoreConstraintDecl *decl,
                                         const SMRange &hoverRange) {
  lsp::Hover hover(lsp::Range(sourceMgr, hoverRange));
  {
    llvm::raw_string_ostream hoverOS(hover.contents.value);
    hoverOS << "**Constraint**: `";
    TypeSwitch<const ast::Decl *>(decl)
        .Case([&](const ast::AttrConstraintDecl *) { hoverOS << "Attr"; })
        .Case([&](const ast::OpConstraintDecl *opCst) {
          hoverOS << "Op";
          if (Optional<StringRef> name = opCst->getName())
            hoverOS << "<" << name << ">";
        })
        .Case([&](const ast::TypeConstraintDecl *) { hoverOS << "Type"; })
        .Case([&](const ast::TypeRangeConstraintDecl *) {
          hoverOS << "TypeRange";
        })
        .Case([&](const ast::ValueConstraintDecl *) { hoverOS << "Value"; })
        .Case([&](const ast::ValueRangeConstraintDecl *) {
          hoverOS << "ValueRange";
        });
    hoverOS << "`\n";
  }
  return hover;
}

template <typename T>
lsp::Hover PDLDocument::buildHoverForUserConstraintOrRewrite(
    StringRef typeName, const T *decl, const SMRange &hoverRange) {
  lsp::Hover hover(lsp::Range(sourceMgr, hoverRange));
  {
    llvm::raw_string_ostream hoverOS(hover.contents.value);
    hoverOS << "**" << typeName << "**: `" << decl->getName().getName()
            << "`\n***\n";
    ArrayRef<ast::VariableDecl *> inputs = decl->getInputs();
    if (!inputs.empty()) {
      hoverOS << "Parameters:\n";
      for (const ast::VariableDecl *input : inputs)
        hoverOS << "* " << input->getName().getName() << ": `"
                << input->getType() << "`\n";
      hoverOS << "***\n";
    }
    ast::Type resultType = decl->getResultType();
    if (auto resultTupleTy = resultType.dyn_cast<ast::TupleType>()) {
      if (!resultTupleTy.empty()) {
        hoverOS << "Results:\n";
        for (auto it : llvm::zip(resultTupleTy.getElementNames(),
                                 resultTupleTy.getElementTypes())) {
          StringRef name = std::get<0>(it);
          hoverOS << "* " << (name.empty() ? "" : (name + ": ")) << "`"
                  << std::get<1>(it) << "`\n";
        }
        hoverOS << "***\n";
      }
    } else {
      hoverOS << "Results:\n* `" << resultType << "`\n";
      hoverOS << "***\n";
    }

    // Format the documentation for the decl.
    if (Optional<std::string> doc = getDocumentationFor(sourceMgr, decl))
      hoverOS << "\n" << *doc << "\n";
  }
  return hover;
}

//===----------------------------------------------------------------------===//
// PDLDocument: Document Symbols
//===----------------------------------------------------------------------===//

void PDLDocument::findDocumentSymbols(
    std::vector<lsp::DocumentSymbol> &symbols) {
  if (failed(astModule))
    return;

  for (const ast::Decl *decl : (*astModule)->getChildren()) {
    if (!isMainFileLoc(sourceMgr, decl->getLoc()))
      continue;

    if (const auto *patternDecl = dyn_cast<ast::PatternDecl>(decl)) {
      const ast::Name *name = patternDecl->getName();

      SMRange nameLoc = name ? name->getLoc() : patternDecl->getLoc();
      SMRange bodyLoc(nameLoc.Start, patternDecl->getBody()->getLoc().End);

      symbols.emplace_back(
          name ? name->getName() : "<pattern>", lsp::SymbolKind::Class,
          lsp::Range(sourceMgr, bodyLoc), lsp::Range(sourceMgr, nameLoc));
    } else if (const auto *cDecl = dyn_cast<ast::UserConstraintDecl>(decl)) {
      // TODO: Add source information for the code block body.
      SMRange nameLoc = cDecl->getName().getLoc();
      SMRange bodyLoc = nameLoc;

      symbols.emplace_back(
          cDecl->getName().getName(), lsp::SymbolKind::Function,
          lsp::Range(sourceMgr, bodyLoc), lsp::Range(sourceMgr, nameLoc));
    } else if (const auto *cDecl = dyn_cast<ast::UserRewriteDecl>(decl)) {
      // TODO: Add source information for the code block body.
      SMRange nameLoc = cDecl->getName().getLoc();
      SMRange bodyLoc = nameLoc;

      symbols.emplace_back(
          cDecl->getName().getName(), lsp::SymbolKind::Function,
          lsp::Range(sourceMgr, bodyLoc), lsp::Range(sourceMgr, nameLoc));
    }
  }
}

//===----------------------------------------------------------------------===//
// PDLDocument: Code Completion
//===----------------------------------------------------------------------===//

namespace {
class LSPCodeCompleteContext : public CodeCompleteContext {
public:
  LSPCodeCompleteContext(SMLoc completeLoc, llvm::SourceMgr &sourceMgr,
                         lsp::CompletionList &completionList,
                         ods::Context &odsContext,
                         ArrayRef<std::string> includeDirs)
      : CodeCompleteContext(completeLoc), sourceMgr(sourceMgr),
        completionList(completionList), odsContext(odsContext),
        includeDirs(includeDirs) {}

  void codeCompleteTupleMemberAccess(ast::TupleType tupleType) final {
    ArrayRef<ast::Type> elementTypes = tupleType.getElementTypes();
    ArrayRef<StringRef> elementNames = tupleType.getElementNames();
    for (unsigned i = 0, e = tupleType.size(); i < e; ++i) {
      // Push back a completion item that uses the result index.
      lsp::CompletionItem item;
      item.label = llvm::formatv("{0} (field #{0})", i).str();
      item.insertText = Twine(i).str();
      item.filterText = item.sortText = item.insertText;
      item.kind = lsp::CompletionItemKind::Field;
      item.detail = llvm::formatv("{0}: {1}", i, elementTypes[i]);
      item.insertTextFormat = lsp::InsertTextFormat::PlainText;
      completionList.items.emplace_back(item);

      // If the element has a name, push back a completion item with that name.
      if (!elementNames[i].empty()) {
        item.label =
            llvm::formatv("{1} (field #{0})", i, elementNames[i]).str();
        item.filterText = item.label;
        item.insertText = elementNames[i].str();
        completionList.items.emplace_back(item);
      }
    }
  }

  void codeCompleteOperationMemberAccess(ast::OperationType opType) final {
    const ods::Operation *odsOp = opType.getODSOperation();
    if (!odsOp)
      return;

    ArrayRef<ods::OperandOrResult> results = odsOp->getResults();
    for (const auto &it : llvm::enumerate(results)) {
      const ods::OperandOrResult &result = it.value();
      const ods::TypeConstraint &constraint = result.getConstraint();

      // Push back a completion item that uses the result index.
      lsp::CompletionItem item;
      item.label = llvm::formatv("{0} (field #{0})", it.index()).str();
      item.insertText = Twine(it.index()).str();
      item.filterText = item.sortText = item.insertText;
      item.kind = lsp::CompletionItemKind::Field;
      switch (result.getVariableLengthKind()) {
      case ods::VariableLengthKind::Single:
        item.detail = llvm::formatv("{0}: Value", it.index()).str();
        break;
      case ods::VariableLengthKind::Optional:
        item.detail = llvm::formatv("{0}: Value?", it.index()).str();
        break;
      case ods::VariableLengthKind::Variadic:
        item.detail = llvm::formatv("{0}: ValueRange", it.index()).str();
        break;
      }
      item.documentation = lsp::MarkupContent{
          lsp::MarkupKind::Markdown,
          llvm::formatv("{0}\n\n```c++\n{1}\n```\n", constraint.getSummary(),
                        constraint.getCppClass())
              .str()};
      item.insertTextFormat = lsp::InsertTextFormat::PlainText;
      completionList.items.emplace_back(item);

      // If the result has a name, push back a completion item with the result
      // name.
      if (!result.getName().empty()) {
        item.label =
            llvm::formatv("{1} (field #{0})", it.index(), result.getName())
                .str();
        item.filterText = item.label;
        item.insertText = result.getName().str();
        completionList.items.emplace_back(item);
      }
    }
  }

  void codeCompleteOperationAttributeName(StringRef opName) final {
    const ods::Operation *odsOp = odsContext.lookupOperation(opName);
    if (!odsOp)
      return;

    for (const ods::Attribute &attr : odsOp->getAttributes()) {
      const ods::AttributeConstraint &constraint = attr.getConstraint();

      lsp::CompletionItem item;
      item.label = attr.getName().str();
      item.kind = lsp::CompletionItemKind::Field;
      item.detail = attr.isOptional() ? "optional" : "";
      item.documentation = lsp::MarkupContent{
          lsp::MarkupKind::Markdown,
          llvm::formatv("{0}\n\n```c++\n{1}\n```\n", constraint.getSummary(),
                        constraint.getCppClass())
              .str()};
      item.insertTextFormat = lsp::InsertTextFormat::PlainText;
      completionList.items.emplace_back(item);
    }
  }

  void codeCompleteConstraintName(ast::Type currentType,
                                  bool allowInlineTypeConstraints,
                                  const ast::DeclScope *scope) final {
    auto addCoreConstraint = [&](StringRef constraint, StringRef mlirType,
                                 StringRef snippetText = "") {
      lsp::CompletionItem item;
      item.label = constraint.str();
      item.kind = lsp::CompletionItemKind::Class;
      item.detail = (constraint + " constraint").str();
      item.documentation = lsp::MarkupContent{
          lsp::MarkupKind::Markdown,
          ("A single entity core constraint of type `" + mlirType + "`").str()};
      item.sortText = "0";
      item.insertText = snippetText.str();
      item.insertTextFormat = snippetText.empty()
                                  ? lsp::InsertTextFormat::PlainText
                                  : lsp::InsertTextFormat::Snippet;
      completionList.items.emplace_back(item);
    };

    // Insert completions for the core constraints. Some core constraints have
    // additional characteristics, so we may add then even if a type has been
    // inferred.
    if (!currentType) {
      addCoreConstraint("Attr", "mlir::Attribute");
      addCoreConstraint("Op", "mlir::Operation *");
      addCoreConstraint("Value", "mlir::Value");
      addCoreConstraint("ValueRange", "mlir::ValueRange");
      addCoreConstraint("Type", "mlir::Type");
      addCoreConstraint("TypeRange", "mlir::TypeRange");
    }
    if (allowInlineTypeConstraints) {
      /// Attr<Type>.
      if (!currentType || currentType.isa<ast::AttributeType>())
        addCoreConstraint("Attr<type>", "mlir::Attribute", "Attr<$1>");
      /// Value<Type>.
      if (!currentType || currentType.isa<ast::ValueType>())
        addCoreConstraint("Value<type>", "mlir::Value", "Value<$1>");
      /// ValueRange<TypeRange>.
      if (!currentType || currentType.isa<ast::ValueRangeType>())
        addCoreConstraint("ValueRange<type>", "mlir::ValueRange",
                          "ValueRange<$1>");
    }

    // If a scope was provided, check it for potential constraints.
    while (scope) {
      for (const ast::Decl *decl : scope->getDecls()) {
        if (const auto *cst = dyn_cast<ast::UserConstraintDecl>(decl)) {
          lsp::CompletionItem item;
          item.label = cst->getName().getName().str();
          item.kind = lsp::CompletionItemKind::Interface;
          item.sortText = "2_" + item.label;

          // Skip constraints that are not single-arg. We currently only
          // complete variable constraints.
          if (cst->getInputs().size() != 1)
            continue;

          // Ensure the input type matched the given type.
          ast::Type constraintType = cst->getInputs()[0]->getType();
          if (currentType && !currentType.refineWith(constraintType))
            continue;

          // Format the constraint signature.
          {
            llvm::raw_string_ostream strOS(item.detail);
            strOS << "(";
            llvm::interleaveComma(
                cst->getInputs(), strOS, [&](const ast::VariableDecl *var) {
                  strOS << var->getName().getName() << ": " << var->getType();
                });
            strOS << ") -> " << cst->getResultType();
          }

          // Format the documentation for the constraint.
          if (Optional<std::string> doc = getDocumentationFor(sourceMgr, cst)) {
            item.documentation =
                lsp::MarkupContent{lsp::MarkupKind::Markdown, std::move(*doc)};
          }

          completionList.items.emplace_back(item);
        }
      }

      scope = scope->getParentScope();
    }
  }

  void codeCompleteDialectName() final {
    // Code complete known dialects.
    for (const ods::Dialect &dialect : odsContext.getDialects()) {
      lsp::CompletionItem item;
      item.label = dialect.getName().str();
      item.kind = lsp::CompletionItemKind::Class;
      item.insertTextFormat = lsp::InsertTextFormat::PlainText;
      completionList.items.emplace_back(item);
    }
  }

  void codeCompleteOperationName(StringRef dialectName) final {
    const ods::Dialect *dialect = odsContext.lookupDialect(dialectName);
    if (!dialect)
      return;

    for (const auto &it : dialect->getOperations()) {
      const ods::Operation &op = *it.second;

      lsp::CompletionItem item;
      item.label = op.getName().drop_front(dialectName.size() + 1).str();
      item.kind = lsp::CompletionItemKind::Field;
      item.insertTextFormat = lsp::InsertTextFormat::PlainText;
      completionList.items.emplace_back(item);
    }
  }

  void codeCompletePatternMetadata() final {
    auto addSimpleConstraint = [&](StringRef constraint, StringRef desc,
                                   StringRef snippetText = "") {
      lsp::CompletionItem item;
      item.label = constraint.str();
      item.kind = lsp::CompletionItemKind::Class;
      item.detail = "pattern metadata";
      item.documentation =
          lsp::MarkupContent{lsp::MarkupKind::Markdown, desc.str()};
      item.insertText = snippetText.str();
      item.insertTextFormat = snippetText.empty()
                                  ? lsp::InsertTextFormat::PlainText
                                  : lsp::InsertTextFormat::Snippet;
      completionList.items.emplace_back(item);
    };

    addSimpleConstraint("benefit", "The `benefit` of matching the pattern.",
                        "benefit($1)");
    addSimpleConstraint("recursion",
                        "The pattern properly handles recursive application.");
  }

  void codeCompleteIncludeFilename(StringRef curPath) final {
    // Normalize the path to allow for interacting with the file system
    // utilities.
    SmallString<128> nativeRelDir(llvm::sys::path::convert_to_slash(curPath));
    llvm::sys::path::native(nativeRelDir);

    // Set of already included completion paths.
    StringSet<> seenResults;

    // Functor used to add a single include completion item.
    auto addIncludeCompletion = [&](StringRef path, bool isDirectory) {
      lsp::CompletionItem item;
      item.label = path.str();
      item.kind = isDirectory ? lsp::CompletionItemKind::Folder
                              : lsp::CompletionItemKind::File;
      if (seenResults.insert(item.label).second)
        completionList.items.emplace_back(item);
    };

    // Process the include directories for this file, adding any potential
    // nested include files or directories.
    for (StringRef includeDir : includeDirs) {
      llvm::SmallString<128> dir = includeDir;
      if (!nativeRelDir.empty())
        llvm::sys::path::append(dir, nativeRelDir);

      std::error_code errorCode;
      for (auto it = llvm::sys::fs::directory_iterator(dir, errorCode),
                e = llvm::sys::fs::directory_iterator();
           !errorCode && it != e; it.increment(errorCode)) {
        StringRef filename = llvm::sys::path::filename(it->path());

        // To know whether a symlink should be treated as file or a directory,
        // we have to stat it. This should be cheap enough as there shouldn't be
        // many symlinks.
        llvm::sys::fs::file_type fileType = it->type();
        if (fileType == llvm::sys::fs::file_type::symlink_file) {
          if (auto fileStatus = it->status())
            fileType = fileStatus->type();
        }

        switch (fileType) {
        case llvm::sys::fs::file_type::directory_file:
          addIncludeCompletion(filename, /*isDirectory=*/true);
          break;
        case llvm::sys::fs::file_type::regular_file: {
          // Only consider concrete files that can actually be included by PDLL.
          if (filename.endswith(".pdll") || filename.endswith(".td"))
            addIncludeCompletion(filename, /*isDirectory=*/false);
          break;
        }
        default:
          break;
        }
      }
    }

    // Sort the completion results to make sure the output is deterministic in
    // the face of different iteration schemes for different platforms.
    llvm::sort(completionList.items, [](const lsp::CompletionItem &lhs,
                                        const lsp::CompletionItem &rhs) {
      return lhs.label < rhs.label;
    });
  }

private:
  llvm::SourceMgr &sourceMgr;
  lsp::CompletionList &completionList;
  ods::Context &odsContext;
  ArrayRef<std::string> includeDirs;
};
} // namespace

lsp::CompletionList
PDLDocument::getCodeCompletion(const lsp::URIForFile &uri,
                               const lsp::Position &completePos) {
  SMLoc posLoc = completePos.getAsSMLoc(sourceMgr);
  if (!posLoc.isValid())
    return lsp::CompletionList();

  // To perform code completion, we run another parse of the module with the
  // code completion context provided.
  ods::Context tmpODSContext;
  lsp::CompletionList completionList;
  LSPCodeCompleteContext lspCompleteContext(posLoc, sourceMgr, completionList,
                                            tmpODSContext,
                                            sourceMgr.getIncludeDirs());

  ast::Context tmpContext(tmpODSContext);
  (void)parsePDLLAST(tmpContext, sourceMgr, /*enableDocumentation=*/true,
                     &lspCompleteContext);

  return completionList;
}

//===----------------------------------------------------------------------===//
// PDLDocument: Signature Help
//===----------------------------------------------------------------------===//

namespace {
class LSPSignatureHelpContext : public CodeCompleteContext {
public:
  LSPSignatureHelpContext(SMLoc completeLoc, llvm::SourceMgr &sourceMgr,
                          lsp::SignatureHelp &signatureHelp,
                          ods::Context &odsContext)
      : CodeCompleteContext(completeLoc), sourceMgr(sourceMgr),
        signatureHelp(signatureHelp), odsContext(odsContext) {}

  void codeCompleteCallSignature(const ast::CallableDecl *callable,
                                 unsigned currentNumArgs) final {
    signatureHelp.activeParameter = currentNumArgs;

    lsp::SignatureInformation signatureInfo;
    {
      llvm::raw_string_ostream strOS(signatureInfo.label);
      strOS << callable->getName()->getName() << "(";
      auto formatParamFn = [&](const ast::VariableDecl *var) {
        unsigned paramStart = strOS.str().size();
        strOS << var->getName().getName() << ": " << var->getType();
        unsigned paramEnd = strOS.str().size();
        signatureInfo.parameters.emplace_back(lsp::ParameterInformation{
            StringRef(strOS.str()).slice(paramStart, paramEnd).str(),
            std::make_pair(paramStart, paramEnd), /*paramDoc*/ std::string()});
      };
      llvm::interleaveComma(callable->getInputs(), strOS, formatParamFn);
      strOS << ") -> " << callable->getResultType();
    }

    // Format the documentation for the callable.
    if (Optional<std::string> doc = getDocumentationFor(sourceMgr, callable))
      signatureInfo.documentation = std::move(*doc);

    signatureHelp.signatures.emplace_back(std::move(signatureInfo));
  }

  void
  codeCompleteOperationOperandsSignature(Optional<StringRef> opName,
                                         unsigned currentNumOperands) final {
    const ods::Operation *odsOp =
        opName ? odsContext.lookupOperation(*opName) : nullptr;
    codeCompleteOperationOperandOrResultSignature(
        opName, odsOp, odsOp ? odsOp->getOperands() : std::nullopt,
        currentNumOperands, "operand", "Value");
  }

  void codeCompleteOperationResultsSignature(Optional<StringRef> opName,
                                             unsigned currentNumResults) final {
    const ods::Operation *odsOp =
        opName ? odsContext.lookupOperation(*opName) : nullptr;
    codeCompleteOperationOperandOrResultSignature(
        opName, odsOp, odsOp ? odsOp->getResults() : std::nullopt,
        currentNumResults, "result", "Type");
  }

  void codeCompleteOperationOperandOrResultSignature(
      Optional<StringRef> opName, const ods::Operation *odsOp,
      ArrayRef<ods::OperandOrResult> values, unsigned currentValue,
      StringRef label, StringRef dataType) {
    signatureHelp.activeParameter = currentValue;

    // If we have ODS information for the operation, add in the ODS signature
    // for the operation. We also verify that the current number of values is
    // not more than what is defined in ODS, as this will result in an error
    // anyways.
    if (odsOp && currentValue < values.size()) {
      lsp::SignatureInformation signatureInfo;

      // Build the signature label.
      {
        llvm::raw_string_ostream strOS(signatureInfo.label);
        strOS << "(";
        auto formatFn = [&](const ods::OperandOrResult &value) {
          unsigned paramStart = strOS.str().size();

          strOS << value.getName() << ": ";

          StringRef constraintDoc = value.getConstraint().getSummary();
          std::string paramDoc;
          switch (value.getVariableLengthKind()) {
          case ods::VariableLengthKind::Single:
            strOS << dataType;
            paramDoc = constraintDoc.str();
            break;
          case ods::VariableLengthKind::Optional:
            strOS << dataType << "?";
            paramDoc = ("optional: " + constraintDoc).str();
            break;
          case ods::VariableLengthKind::Variadic:
            strOS << dataType << "Range";
            paramDoc = ("variadic: " + constraintDoc).str();
            break;
          }

          unsigned paramEnd = strOS.str().size();
          signatureInfo.parameters.emplace_back(lsp::ParameterInformation{
              StringRef(strOS.str()).slice(paramStart, paramEnd).str(),
              std::make_pair(paramStart, paramEnd), paramDoc});
        };
        llvm::interleaveComma(values, strOS, formatFn);
        strOS << ")";
      }
      signatureInfo.documentation =
          llvm::formatv("`op<{0}>` ODS {1} specification", *opName, label)
              .str();
      signatureHelp.signatures.emplace_back(std::move(signatureInfo));
    }

    // If there aren't any arguments yet, we also add the generic signature.
    if (currentValue == 0 && (!odsOp || !values.empty())) {
      lsp::SignatureInformation signatureInfo;
      signatureInfo.label =
          llvm::formatv("(<{0}s>: {1}Range)", label, dataType).str();
      signatureInfo.documentation =
          ("Generic operation " + label + " specification").str();
      signatureInfo.parameters.emplace_back(lsp::ParameterInformation{
          StringRef(signatureInfo.label).drop_front().drop_back().str(),
          std::pair<unsigned, unsigned>(1, signatureInfo.label.size() - 1),
          ("All of the " + label + "s of the operation.").str()});
      signatureHelp.signatures.emplace_back(std::move(signatureInfo));
    }
  }

private:
  llvm::SourceMgr &sourceMgr;
  lsp::SignatureHelp &signatureHelp;
  ods::Context &odsContext;
};
} // namespace

lsp::SignatureHelp PDLDocument::getSignatureHelp(const lsp::URIForFile &uri,
                                                 const lsp::Position &helpPos) {
  SMLoc posLoc = helpPos.getAsSMLoc(sourceMgr);
  if (!posLoc.isValid())
    return lsp::SignatureHelp();

  // To perform code completion, we run another parse of the module with the
  // code completion context provided.
  ods::Context tmpODSContext;
  lsp::SignatureHelp signatureHelp;
  LSPSignatureHelpContext completeContext(posLoc, sourceMgr, signatureHelp,
                                          tmpODSContext);

  ast::Context tmpContext(tmpODSContext);
  (void)parsePDLLAST(tmpContext, sourceMgr, /*enableDocumentation=*/true,
                     &completeContext);

  return signatureHelp;
}

//===----------------------------------------------------------------------===//
// PDLDocument: Inlay Hints
//===----------------------------------------------------------------------===//

/// Returns true if the given name should be added as a hint for `expr`.
static bool shouldAddHintFor(const ast::Expr *expr, StringRef name) {
  if (name.empty())
    return false;

  // If the argument is a reference of the same name, don't add it as a hint.
  if (auto *ref = dyn_cast<ast::DeclRefExpr>(expr)) {
    const ast::Name *declName = ref->getDecl()->getName();
    if (declName && declName->getName() == name)
      return false;
  }

  return true;
}

void PDLDocument::getInlayHints(const lsp::URIForFile &uri,
                                const lsp::Range &range,
                                std::vector<lsp::InlayHint> &inlayHints) {
  if (failed(astModule))
    return;
  SMRange rangeLoc = range.getAsSMRange(sourceMgr);
  if (!rangeLoc.isValid())
    return;
  (*astModule)->walk([&](const ast::Node *node) {
    SMRange loc = node->getLoc();

    // Check that the location of this node is within the input range.
    if (!lsp::contains(rangeLoc, loc.Start) &&
        !lsp::contains(rangeLoc, loc.End))
      return;

    // Handle hints for various types of nodes.
    llvm::TypeSwitch<const ast::Node *>(node)
        .Case<ast::VariableDecl, ast::CallExpr, ast::OperationExpr>(
            [&](const auto *node) {
              this->getInlayHintsFor(node, uri, inlayHints);
            });
  });
}

void PDLDocument::getInlayHintsFor(const ast::VariableDecl *decl,
                                   const lsp::URIForFile &uri,
                                   std::vector<lsp::InlayHint> &inlayHints) {
  // Check to see if the variable has a constraint list, if it does we don't
  // provide initializer hints.
  if (!decl->getConstraints().empty())
    return;

  // Check to see if the variable has an initializer.
  if (const ast::Expr *expr = decl->getInitExpr()) {
    // Don't add hints for operation expression initialized variables given that
    // the type of the variable is easily inferred by the expression operation
    // name.
    if (isa<ast::OperationExpr>(expr))
      return;
  }

  lsp::InlayHint hint(lsp::InlayHintKind::Type,
                      lsp::Position(sourceMgr, decl->getLoc().End));
  {
    llvm::raw_string_ostream labelOS(hint.label);
    labelOS << ": " << decl->getType();
  }

  inlayHints.emplace_back(std::move(hint));
}

void PDLDocument::getInlayHintsFor(const ast::CallExpr *expr,
                                   const lsp::URIForFile &uri,
                                   std::vector<lsp::InlayHint> &inlayHints) {
  // Try to extract the callable of this call.
  const auto *callableRef = dyn_cast<ast::DeclRefExpr>(expr->getCallableExpr());
  const auto *callable =
      callableRef ? dyn_cast<ast::CallableDecl>(callableRef->getDecl())
                  : nullptr;
  if (!callable)
    return;

  // Add hints for the arguments to the call.
  for (const auto &it : llvm::zip(expr->getArguments(), callable->getInputs()))
    addParameterHintFor(inlayHints, std::get<0>(it),
                        std::get<1>(it)->getName().getName());
}

void PDLDocument::getInlayHintsFor(const ast::OperationExpr *expr,
                                   const lsp::URIForFile &uri,
                                   std::vector<lsp::InlayHint> &inlayHints) {
  // Check for ODS information.
  ast::OperationType opType = expr->getType().dyn_cast<ast::OperationType>();
  const auto *odsOp = opType ? opType.getODSOperation() : nullptr;

  auto addOpHint = [&](const ast::Expr *valueExpr, StringRef label) {
    // If the value expression used the same location as the operation, don't
    // add a hint. This expression was materialized during parsing.
    if (expr->getLoc().Start == valueExpr->getLoc().Start)
      return;
    addParameterHintFor(inlayHints, valueExpr, label);
  };

  // Functor used to process hints for the operands and results of the
  // operation. They effectively have the same format, and thus can be processed
  // using the same logic.
  auto addOperandOrResultHints = [&](ArrayRef<ast::Expr *> values,
                                     ArrayRef<ods::OperandOrResult> odsValues,
                                     StringRef allValuesName) {
    if (values.empty())
      return;

    // The values should either map to a single range, or be equivalent to the
    // ODS values.
    if (values.size() != odsValues.size()) {
      // Handle the case of a single element that covers the full range.
      if (values.size() == 1)
        return addOpHint(values.front(), allValuesName);
      return;
    }

    for (const auto &it : llvm::zip(values, odsValues))
      addOpHint(std::get<0>(it), std::get<1>(it).getName());
  };

  // Add hints for the operands and results of the operation.
  addOperandOrResultHints(expr->getOperands(),
                          odsOp ? odsOp->getOperands()
                                : ArrayRef<ods::OperandOrResult>(),
                          "operands");
  addOperandOrResultHints(expr->getResultTypes(),
                          odsOp ? odsOp->getResults()
                                : ArrayRef<ods::OperandOrResult>(),
                          "results");
}

void PDLDocument::addParameterHintFor(std::vector<lsp::InlayHint> &inlayHints,
                                      const ast::Expr *expr, StringRef label) {
  if (!shouldAddHintFor(expr, label))
    return;

  lsp::InlayHint hint(lsp::InlayHintKind::Parameter,
                      lsp::Position(sourceMgr, expr->getLoc().Start));
  hint.label = (label + ":").str();
  hint.paddingRight = true;
  inlayHints.emplace_back(std::move(hint));
}

//===----------------------------------------------------------------------===//
// PDLL ViewOutput
//===----------------------------------------------------------------------===//

void PDLDocument::getPDLLViewOutput(raw_ostream &os,
                                    lsp::PDLLViewOutputKind kind) {
  if (failed(astModule))
    return;
  if (kind == lsp::PDLLViewOutputKind::AST) {
    (*astModule)->print(os);
    return;
  }

  // Generate the MLIR for the ast module. We also capture diagnostics here to
  // show to the user, which may be useful if PDLL isn't capturing constraints
  // expected by PDL.
  MLIRContext mlirContext;
  SourceMgrDiagnosticHandler diagHandler(sourceMgr, &mlirContext, os);
  OwningOpRef<ModuleOp> pdlModule =
      codegenPDLLToMLIR(&mlirContext, astContext, sourceMgr, **astModule);
  if (!pdlModule)
    return;
  if (kind == lsp::PDLLViewOutputKind::MLIR) {
    pdlModule->print(os, OpPrintingFlags().enableDebugInfo());
    return;
  }

  // Otherwise, generate the output for C++.
  assert(kind == lsp::PDLLViewOutputKind::CPP &&
         "unexpected PDLLViewOutputKind");
  codegenPDLLToCPP(**astModule, *pdlModule, os);
}

//===----------------------------------------------------------------------===//
// PDLTextFileChunk
//===----------------------------------------------------------------------===//

namespace {
/// This class represents a single chunk of an PDL text file.
struct PDLTextFileChunk {
  PDLTextFileChunk(uint64_t lineOffset, const lsp::URIForFile &uri,
                   StringRef contents,
                   const std::vector<std::string> &extraDirs,
                   std::vector<lsp::Diagnostic> &diagnostics)
      : lineOffset(lineOffset),
        document(uri, contents, extraDirs, diagnostics) {}

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
  PDLDocument document;
};
} // namespace

//===----------------------------------------------------------------------===//
// PDLTextFile
//===----------------------------------------------------------------------===//

namespace {
/// This class represents a text file containing one or more PDL documents.
class PDLTextFile {
public:
  PDLTextFile(const lsp::URIForFile &uri, StringRef fileContents,
              int64_t version, const std::vector<std::string> &extraDirs,
              std::vector<lsp::Diagnostic> &diagnostics);

  /// Return the current version of this text file.
  int64_t getVersion() const { return version; }

  /// Update the file to the new version using the provided set of content
  /// changes. Returns failure if the update was unsuccessful.
  LogicalResult update(const lsp::URIForFile &uri, int64_t newVersion,
                       ArrayRef<lsp::TextDocumentContentChangeEvent> changes,
                       std::vector<lsp::Diagnostic> &diagnostics);

  //===--------------------------------------------------------------------===//
  // LSP Queries
  //===--------------------------------------------------------------------===//

  void getLocationsOf(const lsp::URIForFile &uri, lsp::Position defPos,
                      std::vector<lsp::Location> &locations);
  void findReferencesOf(const lsp::URIForFile &uri, lsp::Position pos,
                        std::vector<lsp::Location> &references);
  void getDocumentLinks(const lsp::URIForFile &uri,
                        std::vector<lsp::DocumentLink> &links);
  Optional<lsp::Hover> findHover(const lsp::URIForFile &uri,
                                 lsp::Position hoverPos);
  void findDocumentSymbols(std::vector<lsp::DocumentSymbol> &symbols);
  lsp::CompletionList getCodeCompletion(const lsp::URIForFile &uri,
                                        lsp::Position completePos);
  lsp::SignatureHelp getSignatureHelp(const lsp::URIForFile &uri,
                                      lsp::Position helpPos);
  void getInlayHints(const lsp::URIForFile &uri, lsp::Range range,
                     std::vector<lsp::InlayHint> &inlayHints);
  lsp::PDLLViewOutputResult getPDLLViewOutput(lsp::PDLLViewOutputKind kind);

private:
  using ChunkIterator = llvm::pointee_iterator<
      std::vector<std::unique_ptr<PDLTextFileChunk>>::iterator>;

  /// Initialize the text file from the given file contents.
  void initialize(const lsp::URIForFile &uri, int64_t newVersion,
                  std::vector<lsp::Diagnostic> &diagnostics);

  /// Find the PDL document that contains the given position, and update the
  /// position to be anchored at the start of the found chunk instead of the
  /// beginning of the file.
  ChunkIterator getChunkItFor(lsp::Position &pos);
  PDLTextFileChunk &getChunkFor(lsp::Position &pos) {
    return *getChunkItFor(pos);
  }

  /// The full string contents of the file.
  std::string contents;

  /// The version of this file.
  int64_t version = 0;

  /// The number of lines in the file.
  int64_t totalNumLines = 0;

  /// The chunks of this file. The order of these chunks is the order in which
  /// they appear in the text file.
  std::vector<std::unique_ptr<PDLTextFileChunk>> chunks;

  /// The extra set of include directories for this file.
  std::vector<std::string> extraIncludeDirs;
};
} // namespace

PDLTextFile::PDLTextFile(const lsp::URIForFile &uri, StringRef fileContents,
                         int64_t version,
                         const std::vector<std::string> &extraDirs,
                         std::vector<lsp::Diagnostic> &diagnostics)
    : contents(fileContents.str()), extraIncludeDirs(extraDirs) {
  initialize(uri, version, diagnostics);
}

LogicalResult
PDLTextFile::update(const lsp::URIForFile &uri, int64_t newVersion,
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

void PDLTextFile::getLocationsOf(const lsp::URIForFile &uri,
                                 lsp::Position defPos,
                                 std::vector<lsp::Location> &locations) {
  PDLTextFileChunk &chunk = getChunkFor(defPos);
  chunk.document.getLocationsOf(uri, defPos, locations);

  // Adjust any locations within this file for the offset of this chunk.
  if (chunk.lineOffset == 0)
    return;
  for (lsp::Location &loc : locations)
    if (loc.uri == uri)
      chunk.adjustLocForChunkOffset(loc.range);
}

void PDLTextFile::findReferencesOf(const lsp::URIForFile &uri,
                                   lsp::Position pos,
                                   std::vector<lsp::Location> &references) {
  PDLTextFileChunk &chunk = getChunkFor(pos);
  chunk.document.findReferencesOf(uri, pos, references);

  // Adjust any locations within this file for the offset of this chunk.
  if (chunk.lineOffset == 0)
    return;
  for (lsp::Location &loc : references)
    if (loc.uri == uri)
      chunk.adjustLocForChunkOffset(loc.range);
}

void PDLTextFile::getDocumentLinks(const lsp::URIForFile &uri,
                                   std::vector<lsp::DocumentLink> &links) {
  chunks.front()->document.getDocumentLinks(uri, links);
  for (const auto &it : llvm::drop_begin(chunks)) {
    size_t currentNumLinks = links.size();
    it->document.getDocumentLinks(uri, links);

    // Adjust any links within this file to account for the offset of this
    // chunk.
    for (auto &link : llvm::drop_begin(links, currentNumLinks))
      it->adjustLocForChunkOffset(link.range);
  }
}

Optional<lsp::Hover> PDLTextFile::findHover(const lsp::URIForFile &uri,
                                            lsp::Position hoverPos) {
  PDLTextFileChunk &chunk = getChunkFor(hoverPos);
  Optional<lsp::Hover> hoverInfo = chunk.document.findHover(uri, hoverPos);

  // Adjust any locations within this file for the offset of this chunk.
  if (chunk.lineOffset != 0 && hoverInfo && hoverInfo->range)
    chunk.adjustLocForChunkOffset(*hoverInfo->range);
  return hoverInfo;
}

void PDLTextFile::findDocumentSymbols(
    std::vector<lsp::DocumentSymbol> &symbols) {
  if (chunks.size() == 1)
    return chunks.front()->document.findDocumentSymbols(symbols);

  // If there are multiple chunks in this file, we create top-level symbols for
  // each chunk.
  for (unsigned i = 0, e = chunks.size(); i < e; ++i) {
    PDLTextFileChunk &chunk = *chunks[i];
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

lsp::CompletionList PDLTextFile::getCodeCompletion(const lsp::URIForFile &uri,
                                                   lsp::Position completePos) {
  PDLTextFileChunk &chunk = getChunkFor(completePos);
  lsp::CompletionList completionList =
      chunk.document.getCodeCompletion(uri, completePos);

  // Adjust any completion locations.
  for (lsp::CompletionItem &item : completionList.items) {
    if (item.textEdit)
      chunk.adjustLocForChunkOffset(item.textEdit->range);
    for (lsp::TextEdit &edit : item.additionalTextEdits)
      chunk.adjustLocForChunkOffset(edit.range);
  }
  return completionList;
}

lsp::SignatureHelp PDLTextFile::getSignatureHelp(const lsp::URIForFile &uri,
                                                 lsp::Position helpPos) {
  return getChunkFor(helpPos).document.getSignatureHelp(uri, helpPos);
}

void PDLTextFile::getInlayHints(const lsp::URIForFile &uri, lsp::Range range,
                                std::vector<lsp::InlayHint> &inlayHints) {
  auto startIt = getChunkItFor(range.start);
  auto endIt = getChunkItFor(range.end);

  // Functor used to get the chunks for a given file, and fixup any locations
  auto getHintsForChunk = [&](ChunkIterator chunkIt, lsp::Range range) {
    size_t currentNumHints = inlayHints.size();
    chunkIt->document.getInlayHints(uri, range, inlayHints);

    // If this isn't the first chunk, update any positions to account for line
    // number differences.
    if (&*chunkIt != &*chunks.front()) {
      for (auto &hint : llvm::drop_begin(inlayHints, currentNumHints))
        chunkIt->adjustLocForChunkOffset(hint.position);
    }
  };
  // Returns the number of lines held by a given chunk.
  auto getNumLines = [](ChunkIterator chunkIt) {
    return (chunkIt + 1)->lineOffset - chunkIt->lineOffset;
  };

  // Check if the range is fully within a single chunk.
  if (startIt == endIt)
    return getHintsForChunk(startIt, range);

  // Otherwise, the range is split between multiple chunks. The first chunk
  // has the correct range start, but covers the total document.
  getHintsForChunk(startIt, lsp::Range(range.start, getNumLines(startIt)));

  // Every chunk in between uses the full document.
  for (++startIt; startIt != endIt; ++startIt)
    getHintsForChunk(startIt, lsp::Range(0, getNumLines(startIt)));

  // The range for the last chunk starts at the beginning of the document, up
  // through the end of the input range.
  getHintsForChunk(startIt, lsp::Range(0, range.end));
}

lsp::PDLLViewOutputResult
PDLTextFile::getPDLLViewOutput(lsp::PDLLViewOutputKind kind) {
  lsp::PDLLViewOutputResult result;
  {
    llvm::raw_string_ostream outputOS(result.output);
    llvm::interleave(
        llvm::make_pointee_range(chunks),
        [&](PDLTextFileChunk &chunk) {
          chunk.document.getPDLLViewOutput(outputOS, kind);
        },
        [&] { outputOS << "\n// -----\n\n"; });
  }
  return result;
}

void PDLTextFile::initialize(const lsp::URIForFile &uri, int64_t newVersion,
                             std::vector<lsp::Diagnostic> &diagnostics) {
  version = newVersion;
  chunks.clear();

  // Split the file into separate PDL documents.
  // TODO: Find a way to share the split file marker with other tools. We don't
  // want to use `splitAndProcessBuffer` here, but we do want to make sure this
  // marker doesn't go out of sync.
  SmallVector<StringRef, 8> subContents;
  StringRef(contents).split(subContents, "// -----");
  chunks.emplace_back(std::make_unique<PDLTextFileChunk>(
      /*lineOffset=*/0, uri, subContents.front(), extraIncludeDirs,
      diagnostics));

  uint64_t lineOffset = subContents.front().count('\n');
  for (StringRef docContents : llvm::drop_begin(subContents)) {
    unsigned currentNumDiags = diagnostics.size();
    auto chunk = std::make_unique<PDLTextFileChunk>(
        lineOffset, uri, docContents, extraIncludeDirs, diagnostics);
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

PDLTextFile::ChunkIterator PDLTextFile::getChunkItFor(lsp::Position &pos) {
  if (chunks.size() == 1)
    return chunks.begin();

  // Search for the first chunk with a greater line offset, the previous chunk
  // is the one that contains `pos`.
  auto it = llvm::upper_bound(
      chunks, pos, [](const lsp::Position &pos, const auto &chunk) {
        return static_cast<uint64_t>(pos.line) < chunk->lineOffset;
      });
  ChunkIterator chunkIt(it == chunks.end() ? (chunks.end() - 1) : --it);
  pos.line -= chunkIt->lineOffset;
  return chunkIt;
}

//===----------------------------------------------------------------------===//
// PDLLServer::Impl
//===----------------------------------------------------------------------===//

struct lsp::PDLLServer::Impl {
  explicit Impl(const Options &options)
      : options(options), compilationDatabase(options.compilationDatabases) {}

  /// PDLL LSP options.
  const Options &options;

  /// The compilation database containing additional information for files
  /// passed to the server.
  lsp::CompilationDatabase compilationDatabase;

  /// The files held by the server, mapped by their URI file name.
  llvm::StringMap<std::unique_ptr<PDLTextFile>> files;
};

//===----------------------------------------------------------------------===//
// PDLLServer
//===----------------------------------------------------------------------===//

lsp::PDLLServer::PDLLServer(const Options &options)
    : impl(std::make_unique<Impl>(options)) {}
lsp::PDLLServer::~PDLLServer() = default;

void lsp::PDLLServer::addDocument(const URIForFile &uri, StringRef contents,
                                  int64_t version,
                                  std::vector<Diagnostic> &diagnostics) {
  // Build the set of additional include directories.
  std::vector<std::string> additionalIncludeDirs = impl->options.extraDirs;
  const auto &fileInfo = impl->compilationDatabase.getFileInfo(uri.file());
  llvm::append_range(additionalIncludeDirs, fileInfo.includeDirs);

  impl->files[uri.file()] = std::make_unique<PDLTextFile>(
      uri, contents, version, additionalIncludeDirs, diagnostics);
}

void lsp::PDLLServer::updateDocument(
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

Optional<int64_t> lsp::PDLLServer::removeDocument(const URIForFile &uri) {
  auto it = impl->files.find(uri.file());
  if (it == impl->files.end())
    return std::nullopt;

  int64_t version = it->second->getVersion();
  impl->files.erase(it);
  return version;
}

void lsp::PDLLServer::getLocationsOf(const URIForFile &uri,
                                     const Position &defPos,
                                     std::vector<Location> &locations) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    fileIt->second->getLocationsOf(uri, defPos, locations);
}

void lsp::PDLLServer::findReferencesOf(const URIForFile &uri,
                                       const Position &pos,
                                       std::vector<Location> &references) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    fileIt->second->findReferencesOf(uri, pos, references);
}

void lsp::PDLLServer::getDocumentLinks(
    const URIForFile &uri, std::vector<DocumentLink> &documentLinks) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    return fileIt->second->getDocumentLinks(uri, documentLinks);
}

Optional<lsp::Hover> lsp::PDLLServer::findHover(const URIForFile &uri,
                                                const Position &hoverPos) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    return fileIt->second->findHover(uri, hoverPos);
  return std::nullopt;
}

void lsp::PDLLServer::findDocumentSymbols(
    const URIForFile &uri, std::vector<DocumentSymbol> &symbols) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    fileIt->second->findDocumentSymbols(symbols);
}

lsp::CompletionList
lsp::PDLLServer::getCodeCompletion(const URIForFile &uri,
                                   const Position &completePos) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    return fileIt->second->getCodeCompletion(uri, completePos);
  return CompletionList();
}

lsp::SignatureHelp lsp::PDLLServer::getSignatureHelp(const URIForFile &uri,
                                                     const Position &helpPos) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    return fileIt->second->getSignatureHelp(uri, helpPos);
  return SignatureHelp();
}

void lsp::PDLLServer::getInlayHints(const URIForFile &uri, const Range &range,
                                    std::vector<InlayHint> &inlayHints) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt == impl->files.end())
    return;
  fileIt->second->getInlayHints(uri, range, inlayHints);

  // Drop any duplicated hints that may have cropped up.
  llvm::sort(inlayHints);
  inlayHints.erase(std::unique(inlayHints.begin(), inlayHints.end()),
                   inlayHints.end());
}

Optional<lsp::PDLLViewOutputResult>
lsp::PDLLServer::getPDLLViewOutput(const URIForFile &uri,
                                   PDLLViewOutputKind kind) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    return fileIt->second->getPDLLViewOutput(kind);
  return std::nullopt;
}
