//===--- Protocol.h - Language Server Protocol Implementation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains structs based on the LSP specification at
// https://microsoft.github.io/language-server-protocol/specification
//
// This is not meant to be a complete implementation, new interfaces are added
// when they're needed.
//
// Each struct has a toJSON and fromJSON function, that converts between
// the struct and a JSON representation. (See JSON.h)
//
// Some structs also have operator<< serialization. This is for debugging and
// tests, and is not generally machine-readable.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_MLIR_TOOLS_LSPSERVERSUPPORT_PROTOCOL_H_
#define LIB_MLIR_TOOLS_LSPSERVERSUPPORT_PROTOCOL_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <bitset>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mlir {
struct LogicalResult;

namespace lsp {

enum class ErrorCode {
  // Defined by JSON RPC.
  ParseError = -32700,
  InvalidRequest = -32600,
  MethodNotFound = -32601,
  InvalidParams = -32602,
  InternalError = -32603,

  ServerNotInitialized = -32002,
  UnknownErrorCode = -32001,

  // Defined by the protocol.
  RequestCancelled = -32800,
  ContentModified = -32801,
};

/// Defines how the host (editor) should sync document changes to the language
/// server.
enum class TextDocumentSyncKind {
  /// Documents should not be synced at all.
  None = 0,

  /// Documents are synced by always sending the full content of the document.
  Full = 1,

  /// Documents are synced by sending the full content on open. After that only
  /// incremental updates to the document are sent.
  Incremental = 2,
};

//===----------------------------------------------------------------------===//
// LSPError
//===----------------------------------------------------------------------===//

/// This class models an LSP error as an llvm::Error.
class LSPError : public llvm::ErrorInfo<LSPError> {
public:
  std::string message;
  ErrorCode code;
  static char ID;

  LSPError(std::string message, ErrorCode code)
      : message(std::move(message)), code(code) {}

  void log(raw_ostream &os) const override {
    os << int(code) << ": " << message;
  }
  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }
};

//===----------------------------------------------------------------------===//
// URIForFile
//===----------------------------------------------------------------------===//

/// URI in "file" scheme for a file.
class URIForFile {
public:
  URIForFile() = default;

  /// Try to build a URIForFile from the given URI string.
  static llvm::Expected<URIForFile> fromURI(StringRef uri);

  /// Try to build a URIForFile from the given absolute file path.
  static llvm::Expected<URIForFile> fromFile(StringRef absoluteFilepath);

  /// Returns the absolute path to the file.
  StringRef file() const { return filePath; }

  /// Returns the original uri of the file.
  StringRef uri() const { return uriStr; }

  explicit operator bool() const { return !filePath.empty(); }

  friend bool operator==(const URIForFile &lhs, const URIForFile &rhs) {
    return lhs.filePath == rhs.filePath;
  }
  friend bool operator!=(const URIForFile &lhs, const URIForFile &rhs) {
    return !(lhs == rhs);
  }
  friend bool operator<(const URIForFile &lhs, const URIForFile &rhs) {
    return lhs.filePath < rhs.filePath;
  }

private:
  explicit URIForFile(std::string &&filePath, std::string &&uriStr)
      : filePath(std::move(filePath)), uriStr(uriStr) {}

  std::string filePath;
  std::string uriStr;
};

/// Add support for JSON serialization.
llvm::json::Value toJSON(const URIForFile &value);
bool fromJSON(const llvm::json::Value &value, URIForFile &result,
              llvm::json::Path path);
raw_ostream &operator<<(raw_ostream &os, const URIForFile &value);

//===----------------------------------------------------------------------===//
// ClientCapabilities
//===----------------------------------------------------------------------===//

struct ClientCapabilities {
  /// Client supports hierarchical document symbols.
  /// textDocument.documentSymbol.hierarchicalDocumentSymbolSupport
  bool hierarchicalDocumentSymbol = false;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, ClientCapabilities &result,
              llvm::json::Path path);

//===----------------------------------------------------------------------===//
// InitializeParams
//===----------------------------------------------------------------------===//

enum class TraceLevel {
  Off = 0,
  Messages = 1,
  Verbose = 2,
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, TraceLevel &result,
              llvm::json::Path path);

struct InitializeParams {
  /// The capabilities provided by the client (editor or tool).
  ClientCapabilities capabilities;

  /// The initial trace setting. If omitted trace is disabled ('off').
  Optional<TraceLevel> trace;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, InitializeParams &result,
              llvm::json::Path path);

//===----------------------------------------------------------------------===//
// InitializedParams
//===----------------------------------------------------------------------===//

struct NoParams {};
inline bool fromJSON(const llvm::json::Value &, NoParams &, llvm::json::Path) {
  return true;
}
using InitializedParams = NoParams;

//===----------------------------------------------------------------------===//
// TextDocumentItem
//===----------------------------------------------------------------------===//

struct TextDocumentItem {
  /// The text document's URI.
  URIForFile uri;

  /// The text document's language identifier.
  std::string languageId;

  /// The content of the opened text document.
  std::string text;

  /// The version number of this document.
  int64_t version;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, TextDocumentItem &result,
              llvm::json::Path path);

//===----------------------------------------------------------------------===//
// TextDocumentIdentifier
//===----------------------------------------------------------------------===//

struct TextDocumentIdentifier {
  /// The text document's URI.
  URIForFile uri;
};

/// Add support for JSON serialization.
llvm::json::Value toJSON(const TextDocumentIdentifier &value);
bool fromJSON(const llvm::json::Value &value, TextDocumentIdentifier &result,
              llvm::json::Path path);

//===----------------------------------------------------------------------===//
// VersionedTextDocumentIdentifier
//===----------------------------------------------------------------------===//

struct VersionedTextDocumentIdentifier {
  /// The text document's URI.
  URIForFile uri;
  /// The version number of this document.
  int64_t version;
};

/// Add support for JSON serialization.
llvm::json::Value toJSON(const VersionedTextDocumentIdentifier &value);
bool fromJSON(const llvm::json::Value &value,
              VersionedTextDocumentIdentifier &result, llvm::json::Path path);

//===----------------------------------------------------------------------===//
// Position
//===----------------------------------------------------------------------===//

struct Position {
  Position(int line = 0, int character = 0)
      : line(line), character(character) {}

  /// Construct a position from the given source location.
  Position(llvm::SourceMgr &mgr, SMLoc loc) {
    std::pair<unsigned, unsigned> lineAndCol = mgr.getLineAndColumn(loc);
    line = lineAndCol.first - 1;
    character = lineAndCol.second - 1;
  }

  /// Line position in a document (zero-based).
  int line = 0;

  /// Character offset on a line in a document (zero-based).
  int character = 0;

  friend bool operator==(const Position &lhs, const Position &rhs) {
    return std::tie(lhs.line, lhs.character) ==
           std::tie(rhs.line, rhs.character);
  }
  friend bool operator!=(const Position &lhs, const Position &rhs) {
    return !(lhs == rhs);
  }
  friend bool operator<(const Position &lhs, const Position &rhs) {
    return std::tie(lhs.line, lhs.character) <
           std::tie(rhs.line, rhs.character);
  }
  friend bool operator<=(const Position &lhs, const Position &rhs) {
    return std::tie(lhs.line, lhs.character) <=
           std::tie(rhs.line, rhs.character);
  }

  /// Convert this position into a source location in the main file of the given
  /// source manager.
  SMLoc getAsSMLoc(llvm::SourceMgr &mgr) const {
    return mgr.FindLocForLineAndColumn(mgr.getMainFileID(), line + 1,
                                       character + 1);
  }
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, Position &result,
              llvm::json::Path path);
llvm::json::Value toJSON(const Position &value);
raw_ostream &operator<<(raw_ostream &os, const Position &value);

//===----------------------------------------------------------------------===//
// Range
//===----------------------------------------------------------------------===//

struct Range {
  Range() = default;
  Range(Position start, Position end) : start(start), end(end) {}
  Range(Position loc) : Range(loc, loc) {}

  /// Construct a range from the given source range.
  Range(llvm::SourceMgr &mgr, SMRange range)
      : Range(Position(mgr, range.Start), Position(mgr, range.End)) {}

  /// The range's start position.
  Position start;

  /// The range's end position.
  Position end;

  friend bool operator==(const Range &lhs, const Range &rhs) {
    return std::tie(lhs.start, lhs.end) == std::tie(rhs.start, rhs.end);
  }
  friend bool operator!=(const Range &lhs, const Range &rhs) {
    return !(lhs == rhs);
  }
  friend bool operator<(const Range &lhs, const Range &rhs) {
    return std::tie(lhs.start, lhs.end) < std::tie(rhs.start, rhs.end);
  }

  bool contains(Position pos) const { return start <= pos && pos < end; }
  bool contains(Range range) const {
    return start <= range.start && range.end <= end;
  }

  /// Convert this range into a source range in the main file of the given
  /// source manager.
  SMRange getAsSMRange(llvm::SourceMgr &mgr) const {
    SMLoc startLoc = start.getAsSMLoc(mgr);
    SMLoc endLoc = end.getAsSMLoc(mgr);
    // Check that the start and end locations are valid.
    if (!startLoc.isValid() || !endLoc.isValid() ||
        startLoc.getPointer() > endLoc.getPointer())
      return SMRange();
    return SMRange(startLoc, endLoc);
  }
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, Range &result,
              llvm::json::Path path);
llvm::json::Value toJSON(const Range &value);
raw_ostream &operator<<(raw_ostream &os, const Range &value);

//===----------------------------------------------------------------------===//
// Location
//===----------------------------------------------------------------------===//

struct Location {
  Location() = default;
  Location(const URIForFile &uri, Range range) : uri(uri), range(range) {}

  /// Construct a Location from the given source range.
  Location(const URIForFile &uri, llvm::SourceMgr &mgr, SMRange range)
      : Location(uri, Range(mgr, range)) {}

  /// The text document's URI.
  URIForFile uri;
  Range range;

  friend bool operator==(const Location &lhs, const Location &rhs) {
    return lhs.uri == rhs.uri && lhs.range == rhs.range;
  }

  friend bool operator!=(const Location &lhs, const Location &rhs) {
    return !(lhs == rhs);
  }

  friend bool operator<(const Location &lhs, const Location &rhs) {
    return std::tie(lhs.uri, lhs.range) < std::tie(rhs.uri, rhs.range);
  }
};

/// Add support for JSON serialization.
llvm::json::Value toJSON(const Location &value);
raw_ostream &operator<<(raw_ostream &os, const Location &value);

//===----------------------------------------------------------------------===//
// TextDocumentPositionParams
//===----------------------------------------------------------------------===//

struct TextDocumentPositionParams {
  /// The text document.
  TextDocumentIdentifier textDocument;

  /// The position inside the text document.
  Position position;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value,
              TextDocumentPositionParams &result, llvm::json::Path path);

//===----------------------------------------------------------------------===//
// ReferenceParams
//===----------------------------------------------------------------------===//

struct ReferenceContext {
  /// Include the declaration of the current symbol.
  bool includeDeclaration = false;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, ReferenceContext &result,
              llvm::json::Path path);

struct ReferenceParams : public TextDocumentPositionParams {
  ReferenceContext context;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, ReferenceParams &result,
              llvm::json::Path path);

//===----------------------------------------------------------------------===//
// DidOpenTextDocumentParams
//===----------------------------------------------------------------------===//

struct DidOpenTextDocumentParams {
  /// The document that was opened.
  TextDocumentItem textDocument;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, DidOpenTextDocumentParams &result,
              llvm::json::Path path);

//===----------------------------------------------------------------------===//
// DidCloseTextDocumentParams
//===----------------------------------------------------------------------===//

struct DidCloseTextDocumentParams {
  /// The document that was closed.
  TextDocumentIdentifier textDocument;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value,
              DidCloseTextDocumentParams &result, llvm::json::Path path);

//===----------------------------------------------------------------------===//
// DidChangeTextDocumentParams
//===----------------------------------------------------------------------===//

struct TextDocumentContentChangeEvent {
  /// Try to apply this change to the given contents string.
  LogicalResult applyTo(std::string &contents) const;
  /// Try to apply a set of changes to the given contents string.
  static LogicalResult applyTo(ArrayRef<TextDocumentContentChangeEvent> changes,
                               std::string &contents);

  /// The range of the document that changed.
  Optional<Range> range;

  /// The length of the range that got replaced.
  Optional<int> rangeLength;

  /// The new text of the range/document.
  std::string text;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value,
              TextDocumentContentChangeEvent &result, llvm::json::Path path);

struct DidChangeTextDocumentParams {
  /// The document that changed.
  VersionedTextDocumentIdentifier textDocument;

  /// The actual content changes.
  std::vector<TextDocumentContentChangeEvent> contentChanges;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value,
              DidChangeTextDocumentParams &result, llvm::json::Path path);

//===----------------------------------------------------------------------===//
// MarkupContent
//===----------------------------------------------------------------------===//

/// Describes the content type that a client supports in various result literals
/// like `Hover`.
enum class MarkupKind {
  PlainText,
  Markdown,
};
raw_ostream &operator<<(raw_ostream &os, MarkupKind kind);

struct MarkupContent {
  MarkupKind kind = MarkupKind::PlainText;
  std::string value;
};

/// Add support for JSON serialization.
llvm::json::Value toJSON(const MarkupContent &mc);

//===----------------------------------------------------------------------===//
// Hover
//===----------------------------------------------------------------------===//

struct Hover {
  /// Construct a default hover with the given range that uses Markdown content.
  Hover(Range range) : contents{MarkupKind::Markdown, ""}, range(range) {}

  /// The hover's content.
  MarkupContent contents;

  /// An optional range is a range inside a text document that is used to
  /// visualize a hover, e.g. by changing the background color.
  Optional<Range> range;
};

/// Add support for JSON serialization.
llvm::json::Value toJSON(const Hover &hover);

//===----------------------------------------------------------------------===//
// SymbolKind
//===----------------------------------------------------------------------===//

enum class SymbolKind {
  File = 1,
  Module = 2,
  Namespace = 3,
  Package = 4,
  Class = 5,
  Method = 6,
  Property = 7,
  Field = 8,
  Constructor = 9,
  Enum = 10,
  Interface = 11,
  Function = 12,
  Variable = 13,
  Constant = 14,
  String = 15,
  Number = 16,
  Boolean = 17,
  Array = 18,
  Object = 19,
  Key = 20,
  Null = 21,
  EnumMember = 22,
  Struct = 23,
  Event = 24,
  Operator = 25,
  TypeParameter = 26
};

//===----------------------------------------------------------------------===//
// DocumentSymbol
//===----------------------------------------------------------------------===//

/// Represents programming constructs like variables, classes, interfaces etc.
/// that appear in a document. Document symbols can be hierarchical and they
/// have two ranges: one that encloses its definition and one that points to its
/// most interesting range, e.g. the range of an identifier.
struct DocumentSymbol {
  DocumentSymbol() = default;
  DocumentSymbol(DocumentSymbol &&) = default;
  DocumentSymbol(const Twine &name, SymbolKind kind, Range range,
                 Range selectionRange)
      : name(name.str()), kind(kind), range(range),
        selectionRange(selectionRange) {}

  /// The name of this symbol.
  std::string name;

  /// More detail for this symbol, e.g the signature of a function.
  std::string detail;

  /// The kind of this symbol.
  SymbolKind kind;

  /// The range enclosing this symbol not including leading/trailing whitespace
  /// but everything else like comments. This information is typically used to
  /// determine if the clients cursor is inside the symbol to reveal in the
  /// symbol in the UI.
  Range range;

  /// The range that should be selected and revealed when this symbol is being
  /// picked, e.g the name of a function. Must be contained by the `range`.
  Range selectionRange;

  /// Children of this symbol, e.g. properties of a class.
  std::vector<DocumentSymbol> children;
};

/// Add support for JSON serialization.
llvm::json::Value toJSON(const DocumentSymbol &symbol);

//===----------------------------------------------------------------------===//
// DocumentSymbolParams
//===----------------------------------------------------------------------===//

struct DocumentSymbolParams {
  // The text document to find symbols in.
  TextDocumentIdentifier textDocument;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, DocumentSymbolParams &result,
              llvm::json::Path path);

//===----------------------------------------------------------------------===//
// DiagnosticRelatedInformation
//===----------------------------------------------------------------------===//

/// Represents a related message and source code location for a diagnostic.
/// This should be used to point to code locations that cause or related to a
/// diagnostics, e.g. when duplicating a symbol in a scope.
struct DiagnosticRelatedInformation {
  DiagnosticRelatedInformation(Location location, std::string message)
      : location(std::move(location)), message(std::move(message)) {}

  /// The location of this related diagnostic information.
  Location location;
  /// The message of this related diagnostic information.
  std::string message;
};

/// Add support for JSON serialization.
llvm::json::Value toJSON(const DiagnosticRelatedInformation &info);

//===----------------------------------------------------------------------===//
// Diagnostic
//===----------------------------------------------------------------------===//

enum class DiagnosticSeverity {
  /// It is up to the client to interpret diagnostics as error, warning, info or
  /// hint.
  Undetermined = 0,
  Error = 1,
  Warning = 2,
  Information = 3,
  Hint = 4
};

struct Diagnostic {
  /// The source range where the message applies.
  Range range;

  /// The diagnostic's severity. Can be omitted. If omitted it is up to the
  /// client to interpret diagnostics as error, warning, info or hint.
  DiagnosticSeverity severity = DiagnosticSeverity::Undetermined;

  /// A human-readable string describing the source of this diagnostic, e.g.
  /// 'typescript' or 'super lint'.
  std::string source;

  /// The diagnostic's message.
  std::string message;

  /// An array of related diagnostic information, e.g. when symbol-names within
  /// a scope collide all definitions can be marked via this property.
  Optional<std::vector<DiagnosticRelatedInformation>> relatedInformation;

  /// The diagnostic's category. Can be omitted.
  /// An LSP extension that's used to send the name of the category over to the
  /// client. The category typically describes the compilation stage during
  /// which the issue was produced, e.g. "Semantic Issue" or "Parse Issue".
  Optional<std::string> category;
};

/// Add support for JSON serialization.
llvm::json::Value toJSON(const Diagnostic &diag);

//===----------------------------------------------------------------------===//
// PublishDiagnosticsParams
//===----------------------------------------------------------------------===//

struct PublishDiagnosticsParams {
  PublishDiagnosticsParams(URIForFile uri, int64_t version)
      : uri(std::move(uri)), version(version) {}

  /// The URI for which diagnostic information is reported.
  URIForFile uri;
  /// The list of reported diagnostics.
  std::vector<Diagnostic> diagnostics;
  /// The version number of the document the diagnostics are published for.
  int64_t version;
};

/// Add support for JSON serialization.
llvm::json::Value toJSON(const PublishDiagnosticsParams &params);

//===----------------------------------------------------------------------===//
// TextEdit
//===----------------------------------------------------------------------===//

struct TextEdit {
  /// The range of the text document to be manipulated. To insert
  /// text into a document create a range where start === end.
  Range range;

  /// The string to be inserted. For delete operations use an
  /// empty string.
  std::string newText;
};

inline bool operator==(const TextEdit &lhs, const TextEdit &rhs) {
  return std::tie(lhs.newText, lhs.range) == std::tie(rhs.newText, rhs.range);
}

bool fromJSON(const llvm::json::Value &value, TextEdit &result,
              llvm::json::Path path);
llvm::json::Value toJSON(const TextEdit &value);
raw_ostream &operator<<(raw_ostream &os, const TextEdit &value);

//===----------------------------------------------------------------------===//
// CompletionItemKind
//===----------------------------------------------------------------------===//

/// The kind of a completion entry.
enum class CompletionItemKind {
  Missing = 0,
  Text = 1,
  Method = 2,
  Function = 3,
  Constructor = 4,
  Field = 5,
  Variable = 6,
  Class = 7,
  Interface = 8,
  Module = 9,
  Property = 10,
  Unit = 11,
  Value = 12,
  Enum = 13,
  Keyword = 14,
  Snippet = 15,
  Color = 16,
  File = 17,
  Reference = 18,
  Folder = 19,
  EnumMember = 20,
  Constant = 21,
  Struct = 22,
  Event = 23,
  Operator = 24,
  TypeParameter = 25,
};
bool fromJSON(const llvm::json::Value &value, CompletionItemKind &result,
              llvm::json::Path path);

constexpr auto kCompletionItemKindMin =
    static_cast<size_t>(CompletionItemKind::Text);
constexpr auto kCompletionItemKindMax =
    static_cast<size_t>(CompletionItemKind::TypeParameter);
using CompletionItemKindBitset = std::bitset<kCompletionItemKindMax + 1>;
bool fromJSON(const llvm::json::Value &value, CompletionItemKindBitset &result,
              llvm::json::Path path);

CompletionItemKind
adjustKindToCapability(CompletionItemKind kind,
                       CompletionItemKindBitset &supportedCompletionItemKinds);

//===----------------------------------------------------------------------===//
// CompletionItem
//===----------------------------------------------------------------------===//

/// Defines whether the insert text in a completion item should be interpreted
/// as plain text or a snippet.
enum class InsertTextFormat {
  Missing = 0,
  /// The primary text to be inserted is treated as a plain string.
  PlainText = 1,
  /// The primary text to be inserted is treated as a snippet.
  ///
  /// A snippet can define tab stops and placeholders with `$1`, `$2`
  /// and `${3:foo}`. `$0` defines the final tab stop, it defaults to the end
  /// of the snippet. Placeholders with equal identifiers are linked, that is
  /// typing in one will update others too.
  ///
  /// See also:
  /// https//github.com/Microsoft/vscode/blob/master/src/vs/editor/contrib/snippet/common/snippet.md
  Snippet = 2,
};

struct CompletionItem {
  CompletionItem() = default;
  CompletionItem(StringRef label, CompletionItemKind kind)
      : label(label.str()), kind(kind),
        insertTextFormat(InsertTextFormat::PlainText) {}

  /// The label of this completion item. By default also the text that is
  /// inserted when selecting this completion.
  std::string label;

  /// The kind of this completion item. Based of the kind an icon is chosen by
  /// the editor.
  CompletionItemKind kind = CompletionItemKind::Missing;

  /// A human-readable string with additional information about this item, like
  /// type or symbol information.
  std::string detail;

  /// A human-readable string that represents a doc-comment.
  Optional<MarkupContent> documentation;

  /// A string that should be used when comparing this item with other items.
  /// When `falsy` the label is used.
  std::string sortText;

  /// A string that should be used when filtering a set of completion items.
  /// When `falsy` the label is used.
  std::string filterText;

  /// A string that should be inserted to a document when selecting this
  /// completion. When `falsy` the label is used.
  std::string insertText;

  /// The format of the insert text. The format applies to both the `insertText`
  /// property and the `newText` property of a provided `textEdit`.
  InsertTextFormat insertTextFormat = InsertTextFormat::Missing;

  /// An edit which is applied to a document when selecting this completion.
  /// When an edit is provided `insertText` is ignored.
  ///
  /// Note: The range of the edit must be a single line range and it must
  /// contain the position at which completion has been requested.
  Optional<TextEdit> textEdit;

  /// An optional array of additional text edits that are applied when selecting
  /// this completion. Edits must not overlap with the main edit nor with
  /// themselves.
  std::vector<TextEdit> additionalTextEdits;

  /// Indicates if this item is deprecated.
  bool deprecated = false;
};

/// Add support for JSON serialization.
llvm::json::Value toJSON(const CompletionItem &value);
raw_ostream &operator<<(raw_ostream &os, const CompletionItem &value);
bool operator<(const CompletionItem &lhs, const CompletionItem &rhs);

//===----------------------------------------------------------------------===//
// CompletionList
//===----------------------------------------------------------------------===//

/// Represents a collection of completion items to be presented in the editor.
struct CompletionList {
  /// The list is not complete. Further typing should result in recomputing the
  /// list.
  bool isIncomplete = false;

  /// The completion items.
  std::vector<CompletionItem> items;
};

/// Add support for JSON serialization.
llvm::json::Value toJSON(const CompletionList &value);

//===----------------------------------------------------------------------===//
// CompletionContext
//===----------------------------------------------------------------------===//

enum class CompletionTriggerKind {
  /// Completion was triggered by typing an identifier (24x7 code
  /// complete), manual invocation (e.g Ctrl+Space) or via API.
  Invoked = 1,

  /// Completion was triggered by a trigger character specified by
  /// the `triggerCharacters` properties of the `CompletionRegistrationOptions`.
  TriggerCharacter = 2,

  /// Completion was re-triggered as the current completion list is incomplete.
  TriggerTriggerForIncompleteCompletions = 3
};

struct CompletionContext {
  /// How the completion was triggered.
  CompletionTriggerKind triggerKind = CompletionTriggerKind::Invoked;

  /// The trigger character (a single character) that has trigger code complete.
  /// Is undefined if `triggerKind !== CompletionTriggerKind.TriggerCharacter`
  std::string triggerCharacter;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, CompletionContext &result,
              llvm::json::Path path);

//===----------------------------------------------------------------------===//
// CompletionParams
//===----------------------------------------------------------------------===//

struct CompletionParams : TextDocumentPositionParams {
  CompletionContext context;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, CompletionParams &result,
              llvm::json::Path path);

//===----------------------------------------------------------------------===//
// ParameterInformation
//===----------------------------------------------------------------------===//

/// A single parameter of a particular signature.
struct ParameterInformation {
  /// The label of this parameter. Ignored when labelOffsets is set.
  std::string labelString;

  /// Inclusive start and exclusive end offsets withing the containing signature
  /// label.
  Optional<std::pair<unsigned, unsigned>> labelOffsets;

  /// The documentation of this parameter. Optional.
  std::string documentation;
};

/// Add support for JSON serialization.
llvm::json::Value toJSON(const ParameterInformation &value);

//===----------------------------------------------------------------------===//
// SignatureInformation
//===----------------------------------------------------------------------===//

/// Represents the signature of something callable.
struct SignatureInformation {
  /// The label of this signature. Mandatory.
  std::string label;

  /// The documentation of this signature. Optional.
  std::string documentation;

  /// The parameters of this signature.
  std::vector<ParameterInformation> parameters;
};

/// Add support for JSON serialization.
llvm::json::Value toJSON(const SignatureInformation &value);
raw_ostream &operator<<(raw_ostream &os, const SignatureInformation &value);

//===----------------------------------------------------------------------===//
// SignatureHelp
//===----------------------------------------------------------------------===//

/// Represents the signature of a callable.
struct SignatureHelp {
  /// The resulting signatures.
  std::vector<SignatureInformation> signatures;

  /// The active signature.
  int activeSignature = 0;

  /// The active parameter of the active signature.
  int activeParameter = 0;
};

/// Add support for JSON serialization.
llvm::json::Value toJSON(const SignatureHelp &value);

//===----------------------------------------------------------------------===//
// DocumentLinkParams
//===----------------------------------------------------------------------===//

/// Parameters for the document link request.
struct DocumentLinkParams {
  /// The document to provide document links for.
  TextDocumentIdentifier textDocument;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, DocumentLinkParams &result,
              llvm::json::Path path);

//===----------------------------------------------------------------------===//
// DocumentLink
//===----------------------------------------------------------------------===//

/// A range in a text document that links to an internal or external resource,
/// like another text document or a web site.
struct DocumentLink {
  DocumentLink() = default;
  DocumentLink(Range range, URIForFile target)
      : range(range), target(std::move(target)) {}

  /// The range this link applies to.
  Range range;

  /// The uri this link points to. If missing a resolve request is sent later.
  URIForFile target;

  // TODO: The following optional fields defined by the language server protocol
  // are unsupported:
  //
  // data?: any - A data entry field that is preserved on a document link
  //              between a DocumentLinkRequest and a
  //              DocumentLinkResolveRequest.

  friend bool operator==(const DocumentLink &lhs, const DocumentLink &rhs) {
    return lhs.range == rhs.range && lhs.target == rhs.target;
  }

  friend bool operator!=(const DocumentLink &lhs, const DocumentLink &rhs) {
    return !(lhs == rhs);
  }
};

/// Add support for JSON serialization.
llvm::json::Value toJSON(const DocumentLink &value);

//===----------------------------------------------------------------------===//
// InlayHintsParams
//===----------------------------------------------------------------------===//

/// A parameter literal used in inlay hint requests.
struct InlayHintsParams {
  /// The text document.
  TextDocumentIdentifier textDocument;

  /// The visible document range for which inlay hints should be computed.
  Range range;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, InlayHintsParams &result,
              llvm::json::Path path);

//===----------------------------------------------------------------------===//
// InlayHintKind
//===----------------------------------------------------------------------===//

/// Inlay hint kinds.
enum class InlayHintKind {
  /// An inlay hint that for a type annotation.
  ///
  /// An example of a type hint is a hint in this position:
  ///    auto var ^ = expr;
  /// which shows the deduced type of the variable.
  Type = 1,

  /// An inlay hint that is for a parameter.
  ///
  /// An example of a parameter hint is a hint in this position:
  ///    func(^arg);
  /// which shows the name of the corresponding parameter.
  Parameter = 2,
};

//===----------------------------------------------------------------------===//
// InlayHint
//===----------------------------------------------------------------------===//

/// Inlay hint information.
struct InlayHint {
  InlayHint(InlayHintKind kind, Position pos) : position(pos), kind(kind) {}

  /// The position of this hint.
  Position position;

  /// The label of this hint. A human readable string or an array of
  /// InlayHintLabelPart label parts.
  ///
  /// *Note* that neither the string nor the label part can be empty.
  std::string label;

  /// The kind of this hint. Can be omitted in which case the client should fall
  /// back to a reasonable default.
  InlayHintKind kind;

  /// Render padding before the hint.
  ///
  /// Note: Padding should use the editor's background color, not the
  /// background color of the hint itself. That means padding can be used
  /// to visually align/separate an inlay hint.
  bool paddingLeft = false;

  /// Render padding after the hint.
  ///
  /// Note: Padding should use the editor's background color, not the
  /// background color of the hint itself. That means padding can be used
  /// to visually align/separate an inlay hint.
  bool paddingRight = false;
};

/// Add support for JSON serialization.
llvm::json::Value toJSON(const InlayHint &);
bool operator==(const InlayHint &lhs, const InlayHint &rhs);
bool operator<(const InlayHint &lhs, const InlayHint &rhs);
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, InlayHintKind value);

} // namespace lsp
} // namespace mlir

namespace llvm {
template <>
struct format_provider<mlir::lsp::Position> {
  static void format(const mlir::lsp::Position &pos, raw_ostream &os,
                     StringRef style) {
    assert(style.empty() && "style modifiers for this type are not supported");
    os << pos;
  }
};
} // namespace llvm

#endif
