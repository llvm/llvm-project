//===--- Protocol.h - Language Server Protocol Implementation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains structs based on the LSP specification at
// https://github.com/Microsoft/language-server-protocol/blob/master/protocol.md
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

#ifndef LIB_MLIR_TOOLS_MLIRLSPSERVER_LSP_PROTOCOL_H_
#define LIB_MLIR_TOOLS_MLIRLSPSERVER_LSP_PROTOCOL_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <bitset>
#include <memory>
#include <string>
#include <vector>

namespace mlir {
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
// DiagnosticRelatedInformation
//===----------------------------------------------------------------------===//

/// Represents a related message and source code location for a diagnostic.
/// This should be used to point to code locations that cause or related to a
/// diagnostics, e.g. when duplicating a symbol in a scope.
struct DiagnosticRelatedInformation {
  DiagnosticRelatedInformation(Location location, std::string message)
      : location(location), message(std::move(message)) {}

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
      : uri(uri), version(version) {}

  /// The URI for which diagnostic information is reported.
  URIForFile uri;
  /// The list of reported diagnostics.
  std::vector<Diagnostic> diagnostics;
  /// The version number of the document the diagnostics are published for.
  int64_t version;
};

/// Add support for JSON serialization.
llvm::json::Value toJSON(const PublishDiagnosticsParams &params);

} // namespace lsp
} // namespace mlir

namespace llvm {
template <> struct format_provider<mlir::lsp::Position> {
  static void format(const mlir::lsp::Position &pos, raw_ostream &os,
                     StringRef style) {
    assert(style.empty() && "style modifiers for this type are not supported");
    os << pos;
  }
};
} // namespace llvm

#endif
