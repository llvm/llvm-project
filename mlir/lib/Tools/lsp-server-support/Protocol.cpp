//===--- Protocol.cpp - Language Server Protocol Implementation -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the serialization code for the LSP structs.
//
//===----------------------------------------------------------------------===//

#include "Protocol.h"
#include "Logging.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::lsp;

// Helper that doesn't treat `null` and absent fields as failures.
template <typename T>
static bool mapOptOrNull(const llvm::json::Value &params,
                         llvm::StringLiteral prop, T &out,
                         llvm::json::Path path) {
  const llvm::json::Object *o = params.getAsObject();
  assert(o);

  // Field is missing or null.
  auto *v = o->get(prop);
  if (!v || v->getAsNull())
    return true;
  return fromJSON(*v, out, path.field(prop));
}

//===----------------------------------------------------------------------===//
// LSPError
//===----------------------------------------------------------------------===//

char LSPError::ID;

//===----------------------------------------------------------------------===//
// URIForFile
//===----------------------------------------------------------------------===//

static bool isWindowsPath(StringRef path) {
  return path.size() > 1 && llvm::isAlpha(path[0]) && path[1] == ':';
}

static bool isNetworkPath(StringRef path) {
  return path.size() > 2 && path[0] == path[1] &&
         llvm::sys::path::is_separator(path[0]);
}

static bool shouldEscapeInURI(unsigned char c) {
  // Unreserved characters.
  if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
      (c >= '0' && c <= '9'))
    return false;

  switch (c) {
  case '-':
  case '_':
  case '.':
  case '~':
  // '/' is only reserved when parsing.
  case '/':
  // ':' is only reserved for relative URI paths, which we doesn't produce.
  case ':':
    return false;
  }
  return true;
}

/// Encodes a string according to percent-encoding.
/// - Unreserved characters are not escaped.
/// - Reserved characters always escaped with exceptions like '/'.
/// - All other characters are escaped.
static void percentEncode(StringRef content, std::string &out) {
  for (unsigned char c : content) {
    if (shouldEscapeInURI(c)) {
      out.push_back('%');
      out.push_back(llvm::hexdigit(c / 16));
      out.push_back(llvm::hexdigit(c % 16));
    } else {
      out.push_back(c);
    }
  }
}

/// Decodes a string according to percent-encoding.
static std::string percentDecode(StringRef content) {
  std::string result;
  for (auto i = content.begin(), e = content.end(); i != e; ++i) {
    if (*i != '%') {
      result += *i;
      continue;
    }
    if (*i == '%' && i + 2 < content.end() && llvm::isHexDigit(*(i + 1)) &&
        llvm::isHexDigit(*(i + 2))) {
      result.push_back(llvm::hexFromNibbles(*(i + 1), *(i + 2)));
      i += 2;
    } else {
      result.push_back(*i);
    }
  }
  return result;
}

/// Return the set containing the supported URI schemes.
static StringSet<> &getSupportedSchemes() {
  static StringSet<> schemes({"file", "test"});
  return schemes;
}

/// Returns true if the given scheme is structurally valid, i.e. it does not
/// contain any invalid scheme characters. This does not check that the scheme
/// is actually supported.
static bool isStructurallyValidScheme(StringRef scheme) {
  if (scheme.empty())
    return false;
  if (!llvm::isAlpha(scheme[0]))
    return false;
  return llvm::all_of(llvm::drop_begin(scheme), [](char c) {
    return llvm::isAlnum(c) || c == '+' || c == '.' || c == '-';
  });
}

static llvm::Expected<std::string> uriFromAbsolutePath(StringRef absolutePath,
                                                       StringRef scheme) {
  std::string body;
  StringRef authority;
  StringRef root = llvm::sys::path::root_name(absolutePath);
  if (isNetworkPath(root)) {
    // Windows UNC paths e.g. \\server\share => file://server/share
    authority = root.drop_front(2);
    absolutePath.consume_front(root);
  } else if (isWindowsPath(root)) {
    // Windows paths e.g. X:\path => file:///X:/path
    body = "/";
  }
  body += llvm::sys::path::convert_to_slash(absolutePath);

  std::string uri = scheme.str() + ":";
  if (authority.empty() && body.empty())
    return uri;

  // If authority if empty, we only print body if it starts with "/"; otherwise,
  // the URI is invalid.
  if (!authority.empty() || StringRef(body).startswith("/")) {
    uri.append("//");
    percentEncode(authority, uri);
  }
  percentEncode(body, uri);
  return uri;
}

static llvm::Expected<std::string> getAbsolutePath(StringRef authority,
                                                   StringRef body) {
  if (!body.startswith("/"))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "File scheme: expect body to be an absolute path starting "
        "with '/': " +
            body);
  SmallString<128> path;
  if (!authority.empty()) {
    // Windows UNC paths e.g. file://server/share => \\server\share
    ("//" + authority).toVector(path);
  } else if (isWindowsPath(body.substr(1))) {
    // Windows paths e.g. file:///X:/path => X:\path
    body.consume_front("/");
  }
  path.append(body);
  llvm::sys::path::native(path);
  return std::string(path);
}

static llvm::Expected<std::string> parseFilePathFromURI(StringRef origUri) {
  StringRef uri = origUri;

  // Decode the scheme of the URI.
  size_t pos = uri.find(':');
  if (pos == StringRef::npos)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Scheme must be provided in URI: " +
                                       origUri);
  StringRef schemeStr = uri.substr(0, pos);
  std::string uriScheme = percentDecode(schemeStr);
  if (!isStructurallyValidScheme(uriScheme))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Invalid scheme: " + schemeStr +
                                       " (decoded: " + uriScheme + ")");
  uri = uri.substr(pos + 1);

  // Decode the authority of the URI.
  std::string uriAuthority;
  if (uri.consume_front("//")) {
    pos = uri.find('/');
    uriAuthority = percentDecode(uri.substr(0, pos));
    uri = uri.substr(pos);
  }

  // Decode the body of the URI.
  std::string uriBody = percentDecode(uri);

  // Compute the absolute path for this uri.
  if (!getSupportedSchemes().contains(uriScheme)) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unsupported URI scheme `" + uriScheme +
                                       "' for workspace files");
  }
  return getAbsolutePath(uriAuthority, uriBody);
}

llvm::Expected<URIForFile> URIForFile::fromURI(StringRef uri) {
  llvm::Expected<std::string> filePath = parseFilePathFromURI(uri);
  if (!filePath)
    return filePath.takeError();
  return URIForFile(std::move(*filePath), uri.str());
}

llvm::Expected<URIForFile> URIForFile::fromFile(StringRef absoluteFilepath,
                                                StringRef scheme) {
  llvm::Expected<std::string> uri =
      uriFromAbsolutePath(absoluteFilepath, scheme);
  if (!uri)
    return uri.takeError();
  return fromURI(*uri);
}

StringRef URIForFile::scheme() const { return uri().split(':').first; }

void URIForFile::registerSupportedScheme(StringRef scheme) {
  getSupportedSchemes().insert(scheme);
}

bool mlir::lsp::fromJSON(const llvm::json::Value &value, URIForFile &result,
                         llvm::json::Path path) {
  if (Optional<StringRef> str = value.getAsString()) {
    llvm::Expected<URIForFile> expectedURI = URIForFile::fromURI(*str);
    if (!expectedURI) {
      path.report("unresolvable URI");
      consumeError(expectedURI.takeError());
      return false;
    }
    result = std::move(*expectedURI);
    return true;
  }
  return false;
}

llvm::json::Value mlir::lsp::toJSON(const URIForFile &value) {
  return value.uri();
}

raw_ostream &mlir::lsp::operator<<(raw_ostream &os, const URIForFile &value) {
  return os << value.uri();
}

//===----------------------------------------------------------------------===//
// ClientCapabilities
//===----------------------------------------------------------------------===//

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         ClientCapabilities &result, llvm::json::Path path) {
  const llvm::json::Object *o = value.getAsObject();
  if (!o) {
    path.report("expected object");
    return false;
  }
  if (const llvm::json::Object *textDocument = o->getObject("textDocument")) {
    if (const llvm::json::Object *documentSymbol =
            textDocument->getObject("documentSymbol")) {
      if (Optional<bool> hierarchicalSupport =
              documentSymbol->getBoolean("hierarchicalDocumentSymbolSupport"))
        result.hierarchicalDocumentSymbol = *hierarchicalSupport;
    }
    if (auto *codeAction = textDocument->getObject("codeAction")) {
      if (codeAction->getObject("codeActionLiteralSupport"))
        result.codeActionStructure = true;
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// InitializeParams
//===----------------------------------------------------------------------===//

bool mlir::lsp::fromJSON(const llvm::json::Value &value, TraceLevel &result,
                         llvm::json::Path path) {
  if (Optional<StringRef> str = value.getAsString()) {
    if (*str == "off") {
      result = TraceLevel::Off;
      return true;
    }
    if (*str == "messages") {
      result = TraceLevel::Messages;
      return true;
    }
    if (*str == "verbose") {
      result = TraceLevel::Verbose;
      return true;
    }
  }
  return false;
}

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         InitializeParams &result, llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  if (!o)
    return false;
  // We deliberately don't fail if we can't parse individual fields.
  o.map("capabilities", result.capabilities);
  o.map("trace", result.trace);
  return true;
}

//===----------------------------------------------------------------------===//
// TextDocumentItem
//===----------------------------------------------------------------------===//

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         TextDocumentItem &result, llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("uri", result.uri) &&
         o.map("languageId", result.languageId) && o.map("text", result.text) &&
         o.map("version", result.version);
}

//===----------------------------------------------------------------------===//
// TextDocumentIdentifier
//===----------------------------------------------------------------------===//

llvm::json::Value mlir::lsp::toJSON(const TextDocumentIdentifier &value) {
  return llvm::json::Object{{"uri", value.uri}};
}

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         TextDocumentIdentifier &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("uri", result.uri);
}

//===----------------------------------------------------------------------===//
// VersionedTextDocumentIdentifier
//===----------------------------------------------------------------------===//

llvm::json::Value
mlir::lsp::toJSON(const VersionedTextDocumentIdentifier &value) {
  return llvm::json::Object{
      {"uri", value.uri},
      {"version", value.version},
  };
}

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         VersionedTextDocumentIdentifier &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("uri", result.uri) && o.map("version", result.version);
}

//===----------------------------------------------------------------------===//
// Position
//===----------------------------------------------------------------------===//

bool mlir::lsp::fromJSON(const llvm::json::Value &value, Position &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("line", result.line) &&
         o.map("character", result.character);
}

llvm::json::Value mlir::lsp::toJSON(const Position &value) {
  return llvm::json::Object{
      {"line", value.line},
      {"character", value.character},
  };
}

raw_ostream &mlir::lsp::operator<<(raw_ostream &os, const Position &value) {
  return os << value.line << ':' << value.character;
}

//===----------------------------------------------------------------------===//
// Range
//===----------------------------------------------------------------------===//

bool mlir::lsp::fromJSON(const llvm::json::Value &value, Range &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("start", result.start) && o.map("end", result.end);
}

llvm::json::Value mlir::lsp::toJSON(const Range &value) {
  return llvm::json::Object{
      {"start", value.start},
      {"end", value.end},
  };
}

raw_ostream &mlir::lsp::operator<<(raw_ostream &os, const Range &value) {
  return os << value.start << '-' << value.end;
}

//===----------------------------------------------------------------------===//
// Location
//===----------------------------------------------------------------------===//

bool mlir::lsp::fromJSON(const llvm::json::Value &value, Location &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("uri", result.uri) && o.map("range", result.range);
}

llvm::json::Value mlir::lsp::toJSON(const Location &value) {
  return llvm::json::Object{
      {"uri", value.uri},
      {"range", value.range},
  };
}

raw_ostream &mlir::lsp::operator<<(raw_ostream &os, const Location &value) {
  return os << value.range << '@' << value.uri;
}

//===----------------------------------------------------------------------===//
// TextDocumentPositionParams
//===----------------------------------------------------------------------===//

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         TextDocumentPositionParams &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("textDocument", result.textDocument) &&
         o.map("position", result.position);
}

//===----------------------------------------------------------------------===//
// ReferenceParams
//===----------------------------------------------------------------------===//

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         ReferenceContext &result, llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.mapOptional("includeDeclaration", result.includeDeclaration);
}

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         ReferenceParams &result, llvm::json::Path path) {
  TextDocumentPositionParams &base = result;
  llvm::json::ObjectMapper o(value, path);
  return fromJSON(value, base, path) && o &&
         o.mapOptional("context", result.context);
}

//===----------------------------------------------------------------------===//
// DidOpenTextDocumentParams
//===----------------------------------------------------------------------===//

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         DidOpenTextDocumentParams &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("textDocument", result.textDocument);
}

//===----------------------------------------------------------------------===//
// DidCloseTextDocumentParams
//===----------------------------------------------------------------------===//

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         DidCloseTextDocumentParams &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("textDocument", result.textDocument);
}

//===----------------------------------------------------------------------===//
// DidChangeTextDocumentParams
//===----------------------------------------------------------------------===//

LogicalResult
TextDocumentContentChangeEvent::applyTo(std::string &contents) const {
  // If there is no range, the full document changed.
  if (!range) {
    contents = text;
    return success();
  }

  // Try to map the replacement range to the content.
  llvm::SourceMgr tmpScrMgr;
  tmpScrMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(contents),
                               SMLoc());
  SMRange rangeLoc = range->getAsSMRange(tmpScrMgr);
  if (!rangeLoc.isValid())
    return failure();

  contents.replace(rangeLoc.Start.getPointer() - contents.data(),
                   rangeLoc.End.getPointer() - rangeLoc.Start.getPointer(),
                   text);
  return success();
}

LogicalResult TextDocumentContentChangeEvent::applyTo(
    ArrayRef<TextDocumentContentChangeEvent> changes, std::string &contents) {
  for (const auto &change : changes)
    if (failed(change.applyTo(contents)))
      return failure();
  return success();
}

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         TextDocumentContentChangeEvent &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("range", result.range) &&
         o.map("rangeLength", result.rangeLength) && o.map("text", result.text);
}

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         DidChangeTextDocumentParams &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("textDocument", result.textDocument) &&
         o.map("contentChanges", result.contentChanges);
}

//===----------------------------------------------------------------------===//
// MarkupContent
//===----------------------------------------------------------------------===//

static llvm::StringRef toTextKind(MarkupKind kind) {
  switch (kind) {
  case MarkupKind::PlainText:
    return "plaintext";
  case MarkupKind::Markdown:
    return "markdown";
  }
  llvm_unreachable("Invalid MarkupKind");
}

raw_ostream &mlir::lsp::operator<<(raw_ostream &os, MarkupKind kind) {
  return os << toTextKind(kind);
}

llvm::json::Value mlir::lsp::toJSON(const MarkupContent &mc) {
  if (mc.value.empty())
    return nullptr;

  return llvm::json::Object{
      {"kind", toTextKind(mc.kind)},
      {"value", mc.value},
  };
}

//===----------------------------------------------------------------------===//
// Hover
//===----------------------------------------------------------------------===//

llvm::json::Value mlir::lsp::toJSON(const Hover &hover) {
  llvm::json::Object result{{"contents", toJSON(hover.contents)}};
  if (hover.range)
    result["range"] = toJSON(*hover.range);
  return std::move(result);
}

//===----------------------------------------------------------------------===//
// DocumentSymbol
//===----------------------------------------------------------------------===//

llvm::json::Value mlir::lsp::toJSON(const DocumentSymbol &symbol) {
  llvm::json::Object result{{"name", symbol.name},
                            {"kind", static_cast<int>(symbol.kind)},
                            {"range", symbol.range},
                            {"selectionRange", symbol.selectionRange}};

  if (!symbol.detail.empty())
    result["detail"] = symbol.detail;
  if (!symbol.children.empty())
    result["children"] = symbol.children;
  return std::move(result);
}

//===----------------------------------------------------------------------===//
// DocumentSymbolParams
//===----------------------------------------------------------------------===//

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         DocumentSymbolParams &result, llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("textDocument", result.textDocument);
}

//===----------------------------------------------------------------------===//
// DiagnosticRelatedInformation
//===----------------------------------------------------------------------===//

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         DiagnosticRelatedInformation &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("location", result.location) &&
         o.map("message", result.message);
}

llvm::json::Value mlir::lsp::toJSON(const DiagnosticRelatedInformation &info) {
  return llvm::json::Object{
      {"location", info.location},
      {"message", info.message},
  };
}

//===----------------------------------------------------------------------===//
// Diagnostic
//===----------------------------------------------------------------------===//

llvm::json::Value mlir::lsp::toJSON(const Diagnostic &diag) {
  llvm::json::Object result{
      {"range", diag.range},
      {"severity", (int)diag.severity},
      {"message", diag.message},
  };
  if (diag.category)
    result["category"] = *diag.category;
  if (!diag.source.empty())
    result["source"] = diag.source;
  if (diag.relatedInformation)
    result["relatedInformation"] = *diag.relatedInformation;
  return std::move(result);
}

bool mlir::lsp::fromJSON(const llvm::json::Value &value, Diagnostic &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  if (!o)
    return false;
  int severity = 0;
  if (!mapOptOrNull(value, "severity", severity, path))
    return false;
  result.severity = (DiagnosticSeverity)severity;

  return o.map("range", result.range) && o.map("message", result.message) &&
         mapOptOrNull(value, "category", result.category, path) &&
         mapOptOrNull(value, "source", result.source, path) &&
         mapOptOrNull(value, "relatedInformation", result.relatedInformation,
                      path);
}

//===----------------------------------------------------------------------===//
// PublishDiagnosticsParams
//===----------------------------------------------------------------------===//

llvm::json::Value mlir::lsp::toJSON(const PublishDiagnosticsParams &params) {
  return llvm::json::Object{
      {"uri", params.uri},
      {"diagnostics", params.diagnostics},
      {"version", params.version},
  };
}

//===----------------------------------------------------------------------===//
// TextEdit
//===----------------------------------------------------------------------===//

bool mlir::lsp::fromJSON(const llvm::json::Value &value, TextEdit &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("range", result.range) && o.map("newText", result.newText);
}

llvm::json::Value mlir::lsp::toJSON(const TextEdit &value) {
  return llvm::json::Object{
      {"range", value.range},
      {"newText", value.newText},
  };
}

raw_ostream &mlir::lsp::operator<<(raw_ostream &os, const TextEdit &value) {
  os << value.range << " => \"";
  llvm::printEscapedString(value.newText, os);
  return os << '"';
}

//===----------------------------------------------------------------------===//
// CompletionItemKind
//===----------------------------------------------------------------------===//

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         CompletionItemKind &result, llvm::json::Path path) {
  if (Optional<int64_t> intValue = value.getAsInteger()) {
    if (*intValue < static_cast<int>(CompletionItemKind::Text) ||
        *intValue > static_cast<int>(CompletionItemKind::TypeParameter))
      return false;
    result = static_cast<CompletionItemKind>(*intValue);
    return true;
  }
  return false;
}

CompletionItemKind mlir::lsp::adjustKindToCapability(
    CompletionItemKind kind,
    CompletionItemKindBitset &supportedCompletionItemKinds) {
  size_t kindVal = static_cast<size_t>(kind);
  if (kindVal >= kCompletionItemKindMin &&
      kindVal <= supportedCompletionItemKinds.size() &&
      supportedCompletionItemKinds[kindVal])
    return kind;

  // Provide some fall backs for common kinds that are close enough.
  switch (kind) {
  case CompletionItemKind::Folder:
    return CompletionItemKind::File;
  case CompletionItemKind::EnumMember:
    return CompletionItemKind::Enum;
  case CompletionItemKind::Struct:
    return CompletionItemKind::Class;
  default:
    return CompletionItemKind::Text;
  }
}

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         CompletionItemKindBitset &result,
                         llvm::json::Path path) {
  if (const llvm::json::Array *arrayValue = value.getAsArray()) {
    for (size_t i = 0, e = arrayValue->size(); i < e; ++i) {
      CompletionItemKind kindOut;
      if (fromJSON((*arrayValue)[i], kindOut, path.index(i)))
        result.set(size_t(kindOut));
    }
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// CompletionItem
//===----------------------------------------------------------------------===//

llvm::json::Value mlir::lsp::toJSON(const CompletionItem &value) {
  assert(!value.label.empty() && "completion item label is required");
  llvm::json::Object result{{"label", value.label}};
  if (value.kind != CompletionItemKind::Missing)
    result["kind"] = static_cast<int>(value.kind);
  if (!value.detail.empty())
    result["detail"] = value.detail;
  if (value.documentation)
    result["documentation"] = value.documentation;
  if (!value.sortText.empty())
    result["sortText"] = value.sortText;
  if (!value.filterText.empty())
    result["filterText"] = value.filterText;
  if (!value.insertText.empty())
    result["insertText"] = value.insertText;
  if (value.insertTextFormat != InsertTextFormat::Missing)
    result["insertTextFormat"] = static_cast<int>(value.insertTextFormat);
  if (value.textEdit)
    result["textEdit"] = *value.textEdit;
  if (!value.additionalTextEdits.empty()) {
    result["additionalTextEdits"] =
        llvm::json::Array(value.additionalTextEdits);
  }
  if (value.deprecated)
    result["deprecated"] = value.deprecated;
  return std::move(result);
}

raw_ostream &mlir::lsp::operator<<(raw_ostream &os,
                                   const CompletionItem &value) {
  return os << value.label << " - " << toJSON(value);
}

bool mlir::lsp::operator<(const CompletionItem &lhs,
                          const CompletionItem &rhs) {
  return (lhs.sortText.empty() ? lhs.label : lhs.sortText) <
         (rhs.sortText.empty() ? rhs.label : rhs.sortText);
}

//===----------------------------------------------------------------------===//
// CompletionList
//===----------------------------------------------------------------------===//

llvm::json::Value mlir::lsp::toJSON(const CompletionList &value) {
  return llvm::json::Object{
      {"isIncomplete", value.isIncomplete},
      {"items", llvm::json::Array(value.items)},
  };
}

//===----------------------------------------------------------------------===//
// CompletionContext
//===----------------------------------------------------------------------===//

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         CompletionContext &result, llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  int triggerKind;
  if (!o || !o.map("triggerKind", triggerKind) ||
      !mapOptOrNull(value, "triggerCharacter", result.triggerCharacter, path))
    return false;
  result.triggerKind = static_cast<CompletionTriggerKind>(triggerKind);
  return true;
}

//===----------------------------------------------------------------------===//
// CompletionParams
//===----------------------------------------------------------------------===//

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         CompletionParams &result, llvm::json::Path path) {
  if (!fromJSON(value, static_cast<TextDocumentPositionParams &>(result), path))
    return false;
  if (const llvm::json::Value *context = value.getAsObject()->get("context"))
    return fromJSON(*context, result.context, path.field("context"));
  return true;
}

//===----------------------------------------------------------------------===//
// ParameterInformation
//===----------------------------------------------------------------------===//

llvm::json::Value mlir::lsp::toJSON(const ParameterInformation &value) {
  assert((value.labelOffsets || !value.labelString.empty()) &&
         "parameter information label is required");
  llvm::json::Object result;
  if (value.labelOffsets)
    result["label"] = llvm::json::Array(
        {value.labelOffsets->first, value.labelOffsets->second});
  else
    result["label"] = value.labelString;
  if (!value.documentation.empty())
    result["documentation"] = value.documentation;
  return std::move(result);
}

//===----------------------------------------------------------------------===//
// SignatureInformation
//===----------------------------------------------------------------------===//

llvm::json::Value mlir::lsp::toJSON(const SignatureInformation &value) {
  assert(!value.label.empty() && "signature information label is required");
  llvm::json::Object result{
      {"label", value.label},
      {"parameters", llvm::json::Array(value.parameters)},
  };
  if (!value.documentation.empty())
    result["documentation"] = value.documentation;
  return std::move(result);
}

raw_ostream &mlir::lsp::operator<<(raw_ostream &os,
                                   const SignatureInformation &value) {
  return os << value.label << " - " << toJSON(value);
}

//===----------------------------------------------------------------------===//
// SignatureHelp
//===----------------------------------------------------------------------===//

llvm::json::Value mlir::lsp::toJSON(const SignatureHelp &value) {
  assert(value.activeSignature >= 0 &&
         "Unexpected negative value for number of active signatures.");
  assert(value.activeParameter >= 0 &&
         "Unexpected negative value for active parameter index");
  return llvm::json::Object{
      {"activeSignature", value.activeSignature},
      {"activeParameter", value.activeParameter},
      {"signatures", llvm::json::Array(value.signatures)},
  };
}

//===----------------------------------------------------------------------===//
// DocumentLinkParams
//===----------------------------------------------------------------------===//

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         DocumentLinkParams &result, llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("textDocument", result.textDocument);
}

//===----------------------------------------------------------------------===//
// DocumentLink
//===----------------------------------------------------------------------===//

llvm::json::Value mlir::lsp::toJSON(const DocumentLink &value) {
  return llvm::json::Object{
      {"range", value.range},
      {"target", value.target},
  };
}

//===----------------------------------------------------------------------===//
// InlayHintsParams
//===----------------------------------------------------------------------===//

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         InlayHintsParams &result, llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("textDocument", result.textDocument) &&
         o.map("range", result.range);
}

//===----------------------------------------------------------------------===//
// InlayHint
//===----------------------------------------------------------------------===//

llvm::json::Value mlir::lsp::toJSON(const InlayHint &value) {
  return llvm::json::Object{{"position", value.position},
                            {"kind", (int)value.kind},
                            {"label", value.label},
                            {"paddingLeft", value.paddingLeft},
                            {"paddingRight", value.paddingRight}};
}
bool mlir::lsp::operator==(const InlayHint &lhs, const InlayHint &rhs) {
  return std::tie(lhs.position, lhs.kind, lhs.label) ==
         std::tie(rhs.position, rhs.kind, rhs.label);
}
bool mlir::lsp::operator<(const InlayHint &lhs, const InlayHint &rhs) {
  return std::tie(lhs.position, lhs.kind, lhs.label) <
         std::tie(rhs.position, rhs.kind, rhs.label);
}

llvm::raw_ostream &mlir::lsp::operator<<(llvm::raw_ostream &os,
                                         InlayHintKind value) {
  switch (value) {
  case InlayHintKind::Parameter:
    return os << "parameter";
  case InlayHintKind::Type:
    return os << "type";
  }
  llvm_unreachable("Unknown InlayHintKind");
}

//===----------------------------------------------------------------------===//
// CodeActionContext
//===----------------------------------------------------------------------===//

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         CodeActionContext &result, llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  if (!o || !o.map("diagnostics", result.diagnostics))
    return false;
  o.map("only", result.only);
  return true;
}

//===----------------------------------------------------------------------===//
// CodeActionParams
//===----------------------------------------------------------------------===//

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         CodeActionParams &result, llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("textDocument", result.textDocument) &&
         o.map("range", result.range) && o.map("context", result.context);
}

//===----------------------------------------------------------------------===//
// WorkspaceEdit
//===----------------------------------------------------------------------===//

bool mlir::lsp::fromJSON(const llvm::json::Value &value, WorkspaceEdit &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("changes", result.changes);
}

llvm::json::Value mlir::lsp::toJSON(const WorkspaceEdit &value) {
  llvm::json::Object fileChanges;
  for (auto &change : value.changes)
    fileChanges[change.first] = llvm::json::Array(change.second);
  return llvm::json::Object{{"changes", std::move(fileChanges)}};
}

//===----------------------------------------------------------------------===//
// CodeAction
//===----------------------------------------------------------------------===//

const llvm::StringLiteral CodeAction::kQuickFix = "quickfix";
const llvm::StringLiteral CodeAction::kRefactor = "refactor";
const llvm::StringLiteral CodeAction::kInfo = "info";

llvm::json::Value mlir::lsp::toJSON(const CodeAction &value) {
  llvm::json::Object codeAction{{"title", value.title}};
  if (value.kind)
    codeAction["kind"] = *value.kind;
  if (value.diagnostics)
    codeAction["diagnostics"] = llvm::json::Array(*value.diagnostics);
  if (value.isPreferred)
    codeAction["isPreferred"] = true;
  if (value.edit)
    codeAction["edit"] = *value.edit;
  return std::move(codeAction);
}
