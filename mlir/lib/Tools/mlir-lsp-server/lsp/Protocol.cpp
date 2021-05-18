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
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
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
  if (!v || v->getAsNull().hasValue())
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

static bool isValidScheme(StringRef scheme) {
  if (scheme.empty())
    return false;
  if (!llvm::isAlpha(scheme[0]))
    return false;
  return std::all_of(scheme.begin() + 1, scheme.end(), [](char c) {
    return llvm::isAlnum(c) || c == '+' || c == '.' || c == '-';
  });
}

static llvm::Expected<std::string> uriFromAbsolutePath(StringRef absolutePath) {
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

  std::string uri = "file:";
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
  if (!isValidScheme(uriScheme))
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
  if (uriScheme != "file" && uriScheme != "test") {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "mlir-lsp-server only supports 'file' URI scheme for workspace files");
  }
  return getAbsolutePath(uriAuthority, uriBody);
}

llvm::Expected<URIForFile> URIForFile::fromURI(StringRef uri) {
  llvm::Expected<std::string> filePath = parseFilePathFromURI(uri);
  if (!filePath)
    return filePath.takeError();
  return URIForFile(std::move(*filePath), uri.str());
}

llvm::Expected<URIForFile> URIForFile::fromFile(StringRef absoluteFilepath) {
  llvm::Expected<std::string> uri = uriFromAbsolutePath(absoluteFilepath);
  if (!uri)
    return uri.takeError();
  return fromURI(*uri);
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
  if (hover.range.hasValue())
    result["range"] = toJSON(*hover.range);
  return std::move(result);
}

//===----------------------------------------------------------------------===//
// DiagnosticRelatedInformation
//===----------------------------------------------------------------------===//

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
