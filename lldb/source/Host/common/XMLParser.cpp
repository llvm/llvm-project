//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/XMLParser.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ConvertUTF.h"

using namespace lldb_private;
using namespace lldb_private::xml;

namespace {

class Parser {
public:
  explicit Parser(Handler &handler) : m_handler(handler) {}
  llvm::Error Parse(llvm::StringRef input);

private:
  llvm::Error Decode(llvm::StringRef input);
  llvm::Error ParseXMLDecl();
  llvm::Error ParseMisc(bool &found);
  llvm::Error ParseDoctype();
  llvm::Error ParseExternalID();
  llvm::Error ParseElement();
  llvm::Error ParseStartTag(bool &is_empty);
  llvm::Error ParseAttValue(std::string &value);
  llvm::Error ParseReference(std::string &out);
  llvm::Error ParseComment();
  llvm::Error ParsePI();
  llvm::Error ParseName(llvm::StringRef &name);
  llvm::Error ParseQuoted(llvm::StringRef &value);
  llvm::Expected<bool> ParseDeclAttribute(llvm::StringRef name,
                                          llvm::StringRef &value);
  void Characters(llvm::StringRef text) {
    if (!text.empty())
      m_handler.Characters(text);
  }

  llvm::Error MakeError(const llvm::Twine &message) const;
  llvm::Error Expect(llvm::StringRef str);
  bool Consume(llvm::StringRef str);
  bool SkipSpace();
  bool AtEnd() const { return m_pos >= m_text.size(); }
  llvm::StringRef Rest() const {
    return llvm::StringRef(m_text).drop_front(m_pos);
  }
  uint32_t PeekCodePoint(unsigned &length) const;

  Handler &m_handler;
  std::string m_text;
  size_t m_pos = 0;
  std::vector<std::string> m_open;
  bool m_utf16 = false;
  bool m_standalone = false;
  bool m_has_unread_decls = false;
};

} // namespace

Handler::~Handler() = default;

static bool IsSpace(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

static bool IsChar(uint32_t c) {
  return c == 0x9 || c == 0xA || c == 0xD || (c >= 0x20 && c <= 0xD7FF) ||
         (c >= 0xE000 && c <= 0xFFFD) || (c >= 0x10000 && c <= 0x10FFFF);
}

static bool IsNameStartChar(uint32_t c) {
  return c == ':' || (c >= 'A' && c <= 'Z') || c == '_' ||
         (c >= 'a' && c <= 'z') || (c >= 0xC0 && c <= 0xD6) ||
         (c >= 0xD8 && c <= 0xF6) || (c >= 0xF8 && c <= 0x2FF) ||
         (c >= 0x370 && c <= 0x37D) || (c >= 0x37F && c <= 0x1FFF) ||
         (c >= 0x200C && c <= 0x200D) || (c >= 0x2070 && c <= 0x218F) ||
         (c >= 0x2C00 && c <= 0x2FEF) || (c >= 0x3001 && c <= 0xD7FF) ||
         (c >= 0xF900 && c <= 0xFDCF) || (c >= 0xFDF0 && c <= 0xFFFD) ||
         (c >= 0x10000 && c <= 0xEFFFF);
}

static bool IsNameChar(uint32_t c) {
  return IsNameStartChar(c) || c == '-' || c == '.' || (c >= '0' && c <= '9') ||
         c == 0xB7 || (c >= 0x300 && c <= 0x36F) ||
         (c >= 0x203F && c <= 0x2040);
}

static bool IsPubidChar(char c) {
  return c == ' ' || c == '\r' || c == '\n' || llvm::isAlnum(c) ||
         llvm::StringRef("-'()+,./:=?;!*#@$_%").contains(c);
}

static void AppendUTF8(uint32_t c, std::string &out) {
  char buffer[UNI_MAX_UTF8_BYTES_PER_CODE_POINT];
  char *end = buffer;
  llvm::ConvertCodePointToUTF8(c, end);
  out.append(buffer, end);
}

llvm::Error Parser::MakeError(const llvm::Twine &message) const {
  size_t line = llvm::StringRef(m_text).take_front(m_pos).count('\n') + 1;
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "line " + llvm::Twine(line) + ": " + message);
}

bool Parser::Consume(llvm::StringRef str) {
  if (!Rest().starts_with(str))
    return false;
  m_pos += str.size();
  return true;
}

llvm::Error Parser::Expect(llvm::StringRef str) {
  if (Consume(str))
    return llvm::Error::success();
  return MakeError("expected '" + str + "'");
}

bool Parser::SkipSpace() {
  size_t start = m_pos;
  while (!AtEnd() && IsSpace(m_text[m_pos]))
    ++m_pos;
  return m_pos != start;
}

uint32_t Parser::PeekCodePoint(unsigned &length) const {
  const auto *start =
      reinterpret_cast<const llvm::UTF8 *>(m_text.data() + m_pos);
  const llvm::UTF8 *source = start;
  llvm::UTF32 c = 0;
  if (AtEnd() ||
      llvm::convertUTF8Sequence(&source, start + (m_text.size() - m_pos), &c,
                                llvm::strictConversion) != llvm::conversionOK)
    c = 0;
  length = source - start;
  return c;
}

llvm::Error Parser::Decode(llvm::StringRef input) {
  std::string utf8;
  if (input.starts_with("\xFE\xFF") || input.starts_with("\xFF\xFE")) {
    if (input.size() % 2)
      return MakeError("invalid UTF-16 input");
    bool big_endian = input[0] == '\xFE';
    std::vector<llvm::UTF16> units;
    for (size_t i = 2; i < input.size(); i += 2) {
      uint8_t first = input[i], second = input[i + 1];
      units.push_back(big_endian ? (first << 8 | second)
                                 : (second << 8 | first));
    }
    if ((!units.empty() &&
         (units.front() == 0xFEFF || units.front() == 0xFFFE)) ||
        !llvm::convertUTF16ToUTF8String(units, utf8))
      return MakeError("invalid UTF-16 input");
    input = utf8;
    m_utf16 = true;
  } else {
    input.consume_front("\xEF\xBB\xBF");
  }

  m_text.reserve(input.size());
  const auto *source = reinterpret_cast<const llvm::UTF8 *>(input.data());
  const auto *end = source + input.size();
  while (source != end) {
    m_pos = m_text.size();
    const llvm::UTF8 *start = source;
    llvm::UTF32 c;
    if (llvm::convertUTF8Sequence(&source, end, &c, llvm::strictConversion) !=
        llvm::conversionOK)
      return MakeError("invalid UTF-8 input");
    if (!IsChar(c))
      return MakeError("invalid character U+" + llvm::utohexstr(c));
    if (c == '\r') {
      m_text.push_back('\n');
      if (source != end && *source == '\n')
        ++source;
    } else {
      m_text.append(reinterpret_cast<const char *>(start), source - start);
    }
  }
  m_pos = 0;
  return llvm::Error::success();
}

llvm::Error Parser::ParseName(llvm::StringRef &name) {
  size_t start = m_pos;
  unsigned length;
  if (!IsNameStartChar(PeekCodePoint(length)))
    return MakeError("expected a name");
  m_pos += length;
  while (IsNameChar(PeekCodePoint(length)))
    m_pos += length;
  name = llvm::StringRef(m_text).slice(start, m_pos);
  return llvm::Error::success();
}

llvm::Error Parser::ParseQuoted(llvm::StringRef &value) {
  if (AtEnd() || (m_text[m_pos] != '"' && m_text[m_pos] != '\''))
    return MakeError("expected a quoted string");
  char quote = m_text[m_pos++];
  size_t end = m_text.find(quote, m_pos);
  if (end == std::string::npos)
    return MakeError("unterminated quoted string");
  value = llvm::StringRef(m_text).slice(m_pos, end);
  m_pos = end + 1;
  return llvm::Error::success();
}

llvm::Expected<bool> Parser::ParseDeclAttribute(llvm::StringRef name,
                                                llvm::StringRef &value) {
  size_t start = m_pos;
  if (!SkipSpace() || !Consume(name)) {
    m_pos = start;
    return false;
  }
  SkipSpace();
  if (llvm::Error error = Expect("="))
    return std::move(error);
  SkipSpace();
  if (llvm::Error error = ParseQuoted(value))
    return std::move(error);
  return true;
}

llvm::Error Parser::ParseXMLDecl() {
  llvm::StringRef version, encoding, standalone;
  llvm::Expected<bool> found = ParseDeclAttribute("version", version);
  if (!found)
    return found.takeError();
  if (!*found)
    return MakeError("expected a version in the XML declaration");
  if (!version.consume_front("1.") || version.empty() ||
      !llvm::all_of(version, llvm::isDigit))
    return MakeError("unsupported XML version");

  found = ParseDeclAttribute("encoding", encoding);
  if (!found)
    return found.takeError();
  if (*found) {
    if (encoding.empty() || !llvm::isAlpha(encoding[0]) ||
        !llvm::all_of(encoding, [](char c) {
          return llvm::isAlnum(c) || c == '.' || c == '_' || c == '-';
        }))
      return MakeError("invalid encoding name");
    bool matches = m_utf16 ? encoding.equals_insensitive("UTF-16")
                           : encoding.equals_insensitive("UTF-8");
    if (!matches)
      return MakeError("unsupported encoding '" + encoding + "'");
  }

  found = ParseDeclAttribute("standalone", standalone);
  if (!found)
    return found.takeError();
  if (*found && standalone != "yes" && standalone != "no")
    return MakeError("invalid standalone declaration");
  m_standalone = standalone == "yes";

  SkipSpace();
  return Expect("?>");
}

llvm::Error Parser::ParseComment() {
  size_t end = m_text.find("--", m_pos);
  if (end == std::string::npos)
    return MakeError("unterminated comment");
  m_pos = end + 2;
  return Expect(">");
}

llvm::Error Parser::ParsePI() {
  llvm::StringRef target;
  if (llvm::Error error = ParseName(target))
    return error;
  if (target.equals_insensitive("xml"))
    return MakeError("reserved processing instruction target");
  if (Consume("?>"))
    return llvm::Error::success();
  if (!SkipSpace())
    return MakeError("expected whitespace after the processing instruction "
                     "target");
  size_t end = m_text.find("?>", m_pos);
  if (end == std::string::npos)
    return MakeError("unterminated processing instruction");
  m_pos = end + 2;
  return llvm::Error::success();
}

llvm::Error Parser::ParseMisc(bool &found) {
  found = true;
  if (SkipSpace())
    return llvm::Error::success();
  if (Consume("<!--"))
    return ParseComment();
  if (Consume("<?"))
    return ParsePI();
  found = false;
  return llvm::Error::success();
}

llvm::Error Parser::ParseExternalID() {
  llvm::StringRef literal;
  if (Consume("PUBLIC")) {
    if (!SkipSpace())
      return MakeError("expected whitespace");
    if (llvm::Error error = ParseQuoted(literal))
      return error;
    if (!llvm::all_of(literal, IsPubidChar))
      return MakeError("invalid character in public identifier");
  } else if (!Consume("SYSTEM")) {
    return MakeError("expected 'SYSTEM' or 'PUBLIC'");
  }
  if (!SkipSpace())
    return MakeError("expected whitespace");
  return ParseQuoted(literal);
}

llvm::Error Parser::ParseDoctype() {
  llvm::StringRef name;
  if (!SkipSpace())
    return MakeError("expected whitespace");
  if (llvm::Error error = ParseName(name))
    return error;
  bool space = SkipSpace();
  if (space && !Rest().starts_with("[") && !Rest().starts_with(">")) {
    if (llvm::Error error = ParseExternalID())
      return error;
    m_has_unread_decls = !m_standalone;
    SkipSpace();
  }
  if (Rest().starts_with("["))
    return MakeError("internal DTD subsets are not supported");
  return Expect(">");
}

llvm::Error Parser::ParseReference(std::string &out) {
  if (Consume("#")) {
    unsigned radix = Consume("x") ? 16 : 10;
    size_t end = m_text.find(';', m_pos);
    llvm::StringRef digits = llvm::StringRef(m_text).slice(m_pos, end);
    uint32_t c = 0;
    if (end == std::string::npos || digits.empty() ||
        !llvm::all_of(digits, radix == 16 ? llvm::isHexDigit : llvm::isDigit) ||
        digits.getAsInteger(radix, c) || !IsChar(c))
      return MakeError("invalid character reference");
    AppendUTF8(c, out);
    m_pos = end + 1;
    return llvm::Error::success();
  }

  llvm::StringRef name;
  if (llvm::Error error = ParseName(name))
    return error;
  if (llvm::Error error = Expect(";"))
    return error;
  const char *replacement = llvm::StringSwitch<const char *>(name)
                                .Case("lt", "<")
                                .Case("gt", ">")
                                .Case("amp", "&")
                                .Case("apos", "'")
                                .Case("quot", "\"")
                                .Default(nullptr);
  if (replacement)
    out += replacement;
  else if (!m_has_unread_decls)
    return MakeError("undeclared entity '" + name + "'");
  return llvm::Error::success();
}

llvm::Error Parser::ParseAttValue(std::string &value) {
  if (AtEnd() || (m_text[m_pos] != '"' && m_text[m_pos] != '\''))
    return MakeError("expected a quoted attribute value");
  char quote = m_text[m_pos++];
  while (true) {
    if (AtEnd())
      return MakeError("unterminated attribute value");
    char c = m_text[m_pos];
    if (c == quote) {
      ++m_pos;
      return llvm::Error::success();
    }
    if (c == '<')
      return MakeError("'<' in attribute value");
    if (c == '&') {
      ++m_pos;
      if (llvm::Error error = ParseReference(value))
        return error;
      continue;
    }
    value.push_back(IsSpace(c) ? ' ' : c);
    ++m_pos;
  }
}

llvm::Error Parser::ParseStartTag(bool &is_empty) {
  llvm::StringRef name;
  if (llvm::Error error = ParseName(name))
    return error;
  std::string element = name.str();
  std::vector<std::pair<std::string, std::string>> attributes;
  llvm::StringSet<> names;

  while (true) {
    bool space = SkipSpace();
    if (Consume("/>")) {
      is_empty = true;
      break;
    }
    if (Consume(">")) {
      is_empty = false;
      break;
    }
    if (!space)
      return MakeError("expected whitespace, '>' or '/>'");
    if (llvm::Error error = ParseName(name))
      return error;
    if (!names.insert(name).second)
      return MakeError("duplicate attribute '" + name + "'");
    SkipSpace();
    if (llvm::Error error = Expect("="))
      return error;
    SkipSpace();
    std::string value;
    if (llvm::Error error = ParseAttValue(value))
      return error;
    attributes.emplace_back(name.str(), std::move(value));
  }

  std::vector<Attribute> refs;
  for (const auto &[attr_name, value] : attributes)
    refs.push_back({attr_name, value});
  m_handler.StartElement(element, refs);
  if (is_empty)
    m_handler.EndElement(element);
  else
    m_open.push_back(std::move(element));
  return llvm::Error::success();
}

llvm::Error Parser::ParseElement() {
  if (!Consume("<"))
    return MakeError("expected the root element");
  bool is_empty;
  if (llvm::Error error = ParseStartTag(is_empty))
    return error;

  while (!m_open.empty()) {
    if (AtEnd())
      return MakeError("missing end tag for element '" + m_open.back() + "'");
    if (Consume("</")) {
      llvm::StringRef name;
      if (llvm::Error error = ParseName(name))
        return error;
      if (name != m_open.back())
        return MakeError("end tag '" + name + "' does not match '" +
                         m_open.back() + "'");
      SkipSpace();
      if (llvm::Error error = Expect(">"))
        return error;
      m_handler.EndElement(name);
      m_open.pop_back();
    } else if (Consume("<!--")) {
      if (llvm::Error error = ParseComment())
        return error;
    } else if (Consume("<![CDATA[")) {
      size_t end = m_text.find("]]>", m_pos);
      if (end == std::string::npos)
        return MakeError("unterminated CDATA section");
      Characters(llvm::StringRef(m_text).slice(m_pos, end));
      m_pos = end + 3;
    } else if (Consume("<?")) {
      if (llvm::Error error = ParsePI())
        return error;
    } else if (Consume("<")) {
      if (llvm::Error error = ParseStartTag(is_empty))
        return error;
    } else if (Consume("&")) {
      std::string text;
      if (llvm::Error error = ParseReference(text))
        return error;
      Characters(text);
    } else {
      size_t end = std::min(m_text.find_first_of("<&", m_pos), m_text.size());
      llvm::StringRef text = llvm::StringRef(m_text).slice(m_pos, end);
      if (text.contains("]]>"))
        return MakeError("']]>' in character data");
      Characters(text);
      m_pos = end;
    }
  }
  return llvm::Error::success();
}

llvm::Error Parser::Parse(llvm::StringRef input) {
  if (llvm::Error error = Decode(input))
    return error;

  if (Rest().starts_with("<?xml") && Rest().size() > 5 && IsSpace(Rest()[5])) {
    m_pos += 5;
    if (llvm::Error error = ParseXMLDecl())
      return error;
  }

  bool found;
  do {
    if (llvm::Error error = ParseMisc(found))
      return error;
  } while (found);

  if (Consume("<!DOCTYPE")) {
    if (llvm::Error error = ParseDoctype())
      return error;
    do {
      if (llvm::Error error = ParseMisc(found))
        return error;
    } while (found);
  }

  if (llvm::Error error = ParseElement())
    return error;

  do {
    if (llvm::Error error = ParseMisc(found))
      return error;
  } while (found);
  if (!AtEnd())
    return MakeError("extra content at the end of the document");
  return llvm::Error::success();
}

llvm::Error xml::Parse(llvm::StringRef input, Handler &handler) {
  return Parser(handler).Parse(input);
}
