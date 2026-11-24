//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/XMLParser.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
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
  llvm::Error ParseMisc(bool &found);
  llvm::Error ParseElement();
  llvm::Error ParseStartTag(bool &is_empty);
  llvm::Error ParseAttValue(llvm::StringRef &value, size_t &buffer_offset);
  llvm::Error ParseReference(std::string &out);
  llvm::Error ParseComment();
  llvm::Error ParsePI();
  llvm::Error ParseName(llvm::StringRef &name);
  bool IsDuplicateAttribute(llvm::StringRef name);

  /// Character data is accumulated and reported in a single Characters call
  /// before the next start or end tag. When the data is a single contiguous
  /// slice of the input, no copy is made.
  void AppendText(llvm::StringRef text);
  std::string &TextBuffer();
  void FlushText();

  llvm::Error MakeError(const llvm::Twine &message) const;
  llvm::Error Expect(llvm::StringRef str);
  bool Consume(llvm::StringRef str);
  bool SkipSpace();
  bool AtEnd() const { return m_pos >= m_text.size(); }
  llvm::StringRef Rest() const { return m_text.drop_front(m_pos); }
  uint32_t PeekCodePoint(unsigned &length) const;

  Handler &m_handler;
  /// The document, either the input itself or m_storage if line endings had
  /// to be normalized.
  llvm::StringRef m_text;
  std::string m_storage;
  size_t m_pos = 0;
  std::vector<llvm::StringRef> m_open;

  // Scratch buffers reused across elements to avoid per-element allocations.
  llvm::SmallVector<Attribute, 8> m_attributes;
  llvm::DenseSet<llvm::StringRef> m_attribute_names;
  std::string m_attribute_buffer;
  llvm::StringRef m_pending_text;
  std::string m_text_buffer;
  bool m_text_in_buffer = false;
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

static void AppendUTF8(uint32_t c, std::string &out) {
  char buffer[UNI_MAX_UTF8_BYTES_PER_CODE_POINT];
  char *end = buffer;
  llvm::ConvertCodePointToUTF8(c, end);
  out.append(buffer, end);
}

llvm::Error Parser::MakeError(const llvm::Twine &message) const {
  // Decode can fail before line endings are normalized, so count "\r\n",
  // "\r" and "\n" as line breaks.
  llvm::StringRef prefix = m_text.take_front(m_pos);
  size_t line =
      prefix.count('\n') + prefix.count('\r') - prefix.count("\r\n") + 1;
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
  if (!AtEnd() && static_cast<unsigned char>(m_text[m_pos]) < 0x80) {
    length = 1;
    return m_text[m_pos];
  }
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
  input.consume_front("\xEF\xBB\xBF");
  m_text = input;

  // Validate the input in place and only copy it if line endings need to be
  // normalized.
  bool has_cr = false;
  const auto *begin = reinterpret_cast<const llvm::UTF8 *>(input.data());
  const auto *source = begin;
  const auto *end = begin + input.size();
  while (source != end) {
    m_pos = source - begin;
    if (*source < 0x80) {
      if (*source < 0x20 && *source != '\t' && *source != '\n' &&
          *source != '\r')
        return MakeError("invalid character U+" + llvm::utohexstr(*source));
      has_cr |= *source == '\r';
      ++source;
      continue;
    }
    llvm::UTF32 c;
    if (llvm::convertUTF8Sequence(&source, end, &c, llvm::strictConversion) !=
        llvm::conversionOK)
      return MakeError("invalid UTF-8 input");
    if (!IsChar(c))
      return MakeError("invalid character U+" + llvm::utohexstr(c));
  }
  m_pos = 0;

  if (has_cr) {
    m_storage.reserve(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
      if (input[i] != '\r') {
        m_storage.push_back(input[i]);
        continue;
      }
      m_storage.push_back('\n');
      if (i + 1 < input.size() && input[i + 1] == '\n')
        ++i;
    }
    m_text = m_storage;
  }
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
  name = m_text.slice(start, m_pos);
  return llvm::Error::success();
}

llvm::Error Parser::ParseComment() {
  size_t end = m_text.find("--", m_pos);
  if (end == llvm::StringRef::npos)
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
  if (end == llvm::StringRef::npos)
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

llvm::Error Parser::ParseReference(std::string &out) {
  if (Consume("#")) {
    unsigned radix = Consume("x") ? 16 : 10;
    size_t end = m_text.find(';', m_pos);
    llvm::StringRef digits = m_text.slice(m_pos, end);
    uint32_t c = 0;
    if (end == llvm::StringRef::npos || digits.empty() ||
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
  if (!replacement)
    return MakeError("undeclared entity '" + name + "'");
  out += replacement;
  return llvm::Error::success();
}

void Parser::AppendText(llvm::StringRef text) {
  if (text.empty())
    return;
  if (!m_text_in_buffer && m_pending_text.empty()) {
    m_pending_text = text;
    return;
  }
  TextBuffer().append(text);
}

std::string &Parser::TextBuffer() {
  if (!m_text_in_buffer) {
    m_text_buffer.assign(m_pending_text.data(), m_pending_text.size());
    m_text_in_buffer = true;
  }
  return m_text_buffer;
}

void Parser::FlushText() {
  llvm::StringRef text =
      m_text_in_buffer ? llvm::StringRef(m_text_buffer) : m_pending_text;
  if (!text.empty())
    m_handler.Characters(text);
  m_pending_text = {};
  m_text_buffer.clear();
  m_text_in_buffer = false;
}

llvm::Error Parser::ParseAttValue(llvm::StringRef &value,
                                  size_t &buffer_offset) {
  if (AtEnd() || (m_text[m_pos] != '"' && m_text[m_pos] != '\''))
    return MakeError("expected a quoted attribute value");
  char quote = m_text[m_pos++];

  // Most values need neither reference expansion nor whitespace
  // normalization, so they can refer to the input directly.
  size_t end = m_text.find(quote, m_pos);
  if (end != llvm::StringRef::npos) {
    llvm::StringRef raw = m_text.slice(m_pos, end);
    if (raw.find_first_of("<&\t\n") == llvm::StringRef::npos) {
      value = raw;
      buffer_offset = llvm::StringRef::npos;
      m_pos = end + 1;
      return llvm::Error::success();
    }
  }

  // Otherwise, append the normalized value to m_attribute_buffer. Only its
  // offset is returned since the buffer may still grow.
  buffer_offset = m_attribute_buffer.size();
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
      if (llvm::Error error = ParseReference(m_attribute_buffer))
        return error;
      continue;
    }
    m_attribute_buffer.push_back(IsSpace(c) ? ' ' : c);
    ++m_pos;
  }
}

bool Parser::IsDuplicateAttribute(llvm::StringRef name) {
  // A linear scan is cheapest for the handful of attributes elements usually
  // have. Switch to a set for larger counts to avoid quadratic behavior.
  const size_t linear_limit = 16;
  if (m_attributes.size() < linear_limit)
    return llvm::any_of(
        m_attributes, [&](const Attribute &attr) { return attr.name == name; });
  if (m_attribute_names.empty())
    for (const Attribute &attr : m_attributes)
      m_attribute_names.insert(attr.name);
  return !m_attribute_names.insert(name).second;
}

llvm::Error Parser::ParseStartTag(bool &is_empty) {
  llvm::StringRef element;
  if (llvm::Error error = ParseName(element))
    return error;
  m_attributes.clear();
  m_attribute_names.clear();
  m_attribute_buffer.clear();
  // Attribute indices and m_attribute_buffer offsets of normalized values.
  llvm::SmallVector<std::pair<size_t, size_t>, 4> buffered_values;

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
    llvm::StringRef name;
    if (llvm::Error error = ParseName(name))
      return error;
    if (IsDuplicateAttribute(name))
      return MakeError("duplicate attribute '" + name + "'");
    SkipSpace();
    if (llvm::Error error = Expect("="))
      return error;
    SkipSpace();
    llvm::StringRef value;
    size_t buffer_offset;
    if (llvm::Error error = ParseAttValue(value, buffer_offset))
      return error;
    if (buffer_offset != llvm::StringRef::npos)
      buffered_values.emplace_back(m_attributes.size(), buffer_offset);
    m_attributes.push_back({name, value});
  }

  // m_attribute_buffer is complete, so its values can now be referenced. Each
  // value extends to the next buffered value, or to the end of the buffer.
  for (size_t i = 0; i < buffered_values.size(); ++i) {
    size_t end = i + 1 < buffered_values.size() ? buffered_values[i + 1].second
                                                : m_attribute_buffer.size();
    auto [index, begin] = buffered_values[i];
    m_attributes[index].value =
        llvm::StringRef(m_attribute_buffer).slice(begin, end);
  }

  FlushText();
  m_handler.StartElement(element, m_attributes);
  if (is_empty)
    m_handler.EndElement(element);
  else
    m_open.push_back(element);
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
      FlushText();
      m_handler.EndElement(name);
      m_open.pop_back();
    } else if (Consume("<!--")) {
      if (llvm::Error error = ParseComment())
        return error;
    } else if (Consume("<![CDATA[")) {
      size_t end = m_text.find("]]>", m_pos);
      if (end == llvm::StringRef::npos)
        return MakeError("unterminated CDATA section");
      AppendText(m_text.slice(m_pos, end));
      m_pos = end + 3;
    } else if (Consume("<?")) {
      if (llvm::Error error = ParsePI())
        return error;
    } else if (Consume("<")) {
      if (llvm::Error error = ParseStartTag(is_empty))
        return error;
    } else if (Consume("&")) {
      if (llvm::Error error = ParseReference(TextBuffer()))
        return error;
    } else {
      size_t end = std::min(m_text.find_first_of("<&", m_pos), m_text.size());
      llvm::StringRef text = m_text.slice(m_pos, end);
      if (text.contains("]]>"))
        return MakeError("']]>' in character data");
      AppendText(text);
      m_pos = end;
    }
  }
  return llvm::Error::success();
}

llvm::Error Parser::Parse(llvm::StringRef input) {
  if (llvm::Error error = Decode(input))
    return error;

  bool found;
  do {
    if (llvm::Error error = ParseMisc(found))
      return error;
  } while (found);

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
