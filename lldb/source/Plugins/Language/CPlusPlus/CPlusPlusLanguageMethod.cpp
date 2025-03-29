//===-- CPlusPlusLanguageMethod.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CPlusPlusLanguage.h"

#include "lldb/Core/Mangled.h"

#include "CPlusPlusNameParser.h"
#include "MSVCUndecoratedNameParser.h"

using namespace lldb;
using namespace lldb_private;

void CPlusPlusLanguage::MethodName::Clear() {
  m_full.Clear();
  m_basename = llvm::StringRef();
  m_context = llvm::StringRef();
  m_arguments = llvm::StringRef();
  m_qualifiers = llvm::StringRef();
  m_return_type = llvm::StringRef();
  m_parsed = false;
  m_parse_error = false;
}

static bool ReverseFindMatchingChars(const llvm::StringRef &s,
                                     const llvm::StringRef &left_right_chars,
                                     size_t &left_pos, size_t &right_pos,
                                     size_t pos = llvm::StringRef::npos) {
  assert(left_right_chars.size() == 2);
  left_pos = llvm::StringRef::npos;
  const char left_char = left_right_chars[0];
  const char right_char = left_right_chars[1];
  pos = s.find_last_of(left_right_chars, pos);
  if (pos == llvm::StringRef::npos || s[pos] == left_char)
    return false;
  right_pos = pos;
  uint32_t depth = 1;
  while (pos > 0 && depth > 0) {
    pos = s.find_last_of(left_right_chars, pos);
    if (pos == llvm::StringRef::npos)
      return false;
    if (s[pos] == left_char) {
      if (--depth == 0) {
        left_pos = pos;
        return left_pos < right_pos;
      }
    } else if (s[pos] == right_char) {
      ++depth;
    }
  }
  return false;
}

static bool IsTrivialBasename(const llvm::StringRef &basename) {
  // Check that the basename matches with the following regular expression
  // "^~?([A-Za-z_][A-Za-z_0-9]*)$" We are using a hand written implementation
  // because it is significantly more efficient then using the general purpose
  // regular expression library.
  size_t idx = 0;
  if (basename.starts_with('~'))
    idx = 1;

  if (basename.size() <= idx)
    return false; // Empty string or "~"

  if (!std::isalpha(basename[idx]) && basename[idx] != '_')
    return false; // First character (after removing the possible '~'') isn't in
                  // [A-Za-z_]

  // Read all characters matching [A-Za-z_0-9]
  ++idx;
  while (idx < basename.size()) {
    if (!std::isalnum(basename[idx]) && basename[idx] != '_')
      break;
    ++idx;
  }

  // We processed all characters. It is a vaild basename.
  return idx == basename.size();
}

bool CPlusPlusLanguage::MethodName::TrySimplifiedParse() {
  // This method tries to parse simple method definitions which are presumably
  // most comman in user programs. Definitions that can be parsed by this
  // function don't have return types and templates in the name.
  // A::B::C::fun(std::vector<T> &) const
  size_t arg_start, arg_end;
  llvm::StringRef full(m_full.GetCString());
  llvm::StringRef parens("()", 2);
  if (ReverseFindMatchingChars(full, parens, arg_start, arg_end)) {
    m_arguments = full.substr(arg_start, arg_end - arg_start + 1);
    if (arg_end + 1 < full.size())
      m_qualifiers = full.substr(arg_end + 1).ltrim();

    if (arg_start == 0)
      return false;
    size_t basename_end = arg_start;
    size_t context_start = 0;
    size_t context_end = full.rfind(':', basename_end);
    if (context_end == llvm::StringRef::npos)
      m_basename = full.substr(0, basename_end);
    else {
      if (context_start < context_end)
        m_context = full.substr(context_start, context_end - 1 - context_start);
      const size_t basename_begin = context_end + 1;
      m_basename = full.substr(basename_begin, basename_end - basename_begin);
    }

    if (IsTrivialBasename(m_basename)) {
      return true;
    } else {
      // The C++ basename doesn't match our regular expressions so this can't
      // be a valid C++ method, clear everything out and indicate an error
      m_context = llvm::StringRef();
      m_basename = llvm::StringRef();
      m_arguments = llvm::StringRef();
      m_qualifiers = llvm::StringRef();
      m_return_type = llvm::StringRef();
      return false;
    }
  }
  return false;
}

void CPlusPlusLanguage::MethodName::Parse() {
  if (!m_parsed && m_full) {
    if (TrySimplifiedParse()) {
      m_parse_error = false;
    } else {
      CPlusPlusNameParser parser(m_full.GetStringRef());
      if (auto function = parser.ParseAsFunctionDefinition()) {
        m_basename = function->name.basename;
        m_context = function->name.context;
        m_arguments = function->arguments;
        m_qualifiers = function->qualifiers;
        m_return_type = function->return_type;
        m_parse_error = false;
      } else {
        m_parse_error = true;
      }
    }
    m_parsed = true;
  }
}

llvm::StringRef CPlusPlusLanguage::MethodName::GetBasename() {
  if (!m_parsed)
    Parse();
  return m_basename;
}

llvm::StringRef CPlusPlusLanguage::MethodName::GetContext() {
  if (!m_parsed)
    Parse();
  return m_context;
}

llvm::StringRef CPlusPlusLanguage::MethodName::GetArguments() {
  if (!m_parsed)
    Parse();
  return m_arguments;
}

llvm::StringRef CPlusPlusLanguage::MethodName::GetQualifiers() {
  if (!m_parsed)
    Parse();
  return m_qualifiers;
}

llvm::StringRef CPlusPlusLanguage::MethodName::GetReturnType() {
  if (!m_parsed)
    Parse();
  return m_return_type;
}

std::string CPlusPlusLanguage::MethodName::GetScopeQualifiedName() {
  if (!m_parsed)
    Parse();
  if (m_context.empty())
    return std::string(m_basename);

  std::string res;
  res += m_context;
  res += "::";
  res += m_basename;
  return res;
}

llvm::StringRef
CPlusPlusLanguage::MethodName::GetBasenameNoTemplateParameters() {
  llvm::StringRef basename = GetBasename();
  size_t arg_start, arg_end;
  llvm::StringRef parens("<>", 2);
  if (ReverseFindMatchingChars(basename, parens, arg_start, arg_end))
    return basename.substr(0, arg_start);

  return basename;
}

bool CPlusPlusLanguage::MethodName::ContainsPath(llvm::StringRef path) {
  if (!m_parsed)
    Parse();

  // If we can't parse the incoming name, then just check that it contains path.
  if (m_parse_error)
    return m_full.GetStringRef().contains(path);

  llvm::StringRef identifier;
  llvm::StringRef context;
  std::string path_str = path.str();
  bool success = CPlusPlusLanguage::ExtractContextAndIdentifier(
      path_str.c_str(), context, identifier);
  if (!success)
    return m_full.GetStringRef().contains(path);

  // Basename may include template arguments.
  // E.g.,
  // GetBaseName(): func<int>
  // identifier   : func
  //
  // ...but we still want to account for identifiers with template parameter
  // lists, e.g., when users set breakpoints on template specializations.
  //
  // E.g.,
  // GetBaseName(): func<uint32_t>
  // identifier   : func<int32_t*>
  //
  // Try to match the basename with or without template parameters.
  if (GetBasename() != identifier &&
      GetBasenameNoTemplateParameters() != identifier)
    return false;

  // Incoming path only had an identifier, so we match.
  if (context.empty())
    return true;
  // Incoming path has context but this method does not, no match.
  if (m_context.empty())
    return false;

  llvm::StringRef haystack = m_context;
  if (!haystack.consume_back(context))
    return false;
  if (haystack.empty() || !isalnum(haystack.back()))
    return true;

  return false;
}

bool CPlusPlusLanguage::IsCPPMangledName(llvm::StringRef name) {
  // FIXME!! we should really run through all the known C++ Language plugins
  // and ask each one if this is a C++ mangled name

  Mangled::ManglingScheme scheme = Mangled::GetManglingScheme(name);

  if (scheme == Mangled::eManglingSchemeNone)
    return false;

  return true;
}

bool CPlusPlusLanguage::ExtractContextAndIdentifier(
    const char *name, llvm::StringRef &context, llvm::StringRef &identifier) {
  if (MSVCUndecoratedNameParser::IsMSVCUndecoratedName(name))
    return MSVCUndecoratedNameParser::ExtractContextAndIdentifier(name, context,
                                                                  identifier);

  CPlusPlusNameParser parser(name);
  if (auto full_name = parser.ParseAsFullName()) {
    identifier = full_name->basename;
    context = full_name->context;
    return true;
  }
  return false;
}
