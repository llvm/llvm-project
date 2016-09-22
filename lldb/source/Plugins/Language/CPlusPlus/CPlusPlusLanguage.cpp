//===-- CPlusPlusLanguage.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CPlusPlusLanguage.h"

// C Includes
// C++ Includes
#include <cctype>
#include <cstring>
#include <functional>
#include <mutex>

// Other libraries and framework includes
#include "llvm/ADT/StringRef.h"

// Project includes
#include "lldb/Core/ConstString.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/UniqueCStringMap.h"
#include "lldb/DataFormatters/CXXFunctionPointer.h"
#include "lldb/DataFormatters/DataVisualization.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/DataFormatters/VectorType.h"

#include "BlockPointer.h"
#include "CxxStringTypes.h"
#include "LibCxx.h"
#include "LibCxxAtomic.h"
#include "LibStdcpp.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

void CPlusPlusLanguage::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(), "C++ Language",
                                CreateInstance);
}

void CPlusPlusLanguage::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

lldb_private::ConstString CPlusPlusLanguage::GetPluginNameStatic() {
  static ConstString g_name("cplusplus");
  return g_name;
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------

lldb_private::ConstString CPlusPlusLanguage::GetPluginName() {
  return GetPluginNameStatic();
}

uint32_t CPlusPlusLanguage::GetPluginVersion() { return 1; }

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------

Language *CPlusPlusLanguage::CreateInstance(lldb::LanguageType language) {
  if (Language::LanguageIsCPlusPlus(language))
    return new CPlusPlusLanguage();
  return nullptr;
}

void CPlusPlusLanguage::MethodName::Clear() {
  m_full.Clear();
  m_basename = llvm::StringRef();
  m_context = llvm::StringRef();
  m_arguments = llvm::StringRef();
  m_qualifiers = llvm::StringRef();
  m_type = eTypeInvalid;
  m_parsed = false;
  m_parse_error = false;
}

bool ReverseFindMatchingChars(const llvm::StringRef &s,
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

static bool IsValidBasename(const llvm::StringRef &basename) {
  // Check that the basename matches with the following regular expression or is
  // an operator name:
  // "^~?([A-Za-z_][A-Za-z_0-9]*)(<.*>)?$"
  // We are using a hand written implementation because it is significantly more
  // efficient then
  // using the general purpose regular expression library.
  size_t idx = 0;
  if (basename.size() > 0 && basename[0] == '~')
    idx = 1;

  if (basename.size() <= idx)
    return false; // Empty string or "~"

  if (!std::isalpha(basename[idx]) && basename[idx] != '_')
    return false; // First charater (after removing the possible '~'') isn't in
                  // [A-Za-z_]

  // Read all characters matching [A-Za-z_0-9]
  ++idx;
  while (idx < basename.size()) {
    if (!std::isalnum(basename[idx]) && basename[idx] != '_')
      break;
    ++idx;
  }

  // We processed all characters. It is a vaild basename.
  if (idx == basename.size())
    return true;

  // Check for basename with template arguments
  // TODO: Improve the quality of the validation with validating the template
  // arguments
  if (basename[idx] == '<' && basename.back() == '>')
    return true;

  // Check if the basename is a vaild C++ operator name
  if (!basename.startswith("operator"))
    return false;

  static RegularExpression g_operator_regex(
      llvm::StringRef("^(operator)( "
                      "?)([A-Za-z_][A-Za-z_0-9]*|\\(\\)|"
                      "\\[\\]|[\\^<>=!\\/"
                      "*+-]+)(<.*>)?(\\[\\])?$"));
  std::string basename_str(basename.str());
  return g_operator_regex.Execute(basename_str, nullptr);
}

void CPlusPlusLanguage::MethodName::Parse() {
  if (!m_parsed && m_full) {
    //        ConstString mangled;
    //        m_full.GetMangledCounterpart(mangled);
    //        printf ("\n   parsing = '%s'\n", m_full.GetCString());
    //        if (mangled)
    //            printf ("   mangled = '%s'\n", mangled.GetCString());
    m_parse_error = false;
    m_parsed = true;
    llvm::StringRef full(m_full.GetCString());

    size_t arg_start, arg_end;
    llvm::StringRef parens("()", 2);
    if (ReverseFindMatchingChars(full, parens, arg_start, arg_end)) {
      m_arguments = full.substr(arg_start, arg_end - arg_start + 1);
      if (arg_end + 1 < full.size())
        m_qualifiers = full.substr(arg_end + 1);
      if (arg_start > 0) {
        size_t basename_end = arg_start;
        size_t context_start = 0;
        size_t context_end = llvm::StringRef::npos;
        if (basename_end > 0 && full[basename_end - 1] == '>') {
          // TODO: handle template junk...
          // Templated function
          size_t template_start, template_end;
          llvm::StringRef lt_gt("<>", 2);
          if (ReverseFindMatchingChars(full, lt_gt, template_start,
                                       template_end, basename_end)) {
            // Check for templated functions that include return type like:
            // 'void foo<Int>()'
            context_start = full.rfind(' ', template_start);
            if (context_start == llvm::StringRef::npos)
              context_start = 0;
            else
              ++context_start;

            context_end = full.rfind(':', template_start);
            if (context_end == llvm::StringRef::npos ||
                context_end < context_start)
              context_end = context_start;
          } else {
            context_end = full.rfind(':', basename_end);
          }
        } else if (context_end == llvm::StringRef::npos) {
          context_end = full.rfind(':', basename_end);
        }

        if (context_end == llvm::StringRef::npos)
          m_basename = full.substr(0, basename_end);
        else {
          if (context_start < context_end)
            m_context =
                full.substr(context_start, context_end - 1 - context_start);
          const size_t basename_begin = context_end + 1;
          m_basename =
              full.substr(basename_begin, basename_end - basename_begin);
        }
        m_type = eTypeUnknownMethod;
      } else {
        m_parse_error = true;
        return;
      }

      if (!IsValidBasename(m_basename)) {
        // The C++ basename doesn't match our regular expressions so this can't
        // be a valid C++ method, clear everything out and indicate an error
        m_context = llvm::StringRef();
        m_basename = llvm::StringRef();
        m_arguments = llvm::StringRef();
        m_qualifiers = llvm::StringRef();
        m_parse_error = true;
      }
    } else {
      m_parse_error = true;
    }
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

std::string CPlusPlusLanguage::MethodName::GetScopeQualifiedName() {
  if (!m_parsed)
    Parse();
  if (m_basename.empty() || m_context.empty())
    return std::string();

  std::string res;
  res += m_context;
  res += "::";
  res += m_basename;

  return res;
}

bool CPlusPlusLanguage::IsCPPMangledName(const char *name) {
  // FIXME, we should really run through all the known C++ Language plugins and
  // ask each one if
  // this is a C++ mangled name, but we can put that off till there is actually
  // more than one
  // we care about.

  return (name != nullptr && name[0] == '_' && name[1] == 'Z');
}

bool CPlusPlusLanguage::ExtractContextAndIdentifier(
    const char *name, llvm::StringRef &context, llvm::StringRef &identifier) {
  static RegularExpression g_basename_regex(llvm::StringRef(
      "^(([A-Za-z_][A-Za-z_0-9]*::)*)(~?[A-Za-z_~][A-Za-z_0-9]*)$"));
  RegularExpression::Match match(4);
  if (g_basename_regex.Execute(llvm::StringRef::withNullAsEmpty(name),
                               &match)) {
    match.GetMatchAtIndex(name, 1, context);
    match.GetMatchAtIndex(name, 3, identifier);
    return true;
  }
  return false;
}

class CPPRuntimeEquivalents {
public:
  CPPRuntimeEquivalents() {
    m_impl.Append(ConstString("std::basic_string<char, std::char_traits<char>, "
                              "std::allocator<char> >")
                      .AsCString(),
                  ConstString("basic_string<char>"));

    // these two (with a prefixed std::) occur when c++stdlib string class
    // occurs as a template argument in some STL container
    m_impl.Append(ConstString("std::basic_string<char, std::char_traits<char>, "
                              "std::allocator<char> >")
                      .AsCString(),
                  ConstString("std::basic_string<char>"));

    m_impl.Sort();
  }

  void Add(ConstString &type_name, ConstString &type_equivalent) {
    m_impl.Insert(type_name.AsCString(), type_equivalent);
  }

  uint32_t FindExactMatches(ConstString &type_name,
                            std::vector<ConstString> &equivalents) {
    uint32_t count = 0;

    for (ImplData match = m_impl.FindFirstValueForName(type_name.AsCString());
         match != nullptr; match = m_impl.FindNextValueForName(match)) {
      equivalents.push_back(match->value);
      count++;
    }

    return count;
  }

  // partial matches can occur when a name with equivalents is a template
  // argument.
  // e.g. we may have "class Foo" be a match for "struct Bar". if we have a
  // typename
  // such as "class Templatized<class Foo, Anything>" we want this to be
  // replaced with
  // "class Templatized<struct Bar, Anything>". Since partial matching is time
  // consuming
  // once we get a partial match, we add it to the exact matches list for faster
  // retrieval
  uint32_t FindPartialMatches(ConstString &type_name,
                              std::vector<ConstString> &equivalents) {
    uint32_t count = 0;

    const char *type_name_cstr = type_name.AsCString();

    size_t items_count = m_impl.GetSize();

    for (size_t item = 0; item < items_count; item++) {
      const char *key_cstr = m_impl.GetCStringAtIndex(item);
      if (strstr(type_name_cstr, key_cstr)) {
        count += AppendReplacements(type_name_cstr, key_cstr, equivalents);
      }
    }

    return count;
  }

private:
  std::string &replace(std::string &target, std::string &pattern,
                       std::string &with) {
    size_t pos;
    size_t pattern_len = pattern.size();

    while ((pos = target.find(pattern)) != std::string::npos)
      target.replace(pos, pattern_len, with);

    return target;
  }

  uint32_t AppendReplacements(const char *original, const char *matching_key,
                              std::vector<ConstString> &equivalents) {
    std::string matching_key_str(matching_key);
    ConstString original_const(original);

    uint32_t count = 0;

    for (ImplData match = m_impl.FindFirstValueForName(matching_key);
         match != nullptr; match = m_impl.FindNextValueForName(match)) {
      std::string target(original);
      std::string equiv_class(match->value.AsCString());

      replace(target, matching_key_str, equiv_class);

      ConstString target_const(target.c_str());

// you will most probably want to leave this off since it might make this map
// grow indefinitely
#ifdef ENABLE_CPP_EQUIVALENTS_MAP_TO_GROW
      Add(original_const, target_const);
#endif
      equivalents.push_back(target_const);

      count++;
    }

    return count;
  }

  typedef UniqueCStringMap<ConstString> Impl;
  typedef const Impl::Entry *ImplData;
  Impl m_impl;
};

static CPPRuntimeEquivalents &GetEquivalentsMap() {
  static CPPRuntimeEquivalents g_equivalents_map;
  return g_equivalents_map;
}

uint32_t
CPlusPlusLanguage::FindEquivalentNames(ConstString type_name,
                                       std::vector<ConstString> &equivalents) {
  uint32_t count = GetEquivalentsMap().FindExactMatches(type_name, equivalents);

  bool might_have_partials =
      (count == 0) // if we have a full name match just use it
      && (strchr(type_name.AsCString(), '<') !=
              nullptr // we should only have partial matches when templates are
                      // involved, check that we have
          && strchr(type_name.AsCString(), '>') != nullptr); // angle brackets
                                                             // in the type_name
                                                             // before trying to
                                                             // scan for partial
                                                             // matches

  if (might_have_partials)
    count = GetEquivalentsMap().FindPartialMatches(type_name, equivalents);

  return count;
}

static void LoadLibCxxFormatters(lldb::TypeCategoryImplSP cpp_category_sp) {
  if (!cpp_category_sp)
    return;

  TypeSummaryImpl::Flags stl_summary_flags;
  stl_summary_flags.SetCascades(true)
      .SetSkipPointers(false)
      .SetSkipReferences(false)
      .SetDontShowChildren(true)
      .SetDontShowValue(true)
      .SetShowMembersOneLiner(false)
      .SetHideItemNames(false);

#ifndef LLDB_DISABLE_PYTHON
  lldb::TypeSummaryImplSP std_string_summary_sp(new CXXFunctionSummaryFormat(
      stl_summary_flags, lldb_private::formatters::LibcxxStringSummaryProvider,
      "std::string summary provider"));
  lldb::TypeSummaryImplSP std_wstring_summary_sp(new CXXFunctionSummaryFormat(
      stl_summary_flags, lldb_private::formatters::LibcxxWStringSummaryProvider,
      "std::wstring summary provider"));

  cpp_category_sp->GetTypeSummariesContainer()->Add(
      ConstString("std::__1::string"), std_string_summary_sp);
  cpp_category_sp->GetTypeSummariesContainer()->Add(
      ConstString("std::__ndk1::string"), std_string_summary_sp);
  cpp_category_sp->GetTypeSummariesContainer()->Add(
      ConstString("std::__1::basic_string<char, std::__1::char_traits<char>, "
                  "std::__1::allocator<char> >"),
      std_string_summary_sp);
  cpp_category_sp->GetTypeSummariesContainer()->Add(
      ConstString("std::__ndk1::basic_string<char, "
                  "std::__ndk1::char_traits<char>, "
                  "std::__ndk1::allocator<char> >"),
      std_string_summary_sp);

  cpp_category_sp->GetTypeSummariesContainer()->Add(
      ConstString("std::__1::wstring"), std_wstring_summary_sp);
  cpp_category_sp->GetTypeSummariesContainer()->Add(
      ConstString("std::__ndk1::wstring"), std_wstring_summary_sp);
  cpp_category_sp->GetTypeSummariesContainer()->Add(
      ConstString("std::__1::basic_string<wchar_t, "
                  "std::__1::char_traits<wchar_t>, "
                  "std::__1::allocator<wchar_t> >"),
      std_wstring_summary_sp);
  cpp_category_sp->GetTypeSummariesContainer()->Add(
      ConstString("std::__ndk1::basic_string<wchar_t, "
                  "std::__ndk1::char_traits<wchar_t>, "
                  "std::__ndk1::allocator<wchar_t> >"),
      std_wstring_summary_sp);

  SyntheticChildren::Flags stl_synth_flags;
  stl_synth_flags.SetCascades(true).SetSkipPointers(false).SetSkipReferences(
      false);

  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEndCreator,
      "libc++ std::vector<bool> synthetic children",
      ConstString(
          "^std::__(ndk)?1::vector<bool, std::__(ndk)?1::allocator<bool> >$"),
      stl_synth_flags, true);
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxStdVectorSyntheticFrontEndCreator,
      "libc++ std::vector synthetic children",
      ConstString("^std::__(ndk)?1::vector<.+>(( )?&)?$"), stl_synth_flags,
      true);
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxStdListSyntheticFrontEndCreator,
      "libc++ std::list synthetic children",
      ConstString("^std::__(ndk)?1::list<.+>(( )?&)?$"), stl_synth_flags, true);
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxStdMapSyntheticFrontEndCreator,
      "libc++ std::map synthetic children",
      ConstString("^std::__(ndk)?1::map<.+> >(( )?&)?$"), stl_synth_flags,
      true);
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEndCreator,
      "libc++ std::vector<bool> synthetic children",
      ConstString("std::__(ndk)?1::vector<std::__(ndk)?1::allocator<bool> >"),
      stl_synth_flags);
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEndCreator,
      "libc++ std::vector<bool> synthetic children",
      ConstString(
          "std::__(ndk)?1::vector<bool, std::__(ndk)?1::allocator<bool> >"),
      stl_synth_flags);
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxStdMapSyntheticFrontEndCreator,
      "libc++ std::set synthetic children",
      ConstString("^std::__(ndk)?1::set<.+> >(( )?&)?$"), stl_synth_flags,
      true);
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxStdMapSyntheticFrontEndCreator,
      "libc++ std::multiset synthetic children",
      ConstString("^std::__(ndk)?1::multiset<.+> >(( )?&)?$"), stl_synth_flags,
      true);
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxStdMapSyntheticFrontEndCreator,
      "libc++ std::multimap synthetic children",
      ConstString("^std::__(ndk)?1::multimap<.+> >(( )?&)?$"), stl_synth_flags,
      true);
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxStdUnorderedMapSyntheticFrontEndCreator,
      "libc++ std::unordered containers synthetic children",
      ConstString("^(std::__(ndk)?1::)unordered_(multi)?(map|set)<.+> >$"),
      stl_synth_flags, true);
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxInitializerListSyntheticFrontEndCreator,
      "libc++ std::initializer_list synthetic children",
      ConstString("^std::initializer_list<.+>(( )?&)?$"), stl_synth_flags,
      true);
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxAtomicSyntheticFrontEndCreator,
      "libc++ std::atomic synthetic children",
      ConstString("^std::__(ndk)?1::atomic<.+>$"), stl_synth_flags, true);

  cpp_category_sp->GetRegexTypeSyntheticsContainer()->Add(
      RegularExpressionSP(new RegularExpression(
          llvm::StringRef("^(std::__(ndk)?1::)deque<.+>(( )?&)?$"))),
      SyntheticChildrenSP(new ScriptedSyntheticChildren(
          stl_synth_flags,
          "lldb.formatters.cpp.libcxx.stddeque_SynthProvider")));

  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxSharedPtrSyntheticFrontEndCreator,
      "shared_ptr synthetic children",
      ConstString("^(std::__(ndk)?1::)shared_ptr<.+>(( )?&)?$"),
      stl_synth_flags, true);
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxSharedPtrSyntheticFrontEndCreator,
      "weak_ptr synthetic children",
      ConstString("^(std::__(ndk)?1::)weak_ptr<.+>(( )?&)?$"), stl_synth_flags,
      true);

  stl_summary_flags.SetDontShowChildren(false);
  stl_summary_flags.SetSkipPointers(false);
  AddCXXSummary(
      cpp_category_sp, lldb_private::formatters::LibcxxContainerSummaryProvider,
      "libc++ std::vector<bool> summary provider",
      ConstString(
          "std::__(ndk)?1::vector<bool, std::__(ndk)?1::allocator<bool> >"),
      stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxContainerSummaryProvider,
                "libc++ std::vector summary provider",
                ConstString("^std::__(ndk)?1::vector<.+>(( )?&)?$"),
                stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxContainerSummaryProvider,
                "libc++ std::list summary provider",
                ConstString("^std::__(ndk)?1::list<.+>(( )?&)?$"),
                stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxContainerSummaryProvider,
                "libc++ std::map summary provider",
                ConstString("^std::__(ndk)?1::map<.+>(( )?&)?$"),
                stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxContainerSummaryProvider,
                "libc++ std::deque summary provider",
                ConstString("^std::__(ndk)?1::deque<.+>(( )?&)?$"),
                stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxContainerSummaryProvider,
                "libc++ std::set summary provider",
                ConstString("^std::__(ndk)?1::set<.+>(( )?&)?$"),
                stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxContainerSummaryProvider,
                "libc++ std::multiset summary provider",
                ConstString("^std::__(ndk)?1::multiset<.+>(( )?&)?$"),
                stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxContainerSummaryProvider,
                "libc++ std::multimap summary provider",
                ConstString("^std::__(ndk)?1::multimap<.+>(( )?&)?$"),
                stl_summary_flags, true);
  AddCXXSummary(
      cpp_category_sp, lldb_private::formatters::LibcxxContainerSummaryProvider,
      "libc++ std::unordered containers summary provider",
      ConstString("^(std::__(ndk)?1::)unordered_(multi)?(map|set)<.+> >$"),
      stl_summary_flags, true);
  AddCXXSummary(
      cpp_category_sp, lldb_private::formatters::LibCxxAtomicSummaryProvider,
      "libc++ std::atomic summary provider",
      ConstString("^std::__(ndk)?1::atomic<.+>$"), stl_summary_flags, true);

  stl_summary_flags.SetSkipPointers(true);

  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxSmartPointerSummaryProvider,
                "libc++ std::shared_ptr summary provider",
                ConstString("^std::__(ndk)?1::shared_ptr<.+>(( )?&)?$"),
                stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxSmartPointerSummaryProvider,
                "libc++ std::weak_ptr summary provider",
                ConstString("^std::__(ndk)?1::weak_ptr<.+>(( )?&)?$"),
                stl_summary_flags, true);

  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibCxxVectorIteratorSyntheticFrontEndCreator,
      "std::vector iterator synthetic children",
      ConstString("^std::__(ndk)?1::__wrap_iter<.+>$"), stl_synth_flags, true);

  AddCXXSummary(
      cpp_category_sp, lldb_private::formatters::LibcxxContainerSummaryProvider,
      "libc++ std::vector<bool> summary provider",
      ConstString(
          "std::__(ndk)?1::vector<bool, std::__(ndk)?1::allocator<bool> >"),
      stl_summary_flags);
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibCxxMapIteratorSyntheticFrontEndCreator,
      "std::map iterator synthetic children",
      ConstString("^std::__(ndk)?1::__map_iterator<.+>$"), stl_synth_flags,
      true);

  AddCXXSynthetic(
      cpp_category_sp, lldb_private::formatters::LibcxxFunctionFrontEndCreator,
      "std::function synthetic value provider",
      ConstString("^std::__1::function<.+>$"), stl_synth_flags, true);
#endif
}

static void LoadLibStdcppFormatters(lldb::TypeCategoryImplSP cpp_category_sp) {
  if (!cpp_category_sp)
    return;

  TypeSummaryImpl::Flags stl_summary_flags;
  stl_summary_flags.SetCascades(true)
      .SetSkipPointers(false)
      .SetSkipReferences(false)
      .SetDontShowChildren(true)
      .SetDontShowValue(true)
      .SetShowMembersOneLiner(false)
      .SetHideItemNames(false);

  lldb::TypeSummaryImplSP std_string_summary_sp(
      new StringSummaryFormat(stl_summary_flags, "${var._M_dataplus._M_p}"));

  lldb::TypeSummaryImplSP cxx11_string_summary_sp(new CXXFunctionSummaryFormat(
      stl_summary_flags, LibStdcppStringSummaryProvider,
      "libstdc++ c++11 std::string summary provider"));
  lldb::TypeSummaryImplSP cxx11_wstring_summary_sp(new CXXFunctionSummaryFormat(
      stl_summary_flags, LibStdcppWStringSummaryProvider,
      "libstdc++ c++11 std::wstring summary provider"));

  cpp_category_sp->GetTypeSummariesContainer()->Add(ConstString("std::string"),
                                                    std_string_summary_sp);
  cpp_category_sp->GetTypeSummariesContainer()->Add(
      ConstString("std::basic_string<char>"), std_string_summary_sp);
  cpp_category_sp->GetTypeSummariesContainer()->Add(
      ConstString("std::basic_string<char,std::char_traits<char>,std::"
                  "allocator<char> >"),
      std_string_summary_sp);
  cpp_category_sp->GetTypeSummariesContainer()->Add(
      ConstString("std::basic_string<char, std::char_traits<char>, "
                  "std::allocator<char> >"),
      std_string_summary_sp);

  cpp_category_sp->GetTypeSummariesContainer()->Add(
      ConstString("std::__cxx11::string"), cxx11_string_summary_sp);
  cpp_category_sp->GetTypeSummariesContainer()->Add(
      ConstString("std::__cxx11::basic_string<char, std::char_traits<char>, "
                  "std::allocator<char> >"),
      cxx11_string_summary_sp);

  // making sure we force-pick the summary for printing wstring (_M_p is a
  // wchar_t*)
  lldb::TypeSummaryImplSP std_wstring_summary_sp(
      new StringSummaryFormat(stl_summary_flags, "${var._M_dataplus._M_p%S}"));

  cpp_category_sp->GetTypeSummariesContainer()->Add(ConstString("std::wstring"),
                                                    std_wstring_summary_sp);
  cpp_category_sp->GetTypeSummariesContainer()->Add(
      ConstString("std::basic_string<wchar_t>"), std_wstring_summary_sp);
  cpp_category_sp->GetTypeSummariesContainer()->Add(
      ConstString("std::basic_string<wchar_t,std::char_traits<wchar_t>,std::"
                  "allocator<wchar_t> >"),
      std_wstring_summary_sp);
  cpp_category_sp->GetTypeSummariesContainer()->Add(
      ConstString("std::basic_string<wchar_t, std::char_traits<wchar_t>, "
                  "std::allocator<wchar_t> >"),
      std_wstring_summary_sp);

  cpp_category_sp->GetTypeSummariesContainer()->Add(
      ConstString("std::__cxx11::wstring"), cxx11_wstring_summary_sp);
  cpp_category_sp->GetTypeSummariesContainer()->Add(
      ConstString("std::__cxx11::basic_string<wchar_t, "
                  "std::char_traits<wchar_t>, std::allocator<wchar_t> >"),
      cxx11_wstring_summary_sp);

#ifndef LLDB_DISABLE_PYTHON

  SyntheticChildren::Flags stl_synth_flags;
  stl_synth_flags.SetCascades(true).SetSkipPointers(false).SetSkipReferences(
      false);

  cpp_category_sp->GetRegexTypeSyntheticsContainer()->Add(
      RegularExpressionSP(
          new RegularExpression(llvm::StringRef("^std::vector<.+>(( )?&)?$"))),
      SyntheticChildrenSP(new ScriptedSyntheticChildren(
          stl_synth_flags,
          "lldb.formatters.cpp.gnu_libstdcpp.StdVectorSynthProvider")));
  cpp_category_sp->GetRegexTypeSyntheticsContainer()->Add(
      RegularExpressionSP(
          new RegularExpression(llvm::StringRef("^std::map<.+> >(( )?&)?$"))),
      SyntheticChildrenSP(new ScriptedSyntheticChildren(
          stl_synth_flags,
          "lldb.formatters.cpp.gnu_libstdcpp.StdMapSynthProvider")));
  cpp_category_sp->GetRegexTypeSyntheticsContainer()->Add(
      RegularExpressionSP(new RegularExpression(
          llvm::StringRef("^std::(__cxx11::)?list<.+>(( )?&)?$"))),
      SyntheticChildrenSP(new ScriptedSyntheticChildren(
          stl_synth_flags,
          "lldb.formatters.cpp.gnu_libstdcpp.StdListSynthProvider")));
  stl_summary_flags.SetDontShowChildren(false);
  stl_summary_flags.SetSkipPointers(true);
  cpp_category_sp->GetRegexTypeSummariesContainer()->Add(
      RegularExpressionSP(
          new RegularExpression(llvm::StringRef("^std::vector<.+>(( )?&)?$"))),
      TypeSummaryImplSP(
          new StringSummaryFormat(stl_summary_flags, "size=${svar%#}")));
  cpp_category_sp->GetRegexTypeSummariesContainer()->Add(
      RegularExpressionSP(
          new RegularExpression(llvm::StringRef("^std::map<.+> >(( )?&)?$"))),
      TypeSummaryImplSP(
          new StringSummaryFormat(stl_summary_flags, "size=${svar%#}")));
  cpp_category_sp->GetRegexTypeSummariesContainer()->Add(
      RegularExpressionSP(new RegularExpression(
          llvm::StringRef("^std::(__cxx11::)?list<.+>(( )?&)?$"))),
      TypeSummaryImplSP(
          new StringSummaryFormat(stl_summary_flags, "size=${svar%#}")));

  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibStdcppVectorIteratorSyntheticFrontEndCreator,
      "std::vector iterator synthetic children",
      ConstString("^__gnu_cxx::__normal_iterator<.+>$"), stl_synth_flags, true);

  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibstdcppMapIteratorSyntheticFrontEndCreator,
      "std::map iterator synthetic children",
      ConstString("^std::_Rb_tree_iterator<.+>$"), stl_synth_flags, true);

  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibStdcppSharedPtrSyntheticFrontEndCreator,
      "std::shared_ptr synthetic children",
      ConstString("^std::shared_ptr<.+>(( )?&)?$"), stl_synth_flags, true);
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibStdcppSharedPtrSyntheticFrontEndCreator,
      "std::weak_ptr synthetic children",
      ConstString("^std::weak_ptr<.+>(( )?&)?$"), stl_synth_flags, true);

  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibStdcppSmartPointerSummaryProvider,
                "libstdc++ std::shared_ptr summary provider",
                ConstString("^std::shared_ptr<.+>(( )?&)?$"), stl_summary_flags,
                true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibStdcppSmartPointerSummaryProvider,
                "libstdc++ std::weak_ptr summary provider",
                ConstString("^std::weak_ptr<.+>(( )?&)?$"), stl_summary_flags,
                true);
#endif
}

static void LoadSystemFormatters(lldb::TypeCategoryImplSP cpp_category_sp) {
  if (!cpp_category_sp)
    return;

  TypeSummaryImpl::Flags string_flags;
  string_flags.SetCascades(true)
      .SetSkipPointers(true)
      .SetSkipReferences(false)
      .SetDontShowChildren(true)
      .SetDontShowValue(false)
      .SetShowMembersOneLiner(false)
      .SetHideItemNames(false);

  TypeSummaryImpl::Flags string_array_flags;
  string_array_flags.SetCascades(true)
      .SetSkipPointers(true)
      .SetSkipReferences(false)
      .SetDontShowChildren(true)
      .SetDontShowValue(true)
      .SetShowMembersOneLiner(false)
      .SetHideItemNames(false);

#ifndef LLDB_DISABLE_PYTHON
  // FIXME because of a bug in the FormattersContainer we need to add a summary
  // for both X* and const X* (<rdar://problem/12717717>)
  AddCXXSummary(
      cpp_category_sp, lldb_private::formatters::Char16StringSummaryProvider,
      "char16_t * summary provider", ConstString("char16_t *"), string_flags);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::Char16StringSummaryProvider,
                "char16_t [] summary provider",
                ConstString("char16_t \\[[0-9]+\\]"), string_array_flags, true);

  AddCXXSummary(
      cpp_category_sp, lldb_private::formatters::Char32StringSummaryProvider,
      "char32_t * summary provider", ConstString("char32_t *"), string_flags);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::Char32StringSummaryProvider,
                "char32_t [] summary provider",
                ConstString("char32_t \\[[0-9]+\\]"), string_array_flags, true);

  AddCXXSummary(
      cpp_category_sp, lldb_private::formatters::WCharStringSummaryProvider,
      "wchar_t * summary provider", ConstString("wchar_t *"), string_flags);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::WCharStringSummaryProvider,
                "wchar_t * summary provider",
                ConstString("wchar_t \\[[0-9]+\\]"), string_array_flags, true);

  AddCXXSummary(
      cpp_category_sp, lldb_private::formatters::Char16StringSummaryProvider,
      "unichar * summary provider", ConstString("unichar *"), string_flags);

  TypeSummaryImpl::Flags widechar_flags;
  widechar_flags.SetDontShowValue(true)
      .SetSkipPointers(true)
      .SetSkipReferences(false)
      .SetCascades(true)
      .SetDontShowChildren(true)
      .SetHideItemNames(true)
      .SetShowMembersOneLiner(false);

  AddCXXSummary(
      cpp_category_sp, lldb_private::formatters::Char16SummaryProvider,
      "char16_t summary provider", ConstString("char16_t"), widechar_flags);
  AddCXXSummary(
      cpp_category_sp, lldb_private::formatters::Char32SummaryProvider,
      "char32_t summary provider", ConstString("char32_t"), widechar_flags);
  AddCXXSummary(cpp_category_sp, lldb_private::formatters::WCharSummaryProvider,
                "wchar_t summary provider", ConstString("wchar_t"),
                widechar_flags);

  AddCXXSummary(
      cpp_category_sp, lldb_private::formatters::Char16SummaryProvider,
      "unichar summary provider", ConstString("unichar"), widechar_flags);
#endif
}

lldb::TypeCategoryImplSP CPlusPlusLanguage::GetFormatters() {
  static std::once_flag g_initialize;
  static TypeCategoryImplSP g_category;

  std::call_once(g_initialize, [this]() -> void {
    DataVisualization::Categories::GetCategory(GetPluginName(), g_category);
    if (g_category) {
      LoadLibCxxFormatters(g_category);
      LoadLibStdcppFormatters(g_category);
      LoadSystemFormatters(g_category);
    }
  });
  return g_category;
}

HardcodedFormatters::HardcodedSummaryFinder
CPlusPlusLanguage::GetHardcodedSummaries() {
  static std::once_flag g_initialize;
  static ConstString g_vectortypes("VectorTypes");
  static HardcodedFormatters::HardcodedSummaryFinder g_formatters;

  std::call_once(g_initialize, []() -> void {
    g_formatters.push_back(
        [](lldb_private::ValueObject &valobj, lldb::DynamicValueType,
           FormatManager &) -> TypeSummaryImpl::SharedPointer {
          static CXXFunctionSummaryFormat::SharedPointer formatter_sp(
              new CXXFunctionSummaryFormat(
                  TypeSummaryImpl::Flags(),
                  lldb_private::formatters::CXXFunctionPointerSummaryProvider,
                  "Function pointer summary provider"));
          if (valobj.GetCompilerType().IsFunctionPointerType()) {
            return formatter_sp;
          }
          return nullptr;
        });
    g_formatters.push_back(
        [](lldb_private::ValueObject &valobj, lldb::DynamicValueType,
           FormatManager &fmt_mgr) -> TypeSummaryImpl::SharedPointer {
          static CXXFunctionSummaryFormat::SharedPointer formatter_sp(
              new CXXFunctionSummaryFormat(
                  TypeSummaryImpl::Flags()
                      .SetCascades(true)
                      .SetDontShowChildren(true)
                      .SetHideItemNames(true)
                      .SetShowMembersOneLiner(true)
                      .SetSkipPointers(true)
                      .SetSkipReferences(false),
                  lldb_private::formatters::VectorTypeSummaryProvider,
                  "vector_type pointer summary provider"));
          if (valobj.GetCompilerType().IsVectorType(nullptr, nullptr)) {
            if (fmt_mgr.GetCategory(g_vectortypes)->IsEnabled())
              return formatter_sp;
          }
          return nullptr;
        });
    g_formatters.push_back(
        [](lldb_private::ValueObject &valobj, lldb::DynamicValueType,
           FormatManager &fmt_mgr) -> TypeSummaryImpl::SharedPointer {
          static CXXFunctionSummaryFormat::SharedPointer formatter_sp(
              new CXXFunctionSummaryFormat(
                  TypeSummaryImpl::Flags()
                      .SetCascades(true)
                      .SetDontShowChildren(true)
                      .SetHideItemNames(true)
                      .SetShowMembersOneLiner(true)
                      .SetSkipPointers(true)
                      .SetSkipReferences(false),
                  lldb_private::formatters::BlockPointerSummaryProvider,
                  "block pointer summary provider"));
          if (valobj.GetCompilerType().IsBlockPointerType(nullptr)) {
            return formatter_sp;
          }
          return nullptr;
        });
  });

  return g_formatters;
}

HardcodedFormatters::HardcodedSyntheticFinder
CPlusPlusLanguage::GetHardcodedSynthetics() {
  static std::once_flag g_initialize;
  static ConstString g_vectortypes("VectorTypes");
  static HardcodedFormatters::HardcodedSyntheticFinder g_formatters;

  std::call_once(g_initialize, []() -> void {
    g_formatters.push_back([](lldb_private::ValueObject &valobj,
                              lldb::DynamicValueType,
                              FormatManager &
                                  fmt_mgr) -> SyntheticChildren::SharedPointer {
      static CXXSyntheticChildren::SharedPointer formatter_sp(
          new CXXSyntheticChildren(
              SyntheticChildren::Flags()
                  .SetCascades(true)
                  .SetSkipPointers(true)
                  .SetSkipReferences(true)
                  .SetNonCacheable(true),
              "vector_type synthetic children",
              lldb_private::formatters::VectorTypeSyntheticFrontEndCreator));
      if (valobj.GetCompilerType().IsVectorType(nullptr, nullptr)) {
        if (fmt_mgr.GetCategory(g_vectortypes)->IsEnabled())
          return formatter_sp;
      }
      return nullptr;
    });
    g_formatters.push_back([](lldb_private::ValueObject &valobj,
                              lldb::DynamicValueType,
                              FormatManager &
                                  fmt_mgr) -> SyntheticChildren::SharedPointer {
      static CXXSyntheticChildren::SharedPointer formatter_sp(
          new CXXSyntheticChildren(
              SyntheticChildren::Flags()
                  .SetCascades(true)
                  .SetSkipPointers(true)
                  .SetSkipReferences(true)
                  .SetNonCacheable(true),
              "block pointer synthetic children",
              lldb_private::formatters::BlockPointerSyntheticFrontEndCreator));
      if (valobj.GetCompilerType().IsBlockPointerType(nullptr)) {
        return formatter_sp;
      }
      return nullptr;
    });

  });

  return g_formatters;
}
