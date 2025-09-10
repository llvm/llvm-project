//===-- CPlusPlusLanguage.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CPlusPlusLanguage.h"

#include <cctype>
#include <cstring>

#include <functional>
#include <memory>
#include <mutex>
#include <set>

#include "llvm/ADT/StringRef.h"
#include "llvm/Demangle/ItaniumDemangle.h"

#include "lldb/Core/DemangledNameInfo.h"
#include "lldb/Core/Mangled.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/UniqueCStringMap.h"
#include "lldb/DataFormatters/CXXFunctionPointer.h"
#include "lldb/DataFormatters/DataVisualization.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/DataFormatters/VectorType.h"
#include "lldb/Interpreter/OptionValueProperties.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegularExpression.h"
#include "lldb/ValueObject/ValueObjectVariable.h"

#include "BlockPointer.h"
#include "CPlusPlusNameParser.h"
#include "Coroutines.h"
#include "CxxStringTypes.h"
#include "Generic.h"
#include "LibCxx.h"
#include "LibCxxAtomic.h"
#include "LibCxxVariant.h"
#include "LibStdcpp.h"
#include "MSVCUndecoratedNameParser.h"
#include "MsvcStl.h"
#include "lldb/lldb-enumerations.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

LLDB_PLUGIN_DEFINE(CPlusPlusLanguage)

void CPlusPlusLanguage::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(), "C++ Language",
                                CreateInstance, &DebuggerInitialize);
}

void CPlusPlusLanguage::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

std::unique_ptr<Language::MethodName>
CPlusPlusLanguage::GetMethodName(ConstString full_name) const {
  std::unique_ptr<CxxMethodName> cpp_method =
      std::make_unique<CxxMethodName>(full_name);
  cpp_method->IsValid();
  return cpp_method;
}

std::pair<FunctionNameType, std::optional<ConstString>>
CPlusPlusLanguage::GetFunctionNameInfo(ConstString name) const {
  if (Mangled::IsMangledName(name.GetCString()))
    return {eFunctionNameTypeFull, std::nullopt};

  FunctionNameType func_name_type = eFunctionNameTypeNone;
  CxxMethodName method(name);
  llvm::StringRef basename = method.GetBasename();
  if (basename.empty()) {
    llvm::StringRef context;
    func_name_type |=
        (ExtractContextAndIdentifier(name.GetCString(), context, basename)
             ? (eFunctionNameTypeMethod | eFunctionNameTypeBase)
             : eFunctionNameTypeFull);
  } else {
    func_name_type |= (eFunctionNameTypeMethod | eFunctionNameTypeBase);
  }

  if (!method.GetQualifiers().empty()) {
    // There is a 'const' or other qualifier following the end of the function
    // parens, this can't be a eFunctionNameTypeBase.
    func_name_type &= ~(eFunctionNameTypeBase);
  }

  if (basename.empty())
    return {func_name_type, std::nullopt};
  else
    return {func_name_type, ConstString(basename)};
}

bool CPlusPlusLanguage::SymbolNameFitsToLanguage(Mangled mangled) const {
  const char *mangled_name = mangled.GetMangledName().GetCString();
  auto mangling_scheme = Mangled::GetManglingScheme(mangled_name);
  return mangled_name && (mangling_scheme == Mangled::eManglingSchemeItanium ||
                          mangling_scheme == Mangled::eManglingSchemeMSVC);
}

ConstString CPlusPlusLanguage::GetDemangledFunctionNameWithoutArguments(
    Mangled mangled) const {
  const char *mangled_name_cstr = mangled.GetMangledName().GetCString();
  ConstString demangled_name = mangled.GetDemangledName();
  if (demangled_name && mangled_name_cstr && mangled_name_cstr[0]) {
    if (mangled_name_cstr[0] == '_' && mangled_name_cstr[1] == 'Z' &&
        (mangled_name_cstr[2] != 'T' && // avoid virtual table, VTT structure,
                                        // typeinfo structure, and typeinfo
                                        // mangled_name
         mangled_name_cstr[2] != 'G' && // avoid guard variables
         mangled_name_cstr[2] != 'Z'))  // named local entities (if we
                                        // eventually handle eSymbolTypeData,
                                        // we will want this back)
    {
      CxxMethodName cxx_method(demangled_name);
      if (!cxx_method.GetBasename().empty()) {
        std::string shortname;
        if (!cxx_method.GetContext().empty())
          shortname = cxx_method.GetContext().str() + "::";
        shortname += cxx_method.GetBasename().str();
        return ConstString(shortname);
      }
    }
  }
  if (demangled_name)
    return demangled_name;
  return mangled.GetMangledName();
}

// Static Functions

Language *CPlusPlusLanguage::CreateInstance(lldb::LanguageType language) {
  // Use plugin for C++ but not for Objective-C++ (which has its own plugin).
  if (Language::LanguageIsCPlusPlus(language) &&
      language != eLanguageTypeObjC_plus_plus)
    return new CPlusPlusLanguage();
  return nullptr;
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

/// Writes out the function name in 'full_name' to 'out_stream'
/// but replaces each argument type with the variable name
/// and the corresponding pretty-printed value
static bool PrettyPrintFunctionNameWithArgs(Stream &out_stream,
                                            char const *full_name,
                                            ExecutionContextScope *exe_scope,
                                            VariableList const &args) {
  CPlusPlusLanguage::CxxMethodName cpp_method{ConstString(full_name)};

  if (!cpp_method.IsValid())
    return false;

  llvm::StringRef return_type = cpp_method.GetReturnType();
  if (!return_type.empty()) {
    out_stream.PutCString(return_type);
    out_stream.PutChar(' ');
  }

  out_stream.PutCString(cpp_method.GetScopeQualifiedName());
  out_stream.PutChar('(');

  FormatEntity::PrettyPrintFunctionArguments(out_stream, args, exe_scope);

  out_stream.PutChar(')');

  llvm::StringRef qualifiers = cpp_method.GetQualifiers();
  if (!qualifiers.empty()) {
    out_stream.PutChar(' ');
    out_stream.PutCString(qualifiers);
  }

  return true;
}

static llvm::Expected<std::pair<llvm::StringRef, DemangledNameInfo>>
GetAndValidateInfo(const SymbolContext &sc) {
  Mangled mangled = sc.GetPossiblyInlinedFunctionName();
  if (!mangled)
    return llvm::createStringError("Function does not have a mangled name.");

  auto demangled_name = mangled.GetDemangledName().GetStringRef();
  if (demangled_name.empty())
    return llvm::createStringError(
        "Function '%s' does not have a demangled name.",
        mangled.GetMangledName().AsCString(""));

  const std::optional<DemangledNameInfo> &info = mangled.GetDemangledInfo();
  if (!info)
    return llvm::createStringError(
        "Function '%s' does not have demangled info.", demangled_name.data());

  // Function without a basename is nonsense.
  if (!info->hasBasename())
    return llvm::createStringError(
        "DemangledInfo for '%s does not have basename range.",
        demangled_name.data());

  return std::make_pair(demangled_name, *info);
}

static llvm::Expected<llvm::StringRef>
GetDemangledBasename(const SymbolContext &sc) {
  auto info_or_err = GetAndValidateInfo(sc);
  if (!info_or_err)
    return info_or_err.takeError();

  auto [demangled_name, info] = *info_or_err;

  return CPlusPlusLanguage::GetDemangledBasename(demangled_name, info);
}

llvm::StringRef
CPlusPlusLanguage::GetDemangledBasename(llvm::StringRef demangled,
                                        const DemangledNameInfo &info) {
  assert(info.hasBasename());
  return demangled.slice(info.BasenameRange.first, info.BasenameRange.second);
}

llvm::Expected<llvm::StringRef>
CPlusPlusLanguage::GetDemangledTemplateArguments(
    llvm::StringRef demangled, const DemangledNameInfo &info) {
  if (!info.hasTemplateArguments())
    return llvm::createStringError(
        "Template arguments range for '%s' is invalid.", demangled.data());

  return demangled.slice(info.TemplateArgumentsRange.first,
                         info.TemplateArgumentsRange.second);
}

static llvm::Expected<llvm::StringRef>
GetDemangledTemplateArguments(const SymbolContext &sc) {
  auto info_or_err = GetAndValidateInfo(sc);
  if (!info_or_err)
    return info_or_err.takeError();

  auto [demangled_name, info] = *info_or_err;

  return CPlusPlusLanguage::GetDemangledTemplateArguments(demangled_name, info);
}

llvm::Expected<llvm::StringRef>
CPlusPlusLanguage::GetDemangledReturnTypeLHS(llvm::StringRef demangled,
                                             const DemangledNameInfo &info) {
  if (info.ScopeRange.first >= demangled.size())
    return llvm::createStringError(
        "Scope range for '%s' LHS return type is invalid.", demangled.data());

  return demangled.substr(0, info.ScopeRange.first);
}

static llvm::Expected<llvm::StringRef>
GetDemangledReturnTypeLHS(const SymbolContext &sc) {
  auto info_or_err = GetAndValidateInfo(sc);
  if (!info_or_err)
    return info_or_err.takeError();

  auto [demangled_name, info] = *info_or_err;

  return CPlusPlusLanguage::GetDemangledReturnTypeLHS(demangled_name, info);
}

llvm::Expected<llvm::StringRef>
CPlusPlusLanguage::GetDemangledFunctionQualifiers(
    llvm::StringRef demangled, const DemangledNameInfo &info) {
  if (!info.hasQualifiers())
    return llvm::createStringError("Qualifiers range for '%s' is invalid.",
                                   demangled.data());

  return demangled.slice(info.QualifiersRange.first,
                         info.QualifiersRange.second);
}

static llvm::Expected<llvm::StringRef>
GetDemangledFunctionQualifiers(const SymbolContext &sc) {
  auto info_or_err = GetAndValidateInfo(sc);
  if (!info_or_err)
    return info_or_err.takeError();

  auto [demangled_name, info] = *info_or_err;

  return CPlusPlusLanguage::GetDemangledFunctionQualifiers(demangled_name,
                                                           info);
}

llvm::Expected<llvm::StringRef>
CPlusPlusLanguage::GetDemangledReturnTypeRHS(llvm::StringRef demangled,
                                             const DemangledNameInfo &info) {
  if (info.QualifiersRange.first < info.ArgumentsRange.second)
    return llvm::createStringError(
        "Qualifiers range for '%s' RHS return type  is invalid.",
        demangled.data());

  return demangled.slice(info.ArgumentsRange.second,
                         info.QualifiersRange.first);
}

static llvm::Expected<llvm::StringRef>
GetDemangledReturnTypeRHS(const SymbolContext &sc) {
  auto info_or_err = GetAndValidateInfo(sc);
  if (!info_or_err)
    return info_or_err.takeError();

  auto [demangled_name, info] = *info_or_err;

  return CPlusPlusLanguage::GetDemangledReturnTypeRHS(demangled_name, info);
}

llvm::Expected<llvm::StringRef>
CPlusPlusLanguage::GetDemangledScope(llvm::StringRef demangled,
                                     const DemangledNameInfo &info) {
  if (!info.hasScope())
    return llvm::createStringError("Scope range for '%s' is invalid.",
                                   demangled.data());

  return demangled.slice(info.ScopeRange.first, info.ScopeRange.second);
}

static llvm::Expected<llvm::StringRef>
GetDemangledScope(const SymbolContext &sc) {
  auto info_or_err = GetAndValidateInfo(sc);
  if (!info_or_err)
    return info_or_err.takeError();

  auto [demangled_name, info] = *info_or_err;

  return CPlusPlusLanguage::GetDemangledScope(demangled_name, info);
}

llvm::Expected<llvm::StringRef>
CPlusPlusLanguage::GetDemangledFunctionSuffix(llvm::StringRef demangled,
                                              const DemangledNameInfo &info) {
  if (!info.hasSuffix())
    return llvm::createStringError("Suffix range for '%s' is invalid.",
                                   demangled.data());

  return demangled.slice(info.SuffixRange.first, info.SuffixRange.second);
}

static llvm::Expected<llvm::StringRef>
GetDemangledFunctionSuffix(const SymbolContext &sc) {
  auto info_or_err = GetAndValidateInfo(sc);
  if (!info_or_err)
    return info_or_err.takeError();

  auto [demangled_name, info] = *info_or_err;

  return CPlusPlusLanguage::GetDemangledFunctionSuffix(demangled_name, info);
}

llvm::Expected<llvm::StringRef>
CPlusPlusLanguage::GetDemangledFunctionArguments(
    llvm::StringRef demangled, const DemangledNameInfo &info) {
  if (!info.hasArguments())
    return llvm::createStringError(
        "Function arguments range for '%s' is invalid.", demangled.data());

  return demangled.slice(info.ArgumentsRange.first, info.ArgumentsRange.second);
}

static bool PrintDemangledArgumentList(Stream &s, const SymbolContext &sc) {
  assert(sc.symbol);

  auto info_or_err = GetAndValidateInfo(sc);
  if (!info_or_err) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Language), info_or_err.takeError(),
                   "Failed to handle ${{function.formatted-arguments}} "
                   "frame-format variable: {0}");
    return false;
  }

  auto [demangled_name, info] = *info_or_err;

  auto args_or_err =
      CPlusPlusLanguage::GetDemangledFunctionArguments(demangled_name, info);
  if (!args_or_err) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Language), args_or_err.takeError(),
                   "Failed to handle ${{function.formatted-arguments}} "
                   "frame-format variable: {0}");
    return false;
  }

  s << *args_or_err;

  return true;
}

bool CPlusPlusLanguage::CxxMethodName::TrySimplifiedParse() {
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

void CPlusPlusLanguage::CxxMethodName::Parse() {
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
    if (m_context.empty()) {
      m_scope_qualified = std::string(m_basename);
    } else {
      m_scope_qualified = m_context;
      m_scope_qualified += "::";
      m_scope_qualified += m_basename;
    }
    m_parsed = true;
  }
}

llvm::StringRef
CPlusPlusLanguage::CxxMethodName::GetBasenameNoTemplateParameters() {
  llvm::StringRef basename = GetBasename();
  size_t arg_start, arg_end;
  llvm::StringRef parens("<>", 2);
  if (ReverseFindMatchingChars(basename, parens, arg_start, arg_end))
    return basename.substr(0, arg_start);

  return basename;
}

bool CPlusPlusLanguage::CxxMethodName::ContainsPath(llvm::StringRef path) {
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

bool CPlusPlusLanguage::DemangledNameContainsPath(llvm::StringRef path,
                                                  ConstString demangled) const {
  CxxMethodName demangled_name(demangled);
  return demangled_name.ContainsPath(path);
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

std::vector<ConstString> CPlusPlusLanguage::GenerateAlternateFunctionManglings(
    const ConstString mangled_name) const {
  std::vector<ConstString> alternates;

  /// Get a basic set of alternative manglings for the given symbol `name`, by
  /// making a few basic possible substitutions on basic types, storage duration
  /// and `const`ness for the given symbol. The output parameter `alternates`
  /// is filled with a best-guess, non-exhaustive set of different manglings
  /// for the given name.

  // Maybe we're looking for a const symbol but the debug info told us it was
  // non-const...
  if (!strncmp(mangled_name.GetCString(), "_ZN", 3) &&
      strncmp(mangled_name.GetCString(), "_ZNK", 4)) {
    std::string fixed_scratch("_ZNK");
    fixed_scratch.append(mangled_name.GetCString() + 3);
    alternates.push_back(ConstString(fixed_scratch));
  }

  // Maybe we're looking for a static symbol but we thought it was global...
  if (!strncmp(mangled_name.GetCString(), "_Z", 2) &&
      strncmp(mangled_name.GetCString(), "_ZL", 3)) {
    std::string fixed_scratch("_ZL");
    fixed_scratch.append(mangled_name.GetCString() + 2);
    alternates.push_back(ConstString(fixed_scratch));
  }

  auto *log = GetLog(LLDBLog::Language);

  // `char` is implementation defined as either `signed` or `unsigned`.  As a
  // result a char parameter has 3 possible manglings: 'c'-char, 'a'-signed
  // char, 'h'-unsigned char.  If we're looking for symbols with a signed char
  // parameter, try finding matches which have the general case 'c'.
  if (auto char_fixup_or_err =
          SubstituteType_ItaniumMangle(mangled_name.GetStringRef(), "a", "c")) {
    // LLDB_LOG(log, "Substituted mangling {0} -> {1}", Mangled, Result);
    if (*char_fixup_or_err)
      alternates.push_back(*char_fixup_or_err);
  } else
    LLDB_LOG_ERROR(log, char_fixup_or_err.takeError(),
                   "Failed to substitute 'char' type mangling: {0}");

  // long long parameter mangling 'x', may actually just be a long 'l' argument
  if (auto long_fixup_or_err =
          SubstituteType_ItaniumMangle(mangled_name.GetStringRef(), "x", "l")) {
    if (*long_fixup_or_err)
      alternates.push_back(*long_fixup_or_err);
  } else
    LLDB_LOG_ERROR(log, long_fixup_or_err.takeError(),
                   "Failed to substitute 'long long' type mangling: {0}");

  // unsigned long long parameter mangling 'y', may actually just be unsigned
  // long 'm' argument
  if (auto ulong_fixup_or_err =
          SubstituteType_ItaniumMangle(mangled_name.GetStringRef(), "y", "m")) {
    if (*ulong_fixup_or_err)
      alternates.push_back(*ulong_fixup_or_err);
  } else
    LLDB_LOG_ERROR(
        log, ulong_fixup_or_err.takeError(),
        "Failed to substitute 'unsigned long long' type mangling: {0}");

  if (auto ctor_fixup_or_err = SubstituteStructorAliases_ItaniumMangle(
          mangled_name.GetStringRef())) {
    if (*ctor_fixup_or_err) {
      alternates.push_back(*ctor_fixup_or_err);
    }
  } else
    LLDB_LOG_ERROR(log, ctor_fixup_or_err.takeError(),
                   "Failed to substitute structor alias manglings: {0}");

  return alternates;
}

ConstString CPlusPlusLanguage::FindBestAlternateFunctionMangledName(
    const Mangled mangled, const SymbolContext &sym_ctx) const {
  ConstString demangled = mangled.GetDemangledName();
  if (!demangled)
    return ConstString();

  CxxMethodName cpp_name(demangled);
  std::string scope_qualified_name = cpp_name.GetScopeQualifiedName();

  if (!scope_qualified_name.size())
    return ConstString();

  if (!sym_ctx.module_sp)
    return ConstString();

  lldb_private::SymbolFile *sym_file = sym_ctx.module_sp->GetSymbolFile();
  if (!sym_file)
    return ConstString();

  std::vector<ConstString> alternates;
  sym_file->GetMangledNamesForFunction(scope_qualified_name, alternates);

  std::vector<ConstString> param_and_qual_matches;
  std::vector<ConstString> param_matches;
  for (size_t i = 0; i < alternates.size(); i++) {
    ConstString alternate_mangled_name = alternates[i];
    Mangled mangled(alternate_mangled_name);
    ConstString demangled = mangled.GetDemangledName();

    CxxMethodName alternate_cpp_name(demangled);
    if (!cpp_name.IsValid())
      continue;

    if (alternate_cpp_name.GetArguments() == cpp_name.GetArguments()) {
      if (alternate_cpp_name.GetQualifiers() == cpp_name.GetQualifiers())
        param_and_qual_matches.push_back(alternate_mangled_name);
      else
        param_matches.push_back(alternate_mangled_name);
    }
  }

  if (param_and_qual_matches.size())
    return param_and_qual_matches[0]; // It is assumed that there will be only
                                      // one!
  else if (param_matches.size())
    return param_matches[0]; // Return one of them as a best match
  else
    return ConstString();
}

static void LoadLibCxxFormatters(lldb::TypeCategoryImplSP cpp_category_sp) {
  if (!cpp_category_sp)
    return;

  TypeSummaryImpl::Flags stl_summary_flags;
  stl_summary_flags.SetCascades(true)
      .SetSkipPointers(false)
      .SetSkipReferences(false)
      .SetDontShowChildren(true)
      .SetDontShowValue(false)
      .SetShowMembersOneLiner(false)
      .SetHideItemNames(false);

  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxStringSummaryProviderASCII,
                "std::string summary provider", "^std::__[[:alnum:]]+::string$",
                stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxStringSummaryProviderASCII,
                "std::string summary provider",
                "^std::__[[:alnum:]]+::basic_string<char, "
                "std::__[[:alnum:]]+::char_traits<char>,.*>$",
                stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxStringSummaryProviderASCII,
                "std::string summary provider",
                "^std::__[[:alnum:]]+::basic_string<unsigned char, "
                "std::__[[:alnum:]]+::char_traits<unsigned char>,.*>$",
                stl_summary_flags, true);

  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxStringSummaryProviderUTF16,
                "std::u16string summary provider",
                "^std::__[[:alnum:]]+::basic_string<char16_t, "
                "std::__[[:alnum:]]+::char_traits<char16_t>,.*>$",
                stl_summary_flags, true);

  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxStringSummaryProviderUTF32,
                "std::u32string summary provider",
                "^std::__[[:alnum:]]+::basic_string<char32_t, "
                "std::__[[:alnum:]]+::char_traits<char32_t>,.*>$",
                stl_summary_flags, true);

  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxWStringSummaryProvider,
                "std::wstring summary provider",
                "^std::__[[:alnum:]]+::wstring$", stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxWStringSummaryProvider,
                "std::wstring summary provider",
                "^std::__[[:alnum:]]+::basic_string<wchar_t, "
                "std::__[[:alnum:]]+::char_traits<wchar_t>,.*>$",
                stl_summary_flags, true);

  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxStringViewSummaryProviderASCII,
                "std::string_view summary provider",
                "^std::__[[:alnum:]]+::string_view$", stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxStringViewSummaryProviderASCII,
                "std::string_view summary provider",
                "^std::__[[:alnum:]]+::basic_string_view<char, "
                "std::__[[:alnum:]]+::char_traits<char> >$",
                stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxStringViewSummaryProviderASCII,
                "std::string_view summary provider",
                "^std::__[[:alnum:]]+::basic_string_view<unsigned char, "
                "std::__[[:alnum:]]+::char_traits<unsigned char> >$",
                stl_summary_flags, true);

  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxStringViewSummaryProviderUTF16,
                "std::u16string_view summary provider",
                "^std::__[[:alnum:]]+::basic_string_view<char16_t, "
                "std::__[[:alnum:]]+::char_traits<char16_t> >$",
                stl_summary_flags, true);

  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxStringViewSummaryProviderUTF32,
                "std::u32string_view summary provider",
                "^std::__[[:alnum:]]+::basic_string_view<char32_t, "
                "std::__[[:alnum:]]+::char_traits<char32_t> >$",
                stl_summary_flags, true);

  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxWStringViewSummaryProvider,
                "std::wstring_view summary provider",
                "^std::__[[:alnum:]]+::wstring_view$", stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxWStringViewSummaryProvider,
                "std::wstring_view summary provider",
                "^std::__[[:alnum:]]+::basic_string_view<wchar_t, "
                "std::__[[:alnum:]]+::char_traits<wchar_t> >$",
                stl_summary_flags, true);

  SyntheticChildren::Flags stl_synth_flags;
  stl_synth_flags.SetCascades(true).SetSkipPointers(false).SetSkipReferences(
      false);
  SyntheticChildren::Flags stl_deref_flags = stl_synth_flags;
  stl_deref_flags.SetFrontEndWantsDereference();

  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxBitsetSyntheticFrontEndCreator,
      "libc++ std::bitset synthetic children",
      "^std::__[[:alnum:]]+::bitset<.+>$", stl_deref_flags, true);
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxStdVectorSyntheticFrontEndCreator,
      "libc++ std::vector synthetic children",
      "^std::__[[:alnum:]]+::vector<.+>$", stl_deref_flags, true);
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxStdValarraySyntheticFrontEndCreator,
      "libc++ std::valarray synthetic children",
      "^std::__[[:alnum:]]+::valarray<.+>$", stl_deref_flags, true);
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxStdSliceArraySyntheticFrontEndCreator,
      "libc++ std::slice_array synthetic children",
      "^std::__[[:alnum:]]+::slice_array<.+>$", stl_deref_flags, true);
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxStdProxyArraySyntheticFrontEndCreator,
      "libc++ synthetic children for the valarray proxy arrays",
      "^std::__[[:alnum:]]+::(gslice|mask|indirect)_array<.+>$",
      stl_deref_flags, true);
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxStdForwardListSyntheticFrontEndCreator,
      "libc++ std::forward_list synthetic children",
      "^std::__[[:alnum:]]+::forward_list<.+>$", stl_synth_flags, true);
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxStdListSyntheticFrontEndCreator,
      "libc++ std::list synthetic children",
      // A POSIX variant of: "^std::__(?!cxx11:)[[:alnum:]]+::list<.+>$"
      // so that it does not clash with: "^std::(__cxx11::)?list<.+>$"
      "^std::__([A-Zabd-z0-9]|cx?[A-Za-wyz0-9]|cxx1?[A-Za-z02-9]|"
      "cxx11[[:alnum:]])[[:alnum:]]*::list<.+>$",
      stl_deref_flags, true);
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxStdMapSyntheticFrontEndCreator,
      "libc++ std::map synthetic children", "^std::__[[:alnum:]]+::map<.+> >$",
      stl_synth_flags, true);
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxStdMapSyntheticFrontEndCreator,
      "libc++ std::set synthetic children", "^std::__[[:alnum:]]+::set<.+> >$",
      stl_deref_flags, true);
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxStdMapSyntheticFrontEndCreator,
      "libc++ std::multiset synthetic children",
      "^std::__[[:alnum:]]+::multiset<.+> >$", stl_deref_flags, true);
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxStdMapSyntheticFrontEndCreator,
      "libc++ std::multimap synthetic children",
      "^std::__[[:alnum:]]+::multimap<.+> >$", stl_synth_flags, true);
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxStdUnorderedMapSyntheticFrontEndCreator,
      "libc++ std::unordered containers synthetic children",
      "^std::__[[:alnum:]]+::unordered_(multi)?(map|set)<.+> >$",
      stl_synth_flags, true);
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxInitializerListSyntheticFrontEndCreator,
      "libc++ std::initializer_list synthetic children",
      "^std::initializer_list<.+>$", stl_synth_flags, true);
  AddCXXSynthetic(cpp_category_sp, LibcxxQueueFrontEndCreator,
                  "libc++ std::queue synthetic children",
                  "^std::__[[:alnum:]]+::queue<.+>$", stl_synth_flags, true);
  AddCXXSynthetic(cpp_category_sp, LibcxxTupleFrontEndCreator,
                  "libc++ std::tuple synthetic children",
                  "^std::__[[:alnum:]]+::tuple<.*>$", stl_synth_flags, true);
  AddCXXSynthetic(cpp_category_sp, LibcxxOptionalSyntheticFrontEndCreator,
                  "libc++ std::optional synthetic children",
                  "^std::__[[:alnum:]]+::optional<.+>$", stl_synth_flags, true);
  AddCXXSynthetic(cpp_category_sp, LibcxxVariantFrontEndCreator,
                  "libc++ std::variant synthetic children",
                  "^std::__[[:alnum:]]+::variant<.+>$", stl_synth_flags, true);
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxAtomicSyntheticFrontEndCreator,
      "libc++ std::atomic synthetic children",
      "^std::__[[:alnum:]]+::atomic<.+>$", stl_synth_flags, true);
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxStdSpanSyntheticFrontEndCreator,
      "libc++ std::span synthetic children", "^std::__[[:alnum:]]+::span<.+>$",
      stl_deref_flags, true);
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxStdRangesRefViewSyntheticFrontEndCreator,
      "libc++ std::ranges::ref_view synthetic children",
      "^std::__[[:alnum:]]+::ranges::ref_view<.+>$", stl_deref_flags, true);

  cpp_category_sp->AddTypeSynthetic(
      "^std::__[[:alnum:]]+::deque<.+>$", eFormatterMatchRegex,
      SyntheticChildrenSP(new ScriptedSyntheticChildren(
          stl_synth_flags,
          "lldb.formatters.cpp.libcxx.stddeque_SynthProvider")));

  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxSharedPtrSyntheticFrontEndCreator,
      "shared_ptr synthetic children", "^std::__[[:alnum:]]+::shared_ptr<.+>$",
      stl_synth_flags, true);

  static constexpr const char *const libcxx_std_unique_ptr_regex =
      "^std::__[[:alnum:]]+::unique_ptr<.+>$";
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxUniquePtrSyntheticFrontEndCreator,
      "unique_ptr synthetic children", libcxx_std_unique_ptr_regex,
      stl_synth_flags, true);

  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibcxxSharedPtrSyntheticFrontEndCreator,
      "weak_ptr synthetic children", "^std::__[[:alnum:]]+::weak_ptr<.+>$",
      stl_synth_flags, true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxFunctionSummaryProvider,
                "libc++ std::function summary provider",
                "^std::__[[:alnum:]]+::function<.+>$", stl_summary_flags, true);

  static constexpr const char *const libcxx_std_coroutine_handle_regex =
      "^std::__[[:alnum:]]+::coroutine_handle<.+>$";
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::StdlibCoroutineHandleSyntheticFrontEndCreator,
      "coroutine_handle synthetic children", libcxx_std_coroutine_handle_regex,
      stl_deref_flags, true);

  stl_summary_flags.SetDontShowChildren(false);
  stl_summary_flags.SetSkipPointers(false);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::ContainerSizeSummaryProvider,
                "libc++ std::bitset summary provider",
                "^std::__[[:alnum:]]+::bitset<.+>$", stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::ContainerSizeSummaryProvider,
                "libc++ std::vector summary provider",
                "^std::__[[:alnum:]]+::vector<.+>$", stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::ContainerSizeSummaryProvider,
                "libc++ std::valarray summary provider",
                "^std::__[[:alnum:]]+::valarray<.+>$", stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxStdSliceArraySummaryProvider,
                "libc++ std::slice_array summary provider",
                "^std::__[[:alnum:]]+::slice_array<.+>$", stl_summary_flags,
                true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::ContainerSizeSummaryProvider,
                "libc++ summary provider for the valarray proxy arrays",
                "^std::__[[:alnum:]]+::(gslice|mask|indirect)_array<.+>$",
                stl_summary_flags, true);
  AddCXXSummary(
      cpp_category_sp, lldb_private::formatters::ContainerSizeSummaryProvider,
      "libc++ std::list summary provider",
      "^std::__[[:alnum:]]+::forward_list<.+>$", stl_summary_flags, true);
  AddCXXSummary(
      cpp_category_sp, lldb_private::formatters::ContainerSizeSummaryProvider,
      "libc++ std::list summary provider",
      // A POSIX variant of: "^std::__(?!cxx11:)[[:alnum:]]+::list<.+>$"
      // so that it does not clash with: "^std::(__cxx11::)?list<.+>$"
      "^std::__([A-Zabd-z0-9]|cx?[A-Za-wyz0-9]|cxx1?[A-Za-z02-9]|"
      "cxx11[[:alnum:]])[[:alnum:]]*::list<.+>$",
      stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::ContainerSizeSummaryProvider,
                "libc++ std::map summary provider",
                "^std::__[[:alnum:]]+::map<.+>$", stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::ContainerSizeSummaryProvider,
                "libc++ std::deque summary provider",
                "^std::__[[:alnum:]]+::deque<.+>$", stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::ContainerSizeSummaryProvider,
                "libc++ std::queue summary provider",
                "^std::__[[:alnum:]]+::queue<.+>$", stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::ContainerSizeSummaryProvider,
                "libc++ std::set summary provider",
                "^std::__[[:alnum:]]+::set<.+>$", stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::ContainerSizeSummaryProvider,
                "libc++ std::multiset summary provider",
                "^std::__[[:alnum:]]+::multiset<.+>$", stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::ContainerSizeSummaryProvider,
                "libc++ std::multimap summary provider",
                "^std::__[[:alnum:]]+::multimap<.+>$", stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::ContainerSizeSummaryProvider,
                "libc++ std::unordered containers summary provider",
                "^std::__[[:alnum:]]+::unordered_(multi)?(map|set)<.+> >$",
                stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp, ContainerSizeSummaryProvider,
                "libc++ std::tuple summary provider",
                "^std::__[[:alnum:]]+::tuple<.*>$", stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibCxxAtomicSummaryProvider,
                "libc++ std::atomic summary provider",
                "^std::__[[:alnum:]]+::atomic<.+>$", stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::GenericOptionalSummaryProvider,
                "libc++ std::optional summary provider",
                "^std::__[[:alnum:]]+::optional<.+>$", stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxVariantSummaryProvider,
                "libc++ std::variant summary provider",
                "^std::__[[:alnum:]]+::variant<.+>$", stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::ContainerSizeSummaryProvider,
                "libc++ std::span summary provider",
                "^std::__[[:alnum:]]+::span<.+>$", stl_summary_flags, true);

  stl_summary_flags.SetSkipPointers(true);

  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxSmartPointerSummaryProvider,
                "libc++ std::shared_ptr summary provider",
                "^std::__[[:alnum:]]+::shared_ptr<.+>$", stl_summary_flags,
                true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxSmartPointerSummaryProvider,
                "libc++ std::weak_ptr summary provider",
                "^std::__[[:alnum:]]+::weak_ptr<.+>$", stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxUniquePointerSummaryProvider,
                "libc++ std::unique_ptr summary provider",
                libcxx_std_unique_ptr_regex, stl_summary_flags, true);

  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::StdlibCoroutineHandleSummaryProvider,
                "libc++ std::coroutine_handle summary provider",
                libcxx_std_coroutine_handle_regex, stl_summary_flags, true);

  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibCxxVectorIteratorSyntheticFrontEndCreator,
      "std::vector iterator synthetic children",
      "^std::__[[:alnum:]]+::__wrap_iter<.+>$", stl_synth_flags, true);

  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibCxxMapIteratorSyntheticFrontEndCreator,
      "std::map iterator synthetic children",
      "^std::__[[:alnum:]]+::__map_(const_)?iterator<.+>$", stl_synth_flags,
      true);

  AddCXXSynthetic(cpp_category_sp,
                  lldb_private::formatters::
                      LibCxxUnorderedMapIteratorSyntheticFrontEndCreator,
                  "std::unordered_map iterator synthetic children",
                  "^std::__[[:alnum:]]+::__hash_map_(const_)?iterator<.+>$",
                  stl_synth_flags, true);
  // Chrono duration typedefs
  cpp_category_sp->AddTypeSummary(
      "^std::__[[:alnum:]]+::chrono::nanoseconds", eFormatterMatchRegex,
      TypeSummaryImplSP(new StringSummaryFormat(
          eTypeOptionHideChildren | eTypeOptionHideValue, "${var.__rep_} ns")));
  cpp_category_sp->AddTypeSummary(
      "^std::__[[:alnum:]]+::chrono::microseconds", eFormatterMatchRegex,
      TypeSummaryImplSP(new StringSummaryFormat(
          eTypeOptionHideChildren | eTypeOptionHideValue, "${var.__rep_} µs")));
  cpp_category_sp->AddTypeSummary(
      "^std::__[[:alnum:]]+::chrono::milliseconds", eFormatterMatchRegex,
      TypeSummaryImplSP(new StringSummaryFormat(
          eTypeOptionHideChildren | eTypeOptionHideValue, "${var.__rep_} ms")));
  cpp_category_sp->AddTypeSummary(
      "^std::__[[:alnum:]]+::chrono::seconds", eFormatterMatchRegex,
      TypeSummaryImplSP(new StringSummaryFormat(
          eTypeOptionHideChildren | eTypeOptionHideValue, "${var.__rep_} s")));
  cpp_category_sp->AddTypeSummary(
      "^std::__[[:alnum:]]+::chrono::minutes", eFormatterMatchRegex,
      TypeSummaryImplSP(new StringSummaryFormat(eTypeOptionHideChildren |
                                                    eTypeOptionHideValue,
                                                "${var.__rep_} min")));
  cpp_category_sp->AddTypeSummary(
      "^std::__[[:alnum:]]+::chrono::hours", eFormatterMatchRegex,
      TypeSummaryImplSP(new StringSummaryFormat(
          eTypeOptionHideChildren | eTypeOptionHideValue, "${var.__rep_} h")));

  cpp_category_sp->AddTypeSummary(
      "^std::__[[:alnum:]]+::chrono::days", eFormatterMatchRegex,
      TypeSummaryImplSP(new StringSummaryFormat(eTypeOptionHideChildren |
                                                    eTypeOptionHideValue,
                                                "${var.__rep_} days")));
  cpp_category_sp->AddTypeSummary(
      "^std::__[[:alnum:]]+::chrono::weeks", eFormatterMatchRegex,
      TypeSummaryImplSP(new StringSummaryFormat(eTypeOptionHideChildren |
                                                    eTypeOptionHideValue,
                                                "${var.__rep_} weeks")));
  cpp_category_sp->AddTypeSummary(
      "^std::__[[:alnum:]]+::chrono::months", eFormatterMatchRegex,
      TypeSummaryImplSP(new StringSummaryFormat(eTypeOptionHideChildren |
                                                    eTypeOptionHideValue,
                                                "${var.__rep_} months")));
  cpp_category_sp->AddTypeSummary(
      "^std::__[[:alnum:]]+::chrono::years", eFormatterMatchRegex,
      TypeSummaryImplSP(new StringSummaryFormat(eTypeOptionHideChildren |
                                                    eTypeOptionHideValue,
                                                "${var.__rep_} years")));
  cpp_category_sp->AddTypeSummary(
      "^std::__[[:alnum:]]+::chrono::seconds", eFormatterMatchRegex,
      TypeSummaryImplSP(new StringSummaryFormat(
          eTypeOptionHideChildren | eTypeOptionHideValue, "${var.__rep_} s")));

  // Chrono time point types

  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxChronoSysSecondsSummaryProvider,
                "libc++ std::chrono::sys_seconds summary provider",
                "^std::__[[:alnum:]]+::chrono::time_point<"
                "std::__[[:alnum:]]+::chrono::system_clock, "
                "std::__[[:alnum:]]+::chrono::duration<.*, "
                "std::__[[:alnum:]]+::ratio<1, 1> "
                "> >$",
                eTypeOptionHideChildren | eTypeOptionHideValue |
                    eTypeOptionCascade,
                true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxChronoSysDaysSummaryProvider,
                "libc++ std::chrono::sys_seconds summary provider",
                "^std::__[[:alnum:]]+::chrono::time_point<"
                "std::__[[:alnum:]]+::chrono::system_clock, "
                "std::__[[:alnum:]]+::chrono::duration<int, "
                "std::__[[:alnum:]]+::ratio<86400, 1> "
                "> >$",
                eTypeOptionHideChildren | eTypeOptionHideValue |
                    eTypeOptionCascade,
                true);

  AddCXXSummary(
      cpp_category_sp,
      lldb_private::formatters::LibcxxChronoLocalSecondsSummaryProvider,
      "libc++ std::chrono::local_seconds summary provider",
      "^std::__[[:alnum:]]+::chrono::time_point<"
      "std::__[[:alnum:]]+::chrono::local_t, "
      "std::__[[:alnum:]]+::chrono::duration<.*, "
      "std::__[[:alnum:]]+::ratio<1, 1> "
      "> >$",
      eTypeOptionHideChildren | eTypeOptionHideValue | eTypeOptionCascade,
      true);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxChronoLocalDaysSummaryProvider,
                "libc++ std::chrono::local_seconds summary provider",
                "^std::__[[:alnum:]]+::chrono::time_point<"
                "std::__[[:alnum:]]+::chrono::local_t, "
                "std::__[[:alnum:]]+::chrono::duration<int, "
                "std::__[[:alnum:]]+::ratio<86400, 1> "
                "> >$",
                eTypeOptionHideChildren | eTypeOptionHideValue |
                    eTypeOptionCascade,
                true);

  // Chrono calendar types

  cpp_category_sp->AddTypeSummary(
      "^std::__[[:alnum:]]+::chrono::day$", eFormatterMatchRegex,
      TypeSummaryImplSP(new StringSummaryFormat(eTypeOptionHideChildren |
                                                    eTypeOptionHideValue,
                                                "day=${var.__d_%u}")));

  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxChronoMonthSummaryProvider,
                "libc++ std::chrono::month summary provider",
                "^std::__[[:alnum:]]+::chrono::month$",
                eTypeOptionHideChildren | eTypeOptionHideValue, true);

  cpp_category_sp->AddTypeSummary(
      "^std::__[[:alnum:]]+::chrono::year$", eFormatterMatchRegex,
      TypeSummaryImplSP(new StringSummaryFormat(
          eTypeOptionHideChildren | eTypeOptionHideValue, "year=${var.__y_}")));

  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::LibcxxChronoWeekdaySummaryProvider,
                "libc++ std::chrono::weekday summary provider",
                "^std::__[[:alnum:]]+::chrono::weekday$",
                eTypeOptionHideChildren | eTypeOptionHideValue, true);

  cpp_category_sp->AddTypeSummary(
      "^std::__[[:alnum:]]+::chrono::weekday_indexed$", eFormatterMatchRegex,
      TypeSummaryImplSP(new StringSummaryFormat(
          eTypeOptionHideChildren | eTypeOptionHideValue,
          "${var.__wd_} index=${var.__idx_%u}")));

  cpp_category_sp->AddTypeSummary(
      "^std::__[[:alnum:]]+::chrono::weekday_last$", eFormatterMatchRegex,
      TypeSummaryImplSP(new StringSummaryFormat(eTypeOptionHideChildren |
                                                    eTypeOptionHideValue,
                                                "${var.__wd_} index=last")));
  cpp_category_sp->AddTypeSummary(
      "^std::__[[:alnum:]]+::chrono::month_day$", eFormatterMatchRegex,
      TypeSummaryImplSP(new StringSummaryFormat(eTypeOptionHideChildren |
                                                    eTypeOptionHideValue,
                                                "${var.__m_} ${var.__d_}")));
  cpp_category_sp->AddTypeSummary(
      "^std::__[[:alnum:]]+::chrono::month_day_last$", eFormatterMatchRegex,
      TypeSummaryImplSP(new StringSummaryFormat(eTypeOptionHideChildren |
                                                    eTypeOptionHideValue,
                                                "${var.__m_} day=last")));

  cpp_category_sp->AddTypeSummary(
      "^std::__[[:alnum:]]+::chrono::month_weekday$", eFormatterMatchRegex,
      TypeSummaryImplSP(new StringSummaryFormat(eTypeOptionHideChildren |
                                                    eTypeOptionHideValue,
                                                "${var.__m_} ${var.__wdi_}")));

  cpp_category_sp->AddTypeSummary(
      "^std::__[[:alnum:]]+::chrono::month_weekday_last$", eFormatterMatchRegex,
      TypeSummaryImplSP(new StringSummaryFormat(eTypeOptionHideChildren |
                                                    eTypeOptionHideValue,
                                                "${var.__m_} ${var.__wdl_}")));

  cpp_category_sp->AddTypeSummary(
      "^std::__[[:alnum:]]+::chrono::year_month$", eFormatterMatchRegex,
      TypeSummaryImplSP(new StringSummaryFormat(eTypeOptionHideChildren |
                                                    eTypeOptionHideValue,
                                                "${var.__y_} ${var.__m_}")));

  AddCXXSummary(
      cpp_category_sp,
      lldb_private::formatters::LibcxxChronoYearMonthDaySummaryProvider,
      "libc++ std::chrono::year_month_day summary provider",
      "^std::__[[:alnum:]]+::chrono::year_month_day$",
      eTypeOptionHideChildren | eTypeOptionHideValue, true);

  cpp_category_sp->AddTypeSummary(
      "^std::__[[:alnum:]]+::chrono::year_month_day_last$",
      eFormatterMatchRegex,
      TypeSummaryImplSP(new StringSummaryFormat(eTypeOptionHideChildren |
                                                    eTypeOptionHideValue,
                                                "${var.__y_} ${var.__mdl_}")));

  cpp_category_sp->AddTypeSummary(
      "^std::__[[:alnum:]]+::chrono::year_month_weekday$", eFormatterMatchRegex,
      TypeSummaryImplSP(new StringSummaryFormat(
          eTypeOptionHideChildren | eTypeOptionHideValue,
          "${var.__y_} ${var.__m_} ${var.__wdi_}")));

  cpp_category_sp->AddTypeSummary(
      "^std::__[[:alnum:]]+::chrono::year_month_weekday_last$",
      eFormatterMatchRegex,
      TypeSummaryImplSP(new StringSummaryFormat(
          eTypeOptionHideChildren | eTypeOptionHideValue,
          "${var.__y_} ${var.__m_} ${var.__wdl_}")));
}

static void RegisterStdStringSummaryProvider(
    const lldb::TypeCategoryImplSP &category_sp, llvm::StringRef string_ty,
    llvm::StringRef char_ty, lldb::TypeSummaryImplSP summary_sp) {
  auto makeSpecifier = [](llvm::StringRef name) {
    return std::make_shared<lldb_private::TypeNameSpecifierImpl>(
        name, eFormatterMatchExact);
  };

  category_sp->AddTypeSummary(makeSpecifier(string_ty), summary_sp);

  category_sp->AddTypeSummary(
      makeSpecifier(llvm::formatv("std::basic_string<{}>", char_ty).str()),
      summary_sp);

  category_sp->AddTypeSummary(
      std::make_shared<lldb_private::TypeNameSpecifierImpl>(
          llvm::formatv("^std::basic_string<{0}, ?std::char_traits<{0}>,.*>$",
                        char_ty)
              .str(),
          eFormatterMatchRegex),
      summary_sp);
}

static void RegisterStdStringViewSummaryProvider(
    const lldb::TypeCategoryImplSP &category_sp, llvm::StringRef string_ty,
    llvm::StringRef char_ty, lldb::TypeSummaryImplSP summary_sp) {
  // std::string_view
  category_sp->AddTypeSummary(
      std::make_shared<lldb_private::TypeNameSpecifierImpl>(
          string_ty, eFormatterMatchExact),
      summary_sp);

  // std::basic_string_view<char, std::char_traits<char>>
  // NativePDB has spaces at different positions compared to PDB and DWARF, so
  // use a regex and make them optional.
  category_sp->AddTypeSummary(
      std::make_shared<lldb_private::TypeNameSpecifierImpl>(
          llvm::formatv(
              "^std::basic_string_view<{0}, ?std::char_traits<{0}> ?>$",
              char_ty)
              .str(),
          eFormatterMatchRegex),
      summary_sp);
}

static void LoadLibStdcppFormatters(lldb::TypeCategoryImplSP cpp_category_sp) {
  if (!cpp_category_sp)
    return;

  TypeSummaryImpl::Flags stl_summary_flags;
  stl_summary_flags.SetCascades(true)
      .SetSkipPointers(false)
      .SetSkipReferences(false)
      .SetDontShowChildren(true)
      .SetDontShowValue(false)
      .SetShowMembersOneLiner(false)
      .SetHideItemNames(false);

  lldb::TypeSummaryImplSP string_summary_sp(new CXXFunctionSummaryFormat(
      stl_summary_flags, LibStdcppStringSummaryProvider,
      "libstdc++ std::(w)string summary provider"));
  cpp_category_sp->AddTypeSummary("std::__cxx11::string", eFormatterMatchExact,
                                  string_summary_sp);
  cpp_category_sp->AddTypeSummary(
      "^std::__cxx11::basic_string<char, std::char_traits<char>,.*>$",
      eFormatterMatchRegex, string_summary_sp);
  cpp_category_sp->AddTypeSummary("^std::__cxx11::basic_string<unsigned char, "
                                  "std::char_traits<unsigned char>,.*>$",
                                  eFormatterMatchRegex, string_summary_sp);

  cpp_category_sp->AddTypeSummary("std::__cxx11::wstring", eFormatterMatchExact,
                                  string_summary_sp);
  cpp_category_sp->AddTypeSummary(
      "^std::__cxx11::basic_string<wchar_t, std::char_traits<wchar_t>,.*>$",
      eFormatterMatchRegex, string_summary_sp);

  SyntheticChildren::Flags stl_synth_flags;
  stl_synth_flags.SetCascades(true).SetSkipPointers(false).SetSkipReferences(
      false);
  SyntheticChildren::Flags stl_deref_flags = stl_synth_flags;
  stl_deref_flags.SetFrontEndWantsDereference();

  cpp_category_sp->AddTypeSynthetic(
      "^std::__debug::vector<.+>(( )?&)?$", eFormatterMatchRegex,
      SyntheticChildrenSP(new ScriptedSyntheticChildren(
          stl_synth_flags,
          "lldb.formatters.cpp.gnu_libstdcpp.StdVectorSynthProvider")));
  cpp_category_sp->AddTypeSynthetic(
      "^std::__debug::map<.+> >(( )?&)?$", eFormatterMatchRegex,
      SyntheticChildrenSP(new ScriptedSyntheticChildren(
          stl_synth_flags,
          "lldb.formatters.cpp.gnu_libstdcpp.StdMapLikeSynthProvider")));
  cpp_category_sp->AddTypeSynthetic(
      "^std::__debug::deque<.+>(( )?&)?$", eFormatterMatchRegex,
      SyntheticChildrenSP(new ScriptedSyntheticChildren(
          stl_deref_flags,
          "lldb.formatters.cpp.gnu_libstdcpp.StdDequeSynthProvider")));
  cpp_category_sp->AddTypeSynthetic(
      "^std::__debug::set<.+> >(( )?&)?$", eFormatterMatchRegex,
      SyntheticChildrenSP(new ScriptedSyntheticChildren(
          stl_deref_flags,
          "lldb.formatters.cpp.gnu_libstdcpp.StdMapLikeSynthProvider")));
  cpp_category_sp->AddTypeSynthetic(
      "^std::__debug::multimap<.+> >(( )?&)?$", eFormatterMatchRegex,
      SyntheticChildrenSP(new ScriptedSyntheticChildren(
          stl_deref_flags,
          "lldb.formatters.cpp.gnu_libstdcpp.StdMapLikeSynthProvider")));
  cpp_category_sp->AddTypeSynthetic(
      "^std::__debug::multiset<.+> >(( )?&)?$", eFormatterMatchRegex,
      SyntheticChildrenSP(new ScriptedSyntheticChildren(
          stl_deref_flags,
          "lldb.formatters.cpp.gnu_libstdcpp.StdMapLikeSynthProvider")));
  cpp_category_sp->AddTypeSynthetic(
      "^std::__debug::unordered_(multi)?(map|set)<.+> >$", eFormatterMatchRegex,
      SyntheticChildrenSP(new ScriptedSyntheticChildren(
          stl_deref_flags,
          "lldb.formatters.cpp.gnu_libstdcpp.StdUnorderedMapSynthProvider")));
  cpp_category_sp->AddTypeSynthetic(
      "^std::__(debug|cxx11)::list<.+>(( )?&)?$", eFormatterMatchRegex,
      SyntheticChildrenSP(new ScriptedSyntheticChildren(
          stl_deref_flags,
          "lldb.formatters.cpp.gnu_libstdcpp.StdListSynthProvider")));
  cpp_category_sp->AddTypeSynthetic(
      "^std::__(debug|cxx11)::forward_list<.+>(( )?&)?$", eFormatterMatchRegex,
      SyntheticChildrenSP(new ScriptedSyntheticChildren(
          stl_synth_flags,
          "lldb.formatters.cpp.gnu_libstdcpp.StdForwardListSynthProvider")));

  stl_summary_flags.SetDontShowChildren(false);
  stl_summary_flags.SetSkipPointers(false);

  AddCXXSummary(
      cpp_category_sp, lldb_private::formatters::ContainerSizeSummaryProvider,
      "libstdc++ std::bitset summary provider",
      "^std::(__debug::)?bitset<.+>(( )?&)?$", stl_summary_flags, true);

  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::ContainerSizeSummaryProvider,
                "libstdc++ std::__debug::vector summary provider",
                "^std::__debug::vector<.+>(( )?&)?$", stl_summary_flags, true);

  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::ContainerSizeSummaryProvider,
                "libstdc++ debug std::map summary provider",
                "^std::__debug::map<.+> >(( )?&)?$", stl_summary_flags, true);

  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::ContainerSizeSummaryProvider,
                "libstdc++ debug std::set summary provider",
                "^std::__debug::set<.+> >(( )?&)?$", stl_summary_flags, true);

  AddCXXSummary(cpp_category_sp, ContainerSizeSummaryProvider,
                "libstdc++ debug std::deque summary provider",
                "^std::__debug::deque<.+>(( )?&)?$", stl_summary_flags, true);

  AddCXXSummary(
      cpp_category_sp, lldb_private::formatters::ContainerSizeSummaryProvider,
      "libstdc++ debug std::multimap summary provider",
      "^std::__debug::multimap<.+> >(( )?&)?$", stl_summary_flags, true);

  AddCXXSummary(
      cpp_category_sp, lldb_private::formatters::ContainerSizeSummaryProvider,
      "libstdc++ debug std::multiset summary provider",
      "^std::__debug::multiset<.+> >(( )?&)?$", stl_summary_flags, true);

  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::ContainerSizeSummaryProvider,
                "libstdc++ debug std unordered container summary provider",
                "^std::__debug::unordered_(multi)?(map|set)<.+> >$",
                stl_summary_flags, true);

  AddCXXSummary(
      cpp_category_sp, lldb_private::formatters::ContainerSizeSummaryProvider,
      "libstdc++ debug std::list summary provider",
      "^std::__(debug|cxx11)::list<.+>(( )?&)?$", stl_summary_flags, true);

  cpp_category_sp->AddTypeSummary(
      "^std::__(debug|cxx11)::forward_list<.+>(( )?&)?$", eFormatterMatchRegex,
      TypeSummaryImplSP(new ScriptSummaryFormat(
          stl_summary_flags,
          "lldb.formatters.cpp.gnu_libstdcpp.ForwardListSummaryProvider")));

  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibStdcppVectorIteratorSyntheticFrontEndCreator,
      "std::vector iterator synthetic children",
      "^__gnu_cxx::__normal_iterator<.+>$", stl_synth_flags, true);

  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibstdcppMapIteratorSyntheticFrontEndCreator,
      "std::map iterator synthetic children",
      "^std::_Rb_tree_(const_)?iterator<.+>$", stl_synth_flags, true);

  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibStdcppUniquePtrSyntheticFrontEndCreator,
      "std::unique_ptr synthetic children", "^std::unique_ptr<.+>(( )?&)?$",
      stl_synth_flags, true);

  static constexpr const char *const libstdcpp_std_coroutine_handle_regex =
      "^std::coroutine_handle<.+>(( )?&)?$";
  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::StdlibCoroutineHandleSyntheticFrontEndCreator,
      "std::coroutine_handle synthetic children",
      libstdcpp_std_coroutine_handle_regex, stl_deref_flags, true);

  AddCXXSynthetic(
      cpp_category_sp,
      lldb_private::formatters::LibStdcppBitsetSyntheticFrontEndCreator,
      "std::bitset synthetic child", "^std::(__debug::)?bitset<.+>(( )?&)?$",
      stl_deref_flags, true);

  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::StdlibCoroutineHandleSummaryProvider,
                "libstdc++ std::coroutine_handle summary provider",
                libstdcpp_std_coroutine_handle_regex, stl_summary_flags, true);
}

static lldb_private::SyntheticChildrenFrontEnd *
GenericSmartPointerSyntheticFrontEndCreator(CXXSyntheticChildren *children,
                                            lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;

  if (IsMsvcStlSmartPointer(*valobj_sp))
    return MsvcStlSmartPointerSyntheticFrontEndCreator(valobj_sp);
  return LibStdcppSharedPtrSyntheticFrontEndCreator(children, valobj_sp);
}

static bool
GenericSmartPointerSummaryProvider(ValueObject &valobj, Stream &stream,
                                   const TypeSummaryOptions &options) {
  if (IsMsvcStlSmartPointer(valobj))
    return MsvcStlSmartPointerSummaryProvider(valobj, stream, options);
  return LibStdcppSmartPointerSummaryProvider(valobj, stream, options);
}

static lldb_private::SyntheticChildrenFrontEnd *
GenericUniquePtrSyntheticFrontEndCreator(CXXSyntheticChildren *children,
                                         lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;

  if (IsMsvcStlUniquePtr(*valobj_sp))
    return MsvcStlUniquePtrSyntheticFrontEndCreator(valobj_sp);
  return LibStdcppUniquePtrSyntheticFrontEndCreator(children, valobj_sp);
}

static bool GenericUniquePtrSummaryProvider(ValueObject &valobj, Stream &stream,
                                            const TypeSummaryOptions &options) {
  if (IsMsvcStlUniquePtr(valobj))
    return MsvcStlUniquePtrSummaryProvider(valobj, stream, options);
  return LibStdcppUniquePointerSummaryProvider(valobj, stream, options);
}

static SyntheticChildrenFrontEnd *
GenericTupleSyntheticFrontEndCreator(CXXSyntheticChildren *children,
                                     lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;

  if (IsMsvcStlTuple(*valobj_sp))
    return MsvcStlTupleSyntheticFrontEndCreator(children, valobj_sp);
  return LibStdcppTupleSyntheticFrontEndCreator(children, valobj_sp);
}

static SyntheticChildrenFrontEnd *
GenericVectorSyntheticFrontEndCreator(CXXSyntheticChildren *children,
                                      lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;

  // checks for vector<T> and vector<bool>
  if (auto *msvc = MsvcStlVectorSyntheticFrontEndCreator(valobj_sp))
    return msvc;

  return new ScriptedSyntheticChildren::FrontEnd(
      "lldb.formatters.cpp.gnu_libstdcpp.StdVectorSynthProvider", *valobj_sp);
}

static SyntheticChildrenFrontEnd *
GenericListSyntheticFrontEndCreator(CXXSyntheticChildren *children,
                                    lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;

  if (IsMsvcStlList(*valobj_sp))
    return MsvcStlListSyntheticFrontEndCreator(children, valobj_sp);
  return new ScriptedSyntheticChildren::FrontEnd(
      "lldb.formatters.cpp.gnu_libstdcpp.StdListSynthProvider", *valobj_sp);
}

static SyntheticChildrenFrontEnd *
GenericForwardListSyntheticFrontEndCreator(CXXSyntheticChildren *children,
                                           lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;

  if (IsMsvcStlList(*valobj_sp))
    return MsvcStlForwardListSyntheticFrontEndCreator(children, valobj_sp);
  return new ScriptedSyntheticChildren::FrontEnd(
      "lldb.formatters.cpp.gnu_libstdcpp.StdForwardListSynthProvider",
      *valobj_sp);
}

static SyntheticChildrenFrontEnd *
GenericOptionalSyntheticFrontEndCreator(CXXSyntheticChildren *children,
                                        lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;

  if (IsMsvcStlOptional(*valobj_sp))
    return MsvcStlOptionalSyntheticFrontEndCreator(children, valobj_sp);
  return LibStdcppOptionalSyntheticFrontEndCreator(children, valobj_sp);
}

static SyntheticChildrenFrontEnd *
GenericVariantSyntheticFrontEndCreator(CXXSyntheticChildren *children,
                                       lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;

  if (IsMsvcStlVariant(*valobj_sp))
    return MsvcStlVariantSyntheticFrontEndCreator(children, valobj_sp);
  return new ScriptedSyntheticChildren::FrontEnd(
      "lldb.formatters.cpp.gnu_libstdcpp.VariantSynthProvider", *valobj_sp);
}

static bool GenericVariantSummaryProvider(ValueObject &valobj, Stream &stream,
                                          const TypeSummaryOptions &options) {
  if (IsMsvcStlVariant(valobj))
    return MsvcStlVariantSummaryProvider(valobj, stream, options);
  return LibStdcppVariantSummaryProvider(valobj, stream, options);
}

static SyntheticChildrenFrontEnd *
GenericUnorderedSyntheticFrontEndCreator(CXXSyntheticChildren *children,
                                         ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;

  if (IsMsvcStlUnordered(*valobj_sp))
    return MsvcStlUnorderedSyntheticFrontEndCreator(children, valobj_sp);
  return new ScriptedSyntheticChildren::FrontEnd(
      "lldb.formatters.cpp.gnu_libstdcpp.StdUnorderedMapSynthProvider",
      *valobj_sp);
}

static SyntheticChildrenFrontEnd *
GenericMapLikeSyntheticFrontEndCreator(CXXSyntheticChildren *children,
                                       ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;

  if (IsMsvcStlMapLike(*valobj_sp))
    return MsvcStlMapLikeSyntheticFrontEndCreator(valobj_sp);
  return new ScriptedSyntheticChildren::FrontEnd(
      "lldb.formatters.cpp.gnu_libstdcpp.StdMapLikeSynthProvider", *valobj_sp);
}

static SyntheticChildrenFrontEnd *
GenericDequeSyntheticFrontEndCreator(CXXSyntheticChildren *children,
                                     ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;

  if (IsMsvcStlDeque(*valobj_sp))
    return MsvcStlDequeSyntheticFrontEndCreator(children, valobj_sp);
  return new ScriptedSyntheticChildren::FrontEnd(
      "lldb.formatters.cpp.gnu_libstdcpp.StdDequeSynthProvider", *valobj_sp);
}

/// Load formatters that are formatting types from more than one STL
static void LoadCommonStlFormatters(lldb::TypeCategoryImplSP cpp_category_sp) {
  if (!cpp_category_sp)
    return;

  TypeSummaryImpl::Flags stl_summary_flags;
  stl_summary_flags.SetCascades(true)
      .SetSkipPointers(false)
      .SetSkipReferences(false)
      .SetDontShowChildren(true)
      .SetDontShowValue(false)
      .SetShowMembersOneLiner(false)
      .SetHideItemNames(false);
  SyntheticChildren::Flags stl_synth_flags;
  stl_synth_flags.SetCascades(true).SetSkipPointers(false).SetSkipReferences(
      false);

  using StringElementType = StringPrinter::StringElementType;

  RegisterStdStringSummaryProvider(
      cpp_category_sp, "std::string", "char",
      std::make_shared<CXXFunctionSummaryFormat>(
          stl_summary_flags,
          [](ValueObject &valobj, Stream &stream,
             const TypeSummaryOptions &options) {
            if (IsMsvcStlStringType(valobj))
              return MsvcStlStringSummaryProvider<StringElementType::ASCII>(
                  valobj, stream, options);
            return LibStdcppStringSummaryProvider(valobj, stream, options);
          },
          "MSVC STL/libstdc++ std::string summary provider"));
  RegisterStdStringSummaryProvider(
      cpp_category_sp, "std::wstring", "wchar_t",
      std::make_shared<CXXFunctionSummaryFormat>(
          stl_summary_flags,
          [](ValueObject &valobj, Stream &stream,
             const TypeSummaryOptions &options) {
            if (IsMsvcStlStringType(valobj))
              return MsvcStlWStringSummaryProvider(valobj, stream, options);
            return LibStdcppStringSummaryProvider(valobj, stream, options);
          },
          "MSVC STL/libstdc++ std::wstring summary provider"));

  stl_summary_flags.SetDontShowChildren(false);
  stl_summary_flags.SetSkipPointers(false);

  AddCXXSynthetic(cpp_category_sp, GenericSmartPointerSyntheticFrontEndCreator,
                  "std::shared_ptr synthetic children",
                  "^std::shared_ptr<.+>(( )?&)?$", stl_synth_flags, true);
  AddCXXSynthetic(cpp_category_sp, GenericSmartPointerSyntheticFrontEndCreator,
                  "std::weak_ptr synthetic children",
                  "^std::weak_ptr<.+>(( )?&)?$", stl_synth_flags, true);
  AddCXXSynthetic(cpp_category_sp, GenericUniquePtrSyntheticFrontEndCreator,
                  "std::unique_ptr synthetic children",
                  "^std::unique_ptr<.+>(( )?&)?$", stl_synth_flags, true);
  AddCXXSynthetic(cpp_category_sp, GenericTupleSyntheticFrontEndCreator,
                  "std::tuple synthetic children", "^std::tuple<.*>(( )?&)?$",
                  stl_synth_flags, true);
  AddCXXSynthetic(cpp_category_sp, GenericListSyntheticFrontEndCreator,
                  "std::list synthetic children", "^std::list<.+>(( )?&)?$",
                  stl_synth_flags, true);
  AddCXXSynthetic(cpp_category_sp, GenericForwardListSyntheticFrontEndCreator,
                  "std::forward_list synthetic children",
                  "^std::forward_list<.+>(( )?&)?$", stl_synth_flags, true);
  AddCXXSynthetic(cpp_category_sp, GenericVariantSyntheticFrontEndCreator,
                  "std::variant synthetic children", "^std::variant<.*>$",
                  stl_synth_flags, true);
  AddCXXSynthetic(cpp_category_sp, GenericUnorderedSyntheticFrontEndCreator,
                  "std::unordered container synthetic children",
                  "^std::unordered_(multi)?(map|set)<.+> ?>$", stl_synth_flags,
                  true);

  SyntheticChildren::Flags stl_deref_flags = stl_synth_flags;
  stl_deref_flags.SetFrontEndWantsDereference();
  AddCXXSynthetic(cpp_category_sp, GenericOptionalSyntheticFrontEndCreator,
                  "std::optional synthetic children",
                  "^std::optional<.+>(( )?&)?$", stl_deref_flags, true);
  AddCXXSynthetic(cpp_category_sp, GenericDequeSyntheticFrontEndCreator,
                  "std::deque container synthetic children",
                  "^std::deque<.+>(( )?&)?$", stl_deref_flags, true);

  AddCXXSynthetic(cpp_category_sp, GenericMapLikeSyntheticFrontEndCreator,
                  "std::(multi)?map/set synthetic children",
                  "^std::(multi)?(map|set)<.+>(( )?&)?$", stl_synth_flags,
                  true);

  AddCXXSummary(cpp_category_sp, GenericSmartPointerSummaryProvider,
                "MSVC STL/libstdc++ std::shared_ptr summary provider",
                "^std::shared_ptr<.+>(( )?&)?$", stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp, GenericSmartPointerSummaryProvider,
                "MSVC STL/libstdc++ std::weak_ptr summary provider",
                "^std::weak_ptr<.+>(( )?&)?$", stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp, GenericUniquePtrSummaryProvider,
                "MSVC STL/libstdc++ std::unique_ptr summary provider",
                "^std::unique_ptr<.+>(( )?&)?$", stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp, ContainerSizeSummaryProvider,
                "MSVC STL/libstdc++ std::tuple summary provider",
                "^std::tuple<.*>(( )?&)?$", stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp, ContainerSizeSummaryProvider,
                "MSVC/libstdc++ std::vector summary provider",
                "^std::vector<.+>(( )?&)?$", stl_summary_flags, true);
  AddCXXSynthetic(cpp_category_sp, GenericVectorSyntheticFrontEndCreator,
                  "MSVC/libstdc++ std::vector synthetic provider",
                  "^std::vector<.+>(( )?&)?$", stl_synth_flags, true);
  AddCXXSummary(cpp_category_sp, ContainerSizeSummaryProvider,
                "MSVC STL/libstdc++ std::list summary provider",
                "^std::list<.+>(( )?&)?$", stl_summary_flags, true);
  cpp_category_sp->AddTypeSummary(
      "^std::forward_list<.+>(( )?&)?$", eFormatterMatchRegex,
      TypeSummaryImplSP(new ScriptSummaryFormat(
          stl_summary_flags,
          "lldb.formatters.cpp.gnu_libstdcpp.ForwardListSummaryProvider")));
  AddCXXSummary(cpp_category_sp, GenericOptionalSummaryProvider,
                "MSVC STL/libstd++ std::optional summary provider",
                "^std::optional<.+>(( )?&)?$", stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp, GenericVariantSummaryProvider,
                "MSVC STL/libstdc++ std::variant summary provider",
                "^std::variant<.*>$", stl_summary_flags, true);
  AddCXXSummary(cpp_category_sp, ContainerSizeSummaryProvider,
                "MSVC STL/libstdc++ std unordered container summary provider",
                "^std::unordered_(multi)?(map|set)<.+> ?>$", stl_summary_flags,
                true);
  AddCXXSummary(cpp_category_sp, ContainerSizeSummaryProvider,
                "MSVC STL/libstdc++ std::(multi)?map/set summary provider",
                "^std::(multi)?(map|set)<.+>(( )?&)?$", stl_summary_flags,
                true);
  AddCXXSummary(cpp_category_sp, ContainerSizeSummaryProvider,
                "MSVC STL/libstd++ std::deque summary provider",
                "^std::deque<.+>(( )?&)?$", stl_summary_flags, true);
}

static void LoadMsvcStlFormatters(lldb::TypeCategoryImplSP cpp_category_sp) {
  if (!cpp_category_sp)
    return;

  TypeSummaryImpl::Flags stl_summary_flags;
  stl_summary_flags.SetCascades(true)
      .SetSkipPointers(false)
      .SetSkipReferences(false)
      .SetDontShowChildren(true)
      .SetDontShowValue(false)
      .SetShowMembersOneLiner(false)
      .SetHideItemNames(false);
  SyntheticChildren::Flags stl_synth_flags;
  stl_synth_flags.SetCascades(true).SetSkipPointers(false).SetSkipReferences(
      false);

  using StringElementType = StringPrinter::StringElementType;

  RegisterStdStringSummaryProvider(
      cpp_category_sp, "std::u8string", "char8_t",
      std::make_shared<CXXFunctionSummaryFormat>(
          stl_summary_flags,
          MsvcStlStringSummaryProvider<StringElementType::UTF8>,
          "MSVC STL std::u8string summary provider"));
  RegisterStdStringSummaryProvider(
      cpp_category_sp, "std::u16string", "char16_t",
      std::make_shared<CXXFunctionSummaryFormat>(
          stl_summary_flags,
          MsvcStlStringSummaryProvider<StringElementType::UTF16>,
          "MSVC STL std::u16string summary provider"));
  RegisterStdStringSummaryProvider(
      cpp_category_sp, "std::u32string", "char32_t",
      std::make_shared<CXXFunctionSummaryFormat>(
          stl_summary_flags,
          MsvcStlStringSummaryProvider<StringElementType::UTF32>,
          "MSVC STL std::u32string summary provider"));

  RegisterStdStringViewSummaryProvider(
      cpp_category_sp, "std::string_view", "char",
      std::make_shared<CXXFunctionSummaryFormat>(
          stl_summary_flags,
          MsvcStlStringViewSummaryProvider<StringElementType::ASCII>,
          "MSVC STL std::string_view summary provider"));
  RegisterStdStringViewSummaryProvider(
      cpp_category_sp, "std::u8string_view", "char8_t",
      std::make_shared<CXXFunctionSummaryFormat>(
          stl_summary_flags,
          MsvcStlStringViewSummaryProvider<StringElementType::UTF8>,
          "MSVC STL std::u8string_view summary provider"));
  RegisterStdStringViewSummaryProvider(
      cpp_category_sp, "std::u16string_view", "char16_t",
      std::make_shared<CXXFunctionSummaryFormat>(
          stl_summary_flags,
          MsvcStlStringViewSummaryProvider<StringElementType::UTF16>,
          "MSVC STL std::u16string_view summary provider"));
  RegisterStdStringViewSummaryProvider(
      cpp_category_sp, "std::u32string_view", "char32_t",
      std::make_shared<CXXFunctionSummaryFormat>(
          stl_summary_flags,
          MsvcStlStringViewSummaryProvider<StringElementType::UTF32>,
          "MSVC STL std::u32string_view summary provider"));
  RegisterStdStringViewSummaryProvider(
      cpp_category_sp, "std::wstring_view", "wchar_t",
      std::make_shared<CXXFunctionSummaryFormat>(
          stl_summary_flags, MsvcStlWStringViewSummaryProvider,
          "MSVC STL std::wstring_view summary provider"));

  stl_summary_flags.SetDontShowChildren(false);

  AddCXXSynthetic(cpp_category_sp, MsvcStlAtomicSyntheticFrontEndCreator,
                  "MSVC STL std::atomic synthetic children",
                  "^std::atomic<.+>$", stl_synth_flags, true);

  AddCXXSummary(cpp_category_sp, MsvcStlAtomicSummaryProvider,
                "MSVC STL std::atomic summary provider", "^std::atomic<.+>$",
                stl_summary_flags, true);
  AddCXXSynthetic(cpp_category_sp, MsvcStlTreeIterSyntheticFrontEndCreator,
                  "MSVC STL tree iterator synthetic children",
                  "^std::_Tree(_const)?_iterator<.+>(( )?&)?$", stl_synth_flags,
                  true);
  AddCXXSummary(cpp_category_sp, MsvcStlTreeIterSummaryProvider,
                "MSVC STL tree iterator summary",
                "^std::_Tree(_const)?_iterator<.+>(( )?&)?$", stl_summary_flags,
                true);
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

  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::Char8StringSummaryProvider,
                "char8_t * summary provider", "char8_t *", string_flags);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::Char8StringSummaryProvider,
                "char8_t [] summary provider", "char8_t ?\\[[0-9]+\\]",
                string_array_flags, true);

  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::Char16StringSummaryProvider,
                "char16_t * summary provider", "char16_t *", string_flags);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::Char16StringSummaryProvider,
                "char16_t [] summary provider", "char16_t ?\\[[0-9]+\\]",
                string_array_flags, true);

  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::Char32StringSummaryProvider,
                "char32_t * summary provider", "char32_t *", string_flags);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::Char32StringSummaryProvider,
                "char32_t [] summary provider", "char32_t ?\\[[0-9]+\\]",
                string_array_flags, true);

  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::WCharStringSummaryProvider,
                "wchar_t * summary provider", "wchar_t *", string_flags);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::WCharStringSummaryProvider,
                "wchar_t * summary provider", "wchar_t ?\\[[0-9]+\\]",
                string_array_flags, true);

  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::Char16StringSummaryProvider,
                "unichar * summary provider", "unichar *", string_flags);

  TypeSummaryImpl::Flags widechar_flags;
  widechar_flags.SetDontShowValue(true)
      .SetSkipPointers(true)
      .SetSkipReferences(false)
      .SetCascades(true)
      .SetDontShowChildren(true)
      .SetHideItemNames(true)
      .SetShowMembersOneLiner(false);

  AddCXXSummary(cpp_category_sp, lldb_private::formatters::Char8SummaryProvider,
                "char8_t summary provider", "char8_t", widechar_flags);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::Char16SummaryProvider,
                "char16_t summary provider", "char16_t", widechar_flags);
  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::Char32SummaryProvider,
                "char32_t summary provider", "char32_t", widechar_flags);
  AddCXXSummary(cpp_category_sp, lldb_private::formatters::WCharSummaryProvider,
                "wchar_t summary provider", "wchar_t", widechar_flags);

  AddCXXSummary(cpp_category_sp,
                lldb_private::formatters::Char16SummaryProvider,
                "unichar summary provider", "unichar", widechar_flags);
}

std::unique_ptr<Language::TypeScavenger> CPlusPlusLanguage::GetTypeScavenger() {
  class CPlusPlusTypeScavenger : public Language::ImageListTypeScavenger {
  public:
    CompilerType AdjustForInclusion(CompilerType &candidate) override {
      LanguageType lang_type(candidate.GetMinimumLanguage());
      if (!Language::LanguageIsC(lang_type) &&
          !Language::LanguageIsCPlusPlus(lang_type))
        return CompilerType();
      if (candidate.IsTypedefType())
        return candidate.GetTypedefedType();
      return candidate;
    }
  };

  return std::unique_ptr<TypeScavenger>(new CPlusPlusTypeScavenger());
}

lldb::TypeCategoryImplSP CPlusPlusLanguage::GetFormatters() {
  static llvm::once_flag g_initialize;
  static TypeCategoryImplSP g_category;

  llvm::call_once(g_initialize, [this]() -> void {
    DataVisualization::Categories::GetCategory(ConstString(GetPluginName()),
                                               g_category);
    if (g_category) {
      // NOTE: the libstdcpp formatters are loaded after libcxx formatters
      // because we don't want to the libcxx formatters to match the potential
      // `__debug` inline namespace that libstdcpp may use.
      // LLDB prioritizes the last loaded matching formatter.
      LoadLibCxxFormatters(g_category);
      LoadLibStdcppFormatters(g_category);
      LoadMsvcStlFormatters(g_category);
      LoadCommonStlFormatters(g_category);
      LoadSystemFormatters(g_category);
    }
  });
  return g_category;
}

HardcodedFormatters::HardcodedSummaryFinder
CPlusPlusLanguage::GetHardcodedSummaries() {
  static llvm::once_flag g_initialize;
  static ConstString g_vectortypes("VectorTypes");
  static HardcodedFormatters::HardcodedSummaryFinder g_formatters;

  llvm::call_once(g_initialize, []() -> void {
    g_formatters.push_back(
        [](lldb_private::ValueObject &valobj, lldb::DynamicValueType,
           FormatManager &) -> TypeSummaryImpl::SharedPointer {
          static CXXFunctionSummaryFormat::SharedPointer formatter_sp(
              new CXXFunctionSummaryFormat(
                  TypeSummaryImpl::Flags(),
                  lldb_private::formatters::CXXFunctionPointerSummaryProvider,
                  "Function pointer summary provider"));
          if (CompilerType CT = valobj.GetCompilerType();
              CT.IsFunctionPointerType() || CT.IsMemberFunctionPointerType() ||
              valobj.GetValueType() == lldb::eValueTypeVTableEntry) {
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
          if (valobj.GetCompilerType().IsVectorType()) {
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
          if (valobj.GetCompilerType().IsBlockPointerType()) {
            return formatter_sp;
          }
          return nullptr;
        });
  });

  return g_formatters;
}

HardcodedFormatters::HardcodedSyntheticFinder
CPlusPlusLanguage::GetHardcodedSynthetics() {
  static llvm::once_flag g_initialize;
  static ConstString g_vectortypes("VectorTypes");
  static HardcodedFormatters::HardcodedSyntheticFinder g_formatters;

  llvm::call_once(g_initialize, []() -> void {
    g_formatters.push_back([](lldb_private::ValueObject &valobj,
                              lldb::DynamicValueType, FormatManager &fmt_mgr)
                               -> SyntheticChildren::SharedPointer {
      static CXXSyntheticChildren::SharedPointer formatter_sp(
          new CXXSyntheticChildren(
              SyntheticChildren::Flags()
                  .SetCascades(true)
                  .SetSkipPointers(true)
                  .SetSkipReferences(true)
                  .SetNonCacheable(true),
              "vector_type synthetic children",
              lldb_private::formatters::VectorTypeSyntheticFrontEndCreator));
      if (valobj.GetCompilerType().IsVectorType()) {
        if (fmt_mgr.GetCategory(g_vectortypes)->IsEnabled())
          return formatter_sp;
      }
      return nullptr;
    });
    g_formatters.push_back([](lldb_private::ValueObject &valobj,
                              lldb::DynamicValueType, FormatManager &fmt_mgr)
                               -> SyntheticChildren::SharedPointer {
      static CXXSyntheticChildren::SharedPointer formatter_sp(
          new CXXSyntheticChildren(
              SyntheticChildren::Flags()
                  .SetCascades(true)
                  .SetSkipPointers(true)
                  .SetSkipReferences(true)
                  .SetNonCacheable(true),
              "block pointer synthetic children",
              lldb_private::formatters::BlockPointerSyntheticFrontEndCreator));
      if (valobj.GetCompilerType().IsBlockPointerType()) {
        return formatter_sp;
      }
      return nullptr;
    });
  });

  return g_formatters;
}

bool CPlusPlusLanguage::IsNilReference(ValueObject &valobj) {
  if (!Language::LanguageIsCPlusPlus(valobj.GetObjectRuntimeLanguage()) ||
      !valobj.IsPointerType())
    return false;
  bool canReadValue = true;
  bool isZero = valobj.GetValueAsUnsigned(0, &canReadValue) == 0;
  return canReadValue && isZero;
}

bool CPlusPlusLanguage::IsSourceFile(llvm::StringRef file_path) const {
  const auto suffixes = {".cpp", ".cxx", ".c++", ".cc",  ".c",
                         ".h",   ".hh",  ".hpp", ".hxx", ".h++"};
  for (auto suffix : suffixes) {
    if (file_path.ends_with_insensitive(suffix))
      return true;
  }

  // Check if we're in a STL path (where the files usually have no extension
  // that we could check for.
  return file_path.contains("/usr/include/c++/");
}

static VariableListSP GetFunctionVariableList(const SymbolContext &sc) {
  assert(sc.function);

  if (sc.block)
    if (Block *inline_block = sc.block->GetContainingInlinedBlock())
      return inline_block->GetBlockVariableList(true);

  return sc.function->GetBlock(true).GetBlockVariableList(true);
}

static bool PrintFunctionNameWithArgs(Stream &s,
                                      const ExecutionContext *exe_ctx,
                                      const SymbolContext &sc) {
  assert(sc.function);

  ExecutionContextScope *exe_scope =
      exe_ctx ? exe_ctx->GetBestExecutionContextScope() : nullptr;

  const char *cstr = sc.GetPossiblyInlinedFunctionName()
                         .GetName(Mangled::NamePreference::ePreferDemangled)
                         .AsCString();
  if (!cstr)
    return false;

  VariableList args;
  if (auto variable_list_sp = GetFunctionVariableList(sc))
    variable_list_sp->AppendVariablesWithScope(eValueTypeVariableArgument,
                                               args);

  if (args.GetSize() > 0)
    return PrettyPrintFunctionNameWithArgs(s, cstr, exe_scope, args);

  // FIXME: can we just unconditionally call PrettyPrintFunctionNameWithArgs?
  // It should be able to handle the "no arguments" case.
  s.PutCString(cstr);

  return true;
}

bool CPlusPlusLanguage::GetFunctionDisplayName(
    const SymbolContext &sc, const ExecutionContext *exe_ctx,
    FunctionNameRepresentation representation, Stream &s) {
  switch (representation) {
  case FunctionNameRepresentation::eNameWithArgs: {
    // Print the function name with arguments in it
    if (sc.function)
      return PrintFunctionNameWithArgs(s, exe_ctx, sc);

    if (!sc.symbol)
      return false;

    const char *cstr = sc.symbol->GetName().AsCString(nullptr);
    if (!cstr)
      return false;

    s.PutCString(cstr);

    return true;
  }
  case FunctionNameRepresentation::eNameWithNoArgs:
  case FunctionNameRepresentation::eName:
    return false;
  }
}

bool CPlusPlusLanguage::HandleFrameFormatVariable(
    const SymbolContext &sc, const ExecutionContext *exe_ctx,
    FormatEntity::Entry::Type type, Stream &s) {
  switch (type) {
  case FormatEntity::Entry::Type::FunctionScope: {
    auto scope_or_err = ::GetDemangledScope(sc);
    if (!scope_or_err) {
      LLDB_LOG_ERROR(
          GetLog(LLDBLog::Language), scope_or_err.takeError(),
          "Failed to handle ${{function.scope}} frame-format variable: {0}");
      return false;
    }

    s << *scope_or_err;

    return true;
  }

  case FormatEntity::Entry::Type::FunctionBasename: {
    auto name_or_err = ::GetDemangledBasename(sc);
    if (!name_or_err) {
      LLDB_LOG_ERROR(
          GetLog(LLDBLog::Language), name_or_err.takeError(),
          "Failed to handle ${{function.basename}} frame-format variable: {0}");
      return false;
    }

    s << *name_or_err;

    return true;
  }

  case FormatEntity::Entry::Type::FunctionTemplateArguments: {
    auto template_args_or_err = ::GetDemangledTemplateArguments(sc);
    if (!template_args_or_err) {
      LLDB_LOG_ERROR(GetLog(LLDBLog::Language),
                     template_args_or_err.takeError(),
                     "Failed to handle ${{function.template-arguments}} "
                     "frame-format variable: {0}");
      return false;
    }

    s << *template_args_or_err;

    return true;
  }

  case FormatEntity::Entry::Type::FunctionFormattedArguments: {
    // This ensures we print the arguments even when no debug-info is available.
    //
    // FIXME: we should have a Entry::Type::FunctionArguments and
    // use it in the plugin.cplusplus.display.function-name-format
    // once we have a "fallback operator" in the frame-format language.
    if (!sc.function && sc.symbol)
      return PrintDemangledArgumentList(s, sc);

    VariableList args;
    if (auto variable_list_sp = GetFunctionVariableList(sc))
      variable_list_sp->AppendVariablesWithScope(eValueTypeVariableArgument,
                                                 args);

    ExecutionContextScope *exe_scope =
        exe_ctx ? exe_ctx->GetBestExecutionContextScope() : nullptr;

    s << '(';
    FormatEntity::PrettyPrintFunctionArguments(s, args, exe_scope);
    s << ')';

    return true;
  }
  case FormatEntity::Entry::Type::FunctionReturnRight: {
    auto return_rhs_or_err = ::GetDemangledReturnTypeRHS(sc);
    if (!return_rhs_or_err) {
      LLDB_LOG_ERROR(GetLog(LLDBLog::Language), return_rhs_or_err.takeError(),
                     "Failed to handle ${{function.return-right}} frame-format "
                     "variable: {0}");
      return false;
    }

    s << *return_rhs_or_err;

    return true;
  }
  case FormatEntity::Entry::Type::FunctionReturnLeft: {
    auto return_lhs_or_err = ::GetDemangledReturnTypeLHS(sc);
    if (!return_lhs_or_err) {
      LLDB_LOG_ERROR(GetLog(LLDBLog::Language), return_lhs_or_err.takeError(),
                     "Failed to handle ${{function.return-left}} frame-format "
                     "variable: {0}");
      return false;
    }

    s << *return_lhs_or_err;

    return true;
  }
  case FormatEntity::Entry::Type::FunctionQualifiers: {
    auto quals_or_err = ::GetDemangledFunctionQualifiers(sc);
    if (!quals_or_err) {
      LLDB_LOG_ERROR(GetLog(LLDBLog::Language), quals_or_err.takeError(),
                     "Failed to handle ${{function.qualifiers}} frame-format "
                     "variable: {0}");
      return false;
    }

    s << *quals_or_err;

    return true;
  }
  case FormatEntity::Entry::Type::FunctionSuffix: {
    auto suffix_or_err = ::GetDemangledFunctionSuffix(sc);
    if (!suffix_or_err) {
      LLDB_LOG_ERROR(
          GetLog(LLDBLog::Language), suffix_or_err.takeError(),
          "Failed to handle ${{function.suffix}} frame-format variable: {0}");
      return false;
    }

    s << *suffix_or_err;

    return true;
  }
  default:
    return false;
  }
}

namespace {
class NodeAllocator {
  llvm::BumpPtrAllocator Alloc;

public:
  void reset() { Alloc.Reset(); }

  template <typename T, typename... Args> T *makeNode(Args &&...args) {
    return new (Alloc.Allocate(sizeof(T), alignof(T)))
        T(std::forward<Args>(args)...);
  }

  void *allocateNodeArray(size_t sz) {
    return Alloc.Allocate(sizeof(llvm::itanium_demangle::Node *) * sz,
                          alignof(llvm::itanium_demangle::Node *));
  }
};

template <typename Derived>
class ManglingSubstitutor
    : public llvm::itanium_demangle::AbstractManglingParser<Derived,
                                                            NodeAllocator> {
  using Base =
      llvm::itanium_demangle::AbstractManglingParser<Derived, NodeAllocator>;

public:
  ManglingSubstitutor() : Base(nullptr, nullptr) {}

  template <typename... Ts>
  llvm::Expected<ConstString> substitute(llvm::StringRef Mangled,
                                         Ts &&...Vals) {
    this->getDerived().reset(Mangled, std::forward<Ts>(Vals)...);
    return substituteImpl(Mangled);
  }

protected:
  void reset(llvm::StringRef Mangled) {
    Base::reset(Mangled.begin(), Mangled.end());
    Written = Mangled.begin();
    Result.clear();
    Substituted = false;
  }

  llvm::Expected<ConstString> substituteImpl(llvm::StringRef Mangled) {
    if (this->parse() == nullptr)
      return llvm::createStringError(
          llvm::formatv("Failed to substitute mangling in '{0}'", Mangled));

    if (!Substituted)
      return ConstString();

    // Append any trailing unmodified input.
    appendUnchangedInput();
    return ConstString(Result);
  }

  void trySubstitute(llvm::StringRef From, llvm::StringRef To) {
    if (!llvm::StringRef(currentParserPos(), this->numLeft()).starts_with(From))
      return;

    // We found a match. Append unmodified input up to this point.
    appendUnchangedInput();

    // And then perform the replacement.
    Result += To;
    Written += From.size();
    Substituted = true;
  }

private:
  /// Input character until which we have constructed the respective output
  /// already.
  const char *Written = "";

  llvm::SmallString<128> Result;

  /// Whether we have performed any substitutions.
  bool Substituted = false;

  const char *currentParserPos() const { return this->First; }

  void appendUnchangedInput() {
    Result +=
        llvm::StringRef(Written, std::distance(Written, currentParserPos()));
    Written = currentParserPos();
  }
};

/// Given a mangled function `Mangled`, replace all the primitive function type
/// arguments of `Search` with type `Replace`.
class TypeSubstitutor : public ManglingSubstitutor<TypeSubstitutor> {
  llvm::StringRef Search;
  llvm::StringRef Replace;

public:
  void reset(llvm::StringRef Mangled, llvm::StringRef Search,
             llvm::StringRef Replace) {
    ManglingSubstitutor::reset(Mangled);
    this->Search = Search;
    this->Replace = Replace;
  }

  llvm::itanium_demangle::Node *parseType() {
    trySubstitute(Search, Replace);
    return ManglingSubstitutor::parseType();
  }
};

class CtorDtorSubstitutor : public ManglingSubstitutor<CtorDtorSubstitutor> {
  llvm::StringRef Search;
  llvm::StringRef Replace;

public:
  void reset(llvm::StringRef Mangled, llvm::StringRef Search,
             llvm::StringRef Replace) {
    ManglingSubstitutor::reset(Mangled);
    this->Search = Search;
    this->Replace = Replace;
  }

  void reset(llvm::StringRef Mangled) { ManglingSubstitutor::reset(Mangled); }

  llvm::itanium_demangle::Node *
  parseCtorDtorName(llvm::itanium_demangle::Node *&SoFar, NameState *State) {
    if (!Search.empty() && !Replace.empty()) {
      trySubstitute(Search, Replace);
    } else {
      trySubstitute("D1", "D2");
      trySubstitute("C1", "C2");
    }
    return ManglingSubstitutor::parseCtorDtorName(SoFar, State);
  }
};
} // namespace

llvm::Expected<ConstString>
CPlusPlusLanguage::SubstituteType_ItaniumMangle(llvm::StringRef mangled_name,
                                                llvm::StringRef subst_from,
                                                llvm::StringRef subst_to) {
  return TypeSubstitutor().substitute(mangled_name, subst_from, subst_to);
}

llvm::Expected<ConstString> CPlusPlusLanguage::SubstituteStructor_ItaniumMangle(
    llvm::StringRef mangled_name, llvm::StringRef subst_from,
    llvm::StringRef subst_to) {
  return CtorDtorSubstitutor().substitute(mangled_name, subst_from, subst_to);
}

llvm::Expected<ConstString>
CPlusPlusLanguage::SubstituteStructorAliases_ItaniumMangle(
    llvm::StringRef mangled_name) {
  return CtorDtorSubstitutor().substitute(mangled_name);
}

#define LLDB_PROPERTIES_language_cplusplus
#include "LanguageCPlusPlusProperties.inc"

enum {
#define LLDB_PROPERTIES_language_cplusplus
#include "LanguageCPlusPlusPropertiesEnum.inc"
};

namespace {
class PluginProperties : public Properties {
public:
  static llvm::StringRef GetSettingName() { return "display"; }

  PluginProperties() {
    m_collection_sp = std::make_shared<OptionValueProperties>(GetSettingName());
    m_collection_sp->Initialize(g_language_cplusplus_properties);
  }

  FormatEntity::Entry GetFunctionNameFormat() const {
    return GetPropertyAtIndexAs<FormatEntity::Entry>(
        ePropertyFunctionNameFormat, {});
  }
};
} // namespace

static PluginProperties &GetGlobalPluginProperties() {
  static PluginProperties g_settings;
  return g_settings;
}

FormatEntity::Entry CPlusPlusLanguage::GetFunctionNameFormat() const {
  return GetGlobalPluginProperties().GetFunctionNameFormat();
}

void CPlusPlusLanguage::DebuggerInitialize(Debugger &debugger) {
  if (!PluginManager::GetSettingForCPlusPlusLanguagePlugin(
          debugger, PluginProperties::GetSettingName())) {
    PluginManager::CreateSettingForCPlusPlusLanguagePlugin(
        debugger, GetGlobalPluginProperties().GetValueProperties(),
        "Properties for the CPlusPlus language plug-in.",
        /*is_global_property=*/true);
  }
}
