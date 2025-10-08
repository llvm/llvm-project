//===-- Language.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <functional>
#include <map>
#include <mutex>

#include "lldb/Target/Language.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/Interpreter/OptionValueProperties.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/TypeList.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/Stream.h"

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Threading.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

typedef std::unique_ptr<Language> LanguageUP;
typedef std::map<lldb::LanguageType, LanguageUP> LanguagesMap;

#define LLDB_PROPERTIES_language
#include "TargetProperties.inc"

enum {
#define LLDB_PROPERTIES_language
#include "TargetPropertiesEnum.inc"
};

LanguageProperties &Language::GetGlobalLanguageProperties() {
  static LanguageProperties g_settings;
  return g_settings;
}

llvm::StringRef LanguageProperties::GetSettingName() {
  static constexpr llvm::StringLiteral g_setting_name("language");
  return g_setting_name;
}

LanguageProperties::LanguageProperties() {
  m_collection_sp = std::make_shared<OptionValueProperties>(GetSettingName());
  m_collection_sp->Initialize(g_language_properties);
}

bool LanguageProperties::GetEnableFilterForLineBreakpoints() const {
  const uint32_t idx = ePropertyEnableFilterForLineBreakpoints;
  return GetPropertyAtIndexAs<bool>(
      idx, g_language_properties[idx].default_uint_value != 0);
}

static LanguagesMap &GetLanguagesMap() {
  static LanguagesMap *g_map = nullptr;
  static llvm::once_flag g_initialize;

  llvm::call_once(g_initialize, [] {
    g_map = new LanguagesMap(); // NOTE: INTENTIONAL LEAK due to global
                                // destructor chain
  });

  return *g_map;
}
static std::mutex &GetLanguagesMutex() {
  static std::mutex *g_mutex = nullptr;
  static llvm::once_flag g_initialize;

  llvm::call_once(g_initialize, [] {
    g_mutex = new std::mutex(); // NOTE: INTENTIONAL LEAK due to global
                                // destructor chain
  });

  return *g_mutex;
}

Language *Language::FindPlugin(lldb::LanguageType language) {
  std::lock_guard<std::mutex> guard(GetLanguagesMutex());
  LanguagesMap &map(GetLanguagesMap());
  auto iter = map.find(language), end = map.end();
  if (iter != end)
    return iter->second.get();

  Language *language_ptr = nullptr;
  LanguageCreateInstance create_callback;

  for (uint32_t idx = 0;
       (create_callback =
            PluginManager::GetLanguageCreateCallbackAtIndex(idx)) != nullptr;
       ++idx) {
    language_ptr = create_callback(language);

    if (language_ptr) {
      map[language] = std::unique_ptr<Language>(language_ptr);
      return language_ptr;
    }
  }

  return nullptr;
}

Language *Language::FindPlugin(llvm::StringRef file_path) {
  Language *result = nullptr;
  ForEach([&result, file_path](Language *language) {
    if (language->IsSourceFile(file_path)) {
      result = language;
      return false;
    }
    return true;
  });
  return result;
}

Language *Language::FindPlugin(LanguageType language,
                               llvm::StringRef file_path) {
  Language *result = FindPlugin(language);
  // Finding a language by file path is slower, we so we use this as the
  // fallback.
  if (!result)
    result = FindPlugin(file_path);
  return result;
}

void Language::ForEach(std::function<bool(Language *)> callback) {
  // If we want to iterate over all languages, we first have to complete the
  // LanguagesMap.
  static llvm::once_flag g_initialize;
  llvm::call_once(g_initialize, [] {
    for (unsigned lang = eLanguageTypeUnknown; lang < eNumLanguageTypes;
         ++lang) {
      FindPlugin(static_cast<lldb::LanguageType>(lang));
    }
  });

  // callback may call a method in Language that attempts to acquire the same
  // lock (such as Language::ForEach or Language::FindPlugin). To avoid a
  // deadlock, we do not use callback while holding the lock.
  std::vector<Language *> loaded_plugins;
  {
    std::lock_guard<std::mutex> guard(GetLanguagesMutex());
    LanguagesMap &map(GetLanguagesMap());
    for (const auto &entry : map) {
      if (entry.second)
        loaded_plugins.push_back(entry.second.get());
    }
  }

  for (auto *lang : loaded_plugins) {
    if (!callback(lang))
      break;
  }
}

bool Language::IsTopLevelFunction(Function &function) { return false; }

lldb::TypeCategoryImplSP Language::GetFormatters() { return nullptr; }

HardcodedFormatters::HardcodedFormatFinder Language::GetHardcodedFormats() {
  return {};
}

HardcodedFormatters::HardcodedSummaryFinder Language::GetHardcodedSummaries() {
  return {};
}

HardcodedFormatters::HardcodedSyntheticFinder
Language::GetHardcodedSynthetics() {
  return {};
}

std::vector<FormattersMatchCandidate>
Language::GetPossibleFormattersMatches(ValueObject &valobj,
                                       lldb::DynamicValueType use_dynamic) {
  return {};
}

struct language_name_pair {
  const char *name;
  LanguageType type;
};

struct language_name_pair language_names[] = {
    // To allow GetNameForLanguageType to be a simple array lookup, the first
    // part of this array must follow enum LanguageType exactly.
    {"unknown", eLanguageTypeUnknown},
    {"c89", eLanguageTypeC89},
    {"c", eLanguageTypeC},
    {"ada83", eLanguageTypeAda83},
    {"c++", eLanguageTypeC_plus_plus},
    {"cobol74", eLanguageTypeCobol74},
    {"cobol85", eLanguageTypeCobol85},
    {"fortran77", eLanguageTypeFortran77},
    {"fortran90", eLanguageTypeFortran90},
    {"pascal83", eLanguageTypePascal83},
    {"modula2", eLanguageTypeModula2},
    {"java", eLanguageTypeJava},
    {"c99", eLanguageTypeC99},
    {"ada95", eLanguageTypeAda95},
    {"fortran95", eLanguageTypeFortran95},
    {"pli", eLanguageTypePLI},
    {"objective-c", eLanguageTypeObjC},
    {"objective-c++", eLanguageTypeObjC_plus_plus},
    {"upc", eLanguageTypeUPC},
    {"d", eLanguageTypeD},
    {"python", eLanguageTypePython},
    {"opencl", eLanguageTypeOpenCL},
    {"go", eLanguageTypeGo},
    {"modula3", eLanguageTypeModula3},
    {"haskell", eLanguageTypeHaskell},
    {"c++03", eLanguageTypeC_plus_plus_03},
    {"c++11", eLanguageTypeC_plus_plus_11},
    {"ocaml", eLanguageTypeOCaml},
    {"rust", eLanguageTypeRust},
    {"c11", eLanguageTypeC11},
    {"swift", eLanguageTypeSwift},
    {"julia", eLanguageTypeJulia},
    {"dylan", eLanguageTypeDylan},
    {"c++14", eLanguageTypeC_plus_plus_14},
    {"fortran03", eLanguageTypeFortran03},
    {"fortran08", eLanguageTypeFortran08},
    {"renderscript", eLanguageTypeRenderScript},
    {"bliss", eLanguageTypeBLISS},
    {"kotlin", eLanguageTypeKotlin},
    {"zig", eLanguageTypeZig},
    {"crystal", eLanguageTypeCrystal},
    {"<invalid language>",
     static_cast<LanguageType>(
         0x0029)}, // Not yet taken by any language in the DWARF spec
                   // and thus has no entry in LanguageType
    {"c++17", eLanguageTypeC_plus_plus_17},
    {"c++20", eLanguageTypeC_plus_plus_20},
    {"c17", eLanguageTypeC17},
    {"fortran18", eLanguageTypeFortran18},
    {"ada2005", eLanguageTypeAda2005},
    {"ada2012", eLanguageTypeAda2012},
    {"HIP", eLanguageTypeHIP},
    {"assembly", eLanguageTypeAssembly},
    {"c-sharp", eLanguageTypeC_sharp},
    {"mojo", eLanguageTypeMojo},
    // Vendor Extensions
    {"assembler", eLanguageTypeMipsAssembler},
    // Now synonyms, in arbitrary order
    {"objc", eLanguageTypeObjC},
    {"objc++", eLanguageTypeObjC_plus_plus},
    {"pascal", eLanguageTypePascal83}};

static uint32_t num_languages =
    sizeof(language_names) / sizeof(struct language_name_pair);

LanguageType Language::GetLanguageTypeFromString(llvm::StringRef string) {
  for (const auto &L : language_names) {
    if (string.equals_insensitive(L.name))
      return L.type;
  }

  return eLanguageTypeUnknown;
}

const char *Language::GetNameForLanguageType(LanguageType language) {
  if (language < num_languages)
    return language_names[language].name;
  else
    return language_names[eLanguageTypeUnknown].name;
}

llvm::StringRef Language::GetDisplayNameForLanguageType(LanguageType language) {
  return SourceLanguage(language).GetDescription();
}

void Language::PrintSupportedLanguagesForExpressions(Stream &s,
                                                     llvm::StringRef prefix,
                                                     llvm::StringRef suffix) {
  auto supported = Language::GetLanguagesSupportingTypeSystemsForExpressions();
  for (size_t idx = 0; idx < num_languages; ++idx) {
    auto const &lang = language_names[idx];
    if (supported[lang.type])
      s << prefix << lang.name << suffix;
  }
}

void Language::PrintAllLanguages(Stream &s, const char *prefix,
                                 const char *suffix) {
  for (uint32_t i = 1; i < num_languages; i++) {
    s.Printf("%s%s%s", prefix, language_names[i].name, suffix);
  }
}

void Language::ForAllLanguages(
    std::function<bool(lldb::LanguageType)> callback) {
  for (uint32_t i = 1; i < num_languages; i++) {
    if (!callback(language_names[i].type))
      break;
  }
}

bool Language::LanguageIsCPlusPlus(LanguageType language) {
  switch (language) {
  case eLanguageTypeC_plus_plus:
  case eLanguageTypeC_plus_plus_03:
  case eLanguageTypeC_plus_plus_11:
  case eLanguageTypeC_plus_plus_14:
  case eLanguageTypeC_plus_plus_17:
  case eLanguageTypeC_plus_plus_20:
  case eLanguageTypeObjC_plus_plus:
    return true;
  default:
    return false;
  }
}

bool Language::LanguageIsObjC(LanguageType language) {
  switch (language) {
  case eLanguageTypeObjC:
  case eLanguageTypeObjC_plus_plus:
    return true;
  default:
    return false;
  }
}

bool Language::LanguageIsC(LanguageType language) {
  switch (language) {
  case eLanguageTypeC:
  case eLanguageTypeC89:
  case eLanguageTypeC99:
  case eLanguageTypeC11:
    return true;
  default:
    return false;
  }
}

bool Language::LanguageIsCFamily(LanguageType language) {
  switch (language) {
  case eLanguageTypeC:
  case eLanguageTypeC89:
  case eLanguageTypeC99:
  case eLanguageTypeC11:
  case eLanguageTypeC_plus_plus:
  case eLanguageTypeC_plus_plus_03:
  case eLanguageTypeC_plus_plus_11:
  case eLanguageTypeC_plus_plus_14:
  case eLanguageTypeC_plus_plus_17:
  case eLanguageTypeC_plus_plus_20:
  case eLanguageTypeObjC_plus_plus:
  case eLanguageTypeObjC:
    return true;
  default:
    return false;
  }
}

bool Language::LanguageIsPascal(LanguageType language) {
  switch (language) {
  case eLanguageTypePascal83:
    return true;
  default:
    return false;
  }
}

LanguageType Language::GetPrimaryLanguage(LanguageType language) {
  switch (language) {
  case eLanguageTypeC_plus_plus:
  case eLanguageTypeC_plus_plus_03:
  case eLanguageTypeC_plus_plus_11:
  case eLanguageTypeC_plus_plus_14:
  case eLanguageTypeC_plus_plus_17:
  case eLanguageTypeC_plus_plus_20:
    return eLanguageTypeC_plus_plus;
  case eLanguageTypeC:
  case eLanguageTypeC89:
  case eLanguageTypeC99:
  case eLanguageTypeC11:
    return eLanguageTypeC;
  case eLanguageTypeObjC:
  case eLanguageTypeObjC_plus_plus:
    return eLanguageTypeObjC;
  case eLanguageTypePascal83:
  case eLanguageTypeCobol74:
  case eLanguageTypeCobol85:
  case eLanguageTypeFortran77:
  case eLanguageTypeFortran90:
  case eLanguageTypeFortran95:
  case eLanguageTypeFortran03:
  case eLanguageTypeFortran08:
  case eLanguageTypeAda83:
  case eLanguageTypeAda95:
  case eLanguageTypeModula2:
  case eLanguageTypeJava:
  case eLanguageTypePLI:
  case eLanguageTypeUPC:
  case eLanguageTypeD:
  case eLanguageTypePython:
  case eLanguageTypeOpenCL:
  case eLanguageTypeGo:
  case eLanguageTypeModula3:
  case eLanguageTypeHaskell:
  case eLanguageTypeOCaml:
  case eLanguageTypeRust:
  case eLanguageTypeSwift:
  case eLanguageTypeJulia:
  case eLanguageTypeDylan:
  case eLanguageTypeMipsAssembler:
  case eLanguageTypeMojo:
  case eLanguageTypeUnknown:
  default:
    return language;
  }
}

std::set<lldb::LanguageType> Language::GetSupportedLanguages() {
  std::set<lldb::LanguageType> supported_languages;
  ForEach([&](Language *lang) {
    supported_languages.emplace(lang->GetLanguageType());
    return true;
  });
  return supported_languages;
}

LanguageSet Language::GetLanguagesSupportingTypeSystems() {
  return PluginManager::GetAllTypeSystemSupportedLanguagesForTypes();
}

LanguageSet Language::GetLanguagesSupportingTypeSystemsForExpressions() {
  return PluginManager::GetAllTypeSystemSupportedLanguagesForExpressions();
}

LanguageSet Language::GetLanguagesSupportingREPLs() {
  return PluginManager::GetREPLAllTypeSystemSupportedLanguages();
}

std::unique_ptr<Language::TypeScavenger> Language::GetTypeScavenger() {
  return nullptr;
}

const char *Language::GetLanguageSpecificTypeLookupHelp() { return nullptr; }

size_t Language::TypeScavenger::Find(ExecutionContextScope *exe_scope,
                                     const char *key, ResultSet &results,
                                     bool append) {
  if (!exe_scope || !exe_scope->CalculateTarget().get())
    return false;

  if (!key || !key[0])
    return false;

  if (!append)
    results.clear();

  size_t old_size = results.size();

  if (this->Find_Impl(exe_scope, key, results))
    return results.size() - old_size;
  return 0;
}

bool Language::ImageListTypeScavenger::Find_Impl(
    ExecutionContextScope *exe_scope, const char *key, ResultSet &results) {
  bool result = false;

  Target *target = exe_scope->CalculateTarget().get();
  if (target) {
    const auto &images(target->GetImages());
    TypeQuery query(key);
    TypeResults type_results;
    images.FindTypes(nullptr, query, type_results);
    for (const auto &match : type_results.GetTypeMap().Types()) {
      if (match) {
        CompilerType compiler_type(match->GetFullCompilerType());
        compiler_type = AdjustForInclusion(compiler_type);
        if (!compiler_type)
          continue;
        std::unique_ptr<Language::TypeScavenger::Result> scavengeresult(
            new Result(compiler_type));
        results.insert(std::move(scavengeresult));
        result = true;
      }
    }
  }

  return result;
}

std::pair<llvm::StringRef, llvm::StringRef>
Language::GetFormatterPrefixSuffix(llvm::StringRef type_hint) {
  return std::pair<llvm::StringRef, llvm::StringRef>();
}

bool Language::DemangledNameContainsPath(llvm::StringRef path,
                                         ConstString demangled) const {
  // The base implementation does a simple contains comparision:
  if (path.empty())
    return false;
  return demangled.GetStringRef().contains(path);
}

DumpValueObjectOptions::DeclPrintingHelper Language::GetDeclPrintingHelper() {
  return nullptr;
}

LazyBool Language::IsLogicalTrue(ValueObject &valobj, Status &error) {
  return eLazyBoolCalculate;
}

bool Language::IsNilReference(ValueObject &valobj) { return false; }

bool Language::IsUninitializedReference(ValueObject &valobj) { return false; }

bool Language::GetFunctionDisplayName(const SymbolContext &sc,
                                      const ExecutionContext *exe_ctx,
                                      FunctionNameRepresentation representation,
                                      Stream &s) {
  return false;
}

void Language::GetExceptionResolverDescription(bool catch_on, bool throw_on,
                                               Stream &s) {
  GetDefaultExceptionResolverDescription(catch_on, throw_on, s);
}

void Language::GetDefaultExceptionResolverDescription(bool catch_on,
                                                      bool throw_on,
                                                      Stream &s) {
  s.Printf("Exception breakpoint (catch: %s throw: %s)",
           catch_on ? "on" : "off", throw_on ? "on" : "off");
}

std::optional<bool> Language::GetBooleanFromString(llvm::StringRef str) const {
  return llvm::StringSwitch<std::optional<bool>>(str)
      .Case("true", {true})
      .Case("false", {false})
      .Default({});
}

// Constructor
Language::Language() = default;

// Destructor
Language::~Language() = default;

static std::optional<llvm::dwarf::SourceLanguage>
ToDwarfSourceLanguage(lldb::LanguageType language_type) {
  if (language_type <= lldb::eLanguageTypeLastStandardLanguage)
    return static_cast<llvm::dwarf::SourceLanguage>(language_type);

  switch (language_type) {
  case eLanguageTypeMipsAssembler:
    return llvm::dwarf::DW_LANG_Mips_Assembler;
  default:
    return std::nullopt;
  }
}

SourceLanguage::SourceLanguage(lldb::LanguageType language_type) {
  std::optional<llvm::dwarf::SourceLanguage> dwarf_lang =
      ToDwarfSourceLanguage(language_type);
  if (!dwarf_lang)
    return;

  auto lname = llvm::dwarf::toDW_LNAME(*dwarf_lang);
  if (!lname)
    return;
  name = lname->first;
  version = lname->second;
}

lldb::LanguageType SourceLanguage::AsLanguageType() const {
  if (auto lang = llvm::dwarf::toDW_LANG((llvm::dwarf::SourceLanguageName)name,
                                         version))
    return (lldb::LanguageType)*lang;
  return lldb::eLanguageTypeUnknown;
}

llvm::StringRef SourceLanguage::GetDescription() const {
  return llvm::dwarf::LanguageDescription(
      static_cast<llvm::dwarf::SourceLanguageName>(name), version);
}
bool SourceLanguage::IsC() const { return name == llvm::dwarf::DW_LNAME_C; }

bool SourceLanguage::IsObjC() const {
  return name == llvm::dwarf::DW_LNAME_ObjC;
}

bool SourceLanguage::IsCPlusPlus() const {
  return name == llvm::dwarf::DW_LNAME_C_plus_plus;
}

// BEGIN SWIFT
// Implement LanguageCPlusPlus::GetParentNameIfClosure and upstream this.
// rdar://152321823

/// If `sc` represents a "closure"-like function (according to
/// Language::GetParentNameIfClosure), returns sc.function and all parent
/// functions up to and including the first non-closure-like function. If `sc`
/// is not a closure, or if the query does not make sense for `language`,
/// returns a list containing only sc.function.
static llvm::SmallVector<Function *>
GetParentFunctionsWhileClosure(const SymbolContext &sc,
                               const Language &language) {
  // The algorithm below terminates on the assumption that
  // `GetParentNameIfClosure` produces an empty string when composing that
  // function with itself enough times. For safety, define an upper limit.
  constexpr auto upper_limit = 8;

  llvm::SmallVector<Function *> parents;
  Function *root = sc.function;
  if (root == nullptr)
    return parents;

  parents.push_back(root);
  for (int idx = 0; idx < upper_limit; idx++) {
    std::string parent = language.GetParentNameIfClosure(*root);
    if (parent.empty())
      break;

    // Find the enclosing function, if it exists.
    SymbolContextList sc_list;
    Module::LookupInfo lookup_info(
        ConstString(parent), lldb::FunctionNameType::eFunctionNameTypeFull,
        lldb::eLanguageTypeSwift);
    sc.module_sp->FindFunctions(lookup_info, CompilerDeclContext(),
                                ModuleFunctionSearchOptions(), sc_list);
    if (sc_list.GetSize() != 1 || sc_list[0].function == nullptr)
      break;
    parents.push_back(sc_list[0].function);
    root = sc_list[0].function;
  }
  return parents;
}

/// Scans the line table of `function` looking for the first entry whose line
/// number is `line_number`. If no such entry is found, returns the entry
/// closest to but after `line_number`.
static std::optional<Address> FindAddressForLineNumber(Function &function,
                                                       uint32_t line_number) {
  CompileUnit *cu = function.GetCompileUnit();
  LineTable *line_table = cu ? cu->GetLineTable() : nullptr;
  if (line_table == nullptr)
    return std::nullopt;

  // Get the first line entry for this function.
  AddressRange func_range = function.GetAddressRanges().front();
  uint32_t first_entry_idx;
  {
    LineEntry first_line_entry;
    line_table->FindLineEntryByAddress(func_range.GetBaseAddress(),
                                       first_line_entry, &first_entry_idx);
  }

  LineEntry best_match;
  for (uint32_t entry_idx = first_entry_idx; entry_idx < line_table->GetSize();
       entry_idx++) {
    LineEntry next_line;
    line_table->GetLineEntryAtIndex(entry_idx, next_line);

    // Stop if this entry is outside the range of `function`.
    Address base_addr = next_line.range.GetBaseAddress();
    if (!func_range.ContainsFileAddress(base_addr))
      break;

    // Stop on an exact match.
    if (next_line.line == line_number) {
      best_match = next_line;
      break;
    }

    // Otherwise, keep track of the best match so far.
    if (next_line.line > line_number && next_line.line < best_match.line)
      best_match = next_line;
  }

  return best_match.range.GetBaseAddress();
}

/// Given a list of functions, returns a map: Function -> VariableList
/// containing local variables of each function.
static std::map<Function *, VariableList>
GetFuncToLocalVariablesMap(llvm::ArrayRef<Function *> funcs) {
  std::map<Function *, VariableList> map;
  for (Function *function : funcs) {
    VariableList &variable_list = map[function];
    Block &block = function->GetBlock(true /*can_create=*/);
    block.AppendBlockVariables(
        true /*can_create=*/, true /*get_child_block_variables=*/,
        true /*stop_if_child_block_is_inlined_function=*/,
        [](Variable *v) { return true; }, &variable_list);
  }
  return map;
}

/// Returns the first line associated with `function`.
static uint32_t GetLineNumberForFunction(Function &function) {
  SupportFileSP func_decl_file_sp;
  uint32_t line_num = 0;
  function.GetStartLineSourceInfo(func_decl_file_sp, line_num);
  return line_num;
}

/// Checks if `var` is in scope inside `function` at line `line_number`.
/// If this check can't be done, a best-effort comparison of:
///    line_number >= var.line_number
/// is performed.
static bool IsVariableInScopeAtLine(uint32_t line_number,
                                    std::optional<Address> line_addr,
                                    Variable &var) {
  auto fallback_line_comp = [&] {
    return line_number >= var.GetDeclaration().GetLine();
  };

  if (!line_addr)
    return fallback_line_comp();

  Block *defining_block = line_addr->CalculateSymbolContextBlock();
  if (defining_block)
    return var.IsInScope(*defining_block, *line_addr);
  return fallback_line_comp();
}

Function *Language::FindParentOfClosureWithVariable(
    llvm::StringRef variable_name, const SymbolContext &closure_sc) const {
  llvm::SmallVector<Function *> function_chain =
      GetParentFunctionsWhileClosure(closure_sc, *this);
  if (function_chain.empty())
    return nullptr;

  std::map<Function *, VariableList> func_to_locals =
      GetFuncToLocalVariablesMap(function_chain);

  llvm::ArrayRef<Function *> children =
      llvm::ArrayRef(function_chain).drop_back();
  llvm::ArrayRef<Function *> parents =
      llvm::ArrayRef(function_chain).drop_front();

  for (auto [parent, child] : llvm::zip_equal(parents, children)) {
    VariableList &parent_variables = func_to_locals[parent];
    uint32_t child_line_number = GetLineNumberForFunction(*child);
    std::optional<Address> parent_line_addr =
        FindAddressForLineNumber(*parent, child_line_number);

    for (const VariableSP &var : parent_variables)
      if (var->GetName() == variable_name &&
          IsVariableInScopeAtLine(child_line_number, parent_line_addr, *var))
        return parent;
  }
  return nullptr;
}
// END SWIFT
