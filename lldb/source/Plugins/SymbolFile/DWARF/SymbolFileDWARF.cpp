//===-- SymbolFileDWARF.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolFileDWARF.h"

#include "llvm/ADT/Optional.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLoc.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Threading.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Progress.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/Value.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/RegularExpression.h"
#include "lldb/Utility/Scalar.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/Timer.h"

#include "Plugins/ExpressionParser/Clang/ClangModulesDeclVendor.h"
#include "Plugins/Language/CPlusPlus/CPlusPlusLanguage.h"

#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"

#include "lldb/Interpreter/OptionValueFileSpecList.h"
#include "lldb/Interpreter/OptionValueProperties.h"

#include "Plugins/ExpressionParser/Clang/ClangUtil.h"
#include "Plugins/SymbolFile/DWARF/DWARFDebugInfoEntry.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/CompilerDecl.h"
#include "lldb/Symbol/CompilerDeclContext.h"
#include "lldb/Symbol/DebugMacros.h"
#include "lldb/Symbol/LineTable.h"
#include "lldb/Symbol/LocateSymbolFile.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/TypeMap.h"
#include "lldb/Symbol/TypeSystem.h"
#include "lldb/Symbol/VariableList.h"

#include "lldb/Target/Language.h"
#include "lldb/Target/Target.h"

#include "AppleDWARFIndex.h"
#include "DWARFASTParser.h"
#include "DWARFASTParserClang.h"
#include "DWARFCompileUnit.h"
#include "DWARFDebugAbbrev.h"
#include "DWARFDebugAranges.h"
#include "DWARFDebugInfo.h"
#include "DWARFDebugMacro.h"
#include "DWARFDebugRanges.h"
#include "DWARFDeclContext.h"
#include "DWARFFormValue.h"
#include "DWARFTypeUnit.h"
#include "DWARFUnit.h"
#include "DebugNamesDWARFIndex.h"
#include "LogChannelDWARF.h"
#include "ManualDWARFIndex.h"
#include "SymbolFileDWARFDebugMap.h"
#include "SymbolFileDWARFDwo.h"

#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"

#include <algorithm>
#include <map>
#include <memory>

#include <cctype>
#include <cstring>

//#define ENABLE_DEBUG_PRINTF // COMMENT OUT THIS LINE PRIOR TO CHECKIN

#ifdef ENABLE_DEBUG_PRINTF
#include <cstdio>
#define DEBUG_PRINTF(fmt, ...) printf(fmt, __VA_ARGS__)
#else
#define DEBUG_PRINTF(fmt, ...)
#endif

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::dwarf;

LLDB_PLUGIN_DEFINE(SymbolFileDWARF)

char SymbolFileDWARF::ID;

namespace {

#define LLDB_PROPERTIES_symbolfiledwarf
#include "SymbolFileDWARFProperties.inc"

enum {
#define LLDB_PROPERTIES_symbolfiledwarf
#include "SymbolFileDWARFPropertiesEnum.inc"
};

class PluginProperties : public Properties {
public:
  static ConstString GetSettingName() {
    return ConstString(SymbolFileDWARF::GetPluginNameStatic());
  }

  PluginProperties() {
    m_collection_sp = std::make_shared<OptionValueProperties>(GetSettingName());
    m_collection_sp->Initialize(g_symbolfiledwarf_properties);
  }

  bool IgnoreFileIndexes() const {
    return m_collection_sp->GetPropertyAtIndexAsBoolean(
        nullptr, ePropertyIgnoreIndexes, false);
  }
};

static PluginProperties &GetGlobalPluginProperties() {
  static PluginProperties g_settings;
  return g_settings;
}

} // namespace

static const llvm::DWARFDebugLine::LineTable *
ParseLLVMLineTable(lldb_private::DWARFContext &context,
                   llvm::DWARFDebugLine &line, dw_offset_t line_offset,
                   dw_offset_t unit_offset) {
  Log *log = GetLog(DWARFLog::DebugInfo);

  llvm::DWARFDataExtractor data = context.getOrLoadLineData().GetAsLLVM();
  llvm::DWARFContext &ctx = context.GetAsLLVM();
  llvm::Expected<const llvm::DWARFDebugLine::LineTable *> line_table =
      line.getOrParseLineTable(
          data, line_offset, ctx, nullptr, [&](llvm::Error e) {
            LLDB_LOG_ERROR(
                log, std::move(e),
                "SymbolFileDWARF::ParseLineTable failed to parse: {0}");
          });

  if (!line_table) {
    LLDB_LOG_ERROR(log, line_table.takeError(),
                   "SymbolFileDWARF::ParseLineTable failed to parse: {0}");
    return nullptr;
  }
  return *line_table;
}

static bool ParseLLVMLineTablePrologue(lldb_private::DWARFContext &context,
                                       llvm::DWARFDebugLine::Prologue &prologue,
                                       dw_offset_t line_offset,
                                       dw_offset_t unit_offset) {
  Log *log = GetLog(DWARFLog::DebugInfo);
  bool success = true;
  llvm::DWARFDataExtractor data = context.getOrLoadLineData().GetAsLLVM();
  llvm::DWARFContext &ctx = context.GetAsLLVM();
  uint64_t offset = line_offset;
  llvm::Error error = prologue.parse(
      data, &offset,
      [&](llvm::Error e) {
        success = false;
        LLDB_LOG_ERROR(log, std::move(e),
                       "SymbolFileDWARF::ParseSupportFiles failed to parse "
                       "line table prologue: {0}");
      },
      ctx, nullptr);
  if (error) {
    LLDB_LOG_ERROR(log, std::move(error),
                   "SymbolFileDWARF::ParseSupportFiles failed to parse line "
                   "table prologue: {0}");
    return false;
  }
  return success;
}

static llvm::Optional<std::string>
GetFileByIndex(const llvm::DWARFDebugLine::Prologue &prologue, size_t idx,
               llvm::StringRef compile_dir, FileSpec::Style style) {
  // Try to get an absolute path first.
  std::string abs_path;
  auto absolute = llvm::DILineInfoSpecifier::FileLineInfoKind::AbsoluteFilePath;
  if (prologue.getFileNameByIndex(idx, compile_dir, absolute, abs_path, style))
    return std::move(abs_path);

  // Otherwise ask for a relative path.
  std::string rel_path;
  auto relative = llvm::DILineInfoSpecifier::FileLineInfoKind::RawValue;
  if (!prologue.getFileNameByIndex(idx, compile_dir, relative, rel_path, style))
    return {};
  return std::move(rel_path);
}

static FileSpecList
ParseSupportFilesFromPrologue(const lldb::ModuleSP &module,
                              const llvm::DWARFDebugLine::Prologue &prologue,
                              FileSpec::Style style,
                              llvm::StringRef compile_dir = {}) {
  FileSpecList support_files;
  size_t first_file = 0;
  if (prologue.getVersion() <= 4) {
    // File index 0 is not valid before DWARF v5. Add a dummy entry to ensure
    // support file list indices match those we get from the debug info and line
    // tables.
    support_files.Append(FileSpec());
    first_file = 1;
  }

  const size_t number_of_files = prologue.FileNames.size();
  for (size_t idx = first_file; idx <= number_of_files; ++idx) {
    std::string remapped_file;
    if (auto file_path = GetFileByIndex(prologue, idx, compile_dir, style)) {
      if (auto remapped = module->RemapSourceFile(llvm::StringRef(*file_path)))
        remapped_file = *remapped;
      else
        remapped_file = std::move(*file_path);
    }

    // Unconditionally add an entry, so the indices match up.
    support_files.EmplaceBack(remapped_file, style);
  }

  return support_files;
}

void SymbolFileDWARF::Initialize() {
  LogChannelDWARF::Initialize();
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance,
                                DebuggerInitialize);
  SymbolFileDWARFDebugMap::Initialize();
}

void SymbolFileDWARF::DebuggerInitialize(Debugger &debugger) {
  if (!PluginManager::GetSettingForSymbolFilePlugin(
          debugger, PluginProperties::GetSettingName())) {
    const bool is_global_setting = true;
    PluginManager::CreateSettingForSymbolFilePlugin(
        debugger, GetGlobalPluginProperties().GetValueProperties(),
        ConstString("Properties for the dwarf symbol-file plug-in."),
        is_global_setting);
  }
}

void SymbolFileDWARF::Terminate() {
  SymbolFileDWARFDebugMap::Terminate();
  PluginManager::UnregisterPlugin(CreateInstance);
  LogChannelDWARF::Terminate();
}

llvm::StringRef SymbolFileDWARF::GetPluginDescriptionStatic() {
  return "DWARF and DWARF3 debug symbol file reader.";
}

SymbolFile *SymbolFileDWARF::CreateInstance(ObjectFileSP objfile_sp) {
  return new SymbolFileDWARF(std::move(objfile_sp),
                             /*dwo_section_list*/ nullptr);
}

TypeList &SymbolFileDWARF::GetTypeList() {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  if (SymbolFileDWARFDebugMap *debug_map_symfile = GetDebugMapSymfile())
    return debug_map_symfile->GetTypeList();
  return SymbolFileCommon::GetTypeList();
}
void SymbolFileDWARF::GetTypes(const DWARFDIE &die, dw_offset_t min_die_offset,
                               dw_offset_t max_die_offset, uint32_t type_mask,
                               TypeSet &type_set) {
  if (die) {
    const dw_offset_t die_offset = die.GetOffset();

    if (die_offset >= max_die_offset)
      return;

    if (die_offset >= min_die_offset) {
      const dw_tag_t tag = die.Tag();

      bool add_type = false;

      switch (tag) {
      case DW_TAG_array_type:
        add_type = (type_mask & eTypeClassArray) != 0;
        break;
      case DW_TAG_unspecified_type:
      case DW_TAG_base_type:
        add_type = (type_mask & eTypeClassBuiltin) != 0;
        break;
      case DW_TAG_class_type:
        add_type = (type_mask & eTypeClassClass) != 0;
        break;
      case DW_TAG_structure_type:
        add_type = (type_mask & eTypeClassStruct) != 0;
        break;
      case DW_TAG_union_type:
        add_type = (type_mask & eTypeClassUnion) != 0;
        break;
      case DW_TAG_enumeration_type:
        add_type = (type_mask & eTypeClassEnumeration) != 0;
        break;
      case DW_TAG_subroutine_type:
      case DW_TAG_subprogram:
      case DW_TAG_inlined_subroutine:
        add_type = (type_mask & eTypeClassFunction) != 0;
        break;
      case DW_TAG_pointer_type:
        add_type = (type_mask & eTypeClassPointer) != 0;
        break;
      case DW_TAG_rvalue_reference_type:
      case DW_TAG_reference_type:
        add_type = (type_mask & eTypeClassReference) != 0;
        break;
      case DW_TAG_typedef:
        add_type = (type_mask & eTypeClassTypedef) != 0;
        break;
      case DW_TAG_ptr_to_member_type:
        add_type = (type_mask & eTypeClassMemberPointer) != 0;
        break;
      default:
        break;
      }

      if (add_type) {
        const bool assert_not_being_parsed = true;
        Type *type = ResolveTypeUID(die, assert_not_being_parsed);
        if (type)
          type_set.insert(type);
      }
    }

    for (DWARFDIE child_die : die.children()) {
      GetTypes(child_die, min_die_offset, max_die_offset, type_mask, type_set);
    }
  }
}

void SymbolFileDWARF::GetTypes(SymbolContextScope *sc_scope,
                               TypeClass type_mask, TypeList &type_list)

{
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  TypeSet type_set;

  CompileUnit *comp_unit = nullptr;
  if (sc_scope)
    comp_unit = sc_scope->CalculateSymbolContextCompileUnit();

  const auto &get = [&](DWARFUnit *unit) {
    if (!unit)
      return;
    unit = &unit->GetNonSkeletonUnit();
    GetTypes(unit->DIE(), unit->GetOffset(), unit->GetNextUnitOffset(),
             type_mask, type_set);
  };
  if (comp_unit) {
    get(GetDWARFCompileUnit(comp_unit));
  } else {
    DWARFDebugInfo &info = DebugInfo();
    const size_t num_cus = info.GetNumUnits();
    for (size_t cu_idx = 0; cu_idx < num_cus; ++cu_idx)
      get(info.GetUnitAtIndex(cu_idx));
  }

  std::set<CompilerType> compiler_type_set;
  for (Type *type : type_set) {
    CompilerType compiler_type = type->GetForwardCompilerType();
    if (compiler_type_set.find(compiler_type) == compiler_type_set.end()) {
      compiler_type_set.insert(compiler_type);
      type_list.Insert(type->shared_from_this());
    }
  }
}

// Gets the first parent that is a lexical block, function or inlined
// subroutine, or compile unit.
DWARFDIE
SymbolFileDWARF::GetParentSymbolContextDIE(const DWARFDIE &child_die) {
  DWARFDIE die;
  for (die = child_die.GetParent(); die; die = die.GetParent()) {
    dw_tag_t tag = die.Tag();

    switch (tag) {
    case DW_TAG_compile_unit:
    case DW_TAG_partial_unit:
    case DW_TAG_subprogram:
    case DW_TAG_inlined_subroutine:
    case DW_TAG_lexical_block:
      return die;
    default:
      break;
    }
  }
  return DWARFDIE();
}

SymbolFileDWARF::SymbolFileDWARF(ObjectFileSP objfile_sp,
                                 SectionList *dwo_section_list)
    : SymbolFileCommon(std::move(objfile_sp)),
      UserID(0x7fffffff00000000), // Used by SymbolFileDWARFDebugMap to
                                  // when this class parses .o files to
                                  // contain the .o file index/ID
      m_debug_map_module_wp(), m_debug_map_symfile(nullptr),
      m_context(m_objfile_sp->GetModule()->GetSectionList(), dwo_section_list),
      m_fetched_external_modules(false),
      m_supports_DW_AT_APPLE_objc_complete_type(eLazyBoolCalculate) {}

SymbolFileDWARF::~SymbolFileDWARF() = default;

static ConstString GetDWARFMachOSegmentName() {
  static ConstString g_dwarf_section_name("__DWARF");
  return g_dwarf_section_name;
}

UniqueDWARFASTTypeMap &SymbolFileDWARF::GetUniqueDWARFASTTypeMap() {
  SymbolFileDWARFDebugMap *debug_map_symfile = GetDebugMapSymfile();
  if (debug_map_symfile)
    return debug_map_symfile->GetUniqueDWARFASTTypeMap();
  else
    return m_unique_ast_type_map;
}

llvm::Expected<TypeSystem &>
SymbolFileDWARF::GetTypeSystemForLanguage(LanguageType language) {
  if (SymbolFileDWARFDebugMap *debug_map_symfile = GetDebugMapSymfile())
    return debug_map_symfile->GetTypeSystemForLanguage(language);

  auto type_system_or_err =
      m_objfile_sp->GetModule()->GetTypeSystemForLanguage(language);
  if (type_system_or_err) {
    type_system_or_err->SetSymbolFile(this);
  }
  return type_system_or_err;
}

void SymbolFileDWARF::InitializeObject() {
  Log *log = GetLog(DWARFLog::DebugInfo);

  InitializeFirstCodeAddress();

  if (!GetGlobalPluginProperties().IgnoreFileIndexes()) {
    StreamString module_desc;
    GetObjectFile()->GetModule()->GetDescription(module_desc.AsRawOstream(),
                                                 lldb::eDescriptionLevelBrief);
    DWARFDataExtractor apple_names, apple_namespaces, apple_types, apple_objc;
    LoadSectionData(eSectionTypeDWARFAppleNames, apple_names);
    LoadSectionData(eSectionTypeDWARFAppleNamespaces, apple_namespaces);
    LoadSectionData(eSectionTypeDWARFAppleTypes, apple_types);
    LoadSectionData(eSectionTypeDWARFAppleObjC, apple_objc);

    if (apple_names.GetByteSize() > 0 || apple_namespaces.GetByteSize() > 0 ||
        apple_types.GetByteSize() > 0 || apple_objc.GetByteSize() > 0) {
      Progress progress(llvm::formatv("Loading Apple DWARF index for {0}",
                                      module_desc.GetData()));
      m_index = AppleDWARFIndex::Create(
          *GetObjectFile()->GetModule(), apple_names, apple_namespaces,
          apple_types, apple_objc, m_context.getOrLoadStrData());

      if (m_index)
        return;
    }

    DWARFDataExtractor debug_names;
    LoadSectionData(eSectionTypeDWARFDebugNames, debug_names);
    if (debug_names.GetByteSize() > 0) {
      Progress progress(
          llvm::formatv("Loading DWARF5 index for {0}", module_desc.GetData()));
      llvm::Expected<std::unique_ptr<DebugNamesDWARFIndex>> index_or =
          DebugNamesDWARFIndex::Create(*GetObjectFile()->GetModule(),
                                       debug_names,
                                       m_context.getOrLoadStrData(), *this);
      if (index_or) {
        m_index = std::move(*index_or);
        return;
      }
      LLDB_LOG_ERROR(log, index_or.takeError(),
                     "Unable to read .debug_names data: {0}");
    }
  }

  m_index =
      std::make_unique<ManualDWARFIndex>(*GetObjectFile()->GetModule(), *this);
}

void SymbolFileDWARF::InitializeFirstCodeAddress() {
  InitializeFirstCodeAddressRecursive(
      *m_objfile_sp->GetModule()->GetSectionList());
  if (m_first_code_address == LLDB_INVALID_ADDRESS)
    m_first_code_address = 0;
}

void SymbolFileDWARF::InitializeFirstCodeAddressRecursive(
    const lldb_private::SectionList &section_list) {
  for (SectionSP section_sp : section_list) {
    if (section_sp->GetChildren().GetSize() > 0) {
      InitializeFirstCodeAddressRecursive(section_sp->GetChildren());
    } else if (section_sp->GetType() == eSectionTypeCode) {
      m_first_code_address =
          std::min(m_first_code_address, section_sp->GetFileAddress());
    }
  }
}

bool SymbolFileDWARF::SupportedVersion(uint16_t version) {
  return version >= 2 && version <= 5;
}

uint32_t SymbolFileDWARF::CalculateAbilities() {
  uint32_t abilities = 0;
  if (m_objfile_sp != nullptr) {
    const Section *section = nullptr;
    const SectionList *section_list = m_objfile_sp->GetSectionList();
    if (section_list == nullptr)
      return 0;

    uint64_t debug_abbrev_file_size = 0;
    uint64_t debug_info_file_size = 0;
    uint64_t debug_line_file_size = 0;

    section = section_list->FindSectionByName(GetDWARFMachOSegmentName()).get();

    if (section)
      section_list = &section->GetChildren();

    section =
        section_list->FindSectionByType(eSectionTypeDWARFDebugInfo, true).get();
    if (section != nullptr) {
      debug_info_file_size = section->GetFileSize();

      section =
          section_list->FindSectionByType(eSectionTypeDWARFDebugAbbrev, true)
              .get();
      if (section)
        debug_abbrev_file_size = section->GetFileSize();

      DWARFDebugAbbrev *abbrev = DebugAbbrev();
      if (abbrev) {
        std::set<dw_form_t> invalid_forms;
        abbrev->GetUnsupportedForms(invalid_forms);
        if (!invalid_forms.empty()) {
          StreamString error;
          error.Printf("unsupported DW_FORM value%s:",
                       invalid_forms.size() > 1 ? "s" : "");
          for (auto form : invalid_forms)
            error.Printf(" %#x", form);
          m_objfile_sp->GetModule()->ReportWarning(
              "%s", error.GetString().str().c_str());
          return 0;
        }
      }

      section =
          section_list->FindSectionByType(eSectionTypeDWARFDebugLine, true)
              .get();
      if (section)
        debug_line_file_size = section->GetFileSize();
    } else {
      llvm::StringRef symfile_dir =
          m_objfile_sp->GetFileSpec().GetDirectory().GetStringRef();
      if (symfile_dir.contains_insensitive(".dsym")) {
        if (m_objfile_sp->GetType() == ObjectFile::eTypeDebugInfo) {
          // We have a dSYM file that didn't have a any debug info. If the
          // string table has a size of 1, then it was made from an
          // executable with no debug info, or from an executable that was
          // stripped.
          section =
              section_list->FindSectionByType(eSectionTypeDWARFDebugStr, true)
                  .get();
          if (section && section->GetFileSize() == 1) {
            m_objfile_sp->GetModule()->ReportWarning(
                "empty dSYM file detected, dSYM was created with an "
                "executable with no debug info.");
          }
        }
      }
    }

    if (debug_abbrev_file_size > 0 && debug_info_file_size > 0)
      abilities |= CompileUnits | Functions | Blocks | GlobalVariables |
                   LocalVariables | VariableTypes;

    if (debug_line_file_size > 0)
      abilities |= LineTables;
  }
  return abilities;
}

void SymbolFileDWARF::LoadSectionData(lldb::SectionType sect_type,
                                      DWARFDataExtractor &data) {
  ModuleSP module_sp(m_objfile_sp->GetModule());
  const SectionList *section_list = module_sp->GetSectionList();
  if (!section_list)
    return;

  SectionSP section_sp(section_list->FindSectionByType(sect_type, true));
  if (!section_sp)
    return;

  data.Clear();
  m_objfile_sp->ReadSectionData(section_sp.get(), data);
}

DWARFDebugAbbrev *SymbolFileDWARF::DebugAbbrev() {
  if (m_abbr)
    return m_abbr.get();

  const DWARFDataExtractor &debug_abbrev_data = m_context.getOrLoadAbbrevData();
  if (debug_abbrev_data.GetByteSize() == 0)
    return nullptr;

  auto abbr = std::make_unique<DWARFDebugAbbrev>();
  llvm::Error error = abbr->parse(debug_abbrev_data);
  if (error) {
    Log *log = GetLog(DWARFLog::DebugInfo);
    LLDB_LOG_ERROR(log, std::move(error),
                   "Unable to read .debug_abbrev section: {0}");
    return nullptr;
  }

  m_abbr = std::move(abbr);
  return m_abbr.get();
}

DWARFDebugInfo &SymbolFileDWARF::DebugInfo() {
  llvm::call_once(m_info_once_flag, [&] {
    LLDB_SCOPED_TIMERF("%s this = %p", LLVM_PRETTY_FUNCTION,
                       static_cast<void *>(this));
    m_info = std::make_unique<DWARFDebugInfo>(*this, m_context);
  });
  return *m_info;
}

DWARFCompileUnit *SymbolFileDWARF::GetDWARFCompileUnit(CompileUnit *comp_unit) {
  if (!comp_unit)
    return nullptr;

  // The compile unit ID is the index of the DWARF unit.
  DWARFUnit *dwarf_cu = DebugInfo().GetUnitAtIndex(comp_unit->GetID());
  if (dwarf_cu && dwarf_cu->GetUserData() == nullptr)
    dwarf_cu->SetUserData(comp_unit);

  // It must be DWARFCompileUnit when it created a CompileUnit.
  return llvm::cast_or_null<DWARFCompileUnit>(dwarf_cu);
}

DWARFDebugRanges *SymbolFileDWARF::GetDebugRanges() {
  if (!m_ranges) {
    LLDB_SCOPED_TIMERF("%s this = %p", LLVM_PRETTY_FUNCTION,
                       static_cast<void *>(this));

    if (m_context.getOrLoadRangesData().GetByteSize() > 0)
      m_ranges = std::make_unique<DWARFDebugRanges>();

    if (m_ranges)
      m_ranges->Extract(m_context);
  }
  return m_ranges.get();
}

/// Make an absolute path out of \p file_spec and remap it using the
/// module's source remapping dictionary.
static void MakeAbsoluteAndRemap(FileSpec &file_spec, DWARFUnit &dwarf_cu,
                                 const ModuleSP &module_sp) {
  if (!file_spec)
    return;
  // If we have a full path to the compile unit, we don't need to
  // resolve the file.  This can be expensive e.g. when the source
  // files are NFS mounted.
  file_spec.MakeAbsolute(dwarf_cu.GetCompilationDirectory());

  if (auto remapped_file = module_sp->RemapSourceFile(file_spec.GetPath()))
    file_spec.SetFile(*remapped_file, FileSpec::Style::native);
}

/// Return the DW_AT_(GNU_)dwo_name.
static const char *GetDWOName(DWARFCompileUnit &dwarf_cu,
                              const DWARFDebugInfoEntry &cu_die) {
  const char *dwo_name =
      cu_die.GetAttributeValueAsString(&dwarf_cu, DW_AT_GNU_dwo_name, nullptr);
  if (!dwo_name)
    dwo_name =
        cu_die.GetAttributeValueAsString(&dwarf_cu, DW_AT_dwo_name, nullptr);
  return dwo_name;
}

lldb::CompUnitSP SymbolFileDWARF::ParseCompileUnit(DWARFCompileUnit &dwarf_cu) {
  CompUnitSP cu_sp;
  CompileUnit *comp_unit = (CompileUnit *)dwarf_cu.GetUserData();
  if (comp_unit) {
    // We already parsed this compile unit, had out a shared pointer to it
    cu_sp = comp_unit->shared_from_this();
  } else {
    if (dwarf_cu.GetOffset() == 0 && GetDebugMapSymfile()) {
      // Let the debug map create the compile unit
      cu_sp = m_debug_map_symfile->GetCompileUnit(this);
      dwarf_cu.SetUserData(cu_sp.get());
    } else {
      ModuleSP module_sp(m_objfile_sp->GetModule());
      if (module_sp) {
        auto initialize_cu = [&](const FileSpec &file_spec,
                                 LanguageType cu_language) {
          BuildCuTranslationTable();
          cu_sp = std::make_shared<CompileUnit>(
              module_sp, &dwarf_cu, file_spec,
              *GetDWARFUnitIndex(dwarf_cu.GetID()), cu_language,
              eLazyBoolCalculate);

          dwarf_cu.SetUserData(cu_sp.get());

          SetCompileUnitAtIndex(dwarf_cu.GetID(), cu_sp);
        };

        auto lazy_initialize_cu = [&]() {
          // If the version is < 5, we can't do lazy initialization.
          if (dwarf_cu.GetVersion() < 5)
            return false;

          // If there is no DWO, there is no reason to initialize
          // lazily; we will do eager initialization in that case.
          if (GetDebugMapSymfile())
            return false;
          const DWARFBaseDIE cu_die = dwarf_cu.GetUnitDIEOnly();
          if (!cu_die)
            return false;
          if (!GetDWOName(dwarf_cu, *cu_die.GetDIE()))
            return false;

          // With DWARFv5 we can assume that the first support
          // file is also the name of the compile unit. This
          // allows us to avoid loading the non-skeleton unit,
          // which may be in a separate DWO file.
          FileSpecList support_files;
          if (!ParseSupportFiles(dwarf_cu, module_sp, support_files))
            return false;
          if (support_files.GetSize() == 0)
            return false;

          initialize_cu(support_files.GetFileSpecAtIndex(0),
                        eLanguageTypeUnknown);
          cu_sp->SetSupportFiles(std::move(support_files));
          return true;
        };

        if (!lazy_initialize_cu()) {
          // Eagerly initialize compile unit
          const DWARFBaseDIE cu_die =
              dwarf_cu.GetNonSkeletonUnit().GetUnitDIEOnly();
          if (cu_die) {
            LanguageType cu_language = SymbolFileDWARF::LanguageTypeFromDWARF(
                dwarf_cu.GetDWARFLanguageType());

            FileSpec cu_file_spec(cu_die.GetName(), dwarf_cu.GetPathStyle());

            // Path needs to be remapped in this case. In the support files
            // case ParseSupportFiles takes care of the remapping.
            MakeAbsoluteAndRemap(cu_file_spec, dwarf_cu, module_sp);

            initialize_cu(cu_file_spec, cu_language);
          }
        }
      }
    }
  }
  return cu_sp;
}

void SymbolFileDWARF::BuildCuTranslationTable() {
  if (!m_lldb_cu_to_dwarf_unit.empty())
    return;

  DWARFDebugInfo &info = DebugInfo();
  if (!info.ContainsTypeUnits()) {
    // We can use a 1-to-1 mapping. No need to build a translation table.
    return;
  }
  for (uint32_t i = 0, num = info.GetNumUnits(); i < num; ++i) {
    if (auto *cu = llvm::dyn_cast<DWARFCompileUnit>(info.GetUnitAtIndex(i))) {
      cu->SetID(m_lldb_cu_to_dwarf_unit.size());
      m_lldb_cu_to_dwarf_unit.push_back(i);
    }
  }
}

llvm::Optional<uint32_t> SymbolFileDWARF::GetDWARFUnitIndex(uint32_t cu_idx) {
  BuildCuTranslationTable();
  if (m_lldb_cu_to_dwarf_unit.empty())
    return cu_idx;
  if (cu_idx >= m_lldb_cu_to_dwarf_unit.size())
    return llvm::None;
  return m_lldb_cu_to_dwarf_unit[cu_idx];
}

uint32_t SymbolFileDWARF::CalculateNumCompileUnits() {
  BuildCuTranslationTable();
  return m_lldb_cu_to_dwarf_unit.empty() ? DebugInfo().GetNumUnits()
                                         : m_lldb_cu_to_dwarf_unit.size();
}

CompUnitSP SymbolFileDWARF::ParseCompileUnitAtIndex(uint32_t cu_idx) {
  ASSERT_MODULE_LOCK(this);
  if (llvm::Optional<uint32_t> dwarf_idx = GetDWARFUnitIndex(cu_idx)) {
    if (auto *dwarf_cu = llvm::cast_or_null<DWARFCompileUnit>(
            DebugInfo().GetUnitAtIndex(*dwarf_idx)))
      return ParseCompileUnit(*dwarf_cu);
  }
  return {};
}

Function *SymbolFileDWARF::ParseFunction(CompileUnit &comp_unit,
                                         const DWARFDIE &die) {
  ASSERT_MODULE_LOCK(this);
  if (!die.IsValid())
    return nullptr;

  auto type_system_or_err = GetTypeSystemForLanguage(GetLanguage(*die.GetCU()));
  if (auto err = type_system_or_err.takeError()) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Symbols), std::move(err),
                   "Unable to parse function");
    return nullptr;
  }
  DWARFASTParser *dwarf_ast = type_system_or_err->GetDWARFParser();
  if (!dwarf_ast)
    return nullptr;

  DWARFRangeList ranges;
  if (die.GetDIE()->GetAttributeAddressRanges(die.GetCU(), ranges,
                                              /*check_hi_lo_pc=*/true) == 0)
    return nullptr;

  // Union of all ranges in the function DIE (if the function is
  // discontiguous)
  lldb::addr_t lowest_func_addr = ranges.GetMinRangeBase(0);
  lldb::addr_t highest_func_addr = ranges.GetMaxRangeEnd(0);
  if (lowest_func_addr == LLDB_INVALID_ADDRESS ||
      lowest_func_addr >= highest_func_addr ||
      lowest_func_addr < m_first_code_address)
    return nullptr;

  ModuleSP module_sp(die.GetModule());
  AddressRange func_range;
  func_range.GetBaseAddress().ResolveAddressUsingFileSections(
      lowest_func_addr, module_sp->GetSectionList());
  if (!func_range.GetBaseAddress().IsValid())
    return nullptr;

  func_range.SetByteSize(highest_func_addr - lowest_func_addr);
  if (!FixupAddress(func_range.GetBaseAddress()))
    return nullptr;

  return dwarf_ast->ParseFunctionFromDWARF(comp_unit, die, func_range);
}

ConstString
SymbolFileDWARF::ConstructFunctionDemangledName(const DWARFDIE &die) {
  ASSERT_MODULE_LOCK(this);
  if (!die.IsValid()) {
    return ConstString();
  }

  auto type_system_or_err = GetTypeSystemForLanguage(GetLanguage(*die.GetCU()));
  if (auto err = type_system_or_err.takeError()) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Symbols), std::move(err),
                   "Unable to construct demangled name for function");
    return ConstString();
  }

  DWARFASTParser *dwarf_ast = type_system_or_err->GetDWARFParser();
  if (!dwarf_ast)
    return ConstString();

  return dwarf_ast->ConstructDemangledNameFromDWARF(die);
}

lldb::addr_t SymbolFileDWARF::FixupAddress(lldb::addr_t file_addr) {
  SymbolFileDWARFDebugMap *debug_map_symfile = GetDebugMapSymfile();
  if (debug_map_symfile)
    return debug_map_symfile->LinkOSOFileAddress(this, file_addr);
  return file_addr;
}

bool SymbolFileDWARF::FixupAddress(Address &addr) {
  SymbolFileDWARFDebugMap *debug_map_symfile = GetDebugMapSymfile();
  if (debug_map_symfile) {
    return debug_map_symfile->LinkOSOAddress(addr);
  }
  // This is a normal DWARF file, no address fixups need to happen
  return true;
}
lldb::LanguageType SymbolFileDWARF::ParseLanguage(CompileUnit &comp_unit) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  DWARFUnit *dwarf_cu = GetDWARFCompileUnit(&comp_unit);
  if (dwarf_cu)
    return GetLanguage(dwarf_cu->GetNonSkeletonUnit());
  else
    return eLanguageTypeUnknown;
}

XcodeSDK SymbolFileDWARF::ParseXcodeSDK(CompileUnit &comp_unit) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  DWARFUnit *dwarf_cu = GetDWARFCompileUnit(&comp_unit);
  if (!dwarf_cu)
    return {};
  const DWARFBaseDIE cu_die = dwarf_cu->GetNonSkeletonUnit().GetUnitDIEOnly();
  if (!cu_die)
    return {};
  const char *sdk = cu_die.GetAttributeValueAsString(DW_AT_APPLE_sdk, nullptr);
  if (!sdk)
    return {};
  const char *sysroot =
      cu_die.GetAttributeValueAsString(DW_AT_LLVM_sysroot, "");
  // Register the sysroot path remapping with the module belonging to
  // the CU as well as the one belonging to the symbol file. The two
  // would be different if this is an OSO object and module is the
  // corresponding debug map, in which case both should be updated.
  ModuleSP module_sp = comp_unit.GetModule();
  if (module_sp)
    module_sp->RegisterXcodeSDK(sdk, sysroot);

  ModuleSP local_module_sp = m_objfile_sp->GetModule();
  if (local_module_sp && local_module_sp != module_sp)
    local_module_sp->RegisterXcodeSDK(sdk, sysroot);

  return {sdk};
}

size_t SymbolFileDWARF::ParseFunctions(CompileUnit &comp_unit) {
  LLDB_SCOPED_TIMER();
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  DWARFUnit *dwarf_cu = GetDWARFCompileUnit(&comp_unit);
  if (!dwarf_cu)
    return 0;

  size_t functions_added = 0;
  dwarf_cu = &dwarf_cu->GetNonSkeletonUnit();
  for (DWARFDebugInfoEntry &entry : dwarf_cu->dies()) {
    if (entry.Tag() != DW_TAG_subprogram)
      continue;

    DWARFDIE die(dwarf_cu, &entry);
    if (comp_unit.FindFunctionByUID(die.GetID()))
      continue;
    if (ParseFunction(comp_unit, die))
      ++functions_added;
  }
  // FixupTypes();
  return functions_added;
}

bool SymbolFileDWARF::ForEachExternalModule(
    CompileUnit &comp_unit,
    llvm::DenseSet<lldb_private::SymbolFile *> &visited_symbol_files,
    llvm::function_ref<bool(Module &)> lambda) {
  // Only visit each symbol file once.
  if (!visited_symbol_files.insert(this).second)
    return false;

  UpdateExternalModuleListIfNeeded();
  for (auto &p : m_external_type_modules) {
    ModuleSP module = p.second;
    if (!module)
      continue;

    // Invoke the action and potentially early-exit.
    if (lambda(*module))
      return true;

    for (std::size_t i = 0; i < module->GetNumCompileUnits(); ++i) {
      auto cu = module->GetCompileUnitAtIndex(i);
      bool early_exit = cu->ForEachExternalModule(visited_symbol_files, lambda);
      if (early_exit)
        return true;
    }
  }
  return false;
}

bool SymbolFileDWARF::ParseSupportFiles(CompileUnit &comp_unit,
                                        FileSpecList &support_files) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  DWARFUnit *dwarf_cu = GetDWARFCompileUnit(&comp_unit);
  if (!dwarf_cu)
    return false;

  if (!ParseSupportFiles(*dwarf_cu, comp_unit.GetModule(), support_files))
    return false;

  comp_unit.SetSupportFiles(support_files);
  return true;
}

bool SymbolFileDWARF::ParseSupportFiles(DWARFUnit &dwarf_cu,
                                        const ModuleSP &module,
                                        FileSpecList &support_files) {

  dw_offset_t offset = dwarf_cu.GetLineTableOffset();
  if (offset == DW_INVALID_OFFSET)
    return false;

  ElapsedTime elapsed(m_parse_time);
  llvm::DWARFDebugLine::Prologue prologue;
  if (!ParseLLVMLineTablePrologue(m_context, prologue, offset,
                                  dwarf_cu.GetOffset()))
    return false;

  std::string comp_dir = dwarf_cu.GetCompilationDirectory().GetPath();
  support_files = ParseSupportFilesFromPrologue(
      module, prologue, dwarf_cu.GetPathStyle(), comp_dir);
  return true;
}

FileSpec SymbolFileDWARF::GetFile(DWARFUnit &unit, size_t file_idx) {
  if (auto *dwarf_cu = llvm::dyn_cast<DWARFCompileUnit>(&unit)) {
    if (CompileUnit *lldb_cu = GetCompUnitForDWARFCompUnit(*dwarf_cu))
      return lldb_cu->GetSupportFiles().GetFileSpecAtIndex(file_idx);
    return FileSpec();
  }

  auto &tu = llvm::cast<DWARFTypeUnit>(unit);
  return GetTypeUnitSupportFiles(tu).GetFileSpecAtIndex(file_idx);
}

const FileSpecList &
SymbolFileDWARF::GetTypeUnitSupportFiles(DWARFTypeUnit &tu) {
  static FileSpecList empty_list;

  dw_offset_t offset = tu.GetLineTableOffset();
  if (offset == DW_INVALID_OFFSET ||
      offset == llvm::DenseMapInfo<dw_offset_t>::getEmptyKey() ||
      offset == llvm::DenseMapInfo<dw_offset_t>::getTombstoneKey())
    return empty_list;

  // Many type units can share a line table, so parse the support file list
  // once, and cache it based on the offset field.
  auto iter_bool = m_type_unit_support_files.try_emplace(offset);
  FileSpecList &list = iter_bool.first->second;
  if (iter_bool.second) {
    uint64_t line_table_offset = offset;
    llvm::DWARFDataExtractor data = m_context.getOrLoadLineData().GetAsLLVM();
    llvm::DWARFContext &ctx = m_context.GetAsLLVM();
    llvm::DWARFDebugLine::Prologue prologue;
    auto report = [](llvm::Error error) {
      Log *log = GetLog(DWARFLog::DebugInfo);
      LLDB_LOG_ERROR(log, std::move(error),
                     "SymbolFileDWARF::GetTypeUnitSupportFiles failed to parse "
                     "the line table prologue");
    };
    ElapsedTime elapsed(m_parse_time);
    llvm::Error error = prologue.parse(data, &line_table_offset, report, ctx);
    if (error) {
      report(std::move(error));
    } else {
      list = ParseSupportFilesFromPrologue(GetObjectFile()->GetModule(),
                                           prologue, tu.GetPathStyle());
    }
  }
  return list;
}

bool SymbolFileDWARF::ParseIsOptimized(CompileUnit &comp_unit) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  DWARFUnit *dwarf_cu = GetDWARFCompileUnit(&comp_unit);
  if (dwarf_cu)
    return dwarf_cu->GetNonSkeletonUnit().GetIsOptimized();
  return false;
}

bool SymbolFileDWARF::ParseImportedModules(
    const lldb_private::SymbolContext &sc,
    std::vector<SourceModule> &imported_modules) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  assert(sc.comp_unit);
  DWARFUnit *dwarf_cu = GetDWARFCompileUnit(sc.comp_unit);
  if (!dwarf_cu)
    return false;
  if (!ClangModulesDeclVendor::LanguageSupportsClangModules(
          sc.comp_unit->GetLanguage()))
    return false;
  UpdateExternalModuleListIfNeeded();

  const DWARFDIE die = dwarf_cu->DIE();
  if (!die)
    return false;

  for (DWARFDIE child_die : die.children()) {
    if (child_die.Tag() != DW_TAG_imported_declaration)
      continue;

    DWARFDIE module_die = child_die.GetReferencedDIE(DW_AT_import);
    if (module_die.Tag() != DW_TAG_module)
      continue;

    if (const char *name =
            module_die.GetAttributeValueAsString(DW_AT_name, nullptr)) {
      SourceModule module;
      module.path.push_back(ConstString(name));

      DWARFDIE parent_die = module_die;
      while ((parent_die = parent_die.GetParent())) {
        if (parent_die.Tag() != DW_TAG_module)
          break;
        if (const char *name =
                parent_die.GetAttributeValueAsString(DW_AT_name, nullptr))
          module.path.push_back(ConstString(name));
      }
      std::reverse(module.path.begin(), module.path.end());
      if (const char *include_path = module_die.GetAttributeValueAsString(
              DW_AT_LLVM_include_path, nullptr)) {
        FileSpec include_spec(include_path, dwarf_cu->GetPathStyle());
        MakeAbsoluteAndRemap(include_spec, *dwarf_cu,
                             m_objfile_sp->GetModule());
        module.search_path = ConstString(include_spec.GetPath());
      }
      if (const char *sysroot = dwarf_cu->DIE().GetAttributeValueAsString(
              DW_AT_LLVM_sysroot, nullptr))
        module.sysroot = ConstString(sysroot);
      imported_modules.push_back(module);
    }
  }
  return true;
}

bool SymbolFileDWARF::ParseLineTable(CompileUnit &comp_unit) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  if (comp_unit.GetLineTable() != nullptr)
    return true;

  DWARFUnit *dwarf_cu = GetDWARFCompileUnit(&comp_unit);
  if (!dwarf_cu)
    return false;

  dw_offset_t offset = dwarf_cu->GetLineTableOffset();
  if (offset == DW_INVALID_OFFSET)
    return false;

  ElapsedTime elapsed(m_parse_time);
  llvm::DWARFDebugLine line;
  const llvm::DWARFDebugLine::LineTable *line_table =
      ParseLLVMLineTable(m_context, line, offset, dwarf_cu->GetOffset());

  if (!line_table)
    return false;

  // FIXME: Rather than parsing the whole line table and then copying it over
  // into LLDB, we should explore using a callback to populate the line table
  // while we parse to reduce memory usage.
  std::vector<std::unique_ptr<LineSequence>> sequences;
  // The Sequences view contains only valid line sequences. Don't iterate over
  // the Rows directly.
  for (const llvm::DWARFDebugLine::Sequence &seq : line_table->Sequences) {
    // Ignore line sequences that do not start after the first code address.
    // All addresses generated in a sequence are incremental so we only need
    // to check the first one of the sequence. Check the comment at the
    // m_first_code_address declaration for more details on this.
    if (seq.LowPC < m_first_code_address)
      continue;
    std::unique_ptr<LineSequence> sequence =
        LineTable::CreateLineSequenceContainer();
    for (unsigned idx = seq.FirstRowIndex; idx < seq.LastRowIndex; ++idx) {
      const llvm::DWARFDebugLine::Row &row = line_table->Rows[idx];
      LineTable::AppendLineEntryToSequence(
          sequence.get(), row.Address.Address, row.Line, row.Column, row.File,
          row.IsStmt, row.BasicBlock, row.PrologueEnd, row.EpilogueBegin,
          row.EndSequence);
    }
    sequences.push_back(std::move(sequence));
  }

  std::unique_ptr<LineTable> line_table_up =
      std::make_unique<LineTable>(&comp_unit, std::move(sequences));

  if (SymbolFileDWARFDebugMap *debug_map_symfile = GetDebugMapSymfile()) {
    // We have an object file that has a line table with addresses that are not
    // linked. We need to link the line table and convert the addresses that
    // are relative to the .o file into addresses for the main executable.
    comp_unit.SetLineTable(
        debug_map_symfile->LinkOSOLineTable(this, line_table_up.get()));
  } else {
    comp_unit.SetLineTable(line_table_up.release());
  }

  return true;
}

lldb_private::DebugMacrosSP
SymbolFileDWARF::ParseDebugMacros(lldb::offset_t *offset) {
  auto iter = m_debug_macros_map.find(*offset);
  if (iter != m_debug_macros_map.end())
    return iter->second;

  ElapsedTime elapsed(m_parse_time);
  const DWARFDataExtractor &debug_macro_data = m_context.getOrLoadMacroData();
  if (debug_macro_data.GetByteSize() == 0)
    return DebugMacrosSP();

  lldb_private::DebugMacrosSP debug_macros_sp(new lldb_private::DebugMacros());
  m_debug_macros_map[*offset] = debug_macros_sp;

  const DWARFDebugMacroHeader &header =
      DWARFDebugMacroHeader::ParseHeader(debug_macro_data, offset);
  DWARFDebugMacroEntry::ReadMacroEntries(
      debug_macro_data, m_context.getOrLoadStrData(), header.OffsetIs64Bit(),
      offset, this, debug_macros_sp);

  return debug_macros_sp;
}

bool SymbolFileDWARF::ParseDebugMacros(CompileUnit &comp_unit) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());

  DWARFUnit *dwarf_cu = GetDWARFCompileUnit(&comp_unit);
  if (dwarf_cu == nullptr)
    return false;

  const DWARFBaseDIE dwarf_cu_die = dwarf_cu->GetUnitDIEOnly();
  if (!dwarf_cu_die)
    return false;

  lldb::offset_t sect_offset =
      dwarf_cu_die.GetAttributeValueAsUnsigned(DW_AT_macros, DW_INVALID_OFFSET);
  if (sect_offset == DW_INVALID_OFFSET)
    sect_offset = dwarf_cu_die.GetAttributeValueAsUnsigned(DW_AT_GNU_macros,
                                                           DW_INVALID_OFFSET);
  if (sect_offset == DW_INVALID_OFFSET)
    return false;

  comp_unit.SetDebugMacros(ParseDebugMacros(&sect_offset));

  return true;
}

size_t SymbolFileDWARF::ParseBlocksRecursive(
    lldb_private::CompileUnit &comp_unit, Block *parent_block,
    const DWARFDIE &orig_die, addr_t subprogram_low_pc, uint32_t depth) {
  size_t blocks_added = 0;
  DWARFDIE die = orig_die;
  while (die) {
    dw_tag_t tag = die.Tag();

    switch (tag) {
    case DW_TAG_inlined_subroutine:
    case DW_TAG_subprogram:
    case DW_TAG_lexical_block: {
      Block *block = nullptr;
      if (tag == DW_TAG_subprogram) {
        // Skip any DW_TAG_subprogram DIEs that are inside of a normal or
        // inlined functions. These will be parsed on their own as separate
        // entities.

        if (depth > 0)
          break;

        block = parent_block;
      } else {
        BlockSP block_sp(new Block(die.GetID()));
        parent_block->AddChild(block_sp);
        block = block_sp.get();
      }
      DWARFRangeList ranges;
      const char *name = nullptr;
      const char *mangled_name = nullptr;

      int decl_file = 0;
      int decl_line = 0;
      int decl_column = 0;
      int call_file = 0;
      int call_line = 0;
      int call_column = 0;
      if (die.GetDIENamesAndRanges(name, mangled_name, ranges, decl_file,
                                   decl_line, decl_column, call_file, call_line,
                                   call_column, nullptr)) {
        if (tag == DW_TAG_subprogram) {
          assert(subprogram_low_pc == LLDB_INVALID_ADDRESS);
          subprogram_low_pc = ranges.GetMinRangeBase(0);
        } else if (tag == DW_TAG_inlined_subroutine) {
          // We get called here for inlined subroutines in two ways. The first
          // time is when we are making the Function object for this inlined
          // concrete instance.  Since we're creating a top level block at
          // here, the subprogram_low_pc will be LLDB_INVALID_ADDRESS.  So we
          // need to adjust the containing address. The second time is when we
          // are parsing the blocks inside the function that contains the
          // inlined concrete instance.  Since these will be blocks inside the
          // containing "real" function the offset will be for that function.
          if (subprogram_low_pc == LLDB_INVALID_ADDRESS) {
            subprogram_low_pc = ranges.GetMinRangeBase(0);
          }
        }

        const size_t num_ranges = ranges.GetSize();
        for (size_t i = 0; i < num_ranges; ++i) {
          const DWARFRangeList::Entry &range = ranges.GetEntryRef(i);
          const addr_t range_base = range.GetRangeBase();
          if (range_base >= subprogram_low_pc)
            block->AddRange(Block::Range(range_base - subprogram_low_pc,
                                         range.GetByteSize()));
          else {
            GetObjectFile()->GetModule()->ReportError(
                "0x%8.8" PRIx64 ": adding range [0x%" PRIx64 "-0x%" PRIx64
                ") which has a base that is less than the function's low PC "
                "0x%" PRIx64 ". Please file a bug and attach the file at the "
                "start of this error message",
                block->GetID(), range_base, range.GetRangeEnd(),
                subprogram_low_pc);
          }
        }
        block->FinalizeRanges();

        if (tag != DW_TAG_subprogram &&
            (name != nullptr || mangled_name != nullptr)) {
          std::unique_ptr<Declaration> decl_up;
          if (decl_file != 0 || decl_line != 0 || decl_column != 0)
            decl_up = std::make_unique<Declaration>(
                comp_unit.GetSupportFiles().GetFileSpecAtIndex(decl_file),
                decl_line, decl_column);

          std::unique_ptr<Declaration> call_up;
          if (call_file != 0 || call_line != 0 || call_column != 0)
            call_up = std::make_unique<Declaration>(
                comp_unit.GetSupportFiles().GetFileSpecAtIndex(call_file),
                call_line, call_column);

          block->SetInlinedFunctionInfo(name, mangled_name, decl_up.get(),
                                        call_up.get());
        }

        ++blocks_added;

        if (die.HasChildren()) {
          blocks_added +=
              ParseBlocksRecursive(comp_unit, block, die.GetFirstChild(),
                                   subprogram_low_pc, depth + 1);
        }
      }
    } break;
    default:
      break;
    }

    // Only parse siblings of the block if we are not at depth zero. A depth of
    // zero indicates we are currently parsing the top level DW_TAG_subprogram
    // DIE

    if (depth == 0)
      die.Clear();
    else
      die = die.GetSibling();
  }
  return blocks_added;
}

bool SymbolFileDWARF::ClassOrStructIsVirtual(const DWARFDIE &parent_die) {
  if (parent_die) {
    for (DWARFDIE die : parent_die.children()) {
      dw_tag_t tag = die.Tag();
      bool check_virtuality = false;
      switch (tag) {
      case DW_TAG_inheritance:
      case DW_TAG_subprogram:
        check_virtuality = true;
        break;
      default:
        break;
      }
      if (check_virtuality) {
        if (die.GetAttributeValueAsUnsigned(DW_AT_virtuality, 0) != 0)
          return true;
      }
    }
  }
  return false;
}

void SymbolFileDWARF::ParseDeclsForContext(CompilerDeclContext decl_ctx) {
  auto *type_system = decl_ctx.GetTypeSystem();
  if (type_system != nullptr)
    type_system->GetDWARFParser()->EnsureAllDIEsInDeclContextHaveBeenParsed(
        decl_ctx);
}

user_id_t SymbolFileDWARF::GetUID(DIERef ref) {
  if (GetDebugMapSymfile())
    return GetID() | ref.die_offset();

  lldbassert(GetDwoNum().value_or(0) <= 0x3fffffff);
  return user_id_t(GetDwoNum().value_or(0)) << 32 | ref.die_offset() |
         lldb::user_id_t(GetDwoNum().has_value()) << 62 |
         lldb::user_id_t(ref.section() == DIERef::Section::DebugTypes) << 63;
}

llvm::Optional<SymbolFileDWARF::DecodedUID>
SymbolFileDWARF::DecodeUID(lldb::user_id_t uid) {
  // This method can be called without going through the symbol vendor so we
  // need to lock the module.
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  // Anytime we get a "lldb::user_id_t" from an lldb_private::SymbolFile API we
  // must make sure we use the correct DWARF file when resolving things. On
  // MacOSX, when using SymbolFileDWARFDebugMap, we will use multiple
  // SymbolFileDWARF classes, one for each .o file. We can often end up with
  // references to other DWARF objects and we must be ready to receive a
  // "lldb::user_id_t" that specifies a DIE from another SymbolFileDWARF
  // instance.
  if (SymbolFileDWARFDebugMap *debug_map = GetDebugMapSymfile()) {
    SymbolFileDWARF *dwarf = debug_map->GetSymbolFileByOSOIndex(
        debug_map->GetOSOIndexFromUserID(uid));
    return DecodedUID{
        *dwarf, {llvm::None, DIERef::Section::DebugInfo, dw_offset_t(uid)}};
  }
  dw_offset_t die_offset = uid;
  if (die_offset == DW_INVALID_OFFSET)
    return llvm::None;

  DIERef::Section section =
      uid >> 63 ? DIERef::Section::DebugTypes : DIERef::Section::DebugInfo;

  llvm::Optional<uint32_t> dwo_num;
  bool dwo_valid = uid >> 62 & 1;
  if (dwo_valid)
    dwo_num = uid >> 32 & 0x3fffffff;

  return DecodedUID{*this, {dwo_num, section, die_offset}};
}

DWARFDIE
SymbolFileDWARF::GetDIE(lldb::user_id_t uid) {
  // This method can be called without going through the symbol vendor so we
  // need to lock the module.
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());

  llvm::Optional<DecodedUID> decoded = DecodeUID(uid);

  if (decoded)
    return decoded->dwarf.GetDIE(decoded->ref);

  return DWARFDIE();
}

CompilerDecl SymbolFileDWARF::GetDeclForUID(lldb::user_id_t type_uid) {
  // This method can be called without going through the symbol vendor so we
  // need to lock the module.
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  // Anytime we have a lldb::user_id_t, we must get the DIE by calling
  // SymbolFileDWARF::GetDIE(). See comments inside the
  // SymbolFileDWARF::GetDIE() for details.
  if (DWARFDIE die = GetDIE(type_uid))
    return GetDecl(die);
  return CompilerDecl();
}

CompilerDeclContext
SymbolFileDWARF::GetDeclContextForUID(lldb::user_id_t type_uid) {
  // This method can be called without going through the symbol vendor so we
  // need to lock the module.
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  // Anytime we have a lldb::user_id_t, we must get the DIE by calling
  // SymbolFileDWARF::GetDIE(). See comments inside the
  // SymbolFileDWARF::GetDIE() for details.
  if (DWARFDIE die = GetDIE(type_uid))
    return GetDeclContext(die);
  return CompilerDeclContext();
}

CompilerDeclContext
SymbolFileDWARF::GetDeclContextContainingUID(lldb::user_id_t type_uid) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  // Anytime we have a lldb::user_id_t, we must get the DIE by calling
  // SymbolFileDWARF::GetDIE(). See comments inside the
  // SymbolFileDWARF::GetDIE() for details.
  if (DWARFDIE die = GetDIE(type_uid))
    return GetContainingDeclContext(die);
  return CompilerDeclContext();
}

Type *SymbolFileDWARF::ResolveTypeUID(lldb::user_id_t type_uid) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  // Anytime we have a lldb::user_id_t, we must get the DIE by calling
  // SymbolFileDWARF::GetDIE(). See comments inside the
  // SymbolFileDWARF::GetDIE() for details.
  if (DWARFDIE type_die = GetDIE(type_uid))
    return type_die.ResolveType();
  else
    return nullptr;
}

llvm::Optional<SymbolFile::ArrayInfo>
SymbolFileDWARF::GetDynamicArrayInfoForUID(
    lldb::user_id_t type_uid, const lldb_private::ExecutionContext *exe_ctx) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  if (DWARFDIE type_die = GetDIE(type_uid))
    return DWARFASTParser::ParseChildArrayInfo(type_die, exe_ctx);
  else
    return llvm::None;
}

Type *SymbolFileDWARF::ResolveTypeUID(const DIERef &die_ref) {
  return ResolveType(GetDIE(die_ref), true);
}

Type *SymbolFileDWARF::ResolveTypeUID(const DWARFDIE &die,
                                      bool assert_not_being_parsed) {
  if (die) {
    Log *log = GetLog(DWARFLog::DebugInfo);
    if (log)
      GetObjectFile()->GetModule()->LogMessage(
          log, "SymbolFileDWARF::ResolveTypeUID (die = 0x%8.8x) %s '%s'",
          die.GetOffset(), die.GetTagAsCString(), die.GetName());

    // We might be coming in in the middle of a type tree (a class within a
    // class, an enum within a class), so parse any needed parent DIEs before
    // we get to this one...
    DWARFDIE decl_ctx_die = GetDeclContextDIEContainingDIE(die);
    if (decl_ctx_die) {
      if (log) {
        switch (decl_ctx_die.Tag()) {
        case DW_TAG_structure_type:
        case DW_TAG_union_type:
        case DW_TAG_class_type: {
          // Get the type, which could be a forward declaration
          if (log)
            GetObjectFile()->GetModule()->LogMessage(
                log,
                "SymbolFileDWARF::ResolveTypeUID (die = 0x%8.8x) %s '%s' "
                "resolve parent forward type for 0x%8.8x",
                die.GetOffset(), die.GetTagAsCString(), die.GetName(),
                decl_ctx_die.GetOffset());
        } break;

        default:
          break;
        }
      }
    }
    return ResolveType(die);
  }
  return nullptr;
}

// This function is used when SymbolFileDWARFDebugMap owns a bunch of
// SymbolFileDWARF objects to detect if this DWARF file is the one that can
// resolve a compiler_type.
bool SymbolFileDWARF::HasForwardDeclForClangType(
    const CompilerType &compiler_type) {
  CompilerType compiler_type_no_qualifiers =
      ClangUtil::RemoveFastQualifiers(compiler_type);
  if (GetForwardDeclClangTypeToDie().count(
          compiler_type_no_qualifiers.GetOpaqueQualType())) {
    return true;
  }
  TypeSystem *type_system = compiler_type.GetTypeSystem();

  TypeSystemClang *clang_type_system =
      llvm::dyn_cast_or_null<TypeSystemClang>(type_system);
  if (!clang_type_system)
    return false;
  DWARFASTParserClang *ast_parser =
      static_cast<DWARFASTParserClang *>(clang_type_system->GetDWARFParser());
  return ast_parser->GetClangASTImporter().CanImport(compiler_type);
}

bool SymbolFileDWARF::CompleteType(CompilerType &compiler_type) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());

  TypeSystemClang *clang_type_system =
      llvm::dyn_cast_or_null<TypeSystemClang>(compiler_type.GetTypeSystem());
  if (clang_type_system) {
    DWARFASTParserClang *ast_parser =
        static_cast<DWARFASTParserClang *>(clang_type_system->GetDWARFParser());
    if (ast_parser &&
        ast_parser->GetClangASTImporter().CanImport(compiler_type))
      return ast_parser->GetClangASTImporter().CompleteType(compiler_type);
  }

  // We have a struct/union/class/enum that needs to be fully resolved.
  CompilerType compiler_type_no_qualifiers =
      ClangUtil::RemoveFastQualifiers(compiler_type);
  auto die_it = GetForwardDeclClangTypeToDie().find(
      compiler_type_no_qualifiers.GetOpaqueQualType());
  if (die_it == GetForwardDeclClangTypeToDie().end()) {
    // We have already resolved this type...
    return true;
  }

  DWARFDIE dwarf_die = GetDIE(die_it->getSecond());
  if (dwarf_die) {
    // Once we start resolving this type, remove it from the forward
    // declaration map in case anyone child members or other types require this
    // type to get resolved. The type will get resolved when all of the calls
    // to SymbolFileDWARF::ResolveClangOpaqueTypeDefinition are done.
    GetForwardDeclClangTypeToDie().erase(die_it);

    Type *type = GetDIEToType().lookup(dwarf_die.GetDIE());

    Log *log = GetLog(DWARFLog::DebugInfo | DWARFLog::TypeCompletion);
    if (log)
      GetObjectFile()->GetModule()->LogMessageVerboseBacktrace(
          log, "0x%8.8" PRIx64 ": %s '%s' resolving forward declaration...",
          dwarf_die.GetID(), dwarf_die.GetTagAsCString(),
          type->GetName().AsCString());
    assert(compiler_type);
    if (DWARFASTParser *dwarf_ast = GetDWARFParser(*dwarf_die.GetCU()))
      return dwarf_ast->CompleteTypeFromDWARF(dwarf_die, type, compiler_type);
  }
  return false;
}

Type *SymbolFileDWARF::ResolveType(const DWARFDIE &die,
                                   bool assert_not_being_parsed,
                                   bool resolve_function_context) {
  if (die) {
    Type *type = GetTypeForDIE(die, resolve_function_context).get();

    if (assert_not_being_parsed) {
      if (type != DIE_IS_BEING_PARSED)
        return type;

      GetObjectFile()->GetModule()->ReportError(
          "Parsing a die that is being parsed die: 0x%8.8x: %s %s",
          die.GetOffset(), die.GetTagAsCString(), die.GetName());

    } else
      return type;
  }
  return nullptr;
}

CompileUnit *
SymbolFileDWARF::GetCompUnitForDWARFCompUnit(DWARFCompileUnit &dwarf_cu) {
  if (dwarf_cu.IsDWOUnit()) {
    DWARFCompileUnit *non_dwo_cu =
        static_cast<DWARFCompileUnit *>(dwarf_cu.GetUserData());
    assert(non_dwo_cu);
    return non_dwo_cu->GetSymbolFileDWARF().GetCompUnitForDWARFCompUnit(
        *non_dwo_cu);
  }
  // Check if the symbol vendor already knows about this compile unit?
  if (dwarf_cu.GetUserData() == nullptr) {
    // The symbol vendor doesn't know about this compile unit, we need to parse
    // and add it to the symbol vendor object.
    return ParseCompileUnit(dwarf_cu).get();
  }
  return static_cast<CompileUnit *>(dwarf_cu.GetUserData());
}

void SymbolFileDWARF::GetObjCMethods(
    ConstString class_name, llvm::function_ref<bool(DWARFDIE die)> callback) {
  m_index->GetObjCMethods(class_name, callback);
}

bool SymbolFileDWARF::GetFunction(const DWARFDIE &die, SymbolContext &sc) {
  sc.Clear(false);

  if (die && llvm::isa<DWARFCompileUnit>(die.GetCU())) {
    // Check if the symbol vendor already knows about this compile unit?
    sc.comp_unit =
        GetCompUnitForDWARFCompUnit(llvm::cast<DWARFCompileUnit>(*die.GetCU()));

    sc.function = sc.comp_unit->FindFunctionByUID(die.GetID()).get();
    if (sc.function == nullptr)
      sc.function = ParseFunction(*sc.comp_unit, die);

    if (sc.function) {
      sc.module_sp = sc.function->CalculateSymbolContextModule();
      return true;
    }
  }

  return false;
}

lldb::ModuleSP SymbolFileDWARF::GetExternalModule(ConstString name) {
  UpdateExternalModuleListIfNeeded();
  const auto &pos = m_external_type_modules.find(name);
  if (pos != m_external_type_modules.end())
    return pos->second;
  else
    return lldb::ModuleSP();
}

DWARFDIE
SymbolFileDWARF::GetDIE(const DIERef &die_ref) {
  if (die_ref.dwo_num()) {
    SymbolFileDWARF *dwarf = *die_ref.dwo_num() == 0x3fffffff
                                 ? m_dwp_symfile.get()
                                 : this->DebugInfo()
                                       .GetUnitAtIndex(*die_ref.dwo_num())
                                       ->GetDwoSymbolFile();
    return dwarf->DebugInfo().GetDIE(die_ref);
  }

  return DebugInfo().GetDIE(die_ref);
}

/// Return the DW_AT_(GNU_)dwo_id.
static llvm::Optional<uint64_t> GetDWOId(DWARFCompileUnit &dwarf_cu,
                         const DWARFDebugInfoEntry &cu_die) {
  llvm::Optional<uint64_t> dwo_id =
      cu_die.GetAttributeValueAsOptionalUnsigned(&dwarf_cu, DW_AT_GNU_dwo_id);
  if (dwo_id)
    return dwo_id;
  return cu_die.GetAttributeValueAsOptionalUnsigned(&dwarf_cu, DW_AT_dwo_id);
}

llvm::Optional<uint64_t> SymbolFileDWARF::GetDWOId() {
  if (GetNumCompileUnits() == 1) {
    if (auto comp_unit = GetCompileUnitAtIndex(0))
      if (DWARFCompileUnit *cu = GetDWARFCompileUnit(comp_unit.get()))
        if (DWARFDebugInfoEntry *cu_die = cu->DIE().GetDIE())
          return ::GetDWOId(*cu, *cu_die);
  }
  return {};
}

std::shared_ptr<SymbolFileDWARFDwo>
SymbolFileDWARF::GetDwoSymbolFileForCompileUnit(
    DWARFUnit &unit, const DWARFDebugInfoEntry &cu_die) {
  // If this is a Darwin-style debug map (non-.dSYM) symbol file,
  // never attempt to load ELF-style DWO files since the -gmodules
  // support uses the same DWO mechanism to specify full debug info
  // files for modules. This is handled in
  // UpdateExternalModuleListIfNeeded().
  if (GetDebugMapSymfile())
    return nullptr;

  DWARFCompileUnit *dwarf_cu = llvm::dyn_cast<DWARFCompileUnit>(&unit);
  // Only compile units can be split into two parts.
  if (!dwarf_cu)
    return nullptr;

  const char *dwo_name = GetDWOName(*dwarf_cu, cu_die);
  if (!dwo_name)
    return nullptr;

  if (std::shared_ptr<SymbolFileDWARFDwo> dwp_sp = GetDwpSymbolFile())
    return dwp_sp;

  FileSpec dwo_file(dwo_name);
  FileSystem::Instance().Resolve(dwo_file);
  if (dwo_file.IsRelative()) {
    const char *comp_dir =
        cu_die.GetAttributeValueAsString(dwarf_cu, DW_AT_comp_dir, nullptr);
    if (!comp_dir)
      return nullptr;

    dwo_file.SetFile(comp_dir, FileSpec::Style::native);
    if (dwo_file.IsRelative()) {
      // if DW_AT_comp_dir is relative, it should be relative to the location
      // of the executable, not to the location from which the debugger was
      // launched.
      dwo_file.PrependPathComponent(
          m_objfile_sp->GetFileSpec().GetDirectory().GetStringRef());
    }
    FileSystem::Instance().Resolve(dwo_file);
    dwo_file.AppendPathComponent(dwo_name);
  }

  if (!FileSystem::Instance().Exists(dwo_file)) {
    if (m_dwo_warning_issued.test_and_set(std::memory_order_relaxed) == false) {
      GetObjectFile()->GetModule()->ReportWarning(
          "unable to locate separate debug file (dwo, dwp). Debugging will be "
          "degraded.");
    }
    return nullptr;
  }

  const lldb::offset_t file_offset = 0;
  DataBufferSP dwo_file_data_sp;
  lldb::offset_t dwo_file_data_offset = 0;
  ObjectFileSP dwo_obj_file = ObjectFile::FindPlugin(
      GetObjectFile()->GetModule(), &dwo_file, file_offset,
      FileSystem::Instance().GetByteSize(dwo_file), dwo_file_data_sp,
      dwo_file_data_offset);
  if (dwo_obj_file == nullptr)
    return nullptr;

  return std::make_shared<SymbolFileDWARFDwo>(*this, dwo_obj_file,
                                              dwarf_cu->GetID());
}

void SymbolFileDWARF::UpdateExternalModuleListIfNeeded() {
  if (m_fetched_external_modules)
    return;
  m_fetched_external_modules = true;
  DWARFDebugInfo &debug_info = DebugInfo();

  // Follow DWO skeleton unit breadcrumbs.
  const uint32_t num_compile_units = GetNumCompileUnits();
  for (uint32_t cu_idx = 0; cu_idx < num_compile_units; ++cu_idx) {
    auto *dwarf_cu =
        llvm::dyn_cast<DWARFCompileUnit>(debug_info.GetUnitAtIndex(cu_idx));
    if (!dwarf_cu)
      continue;

    const DWARFBaseDIE die = dwarf_cu->GetUnitDIEOnly();
    if (!die || die.HasChildren() || !die.GetDIE())
      continue;

    const char *name = die.GetAttributeValueAsString(DW_AT_name, nullptr);
    if (!name)
      continue;

    ConstString const_name(name);
    ModuleSP &module_sp = m_external_type_modules[const_name];
    if (module_sp)
      continue;

    const char *dwo_path = GetDWOName(*dwarf_cu, *die.GetDIE());
    if (!dwo_path)
      continue;

    ModuleSpec dwo_module_spec;
    dwo_module_spec.GetFileSpec().SetFile(dwo_path, FileSpec::Style::native);
    if (dwo_module_spec.GetFileSpec().IsRelative()) {
      const char *comp_dir =
          die.GetAttributeValueAsString(DW_AT_comp_dir, nullptr);
      if (comp_dir) {
        dwo_module_spec.GetFileSpec().SetFile(comp_dir,
                                              FileSpec::Style::native);
        FileSystem::Instance().Resolve(dwo_module_spec.GetFileSpec());
        dwo_module_spec.GetFileSpec().AppendPathComponent(dwo_path);
      }
    }
    dwo_module_spec.GetArchitecture() =
        m_objfile_sp->GetModule()->GetArchitecture();

    // When LLDB loads "external" modules it looks at the presence of
    // DW_AT_dwo_name. However, when the already created module
    // (corresponding to .dwo itself) is being processed, it will see
    // the presence of DW_AT_dwo_name (which contains the name of dwo
    // file) and will try to call ModuleList::GetSharedModule
    // again. In some cases (i.e., for empty files) Clang 4.0
    // generates a *.dwo file which has DW_AT_dwo_name, but no
    // DW_AT_comp_dir. In this case the method
    // ModuleList::GetSharedModule will fail and the warning will be
    // printed. However, as one can notice in this case we don't
    // actually need to try to load the already loaded module
    // (corresponding to .dwo) so we simply skip it.
    if (m_objfile_sp->GetFileSpec().GetFileNameExtension() == ".dwo" &&
        llvm::StringRef(m_objfile_sp->GetFileSpec().GetPath())
            .endswith(dwo_module_spec.GetFileSpec().GetPath())) {
      continue;
    }

    Status error = ModuleList::GetSharedModule(dwo_module_spec, module_sp,
                                               nullptr, nullptr, nullptr);
    if (!module_sp) {
      GetObjectFile()->GetModule()->ReportWarning(
          "0x%8.8x: unable to locate module needed for external types: "
          "%s\nerror: %s\nDebugging will be degraded due to missing "
          "types. Rebuilding the project will regenerate the needed "
          "module files.",
          die.GetOffset(), dwo_module_spec.GetFileSpec().GetPath().c_str(),
          error.AsCString("unknown error"));
      continue;
    }

    // Verify the DWO hash.
    // FIXME: Technically "0" is a valid hash.
    llvm::Optional<uint64_t> dwo_id = ::GetDWOId(*dwarf_cu, *die.GetDIE());
    if (!dwo_id)
      continue;

    auto *dwo_symfile =
        llvm::dyn_cast_or_null<SymbolFileDWARF>(module_sp->GetSymbolFile());
    if (!dwo_symfile)
      continue;
    llvm::Optional<uint64_t> dwo_dwo_id = dwo_symfile->GetDWOId();
    if (!dwo_dwo_id)
      continue;

    if (dwo_id != dwo_dwo_id) {
      GetObjectFile()->GetModule()->ReportWarning(
          "0x%8.8x: Module %s is out-of-date (hash mismatch). Type information "
          "from this module may be incomplete or inconsistent with the rest of "
          "the program. Rebuilding the project will regenerate the needed "
          "module files.",
          die.GetOffset(), dwo_module_spec.GetFileSpec().GetPath().c_str());
    }
  }
}

SymbolFileDWARF::GlobalVariableMap &SymbolFileDWARF::GetGlobalAranges() {
  if (!m_global_aranges_up) {
    m_global_aranges_up = std::make_unique<GlobalVariableMap>();

    ModuleSP module_sp = GetObjectFile()->GetModule();
    if (module_sp) {
      const size_t num_cus = module_sp->GetNumCompileUnits();
      for (size_t i = 0; i < num_cus; ++i) {
        CompUnitSP cu_sp = module_sp->GetCompileUnitAtIndex(i);
        if (cu_sp) {
          VariableListSP globals_sp = cu_sp->GetVariableList(true);
          if (globals_sp) {
            const size_t num_globals = globals_sp->GetSize();
            for (size_t g = 0; g < num_globals; ++g) {
              VariableSP var_sp = globals_sp->GetVariableAtIndex(g);
              if (var_sp && !var_sp->GetLocationIsConstantValueData()) {
                const DWARFExpressionList &location =
                    var_sp->LocationExpressionList();
                Value location_result;
                Status error;
                ExecutionContext exe_ctx;
                if (location.Evaluate(&exe_ctx, nullptr, LLDB_INVALID_ADDRESS,
                                      nullptr, nullptr, location_result,
                                      &error)) {
                  if (location_result.GetValueType() ==
                      Value::ValueType::FileAddress) {
                    lldb::addr_t file_addr =
                        location_result.GetScalar().ULongLong();
                    lldb::addr_t byte_size = 1;
                    if (var_sp->GetType())
                      byte_size =
                          var_sp->GetType()->GetByteSize(nullptr).value_or(0);
                    m_global_aranges_up->Append(GlobalVariableMap::Entry(
                        file_addr, byte_size, var_sp.get()));
                  }
                }
              }
            }
          }
        }
      }
    }
    m_global_aranges_up->Sort();
  }
  return *m_global_aranges_up;
}

void SymbolFileDWARF::ResolveFunctionAndBlock(lldb::addr_t file_vm_addr,
                                              bool lookup_block,
                                              SymbolContext &sc) {
  assert(sc.comp_unit);
  DWARFCompileUnit &cu =
      GetDWARFCompileUnit(sc.comp_unit)->GetNonSkeletonUnit();
  DWARFDIE function_die = cu.LookupAddress(file_vm_addr);
  DWARFDIE block_die;
  if (function_die) {
    sc.function = sc.comp_unit->FindFunctionByUID(function_die.GetID()).get();
    if (sc.function == nullptr)
      sc.function = ParseFunction(*sc.comp_unit, function_die);

    if (sc.function && lookup_block)
      block_die = function_die.LookupDeepestBlock(file_vm_addr);
  }

  if (!sc.function || !lookup_block)
    return;

  Block &block = sc.function->GetBlock(true);
  if (block_die)
    sc.block = block.FindBlockByID(block_die.GetID());
  else
    sc.block = block.FindBlockByID(function_die.GetID());
}

uint32_t SymbolFileDWARF::ResolveSymbolContext(const Address &so_addr,
                                               SymbolContextItem resolve_scope,
                                               SymbolContext &sc) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  LLDB_SCOPED_TIMERF("SymbolFileDWARF::"
                     "ResolveSymbolContext (so_addr = { "
                     "section = %p, offset = 0x%" PRIx64
                     " }, resolve_scope = 0x%8.8x)",
                     static_cast<void *>(so_addr.GetSection().get()),
                     so_addr.GetOffset(), resolve_scope);
  uint32_t resolved = 0;
  if (resolve_scope &
      (eSymbolContextCompUnit | eSymbolContextFunction | eSymbolContextBlock |
       eSymbolContextLineEntry | eSymbolContextVariable)) {
    lldb::addr_t file_vm_addr = so_addr.GetFileAddress();

    DWARFDebugInfo &debug_info = DebugInfo();
    const DWARFDebugAranges &aranges = debug_info.GetCompileUnitAranges();
    const dw_offset_t cu_offset = aranges.FindAddress(file_vm_addr);
    if (cu_offset == DW_INVALID_OFFSET) {
      // Global variables are not in the compile unit address ranges. The only
      // way to currently find global variables is to iterate over the
      // .debug_pubnames or the __apple_names table and find all items in there
      // that point to DW_TAG_variable DIEs and then find the address that
      // matches.
      if (resolve_scope & eSymbolContextVariable) {
        GlobalVariableMap &map = GetGlobalAranges();
        const GlobalVariableMap::Entry *entry =
            map.FindEntryThatContains(file_vm_addr);
        if (entry && entry->data) {
          Variable *variable = entry->data;
          SymbolContextScope *scc = variable->GetSymbolContextScope();
          if (scc) {
            scc->CalculateSymbolContext(&sc);
            sc.variable = variable;
          }
          return sc.GetResolvedMask();
        }
      }
    } else {
      uint32_t cu_idx = DW_INVALID_INDEX;
      if (auto *dwarf_cu = llvm::dyn_cast_or_null<DWARFCompileUnit>(
              debug_info.GetUnitAtOffset(DIERef::Section::DebugInfo, cu_offset,
                                         &cu_idx))) {
        sc.comp_unit = GetCompUnitForDWARFCompUnit(*dwarf_cu);
        if (sc.comp_unit) {
          resolved |= eSymbolContextCompUnit;

          bool force_check_line_table = false;
          if (resolve_scope & (eSymbolContextFunction | eSymbolContextBlock)) {
            ResolveFunctionAndBlock(file_vm_addr,
                                    resolve_scope & eSymbolContextBlock, sc);
            if (sc.function)
              resolved |= eSymbolContextFunction;
            else {
              // We might have had a compile unit that had discontiguous address
              // ranges where the gaps are symbols that don't have any debug
              // info. Discontiguous compile unit address ranges should only
              // happen when there aren't other functions from other compile
              // units in these gaps. This helps keep the size of the aranges
              // down.
              force_check_line_table = true;
            }
            if (sc.block)
              resolved |= eSymbolContextBlock;
          }

          if ((resolve_scope & eSymbolContextLineEntry) ||
              force_check_line_table) {
            LineTable *line_table = sc.comp_unit->GetLineTable();
            if (line_table != nullptr) {
              // And address that makes it into this function should be in terms
              // of this debug file if there is no debug map, or it will be an
              // address in the .o file which needs to be fixed up to be in
              // terms of the debug map executable. Either way, calling
              // FixupAddress() will work for us.
              Address exe_so_addr(so_addr);
              if (FixupAddress(exe_so_addr)) {
                if (line_table->FindLineEntryByAddress(exe_so_addr,
                                                       sc.line_entry)) {
                  resolved |= eSymbolContextLineEntry;
                }
              }
            }
          }

          if (force_check_line_table && !(resolved & eSymbolContextLineEntry)) {
            // We might have had a compile unit that had discontiguous address
            // ranges where the gaps are symbols that don't have any debug info.
            // Discontiguous compile unit address ranges should only happen when
            // there aren't other functions from other compile units in these
            // gaps. This helps keep the size of the aranges down.
            sc.comp_unit = nullptr;
            resolved &= ~eSymbolContextCompUnit;
          }
        } else {
          GetObjectFile()->GetModule()->ReportWarning(
              "0x%8.8x: compile unit %u failed to create a valid "
              "lldb_private::CompileUnit class.",
              cu_offset, cu_idx);
        }
      }
    }
  }
  return resolved;
}

uint32_t SymbolFileDWARF::ResolveSymbolContext(
    const SourceLocationSpec &src_location_spec,
    SymbolContextItem resolve_scope, SymbolContextList &sc_list) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  const bool check_inlines = src_location_spec.GetCheckInlines();
  const uint32_t prev_size = sc_list.GetSize();
  if (resolve_scope & eSymbolContextCompUnit) {
    for (uint32_t cu_idx = 0, num_cus = GetNumCompileUnits(); cu_idx < num_cus;
         ++cu_idx) {
      CompileUnit *dc_cu = ParseCompileUnitAtIndex(cu_idx).get();
      if (!dc_cu)
        continue;

      bool file_spec_matches_cu_file_spec = FileSpec::Match(
          src_location_spec.GetFileSpec(), dc_cu->GetPrimaryFile());
      if (check_inlines || file_spec_matches_cu_file_spec) {
        dc_cu->ResolveSymbolContext(src_location_spec, resolve_scope, sc_list);
        if (!check_inlines)
          break;
      }
    }
  }
  return sc_list.GetSize() - prev_size;
}

void SymbolFileDWARF::PreloadSymbols() {
  // Get the symbol table for the symbol file prior to taking the module lock
  // so that it is available without needing to take the module lock. The DWARF
  // indexing might end up needing to relocate items when DWARF sections are
  // loaded as they might end up getting the section contents which can call
  // ObjectFileELF::RelocateSection() which in turn will ask for the symbol
  // table and can cause deadlocks.
  GetSymtab();
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  m_index->Preload();
}

std::recursive_mutex &SymbolFileDWARF::GetModuleMutex() const {
  lldb::ModuleSP module_sp(m_debug_map_module_wp.lock());
  if (module_sp)
    return module_sp->GetMutex();
  return GetObjectFile()->GetModule()->GetMutex();
}

bool SymbolFileDWARF::DeclContextMatchesThisSymbolFile(
    const lldb_private::CompilerDeclContext &decl_ctx) {
  if (!decl_ctx.IsValid()) {
    // Invalid namespace decl which means we aren't matching only things in
    // this symbol file, so return true to indicate it matches this symbol
    // file.
    return true;
  }

  TypeSystem *decl_ctx_type_system = decl_ctx.GetTypeSystem();
  auto type_system_or_err = GetTypeSystemForLanguage(
      decl_ctx_type_system->GetMinimumLanguage(nullptr));
  if (auto err = type_system_or_err.takeError()) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Symbols), std::move(err),
                   "Unable to match namespace decl using TypeSystem");
    return false;
  }

  if (decl_ctx_type_system == &type_system_or_err.get())
    return true; // The type systems match, return true

  // The namespace AST was valid, and it does not match...
  Log *log = GetLog(DWARFLog::Lookups);

  if (log)
    GetObjectFile()->GetModule()->LogMessage(
        log, "Valid namespace does not match symbol file");

  return false;
}

void SymbolFileDWARF::FindGlobalVariables(
    ConstString name, const CompilerDeclContext &parent_decl_ctx,
    uint32_t max_matches, VariableList &variables) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  Log *log = GetLog(DWARFLog::Lookups);

  if (log)
    GetObjectFile()->GetModule()->LogMessage(
        log,
        "SymbolFileDWARF::FindGlobalVariables (name=\"%s\", "
        "parent_decl_ctx=%p, max_matches=%u, variables)",
        name.GetCString(), static_cast<const void *>(&parent_decl_ctx),
        max_matches);

  if (!DeclContextMatchesThisSymbolFile(parent_decl_ctx))
    return;

  // Remember how many variables are in the list before we search.
  const uint32_t original_size = variables.GetSize();

  llvm::StringRef basename;
  llvm::StringRef context;
  bool name_is_mangled = Mangled::GetManglingScheme(name.GetStringRef()) !=
                         Mangled::eManglingSchemeNone;

  if (!CPlusPlusLanguage::ExtractContextAndIdentifier(name.GetCString(),
                                                      context, basename))
    basename = name.GetStringRef();

  // Loop invariant: Variables up to this index have been checked for context
  // matches.
  uint32_t pruned_idx = original_size;

  SymbolContext sc;
  m_index->GetGlobalVariables(ConstString(basename), [&](DWARFDIE die) {
    if (!sc.module_sp)
      sc.module_sp = m_objfile_sp->GetModule();
    assert(sc.module_sp);

    if (die.Tag() != DW_TAG_variable)
      return true;

    auto *dwarf_cu = llvm::dyn_cast<DWARFCompileUnit>(die.GetCU());
    if (!dwarf_cu)
      return true;
    sc.comp_unit = GetCompUnitForDWARFCompUnit(*dwarf_cu);

    if (parent_decl_ctx) {
      if (DWARFASTParser *dwarf_ast = GetDWARFParser(*die.GetCU())) {
        CompilerDeclContext actual_parent_decl_ctx =
            dwarf_ast->GetDeclContextContainingUIDFromDWARF(die);
        if (!actual_parent_decl_ctx ||
            actual_parent_decl_ctx != parent_decl_ctx)
          return true;
      }
    }

    ParseAndAppendGlobalVariable(sc, die, variables);
    while (pruned_idx < variables.GetSize()) {
      VariableSP var_sp = variables.GetVariableAtIndex(pruned_idx);
      if (name_is_mangled ||
          var_sp->GetName().GetStringRef().contains(name.GetStringRef()))
        ++pruned_idx;
      else
        variables.RemoveVariableAtIndex(pruned_idx);
    }

    return variables.GetSize() - original_size < max_matches;
  });

  // Return the number of variable that were appended to the list
  const uint32_t num_matches = variables.GetSize() - original_size;
  if (log && num_matches > 0) {
    GetObjectFile()->GetModule()->LogMessage(
        log,
        "SymbolFileDWARF::FindGlobalVariables (name=\"%s\", "
        "parent_decl_ctx=%p, max_matches=%u, variables) => %u",
        name.GetCString(), static_cast<const void *>(&parent_decl_ctx),
        max_matches, num_matches);
  }
}

void SymbolFileDWARF::FindGlobalVariables(const RegularExpression &regex,
                                          uint32_t max_matches,
                                          VariableList &variables) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  Log *log = GetLog(DWARFLog::Lookups);

  if (log) {
    GetObjectFile()->GetModule()->LogMessage(
        log,
        "SymbolFileDWARF::FindGlobalVariables (regex=\"%s\", "
        "max_matches=%u, variables)",
        regex.GetText().str().c_str(), max_matches);
  }

  // Remember how many variables are in the list before we search.
  const uint32_t original_size = variables.GetSize();

  SymbolContext sc;
  m_index->GetGlobalVariables(regex, [&](DWARFDIE die) {
    if (!sc.module_sp)
      sc.module_sp = m_objfile_sp->GetModule();
    assert(sc.module_sp);

    DWARFCompileUnit *dwarf_cu = llvm::dyn_cast<DWARFCompileUnit>(die.GetCU());
    if (!dwarf_cu)
      return true;
    sc.comp_unit = GetCompUnitForDWARFCompUnit(*dwarf_cu);

    ParseAndAppendGlobalVariable(sc, die, variables);

    return variables.GetSize() - original_size < max_matches;
  });
}

bool SymbolFileDWARF::ResolveFunction(const DWARFDIE &orig_die,
                                      bool include_inlines,
                                      SymbolContextList &sc_list) {
  SymbolContext sc;

  if (!orig_die)
    return false;

  // If we were passed a die that is not a function, just return false...
  if (!(orig_die.Tag() == DW_TAG_subprogram ||
        (include_inlines && orig_die.Tag() == DW_TAG_inlined_subroutine)))
    return false;

  DWARFDIE die = orig_die;
  DWARFDIE inlined_die;
  if (die.Tag() == DW_TAG_inlined_subroutine) {
    inlined_die = die;

    while (true) {
      die = die.GetParent();

      if (die) {
        if (die.Tag() == DW_TAG_subprogram)
          break;
      } else
        break;
    }
  }
  assert(die && die.Tag() == DW_TAG_subprogram);
  if (GetFunction(die, sc)) {
    Address addr;
    // Parse all blocks if needed
    if (inlined_die) {
      Block &function_block = sc.function->GetBlock(true);
      sc.block = function_block.FindBlockByID(inlined_die.GetID());
      if (sc.block == nullptr)
        sc.block = function_block.FindBlockByID(inlined_die.GetOffset());
      if (sc.block == nullptr || !sc.block->GetStartAddress(addr))
        addr.Clear();
    } else {
      sc.block = nullptr;
      addr = sc.function->GetAddressRange().GetBaseAddress();
    }

    sc_list.Append(sc);
    return true;
  }

  return false;
}

bool SymbolFileDWARF::DIEInDeclContext(const CompilerDeclContext &decl_ctx,
                                       const DWARFDIE &die) {
  // If we have no parent decl context to match this DIE matches, and if the
  // parent decl context isn't valid, we aren't trying to look for any
  // particular decl context so any die matches.
  if (!decl_ctx.IsValid())
    return true;

  if (die) {
    if (DWARFASTParser *dwarf_ast = GetDWARFParser(*die.GetCU())) {
      if (CompilerDeclContext actual_decl_ctx =
              dwarf_ast->GetDeclContextContainingUIDFromDWARF(die))
        return decl_ctx.IsContainedInLookup(actual_decl_ctx);
    }
  }
  return false;
}

void SymbolFileDWARF::FindFunctions(const Module::LookupInfo &lookup_info,
                                    const CompilerDeclContext &parent_decl_ctx,
                                    bool include_inlines,
                                    SymbolContextList &sc_list) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  ConstString name = lookup_info.GetLookupName();
  FunctionNameType name_type_mask = lookup_info.GetNameTypeMask();
  LLDB_SCOPED_TIMERF("SymbolFileDWARF::FindFunctions (name = '%s')",
                     name.AsCString());

  // eFunctionNameTypeAuto should be pre-resolved by a call to
  // Module::LookupInfo::LookupInfo()
  assert((name_type_mask & eFunctionNameTypeAuto) == 0);

  Log *log = GetLog(DWARFLog::Lookups);

  if (log) {
    GetObjectFile()->GetModule()->LogMessage(
        log,
        "SymbolFileDWARF::FindFunctions (name=\"%s\", name_type_mask=0x%x, "
        "sc_list)",
        name.GetCString(), name_type_mask);
  }

  if (!DeclContextMatchesThisSymbolFile(parent_decl_ctx))
    return;

  // If name is empty then we won't find anything.
  if (name.IsEmpty())
    return;

  // Remember how many sc_list are in the list before we search in case we are
  // appending the results to a variable list.

  const uint32_t original_size = sc_list.GetSize();

  llvm::DenseSet<const DWARFDebugInfoEntry *> resolved_dies;

  m_index->GetFunctions(lookup_info, *this, parent_decl_ctx, [&](DWARFDIE die) {
    if (resolved_dies.insert(die.GetDIE()).second)
      ResolveFunction(die, include_inlines, sc_list);
    return true;
  });

  // Return the number of variable that were appended to the list
  const uint32_t num_matches = sc_list.GetSize() - original_size;

  if (log && num_matches > 0) {
    GetObjectFile()->GetModule()->LogMessage(
        log,
        "SymbolFileDWARF::FindFunctions (name=\"%s\", "
        "name_type_mask=0x%x, include_inlines=%d, sc_list) => %u",
        name.GetCString(), name_type_mask, include_inlines, num_matches);
  }
}

void SymbolFileDWARF::FindFunctions(const RegularExpression &regex,
                                    bool include_inlines,
                                    SymbolContextList &sc_list) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  LLDB_SCOPED_TIMERF("SymbolFileDWARF::FindFunctions (regex = '%s')",
                     regex.GetText().str().c_str());

  Log *log = GetLog(DWARFLog::Lookups);

  if (log) {
    GetObjectFile()->GetModule()->LogMessage(
        log, "SymbolFileDWARF::FindFunctions (regex=\"%s\", sc_list)",
        regex.GetText().str().c_str());
  }

  llvm::DenseSet<const DWARFDebugInfoEntry *> resolved_dies;
  m_index->GetFunctions(regex, [&](DWARFDIE die) {
    if (resolved_dies.insert(die.GetDIE()).second)
      ResolveFunction(die, include_inlines, sc_list);
    return true;
  });
}

void SymbolFileDWARF::GetMangledNamesForFunction(
    const std::string &scope_qualified_name,
    std::vector<ConstString> &mangled_names) {
  DWARFDebugInfo &info = DebugInfo();
  uint32_t num_comp_units = info.GetNumUnits();
  for (uint32_t i = 0; i < num_comp_units; i++) {
    DWARFUnit *cu = info.GetUnitAtIndex(i);
    if (cu == nullptr)
      continue;

    SymbolFileDWARFDwo *dwo = cu->GetDwoSymbolFile();
    if (dwo)
      dwo->GetMangledNamesForFunction(scope_qualified_name, mangled_names);
  }

  for (DIERef die_ref :
       m_function_scope_qualified_name_map.lookup(scope_qualified_name)) {
    DWARFDIE die = GetDIE(die_ref);
    mangled_names.push_back(ConstString(die.GetMangledName()));
  }
}

void SymbolFileDWARF::FindTypes(
    ConstString name, const CompilerDeclContext &parent_decl_ctx,
    uint32_t max_matches,
    llvm::DenseSet<lldb_private::SymbolFile *> &searched_symbol_files,
    TypeMap &types) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  // Make sure we haven't already searched this SymbolFile before.
  if (!searched_symbol_files.insert(this).second)
    return;

  Log *log = GetLog(DWARFLog::Lookups);

  if (log) {
    if (parent_decl_ctx)
      GetObjectFile()->GetModule()->LogMessage(
          log,
          "SymbolFileDWARF::FindTypes (sc, name=\"%s\", parent_decl_ctx = "
          "%p (\"%s\"), max_matches=%u, type_list)",
          name.GetCString(), static_cast<const void *>(&parent_decl_ctx),
          parent_decl_ctx.GetName().AsCString("<NULL>"), max_matches);
    else
      GetObjectFile()->GetModule()->LogMessage(
          log,
          "SymbolFileDWARF::FindTypes (sc, name=\"%s\", parent_decl_ctx = "
          "NULL, max_matches=%u, type_list)",
          name.GetCString(), max_matches);
  }

  if (!DeclContextMatchesThisSymbolFile(parent_decl_ctx))
    return;

  m_index->GetTypes(name, [&](DWARFDIE die) {
    if (!DIEInDeclContext(parent_decl_ctx, die))
      return true; // The containing decl contexts don't match

    Type *matching_type = ResolveType(die, true, true);
    if (!matching_type)
      return true;

    // We found a type pointer, now find the shared pointer form our type
    // list
    types.InsertUnique(matching_type->shared_from_this());
    return types.GetSize() < max_matches;
  });

  // Next search through the reachable Clang modules. This only applies for
  // DWARF objects compiled with -gmodules that haven't been processed by
  // dsymutil.
  if (types.GetSize() < max_matches) {
    UpdateExternalModuleListIfNeeded();

    for (const auto &pair : m_external_type_modules)
      if (ModuleSP external_module_sp = pair.second)
        if (SymbolFile *sym_file = external_module_sp->GetSymbolFile())
          sym_file->FindTypes(name, parent_decl_ctx, max_matches,
                              searched_symbol_files, types);
  }

  if (log && types.GetSize()) {
    if (parent_decl_ctx) {
      GetObjectFile()->GetModule()->LogMessage(
          log,
          "SymbolFileDWARF::FindTypes (sc, name=\"%s\", parent_decl_ctx "
          "= %p (\"%s\"), max_matches=%u, type_list) => %u",
          name.GetCString(), static_cast<const void *>(&parent_decl_ctx),
          parent_decl_ctx.GetName().AsCString("<NULL>"), max_matches,
          types.GetSize());
    } else {
      GetObjectFile()->GetModule()->LogMessage(
          log,
          "SymbolFileDWARF::FindTypes (sc, name=\"%s\", parent_decl_ctx "
          "= NULL, max_matches=%u, type_list) => %u",
          name.GetCString(), max_matches, types.GetSize());
    }
  }
}

void SymbolFileDWARF::FindTypes(
    llvm::ArrayRef<CompilerContext> pattern, LanguageSet languages,
    llvm::DenseSet<SymbolFile *> &searched_symbol_files, TypeMap &types) {
  // Make sure we haven't already searched this SymbolFile before.
  if (!searched_symbol_files.insert(this).second)
    return;

  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  if (pattern.empty())
    return;

  ConstString name = pattern.back().name;

  if (!name)
    return;

  m_index->GetTypes(name, [&](DWARFDIE die) {
    if (!languages[GetLanguageFamily(*die.GetCU())])
      return true;

    llvm::SmallVector<CompilerContext, 4> die_context;
    die.GetDeclContext(die_context);
    if (!contextMatches(die_context, pattern))
      return true;

    if (Type *matching_type = ResolveType(die, true, true)) {
      // We found a type pointer, now find the shared pointer form our type
      // list.
      types.InsertUnique(matching_type->shared_from_this());
    }
    return true;
  });

  // Next search through the reachable Clang modules. This only applies for
  // DWARF objects compiled with -gmodules that haven't been processed by
  // dsymutil.
  UpdateExternalModuleListIfNeeded();

  for (const auto &pair : m_external_type_modules)
    if (ModuleSP external_module_sp = pair.second)
      external_module_sp->FindTypes(pattern, languages, searched_symbol_files,
                                    types);
}

CompilerDeclContext
SymbolFileDWARF::FindNamespace(ConstString name,
                               const CompilerDeclContext &parent_decl_ctx) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  Log *log = GetLog(DWARFLog::Lookups);

  if (log) {
    GetObjectFile()->GetModule()->LogMessage(
        log, "SymbolFileDWARF::FindNamespace (sc, name=\"%s\")",
        name.GetCString());
  }

  CompilerDeclContext namespace_decl_ctx;

  if (!DeclContextMatchesThisSymbolFile(parent_decl_ctx))
    return namespace_decl_ctx;

  m_index->GetNamespaces(name, [&](DWARFDIE die) {
    if (!DIEInDeclContext(parent_decl_ctx, die))
      return true; // The containing decl contexts don't match

    DWARFASTParser *dwarf_ast = GetDWARFParser(*die.GetCU());
    if (!dwarf_ast)
      return true;

    namespace_decl_ctx = dwarf_ast->GetDeclContextForUIDFromDWARF(die);
    return !namespace_decl_ctx.IsValid();
  });

  if (log && namespace_decl_ctx) {
    GetObjectFile()->GetModule()->LogMessage(
        log,
        "SymbolFileDWARF::FindNamespace (sc, name=\"%s\") => "
        "CompilerDeclContext(%p/%p) \"%s\"",
        name.GetCString(),
        static_cast<const void *>(namespace_decl_ctx.GetTypeSystem()),
        static_cast<const void *>(namespace_decl_ctx.GetOpaqueDeclContext()),
        namespace_decl_ctx.GetName().AsCString("<NULL>"));
  }

  return namespace_decl_ctx;
}

TypeSP SymbolFileDWARF::GetTypeForDIE(const DWARFDIE &die,
                                      bool resolve_function_context) {
  TypeSP type_sp;
  if (die) {
    Type *type_ptr = GetDIEToType().lookup(die.GetDIE());
    if (type_ptr == nullptr) {
      SymbolContextScope *scope;
      if (auto *dwarf_cu = llvm::dyn_cast<DWARFCompileUnit>(die.GetCU()))
        scope = GetCompUnitForDWARFCompUnit(*dwarf_cu);
      else
        scope = GetObjectFile()->GetModule().get();
      assert(scope);
      SymbolContext sc(scope);
      const DWARFDebugInfoEntry *parent_die = die.GetParent().GetDIE();
      while (parent_die != nullptr) {
        if (parent_die->Tag() == DW_TAG_subprogram)
          break;
        parent_die = parent_die->GetParent();
      }
      SymbolContext sc_backup = sc;
      if (resolve_function_context && parent_die != nullptr &&
          !GetFunction(DWARFDIE(die.GetCU(), parent_die), sc))
        sc = sc_backup;

      type_sp = ParseType(sc, die, nullptr);
    } else if (type_ptr != DIE_IS_BEING_PARSED) {
      // Get the original shared pointer for this type
      type_sp = type_ptr->shared_from_this();
    }
  }
  return type_sp;
}

DWARFDIE
SymbolFileDWARF::GetDeclContextDIEContainingDIE(const DWARFDIE &orig_die) {
  if (orig_die) {
    DWARFDIE die = orig_die;

    while (die) {
      // If this is the original DIE that we are searching for a declaration
      // for, then don't look in the cache as we don't want our own decl
      // context to be our decl context...
      if (orig_die != die) {
        switch (die.Tag()) {
        case DW_TAG_compile_unit:
        case DW_TAG_partial_unit:
        case DW_TAG_namespace:
        case DW_TAG_structure_type:
        case DW_TAG_union_type:
        case DW_TAG_class_type:
        case DW_TAG_lexical_block:
        case DW_TAG_subprogram:
          return die;
        case DW_TAG_inlined_subroutine: {
          DWARFDIE abs_die = die.GetReferencedDIE(DW_AT_abstract_origin);
          if (abs_die) {
            return abs_die;
          }
          break;
        }
        default:
          break;
        }
      }

      DWARFDIE spec_die = die.GetReferencedDIE(DW_AT_specification);
      if (spec_die) {
        DWARFDIE decl_ctx_die = GetDeclContextDIEContainingDIE(spec_die);
        if (decl_ctx_die)
          return decl_ctx_die;
      }

      DWARFDIE abs_die = die.GetReferencedDIE(DW_AT_abstract_origin);
      if (abs_die) {
        DWARFDIE decl_ctx_die = GetDeclContextDIEContainingDIE(abs_die);
        if (decl_ctx_die)
          return decl_ctx_die;
      }

      die = die.GetParent();
    }
  }
  return DWARFDIE();
}

Symbol *SymbolFileDWARF::GetObjCClassSymbol(ConstString objc_class_name) {
  Symbol *objc_class_symbol = nullptr;
  if (m_objfile_sp) {
    Symtab *symtab = m_objfile_sp->GetSymtab();
    if (symtab) {
      objc_class_symbol = symtab->FindFirstSymbolWithNameAndType(
          objc_class_name, eSymbolTypeObjCClass, Symtab::eDebugNo,
          Symtab::eVisibilityAny);
    }
  }
  return objc_class_symbol;
}

// Some compilers don't emit the DW_AT_APPLE_objc_complete_type attribute. If
// they don't then we can end up looking through all class types for a complete
// type and never find the full definition. We need to know if this attribute
// is supported, so we determine this here and cache th result. We also need to
// worry about the debug map
// DWARF file
// if we are doing darwin DWARF in .o file debugging.
bool SymbolFileDWARF::Supports_DW_AT_APPLE_objc_complete_type(DWARFUnit *cu) {
  if (m_supports_DW_AT_APPLE_objc_complete_type == eLazyBoolCalculate) {
    m_supports_DW_AT_APPLE_objc_complete_type = eLazyBoolNo;
    if (cu && cu->Supports_DW_AT_APPLE_objc_complete_type())
      m_supports_DW_AT_APPLE_objc_complete_type = eLazyBoolYes;
    else {
      DWARFDebugInfo &debug_info = DebugInfo();
      const uint32_t num_compile_units = GetNumCompileUnits();
      for (uint32_t cu_idx = 0; cu_idx < num_compile_units; ++cu_idx) {
        DWARFUnit *dwarf_cu = debug_info.GetUnitAtIndex(cu_idx);
        if (dwarf_cu != cu &&
            dwarf_cu->Supports_DW_AT_APPLE_objc_complete_type()) {
          m_supports_DW_AT_APPLE_objc_complete_type = eLazyBoolYes;
          break;
        }
      }
    }
    if (m_supports_DW_AT_APPLE_objc_complete_type == eLazyBoolNo &&
        GetDebugMapSymfile())
      return m_debug_map_symfile->Supports_DW_AT_APPLE_objc_complete_type(this);
  }
  return m_supports_DW_AT_APPLE_objc_complete_type == eLazyBoolYes;
}

// This function can be used when a DIE is found that is a forward declaration
// DIE and we want to try and find a type that has the complete definition.
TypeSP SymbolFileDWARF::FindCompleteObjCDefinitionTypeForDIE(
    const DWARFDIE &die, ConstString type_name, bool must_be_implementation) {

  TypeSP type_sp;

  if (!type_name || (must_be_implementation && !GetObjCClassSymbol(type_name)))
    return type_sp;

  m_index->GetCompleteObjCClass(
      type_name, must_be_implementation, [&](DWARFDIE type_die) {
        bool try_resolving_type = false;

        // Don't try and resolve the DIE we are looking for with the DIE
        // itself!
        if (type_die != die) {
          switch (type_die.Tag()) {
          case DW_TAG_class_type:
          case DW_TAG_structure_type:
            try_resolving_type = true;
            break;
          default:
            break;
          }
        }
        if (!try_resolving_type)
          return true;

        if (must_be_implementation &&
            type_die.Supports_DW_AT_APPLE_objc_complete_type())
          try_resolving_type = type_die.GetAttributeValueAsUnsigned(
              DW_AT_APPLE_objc_complete_type, 0);
        if (!try_resolving_type)
          return true;

        Type *resolved_type = ResolveType(type_die, false, true);
        if (!resolved_type || resolved_type == DIE_IS_BEING_PARSED)
          return true;

        DEBUG_PRINTF(
            "resolved 0x%8.8" PRIx64 " from %s to 0x%8.8" PRIx64
            " (cu 0x%8.8" PRIx64 ")\n",
            die.GetID(),
            m_objfile_sp->GetFileSpec().GetFilename().AsCString("<Unknown>"),
            type_die.GetID(), type_cu->GetID());

        if (die)
          GetDIEToType()[die.GetDIE()] = resolved_type;
        type_sp = resolved_type->shared_from_this();
        return false;
      });
  return type_sp;
}

// This function helps to ensure that the declaration contexts match for two
// different DIEs. Often times debug information will refer to a forward
// declaration of a type (the equivalent of "struct my_struct;". There will
// often be a declaration of that type elsewhere that has the full definition.
// When we go looking for the full type "my_struct", we will find one or more
// matches in the accelerator tables and we will then need to make sure the
// type was in the same declaration context as the original DIE. This function
// can efficiently compare two DIEs and will return true when the declaration
// context matches, and false when they don't.
bool SymbolFileDWARF::DIEDeclContextsMatch(const DWARFDIE &die1,
                                           const DWARFDIE &die2) {
  if (die1 == die2)
    return true;

  std::vector<DWARFDIE> decl_ctx_1;
  std::vector<DWARFDIE> decl_ctx_2;
  // The declaration DIE stack is a stack of the declaration context DIEs all
  // the way back to the compile unit. If a type "T" is declared inside a class
  // "B", and class "B" is declared inside a class "A" and class "A" is in a
  // namespace "lldb", and the namespace is in a compile unit, there will be a
  // stack of DIEs:
  //
  //   [0] DW_TAG_class_type for "B"
  //   [1] DW_TAG_class_type for "A"
  //   [2] DW_TAG_namespace  for "lldb"
  //   [3] DW_TAG_compile_unit or DW_TAG_partial_unit for the source file.
  //
  // We grab both contexts and make sure that everything matches all the way
  // back to the compiler unit.

  // First lets grab the decl contexts for both DIEs
  decl_ctx_1 = die1.GetDeclContextDIEs();
  decl_ctx_2 = die2.GetDeclContextDIEs();
  // Make sure the context arrays have the same size, otherwise we are done
  const size_t count1 = decl_ctx_1.size();
  const size_t count2 = decl_ctx_2.size();
  if (count1 != count2)
    return false;

  // Make sure the DW_TAG values match all the way back up the compile unit. If
  // they don't, then we are done.
  DWARFDIE decl_ctx_die1;
  DWARFDIE decl_ctx_die2;
  size_t i;
  for (i = 0; i < count1; i++) {
    decl_ctx_die1 = decl_ctx_1[i];
    decl_ctx_die2 = decl_ctx_2[i];
    if (decl_ctx_die1.Tag() != decl_ctx_die2.Tag())
      return false;
  }
#ifndef NDEBUG

  // Make sure the top item in the decl context die array is always
  // DW_TAG_compile_unit or DW_TAG_partial_unit. If it isn't then
  // something went wrong in the DWARFDIE::GetDeclContextDIEs()
  // function.
  dw_tag_t cu_tag = decl_ctx_1[count1 - 1].Tag();
  UNUSED_IF_ASSERT_DISABLED(cu_tag);
  assert(cu_tag == DW_TAG_compile_unit || cu_tag == DW_TAG_partial_unit);

#endif
  // Always skip the compile unit when comparing by only iterating up to "count
  // - 1". Here we compare the names as we go.
  for (i = 0; i < count1 - 1; i++) {
    decl_ctx_die1 = decl_ctx_1[i];
    decl_ctx_die2 = decl_ctx_2[i];
    const char *name1 = decl_ctx_die1.GetName();
    const char *name2 = decl_ctx_die2.GetName();
    // If the string was from a DW_FORM_strp, then the pointer will often be
    // the same!
    if (name1 == name2)
      continue;

    // Name pointers are not equal, so only compare the strings if both are not
    // NULL.
    if (name1 && name2) {
      // If the strings don't compare, we are done...
      if (strcmp(name1, name2) != 0)
        return false;
    } else {
      // One name was NULL while the other wasn't
      return false;
    }
  }
  // We made it through all of the checks and the declaration contexts are
  // equal.
  return true;
}

TypeSP SymbolFileDWARF::FindDefinitionTypeForDWARFDeclContext(
    const DWARFDeclContext &dwarf_decl_ctx) {
  TypeSP type_sp;

  const uint32_t dwarf_decl_ctx_count = dwarf_decl_ctx.GetSize();
  if (dwarf_decl_ctx_count > 0) {
    const ConstString type_name(dwarf_decl_ctx[0].name);
    const dw_tag_t tag = dwarf_decl_ctx[0].tag;

    if (type_name) {
      Log *log = GetLog(DWARFLog::TypeCompletion | DWARFLog::Lookups);
      if (log) {
        GetObjectFile()->GetModule()->LogMessage(
            log,
            "SymbolFileDWARF::FindDefinitionTypeForDWARFDeclContext(tag=%"
            "s, qualified-name='%s')",
            DW_TAG_value_to_name(dwarf_decl_ctx[0].tag),
            dwarf_decl_ctx.GetQualifiedName());
      }

      // Get the type system that we are looking to find a type for. We will
      // use this to ensure any matches we find are in a language that this
      // type system supports
      const LanguageType language = dwarf_decl_ctx.GetLanguage();
      TypeSystem *type_system = nullptr;
      if (language != eLanguageTypeUnknown) {
        auto type_system_or_err = GetTypeSystemForLanguage(language);
        if (auto err = type_system_or_err.takeError()) {
          LLDB_LOG_ERROR(GetLog(LLDBLog::Symbols), std::move(err),
                         "Cannot get TypeSystem for language {}",
                         Language::GetNameForLanguageType(language));
        } else {
          type_system = &type_system_or_err.get();
        }
      }

      m_index->GetTypes(dwarf_decl_ctx, [&](DWARFDIE type_die) {
        // Make sure type_die's language matches the type system we are
        // looking for. We don't want to find a "Foo" type from Java if we
        // are looking for a "Foo" type for C, C++, ObjC, or ObjC++.
        if (type_system &&
            !type_system->SupportsLanguage(GetLanguage(*type_die.GetCU())))
          return true;
        bool try_resolving_type = false;

        // Don't try and resolve the DIE we are looking for with the DIE
        // itself!
        const dw_tag_t type_tag = type_die.Tag();
        // Make sure the tags match
        if (type_tag == tag) {
          // The tags match, lets try resolving this type
          try_resolving_type = true;
        } else {
          // The tags don't match, but we need to watch our for a forward
          // declaration for a struct and ("struct foo") ends up being a
          // class ("class foo { ... };") or vice versa.
          switch (type_tag) {
          case DW_TAG_class_type:
            // We had a "class foo", see if we ended up with a "struct foo
            // { ... };"
            try_resolving_type = (tag == DW_TAG_structure_type);
            break;
          case DW_TAG_structure_type:
            // We had a "struct foo", see if we ended up with a "class foo
            // { ... };"
            try_resolving_type = (tag == DW_TAG_class_type);
            break;
          default:
            // Tags don't match, don't event try to resolve using this type
            // whose name matches....
            break;
          }
        }

        if (!try_resolving_type) {
          if (log) {
            std::string qualified_name;
            type_die.GetQualifiedName(qualified_name);
            GetObjectFile()->GetModule()->LogMessage(
                log,
                "SymbolFileDWARF::"
                "FindDefinitionTypeForDWARFDeclContext(tag=%s, "
                "qualified-name='%s') ignoring die=0x%8.8x (%s)",
                DW_TAG_value_to_name(dwarf_decl_ctx[0].tag),
                dwarf_decl_ctx.GetQualifiedName(), type_die.GetOffset(),
                qualified_name.c_str());
          }
          return true;
        }

        DWARFDeclContext type_dwarf_decl_ctx = GetDWARFDeclContext(type_die);

        if (log) {
          GetObjectFile()->GetModule()->LogMessage(
              log,
              "SymbolFileDWARF::"
              "FindDefinitionTypeForDWARFDeclContext(tag=%s, "
              "qualified-name='%s') trying die=0x%8.8x (%s)",
              DW_TAG_value_to_name(dwarf_decl_ctx[0].tag),
              dwarf_decl_ctx.GetQualifiedName(), type_die.GetOffset(),
              type_dwarf_decl_ctx.GetQualifiedName());
        }

        // Make sure the decl contexts match all the way up
        if (dwarf_decl_ctx != type_dwarf_decl_ctx)
          return true;

        Type *resolved_type = ResolveType(type_die, false);
        if (!resolved_type || resolved_type == DIE_IS_BEING_PARSED)
          return true;

        type_sp = resolved_type->shared_from_this();
        return false;
      });
    }
  }
  return type_sp;
}

TypeSP SymbolFileDWARF::ParseType(const SymbolContext &sc, const DWARFDIE &die,
                                  bool *type_is_new_ptr) {
  if (!die)
    return {};

  auto type_system_or_err = GetTypeSystemForLanguage(GetLanguage(*die.GetCU()));
  if (auto err = type_system_or_err.takeError()) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Symbols), std::move(err),
                   "Unable to parse type");
    return {};
  }

  DWARFASTParser *dwarf_ast = type_system_or_err->GetDWARFParser();
  if (!dwarf_ast)
    return {};

  TypeSP type_sp = dwarf_ast->ParseTypeFromDWARF(sc, die, type_is_new_ptr);
  if (type_sp) {
    GetTypeList().Insert(type_sp);

    if (die.Tag() == DW_TAG_subprogram) {
      std::string scope_qualified_name(GetDeclContextForUID(die.GetID())
                                           .GetScopeQualifiedName()
                                           .AsCString(""));
      if (scope_qualified_name.size()) {
        m_function_scope_qualified_name_map[scope_qualified_name].insert(
            *die.GetDIERef());
      }
    }
  }

  return type_sp;
}

size_t SymbolFileDWARF::ParseTypes(const SymbolContext &sc,
                                   const DWARFDIE &orig_die,
                                   bool parse_siblings, bool parse_children) {
  size_t types_added = 0;
  DWARFDIE die = orig_die;

  while (die) {
    const dw_tag_t tag = die.Tag();
    bool type_is_new = false;

    Tag dwarf_tag = static_cast<Tag>(tag);

    // TODO: Currently ParseTypeFromDWARF(...) which is called by ParseType(...)
    // does not handle DW_TAG_subrange_type. It is not clear if this is a bug or
    // not.
    if (isType(dwarf_tag) && tag != DW_TAG_subrange_type)
      ParseType(sc, die, &type_is_new);

    if (type_is_new)
      ++types_added;

    if (parse_children && die.HasChildren()) {
      if (die.Tag() == DW_TAG_subprogram) {
        SymbolContext child_sc(sc);
        child_sc.function = sc.comp_unit->FindFunctionByUID(die.GetID()).get();
        types_added += ParseTypes(child_sc, die.GetFirstChild(), true, true);
      } else
        types_added += ParseTypes(sc, die.GetFirstChild(), true, true);
    }

    if (parse_siblings)
      die = die.GetSibling();
    else
      die.Clear();
  }
  return types_added;
}

size_t SymbolFileDWARF::ParseBlocksRecursive(Function &func) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  CompileUnit *comp_unit = func.GetCompileUnit();
  lldbassert(comp_unit);

  DWARFUnit *dwarf_cu = GetDWARFCompileUnit(comp_unit);
  if (!dwarf_cu)
    return 0;

  size_t functions_added = 0;
  const dw_offset_t function_die_offset = func.GetID();
  DWARFDIE function_die =
      dwarf_cu->GetNonSkeletonUnit().GetDIE(function_die_offset);
  if (function_die) {
    ParseBlocksRecursive(*comp_unit, &func.GetBlock(false), function_die,
                         LLDB_INVALID_ADDRESS, 0);
  }

  return functions_added;
}

size_t SymbolFileDWARF::ParseTypes(CompileUnit &comp_unit) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  size_t types_added = 0;
  DWARFUnit *dwarf_cu = GetDWARFCompileUnit(&comp_unit);
  if (dwarf_cu) {
    DWARFDIE dwarf_cu_die = dwarf_cu->DIE();
    if (dwarf_cu_die && dwarf_cu_die.HasChildren()) {
      SymbolContext sc;
      sc.comp_unit = &comp_unit;
      types_added = ParseTypes(sc, dwarf_cu_die.GetFirstChild(), true, true);
    }
  }

  return types_added;
}

size_t SymbolFileDWARF::ParseVariablesForContext(const SymbolContext &sc) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  if (sc.comp_unit != nullptr) {
    if (sc.function) {
      DWARFDIE function_die = GetDIE(sc.function->GetID());

      dw_addr_t func_lo_pc = LLDB_INVALID_ADDRESS;
      DWARFRangeList ranges;
      if (function_die.GetDIE()->GetAttributeAddressRanges(
              function_die.GetCU(), ranges,
              /*check_hi_lo_pc=*/true))
        func_lo_pc = ranges.GetMinRangeBase(0);
      if (func_lo_pc != LLDB_INVALID_ADDRESS) {
        const size_t num_variables =
            ParseVariablesInFunctionContext(sc, function_die, func_lo_pc);

        // Let all blocks know they have parse all their variables
        sc.function->GetBlock(false).SetDidParseVariables(true, true);
        return num_variables;
      }
    } else if (sc.comp_unit) {
      DWARFUnit *dwarf_cu = DebugInfo().GetUnitAtIndex(sc.comp_unit->GetID());

      if (dwarf_cu == nullptr)
        return 0;

      uint32_t vars_added = 0;
      VariableListSP variables(sc.comp_unit->GetVariableList(false));

      if (variables.get() == nullptr) {
        variables = std::make_shared<VariableList>();
        sc.comp_unit->SetVariableList(variables);

        m_index->GetGlobalVariables(*dwarf_cu, [&](DWARFDIE die) {
          VariableSP var_sp(ParseVariableDIECached(sc, die));
          if (var_sp) {
            variables->AddVariableIfUnique(var_sp);
            ++vars_added;
          }
          return true;
        });
      }
      return vars_added;
    }
  }
  return 0;
}

VariableSP SymbolFileDWARF::ParseVariableDIECached(const SymbolContext &sc,
                                                   const DWARFDIE &die) {
  if (!die)
    return nullptr;

  DIEToVariableSP &die_to_variable = die.GetDWARF()->GetDIEToVariable();

  VariableSP var_sp = die_to_variable[die.GetDIE()];
  if (var_sp)
    return var_sp;

  var_sp = ParseVariableDIE(sc, die, LLDB_INVALID_ADDRESS);
  if (var_sp) {
    die_to_variable[die.GetDIE()] = var_sp;
    if (DWARFDIE spec_die = die.GetReferencedDIE(DW_AT_specification))
      die_to_variable[spec_die.GetDIE()] = var_sp;
  }
  return var_sp;
}

VariableSP SymbolFileDWARF::ParseVariableDIE(const SymbolContext &sc,
                                             const DWARFDIE &die,
                                             const lldb::addr_t func_low_pc) {
  if (die.GetDWARF() != this)
    return die.GetDWARF()->ParseVariableDIE(sc, die, func_low_pc);

  if (!die)
    return nullptr;

  const dw_tag_t tag = die.Tag();
  ModuleSP module = GetObjectFile()->GetModule();

  if (tag != DW_TAG_variable && tag != DW_TAG_constant &&
      (tag != DW_TAG_formal_parameter || !sc.function))
    return nullptr;

  DWARFAttributes attributes;
  const size_t num_attributes = die.GetAttributes(attributes);
  const char *name = nullptr;
  const char *mangled = nullptr;
  Declaration decl;
  DWARFFormValue type_die_form;
  DWARFExpressionList location_list(module, DWARFExpression(), die.GetCU());
  bool is_external = false;
  bool is_artificial = false;
  DWARFFormValue const_value_form, location_form;
  Variable::RangeList scope_ranges;

  for (size_t i = 0; i < num_attributes; ++i) {
    dw_attr_t attr = attributes.AttributeAtIndex(i);
    DWARFFormValue form_value;

    if (!attributes.ExtractFormValueAtIndex(i, form_value))
      continue;
    switch (attr) {
    case DW_AT_decl_file:
      decl.SetFile(
          attributes.CompileUnitAtIndex(i)->GetFile(form_value.Unsigned()));
      break;
    case DW_AT_decl_line:
      decl.SetLine(form_value.Unsigned());
      break;
    case DW_AT_decl_column:
      decl.SetColumn(form_value.Unsigned());
      break;
    case DW_AT_name:
      name = form_value.AsCString();
      break;
    case DW_AT_linkage_name:
    case DW_AT_MIPS_linkage_name:
      mangled = form_value.AsCString();
      break;
    case DW_AT_type:
      type_die_form = form_value;
      break;
    case DW_AT_external:
      is_external = form_value.Boolean();
      break;
    case DW_AT_const_value:
      const_value_form = form_value;
      break;
    case DW_AT_location:
      location_form = form_value;
      break;
    case DW_AT_start_scope:
      // TODO: Implement this.
      break;
    case DW_AT_artificial:
      is_artificial = form_value.Boolean();
      break;
    case DW_AT_declaration:
    case DW_AT_description:
    case DW_AT_endianity:
    case DW_AT_segment:
    case DW_AT_specification:
    case DW_AT_visibility:
    default:
    case DW_AT_abstract_origin:
    case DW_AT_sibling:
      break;
    }
  }

  // Prefer DW_AT_location over DW_AT_const_value. Both can be emitted e.g.
  // for static constexpr member variables -- DW_AT_const_value will be
  // present in the class declaration and DW_AT_location in the DIE defining
  // the member.
  bool location_is_const_value_data = false;
  bool has_explicit_location = location_form.IsValid();
  bool use_type_size_for_value = false;
  if (location_form.IsValid()) {
    if (DWARFFormValue::IsBlockForm(location_form.Form())) {
      const DWARFDataExtractor &data = die.GetData();

      uint32_t block_offset = location_form.BlockData() - data.GetDataStart();
      uint32_t block_length = location_form.Unsigned();
      location_list = DWARFExpressionList(
          module, DataExtractor(data, block_offset, block_length), die.GetCU());
    } else {
      DataExtractor data = die.GetCU()->GetLocationData();
      dw_offset_t offset = location_form.Unsigned();
      if (location_form.Form() == DW_FORM_loclistx)
        offset = die.GetCU()->GetLoclistOffset(offset).value_or(-1);
      if (data.ValidOffset(offset)) {
        data = DataExtractor(data, offset, data.GetByteSize() - offset);
        const DWARFUnit *dwarf_cu = location_form.GetUnit();
        if (DWARFExpression::ParseDWARFLocationList(dwarf_cu, data,
                                                    &location_list))
          location_list.SetFuncFileAddress(func_low_pc);
      }
    }
  } else if (const_value_form.IsValid()) {
    location_is_const_value_data = true;
    // The constant value will be either a block, a data value or a
    // string.
    const DWARFDataExtractor &debug_info_data = die.GetData();
    if (DWARFFormValue::IsBlockForm(const_value_form.Form())) {
      // Retrieve the value as a block expression.
      uint32_t block_offset =
          const_value_form.BlockData() - debug_info_data.GetDataStart();
      uint32_t block_length = const_value_form.Unsigned();
      location_list = DWARFExpressionList(
          module, DataExtractor(debug_info_data, block_offset, block_length),
          die.GetCU());
    } else if (DWARFFormValue::IsDataForm(const_value_form.Form())) {
      // Constant value size does not have to match the size of the
      // variable. We will fetch the size of the type after we create
      // it.
      use_type_size_for_value = true;
    } else if (const char *str = const_value_form.AsCString()) {
      uint32_t string_length = strlen(str) + 1;
      location_list = DWARFExpressionList(
          module,
          DataExtractor(str, string_length, die.GetCU()->GetByteOrder(),
                        die.GetCU()->GetAddressByteSize()),
          die.GetCU());
    }
  }

  const DWARFDIE parent_context_die = GetDeclContextDIEContainingDIE(die);
  const DWARFDIE sc_parent_die = GetParentSymbolContextDIE(die);
  const dw_tag_t parent_tag = sc_parent_die.Tag();
  bool is_static_member = (parent_tag == DW_TAG_compile_unit ||
                           parent_tag == DW_TAG_partial_unit) &&
                          (parent_context_die.Tag() == DW_TAG_class_type ||
                           parent_context_die.Tag() == DW_TAG_structure_type);

  ValueType scope = eValueTypeInvalid;
  SymbolContextScope *symbol_context_scope = nullptr;

  bool has_explicit_mangled = mangled != nullptr;
  if (!mangled) {
    // LLDB relies on the mangled name (DW_TAG_linkage_name or
    // DW_AT_MIPS_linkage_name) to generate fully qualified names
    // of global variables with commands like "frame var j". For
    // example, if j were an int variable holding a value 4 and
    // declared in a namespace B which in turn is contained in a
    // namespace A, the command "frame var j" returns
    //   "(int) A::B::j = 4".
    // If the compiler does not emit a linkage name, we should be
    // able to generate a fully qualified name from the
    // declaration context.
    if ((parent_tag == DW_TAG_compile_unit ||
         parent_tag == DW_TAG_partial_unit) &&
        Language::LanguageIsCPlusPlus(GetLanguage(*die.GetCU())))
      mangled =
          GetDWARFDeclContext(die).GetQualifiedNameAsConstString().GetCString();
  }

  if (tag == DW_TAG_formal_parameter)
    scope = eValueTypeVariableArgument;
  else {
    // DWARF doesn't specify if a DW_TAG_variable is a local, global
    // or static variable, so we have to do a little digging:
    // 1) DW_AT_linkage_name implies static lifetime (but may be missing)
    // 2) An empty DW_AT_location is an (optimized-out) static lifetime var.
    // 3) DW_AT_location containing a DW_OP_addr implies static lifetime.
    // Clang likes to combine small global variables into the same symbol
    // with locations like: DW_OP_addr(0x1000), DW_OP_constu(2), DW_OP_plus
    // so we need to look through the whole expression.
    bool is_static_lifetime =
        has_explicit_mangled ||
        (has_explicit_location && !location_list.IsValid());
    // Check if the location has a DW_OP_addr with any address value...
    lldb::addr_t location_DW_OP_addr = LLDB_INVALID_ADDRESS;
    if (!location_is_const_value_data) {
      bool op_error = false;
      const DWARFExpression* location = location_list.GetAlwaysValidExpr();
      if (location)
        location_DW_OP_addr = location->GetLocation_DW_OP_addr(
            location_form.GetUnit(), 0, op_error);
      if (op_error) {
        StreamString strm;
        location->DumpLocation(&strm, eDescriptionLevelFull, nullptr);
        GetObjectFile()->GetModule()->ReportError(
            "0x%8.8x: %s has an invalid location: %s", die.GetOffset(),
            die.GetTagAsCString(), strm.GetData());
      }
      if (location_DW_OP_addr != LLDB_INVALID_ADDRESS)
        is_static_lifetime = true;
    }
    SymbolFileDWARFDebugMap *debug_map_symfile = GetDebugMapSymfile();
    if (debug_map_symfile)
      // Set the module of the expression to the linked module
      // instead of the object file so the relocated address can be
      // found there.
      location_list.SetModule(debug_map_symfile->GetObjectFile()->GetModule());

    if (is_static_lifetime) {
      if (is_external)
        scope = eValueTypeVariableGlobal;
      else
        scope = eValueTypeVariableStatic;

      if (debug_map_symfile) {
        // When leaving the DWARF in the .o files on darwin, when we have a
        // global variable that wasn't initialized, the .o file might not
        // have allocated a virtual address for the global variable. In
        // this case it will have created a symbol for the global variable
        // that is undefined/data and external and the value will be the
        // byte size of the variable. When we do the address map in
        // SymbolFileDWARFDebugMap we rely on having an address, we need to
        // do some magic here so we can get the correct address for our
        // global variable. The address for all of these entries will be
        // zero, and there will be an undefined symbol in this object file,
        // and the executable will have a matching symbol with a good
        // address. So here we dig up the correct address and replace it in
        // the location for the variable, and set the variable's symbol
        // context scope to be that of the main executable so the file
        // address will resolve correctly.
        bool linked_oso_file_addr = false;
        if (is_external && location_DW_OP_addr == 0) {
          // we have a possible uninitialized extern global
          ConstString const_name(mangled ? mangled : name);
          ObjectFile *debug_map_objfile = debug_map_symfile->GetObjectFile();
          if (debug_map_objfile) {
            Symtab *debug_map_symtab = debug_map_objfile->GetSymtab();
            if (debug_map_symtab) {
              Symbol *exe_symbol =
                  debug_map_symtab->FindFirstSymbolWithNameAndType(
                      const_name, eSymbolTypeData, Symtab::eDebugYes,
                      Symtab::eVisibilityExtern);
              if (exe_symbol) {
                if (exe_symbol->ValueIsAddress()) {
                  const addr_t exe_file_addr =
                      exe_symbol->GetAddressRef().GetFileAddress();
                  if (exe_file_addr != LLDB_INVALID_ADDRESS) {
                    DWARFExpression *location =
                        location_list.GetMutableExpressionAtAddress();
                    if (location->Update_DW_OP_addr(exe_file_addr)) {
                      linked_oso_file_addr = true;
                      symbol_context_scope = exe_symbol;
                    }
                  }
                }
              }
            }
          }
        }

        if (!linked_oso_file_addr) {
          // The DW_OP_addr is not zero, but it contains a .o file address
          // which needs to be linked up correctly.
          const lldb::addr_t exe_file_addr =
              debug_map_symfile->LinkOSOFileAddress(this, location_DW_OP_addr);
          if (exe_file_addr != LLDB_INVALID_ADDRESS) {
            // Update the file address for this variable
            DWARFExpression *location =
                location_list.GetMutableExpressionAtAddress();
            location->Update_DW_OP_addr(exe_file_addr);
          } else {
            // Variable didn't make it into the final executable
            return nullptr;
          }
        }
      }
    } else {
      if (location_is_const_value_data &&
          die.GetDIE()->IsGlobalOrStaticScopeVariable())
        scope = eValueTypeVariableStatic;
      else {
        scope = eValueTypeVariableLocal;
        if (debug_map_symfile) {
          // We need to check for TLS addresses that we need to fixup
          if (location_list.ContainsThreadLocalStorage()) {
            location_list.LinkThreadLocalStorage(
                debug_map_symfile->GetObjectFile()->GetModule(),
                [this, debug_map_symfile](
                    lldb::addr_t unlinked_file_addr) -> lldb::addr_t {
                  return debug_map_symfile->LinkOSOFileAddress(
                      this, unlinked_file_addr);
                });
            scope = eValueTypeVariableThreadLocal;
          }
        }
      }
    }
  }

  if (symbol_context_scope == nullptr) {
    switch (parent_tag) {
    case DW_TAG_subprogram:
    case DW_TAG_inlined_subroutine:
    case DW_TAG_lexical_block:
      if (sc.function) {
        symbol_context_scope =
            sc.function->GetBlock(true).FindBlockByID(sc_parent_die.GetID());
        if (symbol_context_scope == nullptr)
          symbol_context_scope = sc.function;
      }
      break;

    default:
      symbol_context_scope = sc.comp_unit;
      break;
    }
  }

  if (!symbol_context_scope) {
    // Not ready to parse this variable yet. It might be a global or static
    // variable that is in a function scope and the function in the symbol
    // context wasn't filled in yet
    return nullptr;
  }

  auto type_sp = std::make_shared<SymbolFileType>(
      *this, GetUID(type_die_form.Reference()));

  if (use_type_size_for_value && type_sp->GetType()) {
    DWARFExpression *location = location_list.GetMutableExpressionAtAddress();
    location->UpdateValue(const_value_form.Unsigned(),
                          type_sp->GetType()->GetByteSize(nullptr).value_or(0),
                          die.GetCU()->GetAddressByteSize());
  }

  return std::make_shared<Variable>(
      die.GetID(), name, mangled, type_sp, scope, symbol_context_scope,
      scope_ranges, &decl, location_list, is_external, is_artificial,
      location_is_const_value_data, is_static_member);
}

DWARFDIE
SymbolFileDWARF::FindBlockContainingSpecification(
    const DIERef &func_die_ref, dw_offset_t spec_block_die_offset) {
  // Give the concrete function die specified by "func_die_offset", find the
  // concrete block whose DW_AT_specification or DW_AT_abstract_origin points
  // to "spec_block_die_offset"
  return FindBlockContainingSpecification(DebugInfo().GetDIE(func_die_ref),
                                          spec_block_die_offset);
}

DWARFDIE
SymbolFileDWARF::FindBlockContainingSpecification(
    const DWARFDIE &die, dw_offset_t spec_block_die_offset) {
  if (die) {
    switch (die.Tag()) {
    case DW_TAG_subprogram:
    case DW_TAG_inlined_subroutine:
    case DW_TAG_lexical_block: {
      if (die.GetReferencedDIE(DW_AT_specification).GetOffset() ==
          spec_block_die_offset)
        return die;

      if (die.GetReferencedDIE(DW_AT_abstract_origin).GetOffset() ==
          spec_block_die_offset)
        return die;
    } break;
    default:
      break;
    }

    // Give the concrete function die specified by "func_die_offset", find the
    // concrete block whose DW_AT_specification or DW_AT_abstract_origin points
    // to "spec_block_die_offset"
    for (DWARFDIE child_die : die.children()) {
      DWARFDIE result_die =
          FindBlockContainingSpecification(child_die, spec_block_die_offset);
      if (result_die)
        return result_die;
    }
  }

  return DWARFDIE();
}

void SymbolFileDWARF::ParseAndAppendGlobalVariable(
    const SymbolContext &sc, const DWARFDIE &die,
    VariableList &cc_variable_list) {
  if (!die)
    return;

  dw_tag_t tag = die.Tag();
  if (tag != DW_TAG_variable && tag != DW_TAG_constant)
    return;

  // Check to see if we have already parsed this variable or constant?
  VariableSP var_sp = GetDIEToVariable()[die.GetDIE()];
  if (var_sp) {
    cc_variable_list.AddVariableIfUnique(var_sp);
    return;
  }

  // We haven't parsed the variable yet, lets do that now. Also, let us include
  // the variable in the relevant compilation unit's variable list, if it
  // exists.
  VariableListSP variable_list_sp;
  DWARFDIE sc_parent_die = GetParentSymbolContextDIE(die);
  dw_tag_t parent_tag = sc_parent_die.Tag();
  switch (parent_tag) {
  case DW_TAG_compile_unit:
  case DW_TAG_partial_unit:
    if (sc.comp_unit != nullptr) {
      variable_list_sp = sc.comp_unit->GetVariableList(false);
    } else {
      GetObjectFile()->GetModule()->ReportError(
          "parent 0x%8.8" PRIx64 " %s with no valid compile unit in "
          "symbol context for 0x%8.8" PRIx64 " %s.\n",
          sc_parent_die.GetID(), sc_parent_die.GetTagAsCString(), die.GetID(),
          die.GetTagAsCString());
      return;
    }
    break;

  default:
    GetObjectFile()->GetModule()->ReportError(
        "didn't find appropriate parent DIE for variable list for "
        "0x%8.8" PRIx64 " %s.\n",
        die.GetID(), die.GetTagAsCString());
    return;
  }

  var_sp = ParseVariableDIECached(sc, die);
  if (!var_sp)
    return;

  cc_variable_list.AddVariableIfUnique(var_sp);
  if (variable_list_sp)
    variable_list_sp->AddVariableIfUnique(var_sp);
}

DIEArray
SymbolFileDWARF::MergeBlockAbstractParameters(const DWARFDIE &block_die,
                                              DIEArray &&variable_dies) {
  // DW_TAG_inline_subroutine objects may omit DW_TAG_formal_parameter in
  // instances of the function when they are unused (i.e., the parameter's
  // location list would be empty). The current DW_TAG_inline_subroutine may
  // refer to another DW_TAG_subprogram that might actually have the definitions
  // of the parameters and we need to include these so they show up in the
  // variables for this function (for example, in a stack trace). Let us try to
  // find the abstract subprogram that might contain the parameter definitions
  // and merge with the concrete parameters.

  // Nothing to merge if the block is not an inlined function.
  if (block_die.Tag() != DW_TAG_inlined_subroutine) {
    return std::move(variable_dies);
  }

  // Nothing to merge if the block does not have abstract parameters.
  DWARFDIE abs_die = block_die.GetReferencedDIE(DW_AT_abstract_origin);
  if (!abs_die || abs_die.Tag() != DW_TAG_subprogram ||
      !abs_die.HasChildren()) {
    return std::move(variable_dies);
  }

  // For each abstract parameter, if we have its concrete counterpart, insert
  // it. Otherwise, insert the abstract parameter.
  DIEArray::iterator concrete_it = variable_dies.begin();
  DWARFDIE abstract_child = abs_die.GetFirstChild();
  DIEArray merged;
  bool did_merge_abstract = false;
  for (; abstract_child; abstract_child = abstract_child.GetSibling()) {
    if (abstract_child.Tag() == DW_TAG_formal_parameter) {
      if (concrete_it == variable_dies.end() ||
          GetDIE(*concrete_it).Tag() != DW_TAG_formal_parameter) {
        // We arrived at the end of the concrete parameter list, so all
        // the remaining abstract parameters must have been omitted.
        // Let us insert them to the merged list here.
        merged.push_back(*abstract_child.GetDIERef());
        did_merge_abstract = true;
        continue;
      }

      DWARFDIE origin_of_concrete =
          GetDIE(*concrete_it).GetReferencedDIE(DW_AT_abstract_origin);
      if (origin_of_concrete == abstract_child) {
        // The current abstract parameter is the origin of the current
        // concrete parameter, just push the concrete parameter.
        merged.push_back(*concrete_it);
        ++concrete_it;
      } else {
        // Otherwise, the parameter must have been omitted from the concrete
        // function, so insert the abstract one.
        merged.push_back(*abstract_child.GetDIERef());
        did_merge_abstract = true;
      }
    }
  }

  // Shortcut if no merging happened.
  if (!did_merge_abstract)
    return std::move(variable_dies);

  // We inserted all the abstract parameters (or their concrete counterparts).
  // Let us insert all the remaining concrete variables to the merged list.
  // During the insertion, let us check there are no remaining concrete
  // formal parameters. If that's the case, then just bailout from the merge -
  // the variable list is malformed.
  for (; concrete_it != variable_dies.end(); ++concrete_it) {
    if (GetDIE(*concrete_it).Tag() == DW_TAG_formal_parameter) {
      return std::move(variable_dies);
    }
    merged.push_back(*concrete_it);
  }
  return merged;
}

size_t SymbolFileDWARF::ParseVariablesInFunctionContext(
    const SymbolContext &sc, const DWARFDIE &die,
    const lldb::addr_t func_low_pc) {
  if (!die || !sc.function)
    return 0;

  DIEArray dummy_block_variables; // The recursive call should not add anything
                                  // to this vector because |die| should be a
                                  // subprogram, so all variables will be added
                                  // to the subprogram's list.
  return ParseVariablesInFunctionContextRecursive(sc, die, func_low_pc,
                                                  dummy_block_variables);
}

// This method parses all the variables in the blocks in the subtree of |die|,
// and inserts them to the variable list for all the nested blocks.
// The uninserted variables for the current block are accumulated in
// |accumulator|.
size_t SymbolFileDWARF::ParseVariablesInFunctionContextRecursive(
    const lldb_private::SymbolContext &sc, const DWARFDIE &die,
    lldb::addr_t func_low_pc, DIEArray &accumulator) {
  size_t vars_added = 0;
  dw_tag_t tag = die.Tag();

  if ((tag == DW_TAG_variable) || (tag == DW_TAG_constant) ||
      (tag == DW_TAG_formal_parameter)) {
    accumulator.push_back(*die.GetDIERef());
  }

  switch (tag) {
  case DW_TAG_subprogram:
  case DW_TAG_inlined_subroutine:
  case DW_TAG_lexical_block: {
    // If we start a new block, compute a new block variable list and recurse.
    Block *block =
        sc.function->GetBlock(/*can_create=*/true).FindBlockByID(die.GetID());
    if (block == nullptr) {
      // This must be a specification or abstract origin with a
      // concrete block counterpart in the current function. We need
      // to find the concrete block so we can correctly add the
      // variable to it.
      const DWARFDIE concrete_block_die = FindBlockContainingSpecification(
          GetDIE(sc.function->GetID()), die.GetOffset());
      if (concrete_block_die)
        block = sc.function->GetBlock(/*can_create=*/true)
                    .FindBlockByID(concrete_block_die.GetID());
    }

    if (block == nullptr)
      return 0;

    const bool can_create = false;
    VariableListSP block_variable_list_sp =
        block->GetBlockVariableList(can_create);
    if (block_variable_list_sp.get() == nullptr) {
      block_variable_list_sp = std::make_shared<VariableList>();
      block->SetVariableList(block_variable_list_sp);
    }

    DIEArray block_variables;
    for (DWARFDIE child = die.GetFirstChild(); child;
         child = child.GetSibling()) {
      vars_added += ParseVariablesInFunctionContextRecursive(
          sc, child, func_low_pc, block_variables);
    }
    block_variables =
        MergeBlockAbstractParameters(die, std::move(block_variables));
    vars_added += PopulateBlockVariableList(*block_variable_list_sp, sc,
                                            block_variables, func_low_pc);
    break;
  }

  default:
    // Recurse to children with the same variable accumulator.
    for (DWARFDIE child = die.GetFirstChild(); child;
         child = child.GetSibling()) {
      vars_added += ParseVariablesInFunctionContextRecursive(
          sc, child, func_low_pc, accumulator);
    }
    break;
  }

  return vars_added;
}

size_t SymbolFileDWARF::PopulateBlockVariableList(
    VariableList &variable_list, const lldb_private::SymbolContext &sc,
    llvm::ArrayRef<DIERef> variable_dies, lldb::addr_t func_low_pc) {
  // Parse the variable DIEs and insert them to the list.
  for (auto &die : variable_dies) {
    if (VariableSP var_sp = ParseVariableDIE(sc, GetDIE(die), func_low_pc)) {
      variable_list.AddVariableIfUnique(var_sp);
    }
  }
  return variable_dies.size();
}

/// Collect call site parameters in a DW_TAG_call_site DIE.
static CallSiteParameterArray
CollectCallSiteParameters(ModuleSP module, DWARFDIE call_site_die) {
  CallSiteParameterArray parameters;
  for (DWARFDIE child : call_site_die.children()) {
    if (child.Tag() != DW_TAG_call_site_parameter &&
        child.Tag() != DW_TAG_GNU_call_site_parameter)
      continue;

    llvm::Optional<DWARFExpressionList> LocationInCallee;
    llvm::Optional<DWARFExpressionList> LocationInCaller;

    DWARFAttributes attributes;
    const size_t num_attributes = child.GetAttributes(attributes);

    // Parse the location at index \p attr_index within this call site parameter
    // DIE, or return None on failure.
    auto parse_simple_location =
        [&](int attr_index) -> llvm::Optional<DWARFExpressionList> {
      DWARFFormValue form_value;
      if (!attributes.ExtractFormValueAtIndex(attr_index, form_value))
        return {};
      if (!DWARFFormValue::IsBlockForm(form_value.Form()))
        return {};
      auto data = child.GetData();
      uint32_t block_offset = form_value.BlockData() - data.GetDataStart();
      uint32_t block_length = form_value.Unsigned();
      return DWARFExpressionList(
          module, DataExtractor(data, block_offset, block_length),
          child.GetCU());
    };

    for (size_t i = 0; i < num_attributes; ++i) {
      dw_attr_t attr = attributes.AttributeAtIndex(i);
      if (attr == DW_AT_location)
        LocationInCallee = parse_simple_location(i);
      if (attr == DW_AT_call_value || attr == DW_AT_GNU_call_site_value)
        LocationInCaller = parse_simple_location(i);
    }

    if (LocationInCallee && LocationInCaller) {
      CallSiteParameter param = {*LocationInCallee, *LocationInCaller};
      parameters.push_back(param);
    }
  }
  return parameters;
}

/// Collect call graph edges present in a function DIE.
std::vector<std::unique_ptr<lldb_private::CallEdge>>
SymbolFileDWARF::CollectCallEdges(ModuleSP module, DWARFDIE function_die) {
  // Check if the function has a supported call site-related attribute.
  // TODO: In the future it may be worthwhile to support call_all_source_calls.
  bool has_call_edges =
      function_die.GetAttributeValueAsUnsigned(DW_AT_call_all_calls, 0) ||
      function_die.GetAttributeValueAsUnsigned(DW_AT_GNU_all_call_sites, 0);
  if (!has_call_edges)
    return {};

  Log *log = GetLog(LLDBLog::Step);
  LLDB_LOG(log, "CollectCallEdges: Found call site info in {0}",
           function_die.GetPubname());

  // Scan the DIE for TAG_call_site entries.
  // TODO: A recursive scan of all blocks in the subprogram is needed in order
  // to be DWARF5-compliant. This may need to be done lazily to be performant.
  // For now, assume that all entries are nested directly under the subprogram
  // (this is the kind of DWARF LLVM produces) and parse them eagerly.
  std::vector<std::unique_ptr<CallEdge>> call_edges;
  for (DWARFDIE child : function_die.children()) {
    if (child.Tag() != DW_TAG_call_site && child.Tag() != DW_TAG_GNU_call_site)
      continue;

    llvm::Optional<DWARFDIE> call_origin;
    llvm::Optional<DWARFExpressionList> call_target;
    addr_t return_pc = LLDB_INVALID_ADDRESS;
    addr_t call_inst_pc = LLDB_INVALID_ADDRESS;
    addr_t low_pc = LLDB_INVALID_ADDRESS;
    bool tail_call = false;

    // Second DW_AT_low_pc may come from DW_TAG_subprogram referenced by
    // DW_TAG_GNU_call_site's DW_AT_abstract_origin overwriting our 'low_pc'.
    // So do not inherit attributes from DW_AT_abstract_origin.
    DWARFAttributes attributes;
    const size_t num_attributes =
        child.GetAttributes(attributes, DWARFDIE::Recurse::no);
    for (size_t i = 0; i < num_attributes; ++i) {
      DWARFFormValue form_value;
      if (!attributes.ExtractFormValueAtIndex(i, form_value)) {
        LLDB_LOG(log, "CollectCallEdges: Could not extract TAG_call_site form");
        break;
      }

      dw_attr_t attr = attributes.AttributeAtIndex(i);

      if (attr == DW_AT_call_tail_call || attr == DW_AT_GNU_tail_call)
        tail_call = form_value.Boolean();

      // Extract DW_AT_call_origin (the call target's DIE).
      if (attr == DW_AT_call_origin || attr == DW_AT_abstract_origin) {
        call_origin = form_value.Reference();
        if (!call_origin->IsValid()) {
          LLDB_LOG(log, "CollectCallEdges: Invalid call origin in {0}",
                   function_die.GetPubname());
          break;
        }
      }

      if (attr == DW_AT_low_pc)
        low_pc = form_value.Address();

      // Extract DW_AT_call_return_pc (the PC the call returns to) if it's
      // available. It should only ever be unavailable for tail call edges, in
      // which case use LLDB_INVALID_ADDRESS.
      if (attr == DW_AT_call_return_pc)
        return_pc = form_value.Address();

      // Extract DW_AT_call_pc (the PC at the call/branch instruction). It
      // should only ever be unavailable for non-tail calls, in which case use
      // LLDB_INVALID_ADDRESS.
      if (attr == DW_AT_call_pc)
        call_inst_pc = form_value.Address();

      // Extract DW_AT_call_target (the location of the address of the indirect
      // call).
      if (attr == DW_AT_call_target || attr == DW_AT_GNU_call_site_target) {
        if (!DWARFFormValue::IsBlockForm(form_value.Form())) {
          LLDB_LOG(log,
                   "CollectCallEdges: AT_call_target does not have block form");
          break;
        }

        auto data = child.GetData();
        uint32_t block_offset = form_value.BlockData() - data.GetDataStart();
        uint32_t block_length = form_value.Unsigned();
        call_target = DWARFExpressionList(
            module, DataExtractor(data, block_offset, block_length),
            child.GetCU());
      }
    }
    if (!call_origin && !call_target) {
      LLDB_LOG(log, "CollectCallEdges: call site without any call target");
      continue;
    }

    addr_t caller_address;
    CallEdge::AddrType caller_address_type;
    if (return_pc != LLDB_INVALID_ADDRESS) {
      caller_address = return_pc;
      caller_address_type = CallEdge::AddrType::AfterCall;
    } else if (low_pc != LLDB_INVALID_ADDRESS) {
      caller_address = low_pc;
      caller_address_type = CallEdge::AddrType::AfterCall;
    } else if (call_inst_pc != LLDB_INVALID_ADDRESS) {
      caller_address = call_inst_pc;
      caller_address_type = CallEdge::AddrType::Call;
    } else {
      LLDB_LOG(log, "CollectCallEdges: No caller address");
      continue;
    }
    // Adjust any PC forms. It needs to be fixed up if the main executable
    // contains a debug map (i.e. pointers to object files), because we need a
    // file address relative to the executable's text section.
    caller_address = FixupAddress(caller_address);

    // Extract call site parameters.
    CallSiteParameterArray parameters =
        CollectCallSiteParameters(module, child);

    std::unique_ptr<CallEdge> edge;
    if (call_origin) {
      LLDB_LOG(log,
               "CollectCallEdges: Found call origin: {0} (retn-PC: {1:x}) "
               "(call-PC: {2:x})",
               call_origin->GetPubname(), return_pc, call_inst_pc);
      edge = std::make_unique<DirectCallEdge>(
          call_origin->GetMangledName(), caller_address_type, caller_address,
          tail_call, std::move(parameters));
    } else {
      if (log) {
        StreamString call_target_desc;
        call_target->GetDescription(&call_target_desc, eDescriptionLevelBrief,
                                    nullptr);
        LLDB_LOG(log, "CollectCallEdges: Found indirect call target: {0}",
                 call_target_desc.GetString());
      }
      edge = std::make_unique<IndirectCallEdge>(
          *call_target, caller_address_type, caller_address, tail_call,
          std::move(parameters));
    }

    if (log && parameters.size()) {
      for (const CallSiteParameter &param : parameters) {
        StreamString callee_loc_desc, caller_loc_desc;
        param.LocationInCallee.GetDescription(&callee_loc_desc,
                                              eDescriptionLevelBrief, nullptr);
        param.LocationInCaller.GetDescription(&caller_loc_desc,
                                              eDescriptionLevelBrief, nullptr);
        LLDB_LOG(log, "CollectCallEdges: \tparam: {0} => {1}",
                 callee_loc_desc.GetString(), caller_loc_desc.GetString());
      }
    }

    call_edges.push_back(std::move(edge));
  }
  return call_edges;
}

std::vector<std::unique_ptr<lldb_private::CallEdge>>
SymbolFileDWARF::ParseCallEdgesInFunction(UserID func_id) {
  // ParseCallEdgesInFunction must be called at the behest of an exclusively
  // locked lldb::Function instance. Storage for parsed call edges is owned by
  // the lldb::Function instance: locking at the SymbolFile level would be too
  // late, because the act of storing results from ParseCallEdgesInFunction
  // would be racy.
  DWARFDIE func_die = GetDIE(func_id.GetID());
  if (func_die.IsValid())
    return CollectCallEdges(GetObjectFile()->GetModule(), func_die);
  return {};
}

void SymbolFileDWARF::Dump(lldb_private::Stream &s) {
  SymbolFileCommon::Dump(s);
  m_index->Dump(s);
}

void SymbolFileDWARF::DumpClangAST(Stream &s) {
  auto ts_or_err = GetTypeSystemForLanguage(eLanguageTypeC_plus_plus);
  if (!ts_or_err)
    return;
  TypeSystemClang *clang =
      llvm::dyn_cast_or_null<TypeSystemClang>(&ts_or_err.get());
  if (!clang)
    return;
  clang->Dump(s.AsRawOstream());
}

SymbolFileDWARFDebugMap *SymbolFileDWARF::GetDebugMapSymfile() {
  if (m_debug_map_symfile == nullptr) {
    lldb::ModuleSP module_sp(m_debug_map_module_wp.lock());
    if (module_sp) {
      m_debug_map_symfile = llvm::cast<SymbolFileDWARFDebugMap>(
          module_sp->GetSymbolFile()->GetBackingSymbolFile());
    }
  }
  return m_debug_map_symfile;
}

const std::shared_ptr<SymbolFileDWARFDwo> &SymbolFileDWARF::GetDwpSymbolFile() {
  llvm::call_once(m_dwp_symfile_once_flag, [this]() {
    ModuleSpec module_spec;
    module_spec.GetFileSpec() = m_objfile_sp->GetFileSpec();
    module_spec.GetSymbolFileSpec() =
        FileSpec(m_objfile_sp->GetModule()->GetFileSpec().GetPath() + ".dwp");

    FileSpecList search_paths = Target::GetDefaultDebugFileSearchPaths();
    FileSpec dwp_filespec =
        Symbols::LocateExecutableSymbolFile(module_spec, search_paths);
    if (FileSystem::Instance().Exists(dwp_filespec)) {
      DataBufferSP dwp_file_data_sp;
      lldb::offset_t dwp_file_data_offset = 0;
      ObjectFileSP dwp_obj_file = ObjectFile::FindPlugin(
          GetObjectFile()->GetModule(), &dwp_filespec, 0,
          FileSystem::Instance().GetByteSize(dwp_filespec), dwp_file_data_sp,
          dwp_file_data_offset);
      if (!dwp_obj_file)
        return;
      m_dwp_symfile =
          std::make_shared<SymbolFileDWARFDwo>(*this, dwp_obj_file, 0x3fffffff);
    }
  });
  return m_dwp_symfile;
}

llvm::Expected<TypeSystem &> SymbolFileDWARF::GetTypeSystem(DWARFUnit &unit) {
  return unit.GetSymbolFileDWARF().GetTypeSystemForLanguage(GetLanguage(unit));
}

DWARFASTParser *SymbolFileDWARF::GetDWARFParser(DWARFUnit &unit) {
  auto type_system_or_err = GetTypeSystem(unit);
  if (auto err = type_system_or_err.takeError()) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Symbols), std::move(err),
                   "Unable to get DWARFASTParser");
    return nullptr;
  }
  return type_system_or_err->GetDWARFParser();
}

CompilerDecl SymbolFileDWARF::GetDecl(const DWARFDIE &die) {
  if (DWARFASTParser *dwarf_ast = GetDWARFParser(*die.GetCU()))
    return dwarf_ast->GetDeclForUIDFromDWARF(die);
  return CompilerDecl();
}

CompilerDeclContext SymbolFileDWARF::GetDeclContext(const DWARFDIE &die) {
  if (DWARFASTParser *dwarf_ast = GetDWARFParser(*die.GetCU()))
    return dwarf_ast->GetDeclContextForUIDFromDWARF(die);
  return CompilerDeclContext();
}

CompilerDeclContext
SymbolFileDWARF::GetContainingDeclContext(const DWARFDIE &die) {
  if (DWARFASTParser *dwarf_ast = GetDWARFParser(*die.GetCU()))
    return dwarf_ast->GetDeclContextContainingUIDFromDWARF(die);
  return CompilerDeclContext();
}

DWARFDeclContext SymbolFileDWARF::GetDWARFDeclContext(const DWARFDIE &die) {
  if (!die.IsValid())
    return {};
  DWARFDeclContext dwarf_decl_ctx =
      die.GetDIE()->GetDWARFDeclContext(die.GetCU());
  dwarf_decl_ctx.SetLanguage(GetLanguage(*die.GetCU()));
  return dwarf_decl_ctx;
}

LanguageType SymbolFileDWARF::LanguageTypeFromDWARF(uint64_t val) {
  // Note: user languages between lo_user and hi_user must be handled
  // explicitly here.
  switch (val) {
  case DW_LANG_Mips_Assembler:
    return eLanguageTypeMipsAssembler;
  case DW_LANG_GOOGLE_RenderScript:
    return eLanguageTypeExtRenderScript;
  default:
    return static_cast<LanguageType>(val);
  }
}

LanguageType SymbolFileDWARF::GetLanguage(DWARFUnit &unit) {
  return LanguageTypeFromDWARF(unit.GetDWARFLanguageType());
}

LanguageType SymbolFileDWARF::GetLanguageFamily(DWARFUnit &unit) {
  auto lang = (llvm::dwarf::SourceLanguage)unit.GetDWARFLanguageType();
  if (llvm::dwarf::isCPlusPlus(lang))
    lang = DW_LANG_C_plus_plus;
  return LanguageTypeFromDWARF(lang);
}

StatsDuration::Duration SymbolFileDWARF::GetDebugInfoIndexTime() {
  if (m_index)
    return m_index->GetIndexTime();
  return {};
}

Status SymbolFileDWARF::GetFrameVariableError(StackFrame &frame) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  CompileUnit *cu = frame.GetSymbolContext(eSymbolContextCompUnit).comp_unit;
  if (!cu)
    return Status();

  DWARFCompileUnit *dwarf_cu = GetDWARFCompileUnit(cu);
  if (!dwarf_cu)
    return Status();

  // Don't return an error for assembly files as they typically don't have
  // varaible information.
  if (dwarf_cu->GetDWARFLanguageType() == DW_LANG_Mips_Assembler)
    return Status();

  // Check if this compile unit has any variable DIEs. If it doesn't then there
  // is not variable information for the entire compile unit.
  if (dwarf_cu->HasAny({DW_TAG_variable, DW_TAG_formal_parameter}))
    return Status();

  return Status("no variable information is available in debug info for this "
                "compile unit");
}
