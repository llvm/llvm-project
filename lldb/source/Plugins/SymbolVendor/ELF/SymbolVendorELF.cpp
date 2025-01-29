//===-- SymbolVendorELF.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolVendorELF.h"

#include <cstring>

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Host/Host.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/Timer.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(SymbolVendorELF)

// SymbolVendorELF constructor
SymbolVendorELF::SymbolVendorELF(const lldb::ModuleSP &module_sp)
    : SymbolVendor(module_sp) {}

void SymbolVendorELF::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance);
}

void SymbolVendorELF::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

llvm::StringRef SymbolVendorELF::GetPluginDescriptionStatic() {
  return "Symbol vendor for ELF that looks for dSYM files that match "
         "executables.";
}

// If this is needed elsewhere, it can be exported/moved.
static bool IsDwpSymbolFile(const lldb::ModuleSP &module_sp,
                            const FileSpec &file_spec) {
  DataBufferSP dwp_file_data_sp;
  lldb::offset_t dwp_file_data_offset = 0;
  // Try to create an ObjectFile from the file_spec.
  ObjectFileSP dwp_obj_file = ObjectFile::FindPlugin(
      module_sp, &file_spec, 0, FileSystem::Instance().GetByteSize(file_spec),
      dwp_file_data_sp, dwp_file_data_offset);
  // The presence of a debug_cu_index section is the key identifying feature of
  // a DWP file. Make sure we don't fill in the section list on dwp_obj_file
  // (by calling GetSectionList(false)) as this function could be called before
  // we may have all the symbol files collected and available.
  return dwp_obj_file && ObjectFileELF::classof(dwp_obj_file.get()) &&
         dwp_obj_file->GetSectionList(false)->FindSectionByType(
             eSectionTypeDWARFDebugCuIndex, false);
}

// CreateInstance
//
// Platforms can register a callback to use when creating symbol vendors to
// allow for complex debug information file setups, and to also allow for
// finding separate debug information files.
SymbolVendor *
SymbolVendorELF::CreateInstance(const lldb::ModuleSP &module_sp,
                                lldb_private::Stream *feedback_strm) {
  if (!module_sp)
    return nullptr;

  ObjectFileELF *obj_file =
      llvm::dyn_cast_or_null<ObjectFileELF>(module_sp->GetObjectFile());
  if (!obj_file)
    return nullptr;

  lldb_private::UUID uuid = obj_file->GetUUID();
  if (!uuid)
    return nullptr;

  // If the main object file already contains debug info, then we are done.
  if (obj_file->GetSectionList()->FindSectionByType(
          lldb::eSectionTypeDWARFDebugInfo, true))
    return nullptr;

  // If the module specified a filespec, use that.
  FileSpec fspec = module_sp->GetSymbolFileFileSpec();
  // Otherwise, try gnu_debuglink, if one exists.
  if (!fspec)
    fspec = obj_file->GetDebugLink().value_or(FileSpec());

  LLDB_SCOPED_TIMERF("SymbolVendorELF::CreateInstance (module = %s)",
                     module_sp->GetFileSpec().GetPath().c_str());

  ModuleSpec module_spec;

  module_spec.GetFileSpec() = obj_file->GetFileSpec();
  FileSystem::Instance().Resolve(module_spec.GetFileSpec());
  module_spec.GetSymbolFileSpec() = fspec;
  module_spec.GetUUID() = uuid;
  FileSpecList search_paths = Target::GetDefaultDebugFileSearchPaths();
  FileSpec dsym_fspec =
      PluginManager::LocateExecutableSymbolFile(module_spec, search_paths);
  if (!dsym_fspec || IsDwpSymbolFile(module_sp, dsym_fspec)) {
    // If we have a stripped binary or if we have a DWP file, SymbolLocator
    // plugins may be able to give us an unstripped binary or an
    // 'only-keep-debug' stripped file.
    ModuleSpec unstripped_spec =
        PluginManager::LocateExecutableObjectFile(module_spec);
    if (!unstripped_spec)
      return nullptr;
    // The default SymbolLocator plugin returns the original binary if no other
    // plugin finds something better.
    if (unstripped_spec.GetFileSpec() == module_spec.GetFileSpec())
      return nullptr;
    dsym_fspec = unstripped_spec.GetFileSpec();
  }

  DataBufferSP dsym_file_data_sp;
  lldb::offset_t dsym_file_data_offset = 0;
  ObjectFileSP dsym_objfile_sp = ObjectFile::FindPlugin(
      module_sp, &dsym_fspec, 0, FileSystem::Instance().GetByteSize(dsym_fspec),
      dsym_file_data_sp, dsym_file_data_offset);
  if (!dsym_objfile_sp)
    return nullptr;

  // This objfile is for debugging purposes. Sadly, ObjectFileELF won't
  // be able to figure this out consistently as the symbol file may not
  // have stripped the code sections, etc.
  dsym_objfile_sp->SetType(ObjectFile::eTypeDebugInfo);

  SymbolVendorELF *symbol_vendor = new SymbolVendorELF(module_sp);

  // Get the module unified section list and add our debug sections to
  // that.
  SectionList *module_section_list = module_sp->GetSectionList();
  SectionList *objfile_section_list = dsym_objfile_sp->GetSectionList();

  if (!module_section_list || !objfile_section_list)
    return nullptr;

  static const SectionType g_sections[] = {
      eSectionTypeDWARFDebugAbbrev,     eSectionTypeDWARFDebugAddr,
      eSectionTypeDWARFDebugAranges,    eSectionTypeDWARFDebugCuIndex,
      eSectionTypeDWARFDebugFrame,      eSectionTypeDWARFDebugInfo,
      eSectionTypeDWARFDebugLine,       eSectionTypeDWARFDebugLineStr,
      eSectionTypeDWARFDebugLoc,        eSectionTypeDWARFDebugLocLists,
      eSectionTypeDWARFDebugMacInfo,    eSectionTypeDWARFDebugMacro,
      eSectionTypeDWARFDebugNames,      eSectionTypeDWARFDebugPubNames,
      eSectionTypeDWARFDebugPubTypes,   eSectionTypeDWARFDebugRanges,
      eSectionTypeDWARFDebugRngLists,   eSectionTypeDWARFDebugStr,
      eSectionTypeDWARFDebugStrOffsets, eSectionTypeDWARFDebugTypes,
      eSectionTypeELFSymbolTable,       eSectionTypeDWARFGNUDebugAltLink,
  };
  for (SectionType section_type : g_sections) {
    if (SectionSP section_sp =
            objfile_section_list->FindSectionByType(section_type, true)) {
      if (SectionSP module_section_sp =
              module_section_list->FindSectionByType(section_type, true))
        module_section_list->ReplaceSection(module_section_sp->GetID(),
                                            section_sp);
      else
        module_section_list->AddSection(section_sp);
    }
  }

  symbol_vendor->AddSymbolFileRepresentation(dsym_objfile_sp);
  return symbol_vendor;
}
