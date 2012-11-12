//===-- SymbolFileDWARF.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SymbolFileDWARF.h"

// Other libraries and framework includes
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Sema/DeclSpec.h"

#include "llvm/Support/Casting.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/Timer.h"
#include "lldb/Core/Value.h"

#include "lldb/Host/Host.h"

#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/ClangExternalASTSourceCallbacks.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/LineTable.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Symbol/VariableList.h"

#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/CPPLanguageRuntime.h"

#include "DWARFCompileUnit.h"
#include "DWARFDebugAbbrev.h"
#include "DWARFDebugAranges.h"
#include "DWARFDebugInfo.h"
#include "DWARFDebugInfoEntry.h"
#include "DWARFDebugLine.h"
#include "DWARFDebugPubnames.h"
#include "DWARFDebugRanges.h"
#include "DWARFDeclContext.h"
#include "DWARFDIECollection.h"
#include "DWARFFormValue.h"
#include "DWARFLocationList.h"
#include "LogChannelDWARF.h"
#include "SymbolFileDWARFDebugMap.h"

#include <map>

//#define ENABLE_DEBUG_PRINTF // COMMENT OUT THIS LINE PRIOR TO CHECKIN

#ifdef ENABLE_DEBUG_PRINTF
#include <stdio.h>
#define DEBUG_PRINTF(fmt, ...) printf(fmt, ## __VA_ARGS__)
#else
#define DEBUG_PRINTF(fmt, ...)
#endif

#define DIE_IS_BEING_PARSED ((lldb_private::Type*)1)

using namespace lldb;
using namespace lldb_private;

//static inline bool
//child_requires_parent_class_union_or_struct_to_be_completed (dw_tag_t tag)
//{
//    switch (tag)
//    {
//    default:
//        break;
//    case DW_TAG_subprogram:
//    case DW_TAG_inlined_subroutine:
//    case DW_TAG_class_type:
//    case DW_TAG_structure_type:
//    case DW_TAG_union_type:
//        return true;
//    }
//    return false;
//}
//
static AccessType
DW_ACCESS_to_AccessType (uint32_t dwarf_accessibility)
{
    switch (dwarf_accessibility)
    {
        case DW_ACCESS_public:      return eAccessPublic;
        case DW_ACCESS_private:     return eAccessPrivate;
        case DW_ACCESS_protected:   return eAccessProtected;
        default:                    break;
    }
    return eAccessNone;
}

#if defined(LLDB_CONFIGURATION_DEBUG) or defined(LLDB_CONFIGURATION_RELEASE)

class DIEStack
{
public:
    
    void Push (DWARFCompileUnit *cu, const DWARFDebugInfoEntry *die)
    {
        m_dies.push_back (DIEInfo(cu, die));
    }

    
    void LogDIEs (Log *log, SymbolFileDWARF *dwarf)
    {
        StreamString log_strm;
        const size_t n = m_dies.size();
        log_strm.Printf("DIEStack[%llu]:\n", (uint64_t)n);
        for (size_t i=0; i<n; i++)
        {
            DWARFCompileUnit *cu = m_dies[i].cu;
            const DWARFDebugInfoEntry *die = m_dies[i].die;
            std::string qualified_name;
            die->GetQualifiedName(dwarf, cu, qualified_name);
            log_strm.Printf ("[%llu] 0x%8.8x: %s name='%s'\n", 
                             (uint64_t)i,
                             die->GetOffset(), 
                             DW_TAG_value_to_name(die->Tag()), 
                             qualified_name.c_str());
        }
        log->PutCString(log_strm.GetData());
    }
    void Pop ()
    {
        m_dies.pop_back();
    }
    
    class ScopedPopper
    {
    public:
        ScopedPopper (DIEStack &die_stack) :
            m_die_stack (die_stack),
            m_valid (false)
        {
        }
        
        void
        Push (DWARFCompileUnit *cu, const DWARFDebugInfoEntry *die)
        {
            m_valid = true;
            m_die_stack.Push (cu, die);
        }
        
        ~ScopedPopper ()
        {
            if (m_valid)
                m_die_stack.Pop();
        }
        
        
        
    protected:
        DIEStack &m_die_stack;
        bool m_valid;
    };

protected:
    struct DIEInfo {
        DIEInfo (DWARFCompileUnit *c, const DWARFDebugInfoEntry *d) :
            cu(c),
            die(d)
        {
        }
        DWARFCompileUnit *cu;
        const DWARFDebugInfoEntry *die;
    };
    typedef std::vector<DIEInfo> Stack;
    Stack m_dies;
};
#endif

void
SymbolFileDWARF::Initialize()
{
    LogChannelDWARF::Initialize();
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   GetPluginDescriptionStatic(),
                                   CreateInstance);
}

void
SymbolFileDWARF::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
    LogChannelDWARF::Initialize();
}


const char *
SymbolFileDWARF::GetPluginNameStatic()
{
    return "dwarf";
}

const char *
SymbolFileDWARF::GetPluginDescriptionStatic()
{
    return "DWARF and DWARF3 debug symbol file reader.";
}


SymbolFile*
SymbolFileDWARF::CreateInstance (ObjectFile* obj_file)
{
    return new SymbolFileDWARF(obj_file);
}

TypeList *          
SymbolFileDWARF::GetTypeList ()
{
    if (GetDebugMapSymfile ())
        return m_debug_map_symfile->GetTypeList();
    return m_obj_file->GetModule()->GetTypeList();

}

//----------------------------------------------------------------------
// Gets the first parent that is a lexical block, function or inlined
// subroutine, or compile unit.
//----------------------------------------------------------------------
static const DWARFDebugInfoEntry *
GetParentSymbolContextDIE(const DWARFDebugInfoEntry *child_die)
{
    const DWARFDebugInfoEntry *die;
    for (die = child_die->GetParent(); die != NULL; die = die->GetParent())
    {
        dw_tag_t tag = die->Tag();

        switch (tag)
        {
        case DW_TAG_compile_unit:
        case DW_TAG_subprogram:
        case DW_TAG_inlined_subroutine:
        case DW_TAG_lexical_block:
            return die;
        }
    }
    return NULL;
}


SymbolFileDWARF::SymbolFileDWARF(ObjectFile* objfile) :
    SymbolFile (objfile),
    UserID (0),  // Used by SymbolFileDWARFDebugMap to when this class parses .o files to contain the .o file index/ID
    m_debug_map_module_wp (),
    m_debug_map_symfile (NULL),
    m_clang_tu_decl (NULL),
    m_flags(),
    m_data_debug_abbrev (),
    m_data_debug_aranges (),
    m_data_debug_frame (),
    m_data_debug_info (),
    m_data_debug_line (),
    m_data_debug_loc (),
    m_data_debug_ranges (),
    m_data_debug_str (),
    m_data_apple_names (),
    m_data_apple_types (),
    m_data_apple_namespaces (),
    m_abbr(),
    m_info(),
    m_line(),
    m_apple_names_ap (),
    m_apple_types_ap (),
    m_apple_namespaces_ap (),
    m_apple_objc_ap (),
    m_function_basename_index(),
    m_function_fullname_index(),
    m_function_method_index(),
    m_function_selector_index(),
    m_objc_class_selectors_index(),
    m_global_index(),
    m_type_index(),
    m_namespace_index(),
    m_indexed (false),
    m_is_external_ast_source (false),
    m_using_apple_tables (false),
    m_supports_DW_AT_APPLE_objc_complete_type (eLazyBoolCalculate),
    m_ranges(),
    m_unique_ast_type_map ()
{
}

SymbolFileDWARF::~SymbolFileDWARF()
{
    if (m_is_external_ast_source)
    {
        ModuleSP module_sp (m_obj_file->GetModule());
        if (module_sp)
            module_sp->GetClangASTContext().RemoveExternalSource ();
    }
}

static const ConstString &
GetDWARFMachOSegmentName ()
{
    static ConstString g_dwarf_section_name ("__DWARF");
    return g_dwarf_section_name;
}

UniqueDWARFASTTypeMap &
SymbolFileDWARF::GetUniqueDWARFASTTypeMap ()
{
    if (GetDebugMapSymfile ())
        return m_debug_map_symfile->GetUniqueDWARFASTTypeMap ();
    return m_unique_ast_type_map;
}

ClangASTContext &       
SymbolFileDWARF::GetClangASTContext ()
{
    if (GetDebugMapSymfile ())
        return m_debug_map_symfile->GetClangASTContext ();

    ClangASTContext &ast = m_obj_file->GetModule()->GetClangASTContext();
    if (!m_is_external_ast_source)
    {
        m_is_external_ast_source = true;
        llvm::OwningPtr<clang::ExternalASTSource> ast_source_ap (
            new ClangExternalASTSourceCallbacks (SymbolFileDWARF::CompleteTagDecl,
                                                 SymbolFileDWARF::CompleteObjCInterfaceDecl,
                                                 SymbolFileDWARF::FindExternalVisibleDeclsByName,
                                                 SymbolFileDWARF::LayoutRecordType,
                                                 this));
        ast.SetExternalSource (ast_source_ap);
    }
    return ast;
}

void
SymbolFileDWARF::InitializeObject()
{
    // Install our external AST source callbacks so we can complete Clang types.
    ModuleSP module_sp (m_obj_file->GetModule());
    if (module_sp)
    {
        const SectionList *section_list = m_obj_file->GetSectionList();

        const Section* section = section_list->FindSectionByName(GetDWARFMachOSegmentName ()).get();

        // Memory map the DWARF mach-o segment so we have everything mmap'ed
        // to keep our heap memory usage down.
        if (section)
            m_obj_file->MemoryMapSectionData(section, m_dwarf_data);
    }
    get_apple_names_data();
    if (m_data_apple_names.GetByteSize() > 0)
    {
        m_apple_names_ap.reset (new DWARFMappedHash::MemoryTable (m_data_apple_names, get_debug_str_data(), ".apple_names"));
        if (m_apple_names_ap->IsValid())
            m_using_apple_tables = true;
        else
            m_apple_names_ap.reset();
    }
    get_apple_types_data();
    if (m_data_apple_types.GetByteSize() > 0)
    {
        m_apple_types_ap.reset (new DWARFMappedHash::MemoryTable (m_data_apple_types, get_debug_str_data(), ".apple_types"));
        if (m_apple_types_ap->IsValid())
            m_using_apple_tables = true;
        else
            m_apple_types_ap.reset();
    }

    get_apple_namespaces_data();
    if (m_data_apple_namespaces.GetByteSize() > 0)
    {
        m_apple_namespaces_ap.reset (new DWARFMappedHash::MemoryTable (m_data_apple_namespaces, get_debug_str_data(), ".apple_namespaces"));
        if (m_apple_namespaces_ap->IsValid())
            m_using_apple_tables = true;
        else
            m_apple_namespaces_ap.reset();
    }

    get_apple_objc_data();
    if (m_data_apple_objc.GetByteSize() > 0)
    {
        m_apple_objc_ap.reset (new DWARFMappedHash::MemoryTable (m_data_apple_objc, get_debug_str_data(), ".apple_objc"));
        if (m_apple_objc_ap->IsValid())
            m_using_apple_tables = true;
        else
            m_apple_objc_ap.reset();
    }
}

bool
SymbolFileDWARF::SupportedVersion(uint16_t version)
{
    return version == 2 || version == 3;
}

uint32_t
SymbolFileDWARF::CalculateAbilities ()
{
    uint32_t abilities = 0;
    if (m_obj_file != NULL)
    {
        const Section* section = NULL;
        const SectionList *section_list = m_obj_file->GetSectionList();
        if (section_list == NULL)
            return 0;

        uint64_t debug_abbrev_file_size = 0;
        uint64_t debug_info_file_size = 0;
        uint64_t debug_line_file_size = 0;

        section = section_list->FindSectionByName(GetDWARFMachOSegmentName ()).get();
        
        if (section)
            section_list = &section->GetChildren ();
        
        section = section_list->FindSectionByType (eSectionTypeDWARFDebugInfo, true).get();
        if (section != NULL)
        {
            debug_info_file_size = section->GetFileSize();

            section = section_list->FindSectionByType (eSectionTypeDWARFDebugAbbrev, true).get();
            if (section)
                debug_abbrev_file_size = section->GetFileSize();
            else
                m_flags.Set (flagsGotDebugAbbrevData);

            section = section_list->FindSectionByType (eSectionTypeDWARFDebugAranges, true).get();
            if (!section)
                m_flags.Set (flagsGotDebugArangesData);

            section = section_list->FindSectionByType (eSectionTypeDWARFDebugFrame, true).get();
            if (!section)
                m_flags.Set (flagsGotDebugFrameData);

            section = section_list->FindSectionByType (eSectionTypeDWARFDebugLine, true).get();
            if (section)
                debug_line_file_size = section->GetFileSize();
            else
                m_flags.Set (flagsGotDebugLineData);

            section = section_list->FindSectionByType (eSectionTypeDWARFDebugLoc, true).get();
            if (!section)
                m_flags.Set (flagsGotDebugLocData);

            section = section_list->FindSectionByType (eSectionTypeDWARFDebugMacInfo, true).get();
            if (!section)
                m_flags.Set (flagsGotDebugMacInfoData);

            section = section_list->FindSectionByType (eSectionTypeDWARFDebugPubNames, true).get();
            if (!section)
                m_flags.Set (flagsGotDebugPubNamesData);

            section = section_list->FindSectionByType (eSectionTypeDWARFDebugPubTypes, true).get();
            if (!section)
                m_flags.Set (flagsGotDebugPubTypesData);

            section = section_list->FindSectionByType (eSectionTypeDWARFDebugRanges, true).get();
            if (!section)
                m_flags.Set (flagsGotDebugRangesData);

            section = section_list->FindSectionByType (eSectionTypeDWARFDebugStr, true).get();
            if (!section)
                m_flags.Set (flagsGotDebugStrData);
        }
        else
        {
            const char *symfile_dir_cstr = m_obj_file->GetFileSpec().GetDirectory().GetCString();
            if (symfile_dir_cstr)
            {
                if (strcasestr(symfile_dir_cstr, ".dsym"))
                {
                    if (m_obj_file->GetType() == ObjectFile::eTypeDebugInfo)
                    {
                        // We have a dSYM file that didn't have a any debug info.
                        // If the string table has a size of 1, then it was made from
                        // an executable with no debug info, or from an executable that
                        // was stripped.
                        section = section_list->FindSectionByType (eSectionTypeDWARFDebugStr, true).get();
                        if (section && section->GetFileSize() == 1)
                        {
                            m_obj_file->GetModule()->ReportWarning ("empty dSYM file detected, dSYM was created with an executable with no debug info.");
                        }
                    }
                }
            }
        }

        if (debug_abbrev_file_size > 0 && debug_info_file_size > 0)
            abilities |= CompileUnits | Functions | Blocks | GlobalVariables | LocalVariables | VariableTypes;

        if (debug_line_file_size > 0)
            abilities |= LineTables;
    }
    return abilities;
}

const DataExtractor&
SymbolFileDWARF::GetCachedSectionData (uint32_t got_flag, SectionType sect_type, DataExtractor &data)
{
    if (m_flags.IsClear (got_flag))
    {
        m_flags.Set (got_flag);
        const SectionList *section_list = m_obj_file->GetSectionList();
        if (section_list)
        {
            SectionSP section_sp (section_list->FindSectionByType(sect_type, true));
            if (section_sp)
            {
                // See if we memory mapped the DWARF segment?
                if (m_dwarf_data.GetByteSize())
                {
                    data.SetData(m_dwarf_data, section_sp->GetOffset (), section_sp->GetFileSize());
                }
                else
                {
                    if (m_obj_file->ReadSectionData (section_sp.get(), data) == 0)
                        data.Clear();
                }
            }
        }
    }
    return data;
}

const DataExtractor&
SymbolFileDWARF::get_debug_abbrev_data()
{
    return GetCachedSectionData (flagsGotDebugAbbrevData, eSectionTypeDWARFDebugAbbrev, m_data_debug_abbrev);
}

const DataExtractor&
SymbolFileDWARF::get_debug_aranges_data()
{
    return GetCachedSectionData (flagsGotDebugArangesData, eSectionTypeDWARFDebugAranges, m_data_debug_aranges);
}

const DataExtractor&
SymbolFileDWARF::get_debug_frame_data()
{
    return GetCachedSectionData (flagsGotDebugFrameData, eSectionTypeDWARFDebugFrame, m_data_debug_frame);
}

const DataExtractor&
SymbolFileDWARF::get_debug_info_data()
{
    return GetCachedSectionData (flagsGotDebugInfoData, eSectionTypeDWARFDebugInfo, m_data_debug_info);
}

const DataExtractor&
SymbolFileDWARF::get_debug_line_data()
{
    return GetCachedSectionData (flagsGotDebugLineData, eSectionTypeDWARFDebugLine, m_data_debug_line);
}

const DataExtractor&
SymbolFileDWARF::get_debug_loc_data()
{
    return GetCachedSectionData (flagsGotDebugLocData, eSectionTypeDWARFDebugLoc, m_data_debug_loc);
}

const DataExtractor&
SymbolFileDWARF::get_debug_ranges_data()
{
    return GetCachedSectionData (flagsGotDebugRangesData, eSectionTypeDWARFDebugRanges, m_data_debug_ranges);
}

const DataExtractor&
SymbolFileDWARF::get_debug_str_data()
{
    return GetCachedSectionData (flagsGotDebugStrData, eSectionTypeDWARFDebugStr, m_data_debug_str);
}

const DataExtractor&
SymbolFileDWARF::get_apple_names_data()
{
    return GetCachedSectionData (flagsGotAppleNamesData, eSectionTypeDWARFAppleNames, m_data_apple_names);
}

const DataExtractor&
SymbolFileDWARF::get_apple_types_data()
{
    return GetCachedSectionData (flagsGotAppleTypesData, eSectionTypeDWARFAppleTypes, m_data_apple_types);
}

const DataExtractor&
SymbolFileDWARF::get_apple_namespaces_data()
{
    return GetCachedSectionData (flagsGotAppleNamespacesData, eSectionTypeDWARFAppleNamespaces, m_data_apple_namespaces);
}

const DataExtractor&
SymbolFileDWARF::get_apple_objc_data()
{
    return GetCachedSectionData (flagsGotAppleObjCData, eSectionTypeDWARFAppleObjC, m_data_apple_objc);
}


DWARFDebugAbbrev*
SymbolFileDWARF::DebugAbbrev()
{
    if (m_abbr.get() == NULL)
    {
        const DataExtractor &debug_abbrev_data = get_debug_abbrev_data();
        if (debug_abbrev_data.GetByteSize() > 0)
        {
            m_abbr.reset(new DWARFDebugAbbrev());
            if (m_abbr.get())
                m_abbr->Parse(debug_abbrev_data);
        }
    }
    return m_abbr.get();
}

const DWARFDebugAbbrev*
SymbolFileDWARF::DebugAbbrev() const
{
    return m_abbr.get();
}


DWARFDebugInfo*
SymbolFileDWARF::DebugInfo()
{
    if (m_info.get() == NULL)
    {
        Timer scoped_timer(__PRETTY_FUNCTION__, "%s this = %p", __PRETTY_FUNCTION__, this);
        if (get_debug_info_data().GetByteSize() > 0)
        {
            m_info.reset(new DWARFDebugInfo());
            if (m_info.get())
            {
                m_info->SetDwarfData(this);
            }
        }
    }
    return m_info.get();
}

const DWARFDebugInfo*
SymbolFileDWARF::DebugInfo() const
{
    return m_info.get();
}

DWARFCompileUnit*
SymbolFileDWARF::GetDWARFCompileUnit(lldb_private::CompileUnit *comp_unit)
{
    DWARFDebugInfo* info = DebugInfo();
    if (info)
    {
        if (GetDebugMapSymfile ())
        {
            // The debug map symbol file made the compile units for this DWARF
            // file which is .o file with DWARF in it, and we should have
            // only 1 compile unit which is at offset zero in the DWARF.
            // TODO: modify to support LTO .o files where each .o file might
            // have multiple DW_TAG_compile_unit tags.
            return info->GetCompileUnit(0).get();
        }
        else
        {
            // Just a normal DWARF file whose user ID for the compile unit is
            // the DWARF offset itself
            return info->GetCompileUnit((dw_offset_t)comp_unit->GetID()).get();
        }
    }
    return NULL;
}


DWARFDebugRanges*
SymbolFileDWARF::DebugRanges()
{
    if (m_ranges.get() == NULL)
    {
        Timer scoped_timer(__PRETTY_FUNCTION__, "%s this = %p", __PRETTY_FUNCTION__, this);
        if (get_debug_ranges_data().GetByteSize() > 0)
        {
            m_ranges.reset(new DWARFDebugRanges());
            if (m_ranges.get())
                m_ranges->Extract(this);
        }
    }
    return m_ranges.get();
}

const DWARFDebugRanges*
SymbolFileDWARF::DebugRanges() const
{
    return m_ranges.get();
}

lldb::CompUnitSP
SymbolFileDWARF::ParseCompileUnit (DWARFCompileUnit* dwarf_cu, uint32_t cu_idx)
{
    CompUnitSP cu_sp;
    if (dwarf_cu)
    {
        CompileUnit *comp_unit = (CompileUnit*)dwarf_cu->GetUserData();
        if (comp_unit)
        {
            // We already parsed this compile unit, had out a shared pointer to it
            cu_sp = comp_unit->shared_from_this();
        }
        else
        {
            if (GetDebugMapSymfile ())
            {
                // Let the debug map create the compile unit
                cu_sp = m_debug_map_symfile->GetCompileUnit(this);
                dwarf_cu->SetUserData(cu_sp.get());
            }
            else
            {
                ModuleSP module_sp (m_obj_file->GetModule());
                if (module_sp)
                {
                    const DWARFDebugInfoEntry * cu_die = dwarf_cu->GetCompileUnitDIEOnly ();
                    if (cu_die)
                    {
                        const char * cu_die_name = cu_die->GetName(this, dwarf_cu);
                        const char * cu_comp_dir = cu_die->GetAttributeValueAsString(this, dwarf_cu, DW_AT_comp_dir, NULL);
                        LanguageType cu_language = (LanguageType)cu_die->GetAttributeValueAsUnsigned(this, dwarf_cu, DW_AT_language, 0);
                        if (cu_die_name)
                        {
                            std::string ramapped_file;
                            FileSpec cu_file_spec;

                            if (cu_die_name[0] == '/' || cu_comp_dir == NULL || cu_comp_dir[0] == '\0')
                            {
                                // If we have a full path to the compile unit, we don't need to resolve
                                // the file.  This can be expensive e.g. when the source files are NFS mounted.
                                if (module_sp->RemapSourceFile(cu_die_name, ramapped_file))
                                    cu_file_spec.SetFile (ramapped_file.c_str(), false);
                                else
                                    cu_file_spec.SetFile (cu_die_name, false);
                            }
                            else
                            {
                                std::string fullpath(cu_comp_dir);
                                if (*fullpath.rbegin() != '/')
                                    fullpath += '/';
                                fullpath += cu_die_name;
                                if (module_sp->RemapSourceFile (fullpath.c_str(), ramapped_file))
                                    cu_file_spec.SetFile (ramapped_file.c_str(), false);
                                else
                                    cu_file_spec.SetFile (fullpath.c_str(), false);
                            }

                            cu_sp.reset(new CompileUnit (module_sp,
                                                         dwarf_cu,
                                                         cu_file_spec, 
                                                         MakeUserID(dwarf_cu->GetOffset()),
                                                         cu_language));
                            if (cu_sp)
                            {
                                dwarf_cu->SetUserData(cu_sp.get());
                                
                                // Figure out the compile unit index if we weren't given one
                                if (cu_idx == UINT32_MAX)
                                    DebugInfo()->GetCompileUnit(dwarf_cu->GetOffset(), &cu_idx);
                                
                                m_obj_file->GetModule()->GetSymbolVendor()->SetCompileUnitAtIndex(cu_idx, cu_sp);
                            }
                        }
                    }
                }
            }
        }
    }
    return cu_sp;
}

uint32_t
SymbolFileDWARF::GetNumCompileUnits()
{
    DWARFDebugInfo* info = DebugInfo();
    if (info)
        return info->GetNumCompileUnits();
    return 0;
}

CompUnitSP
SymbolFileDWARF::ParseCompileUnitAtIndex(uint32_t cu_idx)
{
    CompUnitSP cu_sp;
    DWARFDebugInfo* info = DebugInfo();
    if (info)
    {
        DWARFCompileUnit* dwarf_cu = info->GetCompileUnitAtIndex(cu_idx);
        if (dwarf_cu)
            cu_sp = ParseCompileUnit(dwarf_cu, cu_idx);
    }
    return cu_sp;
}

static void
AddRangesToBlock (Block& block,
                  DWARFDebugRanges::RangeList& ranges,
                  addr_t block_base_addr)
{
    const size_t num_ranges = ranges.GetSize();
    for (size_t i = 0; i<num_ranges; ++i)
    {
        const DWARFDebugRanges::Range &range = ranges.GetEntryRef (i);
        const addr_t range_base = range.GetRangeBase();
        assert (range_base >= block_base_addr);
        block.AddRange(Block::Range (range_base - block_base_addr, range.GetByteSize()));;
    }
    block.FinalizeRanges ();
}


Function *
SymbolFileDWARF::ParseCompileUnitFunction (const SymbolContext& sc, DWARFCompileUnit* dwarf_cu, const DWARFDebugInfoEntry *die)
{
    DWARFDebugRanges::RangeList func_ranges;
    const char *name = NULL;
    const char *mangled = NULL;
    int decl_file = 0;
    int decl_line = 0;
    int decl_column = 0;
    int call_file = 0;
    int call_line = 0;
    int call_column = 0;
    DWARFExpression frame_base;

    assert (die->Tag() == DW_TAG_subprogram);
    
    if (die->Tag() != DW_TAG_subprogram)
        return NULL;

    if (die->GetDIENamesAndRanges (this, 
                                   dwarf_cu, 
                                   name, 
                                   mangled, 
                                   func_ranges, 
                                   decl_file, 
                                   decl_line, 
                                   decl_column, 
                                   call_file, 
                                   call_line, 
                                   call_column, 
                                   &frame_base))
    {
        // Union of all ranges in the function DIE (if the function is discontiguous)
        AddressRange func_range;
        lldb::addr_t lowest_func_addr = func_ranges.GetMinRangeBase (0);
        lldb::addr_t highest_func_addr = func_ranges.GetMaxRangeEnd (0);
        if (lowest_func_addr != LLDB_INVALID_ADDRESS && lowest_func_addr <= highest_func_addr)
        {
            func_range.GetBaseAddress().ResolveAddressUsingFileSections (lowest_func_addr, m_obj_file->GetSectionList());
            if (func_range.GetBaseAddress().IsValid())
                func_range.SetByteSize(highest_func_addr - lowest_func_addr);
        }

        if (func_range.GetBaseAddress().IsValid())
        {
            Mangled func_name;
            if (mangled)
                func_name.SetValue(ConstString(mangled), true);
            else if (name)
                func_name.SetValue(ConstString(name), false);

            FunctionSP func_sp;
            std::auto_ptr<Declaration> decl_ap;
            if (decl_file != 0 || decl_line != 0 || decl_column != 0)
                decl_ap.reset(new Declaration (sc.comp_unit->GetSupportFiles().GetFileSpecAtIndex(decl_file), 
                                               decl_line, 
                                               decl_column));

            // Supply the type _only_ if it has already been parsed
            Type *func_type = m_die_to_type.lookup (die);

            assert(func_type == NULL || func_type != DIE_IS_BEING_PARSED);

            func_range.GetBaseAddress().ResolveLinkedAddress();

            const user_id_t func_user_id = MakeUserID(die->GetOffset());
            func_sp.reset(new Function (sc.comp_unit,
                                        func_user_id,       // UserID is the DIE offset
                                        func_user_id,
                                        func_name,
                                        func_type,
                                        func_range));           // first address range

            if (func_sp.get() != NULL)
            {
                if (frame_base.IsValid())
                    func_sp->GetFrameBaseExpression() = frame_base;
                sc.comp_unit->AddFunction(func_sp);
                return func_sp.get();
            }
        }
    }
    return NULL;
}

lldb::LanguageType
SymbolFileDWARF::ParseCompileUnitLanguage (const SymbolContext& sc)
{
    assert (sc.comp_unit);
    DWARFCompileUnit* dwarf_cu = GetDWARFCompileUnit(sc.comp_unit);
    if (dwarf_cu)
    {
        const DWARFDebugInfoEntry *die = dwarf_cu->GetCompileUnitDIEOnly();
        if (die)
        {
            const uint32_t language = die->GetAttributeValueAsUnsigned(this, dwarf_cu, DW_AT_language, 0);
            if (language)
                return (lldb::LanguageType)language;
        }
    }
    return eLanguageTypeUnknown;
}

size_t
SymbolFileDWARF::ParseCompileUnitFunctions(const SymbolContext &sc)
{
    assert (sc.comp_unit);
    size_t functions_added = 0;
    DWARFCompileUnit* dwarf_cu = GetDWARFCompileUnit(sc.comp_unit);
    if (dwarf_cu)
    {
        DWARFDIECollection function_dies;
        const size_t num_functions = dwarf_cu->AppendDIEsWithTag (DW_TAG_subprogram, function_dies);
        size_t func_idx;
        for (func_idx = 0; func_idx < num_functions; ++func_idx)
        {
            const DWARFDebugInfoEntry *die = function_dies.GetDIEPtrAtIndex(func_idx);
            if (sc.comp_unit->FindFunctionByUID (MakeUserID(die->GetOffset())).get() == NULL)
            {
                if (ParseCompileUnitFunction(sc, dwarf_cu, die))
                    ++functions_added;
            }
        }
        //FixupTypes();
    }
    return functions_added;
}

bool
SymbolFileDWARF::ParseCompileUnitSupportFiles (const SymbolContext& sc, FileSpecList& support_files)
{
    assert (sc.comp_unit);
    DWARFCompileUnit* dwarf_cu = GetDWARFCompileUnit(sc.comp_unit);
    if (dwarf_cu)
    {
        const DWARFDebugInfoEntry * cu_die = dwarf_cu->GetCompileUnitDIEOnly();

        if (cu_die)
        {
            const char * cu_comp_dir = cu_die->GetAttributeValueAsString(this, dwarf_cu, DW_AT_comp_dir, NULL);
            dw_offset_t stmt_list = cu_die->GetAttributeValueAsUnsigned(this, dwarf_cu, DW_AT_stmt_list, DW_INVALID_OFFSET);

            // All file indexes in DWARF are one based and a file of index zero is
            // supposed to be the compile unit itself.
            support_files.Append (*sc.comp_unit);

            return DWARFDebugLine::ParseSupportFiles(sc.comp_unit->GetModule(), get_debug_line_data(), cu_comp_dir, stmt_list, support_files);
        }
    }
    return false;
}

struct ParseDWARFLineTableCallbackInfo
{
    LineTable* line_table;
    const SectionList *section_list;
    lldb::addr_t prev_sect_file_base_addr;
    lldb::addr_t curr_sect_file_base_addr;
    bool is_oso_for_debug_map;
    bool prev_in_final_executable;
    DWARFDebugLine::Row prev_row;
    SectionSP prev_section_sp;
    SectionSP curr_section_sp;
};

//----------------------------------------------------------------------
// ParseStatementTableCallback
//----------------------------------------------------------------------
static void
ParseDWARFLineTableCallback(dw_offset_t offset, const DWARFDebugLine::State& state, void* userData)
{
    LineTable* line_table = ((ParseDWARFLineTableCallbackInfo*)userData)->line_table;
    if (state.row == DWARFDebugLine::State::StartParsingLineTable)
    {
        // Just started parsing the line table
    }
    else if (state.row == DWARFDebugLine::State::DoneParsingLineTable)
    {
        // Done parsing line table, nothing to do for the cleanup
    }
    else
    {
        ParseDWARFLineTableCallbackInfo* info = (ParseDWARFLineTableCallbackInfo*)userData;
        // We have a new row, lets append it

        if (info->curr_section_sp.get() == NULL || info->curr_section_sp->ContainsFileAddress(state.address) == false)
        {
            info->prev_section_sp = info->curr_section_sp;
            info->prev_sect_file_base_addr = info->curr_sect_file_base_addr;
            // If this is an end sequence entry, then we subtract one from the
            // address to make sure we get an address that is not the end of
            // a section.
            if (state.end_sequence && state.address != 0)
                info->curr_section_sp = info->section_list->FindSectionContainingFileAddress (state.address - 1);
            else
                info->curr_section_sp = info->section_list->FindSectionContainingFileAddress (state.address);

            if (info->curr_section_sp.get())
                info->curr_sect_file_base_addr = info->curr_section_sp->GetFileAddress ();
            else
                info->curr_sect_file_base_addr = 0;
        }
        if (info->curr_section_sp.get())
        {
            lldb::addr_t curr_line_section_offset = state.address - info->curr_sect_file_base_addr;
            // Check for the fancy section magic to determine if we

            if (info->is_oso_for_debug_map)
            {
                // When this is a debug map object file that contains DWARF
                // (referenced from an N_OSO debug map nlist entry) we will have
                // a file address in the file range for our section from the
                // original .o file, and a load address in the executable that
                // contains the debug map.
                //
                // If the sections for the file range and load range are
                // different, we have a remapped section for the function and
                // this address is resolved. If they are the same, then the
                // function for this address didn't make it into the final
                // executable.
                bool curr_in_final_executable = (bool) info->curr_section_sp->GetLinkedSection ();

                // If we are doing DWARF with debug map, then we need to carefully
                // add each line table entry as there may be gaps as functions
                // get moved around or removed.
                if (!info->prev_row.end_sequence && info->prev_section_sp.get())
                {
                    if (info->prev_in_final_executable)
                    {
                        bool terminate_previous_entry = false;
                        if (!curr_in_final_executable)
                        {
                            // Check for the case where the previous line entry
                            // in a function made it into the final executable,
                            // yet the current line entry falls in a function
                            // that didn't. The line table used to be contiguous
                            // through this address range but now it isn't. We
                            // need to terminate the previous line entry so
                            // that we can reconstruct the line range correctly
                            // for it and to keep the line table correct.
                            terminate_previous_entry = true;
                        }
                        else if (info->curr_section_sp.get() != info->prev_section_sp.get())
                        {
                            // Check for cases where the line entries used to be
                            // contiguous address ranges, but now they aren't.
                            // This can happen when order files specify the
                            // ordering of the functions.
                            lldb::addr_t prev_line_section_offset = info->prev_row.address - info->prev_sect_file_base_addr;
                            Section *curr_sect = info->curr_section_sp.get();
                            Section *prev_sect = info->prev_section_sp.get();
                            assert (curr_sect->GetLinkedSection());
                            assert (prev_sect->GetLinkedSection());
                            lldb::addr_t object_file_addr_delta = state.address - info->prev_row.address;
                            lldb::addr_t curr_linked_file_addr = curr_sect->GetLinkedFileAddress() + curr_line_section_offset;
                            lldb::addr_t prev_linked_file_addr = prev_sect->GetLinkedFileAddress() + prev_line_section_offset;
                            lldb::addr_t linked_file_addr_delta = curr_linked_file_addr - prev_linked_file_addr;
                            if (object_file_addr_delta != linked_file_addr_delta)
                                terminate_previous_entry = true;
                        }

                        if (terminate_previous_entry)
                        {
                            line_table->InsertLineEntry (info->prev_section_sp,
                                                         state.address - info->prev_sect_file_base_addr,
                                                         info->prev_row.line,
                                                         info->prev_row.column,
                                                         info->prev_row.file,
                                                         false,                 // is_stmt
                                                         false,                 // basic_block
                                                         false,                 // state.prologue_end
                                                         false,                 // state.epilogue_begin
                                                         true);                 // end_sequence);
                        }
                    }
                }

                if (curr_in_final_executable)
                {
                    line_table->InsertLineEntry (info->curr_section_sp,
                                                 curr_line_section_offset,
                                                 state.line,
                                                 state.column,
                                                 state.file,
                                                 state.is_stmt,
                                                 state.basic_block,
                                                 state.prologue_end,
                                                 state.epilogue_begin,
                                                 state.end_sequence);
                    info->prev_section_sp = info->curr_section_sp;
                }
                else
                {
                    // If the current address didn't make it into the final
                    // executable, the current section will be the __text
                    // segment in the .o file, so we need to clear this so
                    // we can catch the next function that did make it into
                    // the final executable.
                    info->prev_section_sp.reset();
                    info->curr_section_sp.reset();
                }

                info->prev_in_final_executable = curr_in_final_executable;
            }
            else
            {
                // We are not in an object file that contains DWARF for an
                // N_OSO, this is just a normal DWARF file. The DWARF spec
                // guarantees that the addresses will be in increasing order
                // so, since we store line tables in file address order, we
                // can always just append the line entry without needing to
                // search for the correct insertion point (we don't need to
                // use LineEntry::InsertLineEntry()).
                line_table->AppendLineEntry (info->curr_section_sp,
                                             curr_line_section_offset,
                                             state.line,
                                             state.column,
                                             state.file,
                                             state.is_stmt,
                                             state.basic_block,
                                             state.prologue_end,
                                             state.epilogue_begin,
                                             state.end_sequence);
            }
        }

        info->prev_row = state;
    }
}

bool
SymbolFileDWARF::ParseCompileUnitLineTable (const SymbolContext &sc)
{
    assert (sc.comp_unit);
    if (sc.comp_unit->GetLineTable() != NULL)
        return true;

    DWARFCompileUnit* dwarf_cu = GetDWARFCompileUnit(sc.comp_unit);
    if (dwarf_cu)
    {
        const DWARFDebugInfoEntry *dwarf_cu_die = dwarf_cu->GetCompileUnitDIEOnly();
        if (dwarf_cu_die)
        {
            const dw_offset_t cu_line_offset = dwarf_cu_die->GetAttributeValueAsUnsigned(this, dwarf_cu, DW_AT_stmt_list, DW_INVALID_OFFSET);
            if (cu_line_offset != DW_INVALID_OFFSET)
            {
                std::auto_ptr<LineTable> line_table_ap(new LineTable(sc.comp_unit));
                if (line_table_ap.get())
                {
                    ParseDWARFLineTableCallbackInfo info = { 
                        line_table_ap.get(), 
                        m_obj_file->GetSectionList(), 
                        0, 
                        0, 
                        GetDebugMapSymfile () != NULL,
                        false, 
                        DWARFDebugLine::Row(), 
                        SectionSP(), 
                        SectionSP()
                    };
                    uint32_t offset = cu_line_offset;
                    DWARFDebugLine::ParseStatementTable(get_debug_line_data(), &offset, ParseDWARFLineTableCallback, &info);
                    sc.comp_unit->SetLineTable(line_table_ap.release());
                    return true;
                }
            }
        }
    }
    return false;
}

size_t
SymbolFileDWARF::ParseFunctionBlocks
(
    const SymbolContext& sc,
    Block *parent_block,
    DWARFCompileUnit* dwarf_cu,
    const DWARFDebugInfoEntry *die,
    addr_t subprogram_low_pc,
    uint32_t depth
)
{
    size_t blocks_added = 0;
    while (die != NULL)
    {
        dw_tag_t tag = die->Tag();

        switch (tag)
        {
        case DW_TAG_inlined_subroutine:
        case DW_TAG_subprogram:
        case DW_TAG_lexical_block:
            {
                Block *block = NULL;
                if (tag == DW_TAG_subprogram)
                {
                    // Skip any DW_TAG_subprogram DIEs that are inside
                    // of a normal or inlined functions. These will be 
                    // parsed on their own as separate entities.

                    if (depth > 0)
                        break;

                    block = parent_block;
                }
                else
                {
                    BlockSP block_sp(new Block (MakeUserID(die->GetOffset())));
                    parent_block->AddChild(block_sp);
                    block = block_sp.get();
                }
                DWARFDebugRanges::RangeList ranges;
                const char *name = NULL;
                const char *mangled_name = NULL;

                int decl_file = 0;
                int decl_line = 0;
                int decl_column = 0;
                int call_file = 0;
                int call_line = 0;
                int call_column = 0;
                if (die->GetDIENamesAndRanges (this, 
                                               dwarf_cu, 
                                               name, 
                                               mangled_name, 
                                               ranges, 
                                               decl_file, decl_line, decl_column,
                                               call_file, call_line, call_column))
                {
                    if (tag == DW_TAG_subprogram)
                    {
                        assert (subprogram_low_pc == LLDB_INVALID_ADDRESS);
                        subprogram_low_pc = ranges.GetMinRangeBase(0);
                    }
                    else if (tag == DW_TAG_inlined_subroutine)
                    {
                        // We get called here for inlined subroutines in two ways.  
                        // The first time is when we are making the Function object 
                        // for this inlined concrete instance.  Since we're creating a top level block at
                        // here, the subprogram_low_pc will be LLDB_INVALID_ADDRESS.  So we need to 
                        // adjust the containing address.
                        // The second time is when we are parsing the blocks inside the function that contains
                        // the inlined concrete instance.  Since these will be blocks inside the containing "real"
                        // function the offset will be for that function.  
                        if (subprogram_low_pc == LLDB_INVALID_ADDRESS)
                        {
                            subprogram_low_pc = ranges.GetMinRangeBase(0);
                        }
                    }
                    
                    AddRangesToBlock (*block, ranges, subprogram_low_pc);

                    if (tag != DW_TAG_subprogram && (name != NULL || mangled_name != NULL))
                    {
                        std::auto_ptr<Declaration> decl_ap;
                        if (decl_file != 0 || decl_line != 0 || decl_column != 0)
                            decl_ap.reset(new Declaration(sc.comp_unit->GetSupportFiles().GetFileSpecAtIndex(decl_file), 
                                                          decl_line, decl_column));

                        std::auto_ptr<Declaration> call_ap;
                        if (call_file != 0 || call_line != 0 || call_column != 0)
                            call_ap.reset(new Declaration(sc.comp_unit->GetSupportFiles().GetFileSpecAtIndex(call_file), 
                                                          call_line, call_column));

                        block->SetInlinedFunctionInfo (name, mangled_name, decl_ap.get(), call_ap.get());
                    }

                    ++blocks_added;

                    if (die->HasChildren())
                    {
                        blocks_added += ParseFunctionBlocks (sc, 
                                                             block, 
                                                             dwarf_cu, 
                                                             die->GetFirstChild(), 
                                                             subprogram_low_pc, 
                                                             depth + 1);
                    }
                }
            }
            break;
        default:
            break;
        }

        // Only parse siblings of the block if we are not at depth zero. A depth
        // of zero indicates we are currently parsing the top level 
        // DW_TAG_subprogram DIE
        
        if (depth == 0)
            die = NULL;
        else
            die = die->GetSibling();
    }
    return blocks_added;
}

bool
SymbolFileDWARF::ParseTemplateDIE (DWARFCompileUnit* dwarf_cu,
                                   const DWARFDebugInfoEntry *die,
                                   ClangASTContext::TemplateParameterInfos &template_param_infos)
{
    const dw_tag_t tag = die->Tag();
    
    switch (tag)
    {
    case DW_TAG_template_type_parameter:
    case DW_TAG_template_value_parameter:
        {
            const uint8_t *fixed_form_sizes = DWARFFormValue::GetFixedFormSizesForAddressSize (dwarf_cu->GetAddressByteSize());

            DWARFDebugInfoEntry::Attributes attributes;
            const size_t num_attributes = die->GetAttributes (this, 
                                                              dwarf_cu, 
                                                              fixed_form_sizes, 
                                                              attributes);
            const char *name = NULL;
            Type *lldb_type = NULL;
            clang_type_t clang_type = NULL;
            uint64_t uval64 = 0;
            bool uval64_valid = false;
            if (num_attributes > 0)
            {
                DWARFFormValue form_value;
                for (size_t i=0; i<num_attributes; ++i)
                {
                    const dw_attr_t attr = attributes.AttributeAtIndex(i);
                    
                    switch (attr)
                    {
                        case DW_AT_name:
                            if (attributes.ExtractFormValueAtIndex(this, i, form_value))
                                name = form_value.AsCString(&get_debug_str_data());
                            break;
                            
                        case DW_AT_type:
                            if (attributes.ExtractFormValueAtIndex(this, i, form_value))
                            {
                                const dw_offset_t type_die_offset = form_value.Reference(dwarf_cu);
                                lldb_type = ResolveTypeUID(type_die_offset);
                                if (lldb_type)
                                    clang_type = lldb_type->GetClangForwardType();
                            }
                            break;
                            
                        case DW_AT_const_value:
                            if (attributes.ExtractFormValueAtIndex(this, i, form_value))
                            {
                                uval64_valid = true;
                                uval64 = form_value.Unsigned();
                            }
                            break;
                        default:
                            break;
                    }
                }
                
                if (name && lldb_type && clang_type)
                {
                    bool is_signed = false;
                    template_param_infos.names.push_back(name);
                    clang::QualType clang_qual_type (clang::QualType::getFromOpaquePtr (clang_type));
                    if (tag == DW_TAG_template_value_parameter && ClangASTContext::IsIntegerType (clang_type, is_signed) && uval64_valid)
                    {
                        llvm::APInt apint (lldb_type->GetByteSize() * 8, uval64, is_signed);
                        template_param_infos.args.push_back (clang::TemplateArgument (*GetClangASTContext().getASTContext(),
                                                                                      llvm::APSInt(apint),
                                                                                      clang_qual_type));
                    }
                    else
                    {
                        template_param_infos.args.push_back (clang::TemplateArgument (clang_qual_type));
                    }
                }
                else
                {
                    return false;
                }
                
            }
        }
        return true;

    default:
        break;
    }
    return false;
}

bool
SymbolFileDWARF::ParseTemplateParameterInfos (DWARFCompileUnit* dwarf_cu,
                                              const DWARFDebugInfoEntry *parent_die,
                                              ClangASTContext::TemplateParameterInfos &template_param_infos)
{

    if (parent_die == NULL)
        return false;
    
    Args template_parameter_names;
    for (const DWARFDebugInfoEntry *die = parent_die->GetFirstChild(); 
         die != NULL; 
         die = die->GetSibling())
    {
        const dw_tag_t tag = die->Tag();
        
        switch (tag)
        {
            case DW_TAG_template_type_parameter:
            case DW_TAG_template_value_parameter:
                ParseTemplateDIE (dwarf_cu, die, template_param_infos);
            break;
                
        default:
            break;
        }
    }
    if (template_param_infos.args.empty())
        return false;
    return template_param_infos.args.size() == template_param_infos.names.size();
}

clang::ClassTemplateDecl *
SymbolFileDWARF::ParseClassTemplateDecl (clang::DeclContext *decl_ctx,
                                         lldb::AccessType access_type,
                                         const char *parent_name,
                                         int tag_decl_kind,
                                         const ClangASTContext::TemplateParameterInfos &template_param_infos)
{
    if (template_param_infos.IsValid())
    {
        std::string template_basename(parent_name);
        template_basename.erase (template_basename.find('<'));
        ClangASTContext &ast = GetClangASTContext();

        return ast.CreateClassTemplateDecl (decl_ctx,
                                            access_type,
                                            template_basename.c_str(), 
                                            tag_decl_kind, 
                                            template_param_infos);
    }
    return NULL;
}

class SymbolFileDWARF::DelayedAddObjCClassProperty
{
public:
    DelayedAddObjCClassProperty
    (
        clang::ASTContext      *ast,
        lldb::clang_type_t      class_opaque_type, 
        const char             *property_name,
        lldb::clang_type_t      property_opaque_type,  // The property type is only required if you don't have an ivar decl
        clang::ObjCIvarDecl    *ivar_decl,   
        const char             *property_setter_name,
        const char             *property_getter_name,
        uint32_t                property_attributes,
        const ClangASTMetadata       *metadata
    ) :
        m_ast                   (ast),
        m_class_opaque_type     (class_opaque_type),
        m_property_name         (property_name),
        m_property_opaque_type  (property_opaque_type),
        m_ivar_decl             (ivar_decl),
        m_property_setter_name  (property_setter_name),
        m_property_getter_name  (property_getter_name),
        m_property_attributes   (property_attributes)
    {
        if (metadata != NULL)
        {
            m_metadata_ap.reset(new ClangASTMetadata());
            *(m_metadata_ap.get()) = *metadata;
        }
    }
    
    DelayedAddObjCClassProperty (const DelayedAddObjCClassProperty &rhs)
    {
        m_ast                  = rhs.m_ast;
        m_class_opaque_type    = rhs.m_class_opaque_type;
        m_property_name        = rhs.m_property_name;
        m_property_opaque_type = rhs.m_property_opaque_type;
        m_ivar_decl            = rhs.m_ivar_decl;
        m_property_setter_name = rhs.m_property_setter_name;
        m_property_getter_name = rhs.m_property_getter_name;
        m_property_attributes  = rhs.m_property_attributes;
        
        if (rhs.m_metadata_ap.get())
        {
            m_metadata_ap.reset (new ClangASTMetadata());
            *(m_metadata_ap.get()) = *(rhs.m_metadata_ap.get());
        }
        
    }
    
    bool Finalize() const
    {
        return ClangASTContext::AddObjCClassProperty(m_ast,
                                                     m_class_opaque_type,
                                                     m_property_name,
                                                     m_property_opaque_type,
                                                     m_ivar_decl,
                                                     m_property_setter_name,
                                                     m_property_getter_name,
                                                     m_property_attributes,
                                                     m_metadata_ap.get());
    }
private:
    clang::ASTContext      *m_ast;
    lldb::clang_type_t      m_class_opaque_type;
    const char             *m_property_name;
    lldb::clang_type_t      m_property_opaque_type;
    clang::ObjCIvarDecl    *m_ivar_decl;
    const char             *m_property_setter_name;
    const char             *m_property_getter_name;
    uint32_t                m_property_attributes;
    std::auto_ptr<ClangASTMetadata>        m_metadata_ap;
};

size_t
SymbolFileDWARF::ParseChildMembers
(
    const SymbolContext& sc,
    DWARFCompileUnit* dwarf_cu,
    const DWARFDebugInfoEntry *parent_die,
    clang_type_t class_clang_type,
    const LanguageType class_language,
    std::vector<clang::CXXBaseSpecifier *>& base_classes,
    std::vector<int>& member_accessibilities,
    DWARFDIECollection& member_function_dies,
    BitfieldMap &bitfield_map,
    DelayedPropertyList& delayed_properties,
    AccessType& default_accessibility,
    bool &is_a_class,
    LayoutInfo &layout_info
)
{
    if (parent_die == NULL)
        return 0;

    size_t count = 0;
    const DWARFDebugInfoEntry *die;
    const uint8_t *fixed_form_sizes = DWARFFormValue::GetFixedFormSizesForAddressSize (dwarf_cu->GetAddressByteSize());
    uint32_t member_idx = 0;

    for (die = parent_die->GetFirstChild(); die != NULL; die = die->GetSibling())
    {
        dw_tag_t tag = die->Tag();

        switch (tag)
        {
        case DW_TAG_member:
        case DW_TAG_APPLE_property:
            {
                DWARFDebugInfoEntry::Attributes attributes;
                const size_t num_attributes = die->GetAttributes (this, 
                                                                  dwarf_cu, 
                                                                  fixed_form_sizes, 
                                                                  attributes);
                if (num_attributes > 0)
                {
                    Declaration decl;
                    //DWARFExpression location;
                    const char *name = NULL;
                    const char *prop_name = NULL;
                    const char *prop_getter_name = NULL;
                    const char *prop_setter_name = NULL;
                    uint32_t        prop_attributes = 0;
                    
                    
                    bool is_artificial = false;
                    lldb::user_id_t encoding_uid = LLDB_INVALID_UID;
                    AccessType accessibility = eAccessNone;
                    uint32_t member_byte_offset = UINT32_MAX;
                    size_t byte_size = 0;
                    size_t bit_offset = 0;
                    size_t bit_size = 0;
                    uint32_t i;
                    for (i=0; i<num_attributes && !is_artificial; ++i)
                    {
                        const dw_attr_t attr = attributes.AttributeAtIndex(i);
                        DWARFFormValue form_value;
                        if (attributes.ExtractFormValueAtIndex(this, i, form_value))
                        {
                            switch (attr)
                            {
                            case DW_AT_decl_file:   decl.SetFile(sc.comp_unit->GetSupportFiles().GetFileSpecAtIndex(form_value.Unsigned())); break;
                            case DW_AT_decl_line:   decl.SetLine(form_value.Unsigned()); break;
                            case DW_AT_decl_column: decl.SetColumn(form_value.Unsigned()); break;
                            case DW_AT_name:        name = form_value.AsCString(&get_debug_str_data()); break;
                            case DW_AT_type:        encoding_uid = form_value.Reference(dwarf_cu); break;
                            case DW_AT_bit_offset:  bit_offset = form_value.Unsigned(); break;
                            case DW_AT_bit_size:    bit_size = form_value.Unsigned(); break;
                            case DW_AT_byte_size:   byte_size = form_value.Unsigned(); break;
                            case DW_AT_data_member_location:
                                if (form_value.BlockData())
                                {
                                    Value initialValue(0);
                                    Value memberOffset(0);
                                    const DataExtractor& debug_info_data = get_debug_info_data();
                                    uint32_t block_length = form_value.Unsigned();
                                    uint32_t block_offset = form_value.BlockData() - debug_info_data.GetDataStart();
                                    if (DWARFExpression::Evaluate(NULL, // ExecutionContext *
                                                                  NULL, // clang::ASTContext *
                                                                  NULL, // ClangExpressionVariableList *
                                                                  NULL, // ClangExpressionDeclMap *
                                                                  NULL, // RegisterContext *
                                                                  debug_info_data, 
                                                                  block_offset, 
                                                                  block_length, 
                                                                  eRegisterKindDWARF, 
                                                                  &initialValue, 
                                                                  memberOffset, 
                                                                  NULL))
                                    {
                                        member_byte_offset = memberOffset.ResolveValue(NULL, NULL).UInt();
                                    }
                                }
                                break;

                            case DW_AT_accessibility: accessibility = DW_ACCESS_to_AccessType (form_value.Unsigned()); break;
                            case DW_AT_artificial: is_artificial = form_value.Unsigned() != 0; break;                            
                            case DW_AT_APPLE_property_name:      prop_name = form_value.AsCString(&get_debug_str_data()); break;
                            case DW_AT_APPLE_property_getter:    prop_getter_name = form_value.AsCString(&get_debug_str_data()); break;
                            case DW_AT_APPLE_property_setter:    prop_setter_name = form_value.AsCString(&get_debug_str_data()); break;
                            case DW_AT_APPLE_property_attribute: prop_attributes = form_value.Unsigned(); break;

                            default:
                            case DW_AT_declaration:
                            case DW_AT_description:
                            case DW_AT_mutable:
                            case DW_AT_visibility:
                            case DW_AT_sibling:
                                break;
                            }
                        }
                    }
                                        
                    if (prop_name)
                    {
                        ConstString fixed_getter;
                        ConstString fixed_setter;

                        // Check if the property getter/setter were provided as full
                        // names.  We want basenames, so we extract them.
                        
                        if (prop_getter_name && prop_getter_name[0] == '-')
                        {
                            ObjCLanguageRuntime::ParseMethodName (prop_getter_name,
                                                                  NULL,
                                                                  &fixed_getter,
                                                                  NULL,
                                                                  NULL);
                            prop_getter_name = fixed_getter.GetCString();
                        }
                        
                        if (prop_setter_name && prop_setter_name[0] == '-')
                        {
                            ObjCLanguageRuntime::ParseMethodName (prop_setter_name,
                                                                  NULL,
                                                                  &fixed_setter,
                                                                  NULL,
                                                                  NULL);
                            prop_setter_name = fixed_setter.GetCString();
                        }
                        
                        // If the names haven't been provided, they need to be
                        // filled in.
                        
                        if (!prop_getter_name)
                        {
                            prop_getter_name = prop_name;
                        }
                        if (!prop_setter_name && prop_name[0] && !(prop_attributes & DW_APPLE_PROPERTY_readonly))
                        {
                            StreamString ss;
                            
                            ss.Printf("set%c%s:",
                                      toupper(prop_name[0]),
                                      &prop_name[1]);
                            
                            fixed_setter.SetCString(ss.GetData());
                            prop_setter_name = fixed_setter.GetCString();
                        }
                    }
                    
                    // Clang has a DWARF generation bug where sometimes it
                    // represents fields that are references with bad byte size
                    // and bit size/offset information such as:
                    //
                    //  DW_AT_byte_size( 0x00 )
                    //  DW_AT_bit_size( 0x40 )
                    //  DW_AT_bit_offset( 0xffffffffffffffc0 )
                    //
                    // So check the bit offset to make sure it is sane, and if 
                    // the values are not sane, remove them. If we don't do this
                    // then we will end up with a crash if we try to use this 
                    // type in an expression when clang becomes unhappy with its
                    // recycled debug info.
                    
                    if (bit_offset > 128)
                    {
                        bit_size = 0;
                        bit_offset = 0;
                    }

                    // FIXME: Make Clang ignore Objective-C accessibility for expressions
                    if (class_language == eLanguageTypeObjC ||
                        class_language == eLanguageTypeObjC_plus_plus)
                        accessibility = eAccessNone; 
                    
                    if (member_idx == 0 && !is_artificial && name && (strstr (name, "_vptr$") == name))
                    {
                        // Not all compilers will mark the vtable pointer
                        // member as artificial (llvm-gcc). We can't have
                        // the virtual members in our classes otherwise it
                        // throws off all child offsets since we end up
                        // having and extra pointer sized member in our 
                        // class layouts.
                        is_artificial = true;
                    }

                    if (is_artificial == false)
                    {
                        Type *member_type = ResolveTypeUID(encoding_uid);
                        clang::FieldDecl *field_decl = NULL;
                        if (tag == DW_TAG_member)
                        {
                            if (member_type)
                            {
                                if (accessibility == eAccessNone)
                                    accessibility = default_accessibility;
                                member_accessibilities.push_back(accessibility);

                                // Code to detect unnamed bitifields
                                if (bit_size > 0 && member_byte_offset != UINT32_MAX)
                                {
                                    // Objective C has invalid DW_AT_bit_offset values so we can't use them to detect
                                    // unnamed bitfields. Once clang is fixed we will enable unnamed bitfields
                                    // in ObjC classes (<rdar://problem/12636970>)
                                    
                                    if (!(class_language == eLanguageTypeObjC || class_language == eLanguageTypeObjC_plus_plus))
                                    {
                                        // We have a bitfield, we need to watch out for
                                        // unnamed bitfields that we need to insert if
                                        // there is a gap in the bytes as many compilers
                                        // doesn't emit DWARF DW_TAG_member tags for
                                        // unnammed bitfields.
                                        BitfieldMap::iterator bit_pos = bitfield_map.find(member_byte_offset);
                                        uint32_t unnamed_bit_size = 0;
                                        uint32_t unnamed_bit_offset = 0;
                                        if (bit_pos == bitfield_map.end())
                                        {
                                            // First bitfield in an integral type.
                                            
                                            // We might need to insert a leading unnamed bitfield
                                            if (bit_offset < byte_size * 8)
                                            {
                                                unnamed_bit_size = byte_size * 8 - (bit_size + bit_offset);
                                                unnamed_bit_offset = byte_size * 8 - unnamed_bit_size;
                                            }
                                            
                                            // Now put the current bitfield info into the map
                                            bitfield_map[member_byte_offset].bit_size = bit_size;
                                            bitfield_map[member_byte_offset].bit_offset = bit_offset;
                                        }
                                        else
                                        {
                                            // Subsequent bitfield in an integral type.

                                            // We have a bitfield that isn't the first for this
                                            // integral type, check to make sure there aren't any
                                            // gaps.
                                            assert (bit_pos->second.bit_size > 0);
                                            if (bit_offset < bit_pos->second.bit_offset)
                                            {
                                                unnamed_bit_size = bit_pos->second.bit_offset - (bit_size + bit_offset);
                                                unnamed_bit_offset = bit_pos->second.bit_offset - unnamed_bit_size;
                                            }

                                            // Now put the current bitfield info into the map
                                            bit_pos->second.bit_size = bit_size;
                                            bit_pos->second.bit_offset = bit_offset;
                                        }
                                        
                                        if (unnamed_bit_size > 0)
                                        {
                                            clang::FieldDecl *unnamed_bitfield_decl = GetClangASTContext().AddFieldToRecordType (class_clang_type,
                                                                                                                                 NULL,
                                                                                                                                 member_type->GetClangLayoutType(),
                                                                                                                                 accessibility,
                                                                                                                                 unnamed_bit_size);
                                            uint64_t total_bit_offset = 0;
                                            
                                            total_bit_offset += (member_byte_offset == UINT32_MAX ? 0 : (member_byte_offset * 8));
                                            
                                            if (GetObjectFile()->GetByteOrder() == eByteOrderLittle)
                                            {
                                                total_bit_offset += byte_size * 8;
                                                total_bit_offset -= (unnamed_bit_offset + unnamed_bit_size);
                                            }
                                            else
                                            {
                                                total_bit_offset += unnamed_bit_size;
                                            }
                                            
                                            layout_info.field_offsets.insert(std::make_pair(unnamed_bitfield_decl, total_bit_offset));
                                        }
                                    }
                                }
                                field_decl = GetClangASTContext().AddFieldToRecordType (class_clang_type,
                                                                                        name, 
                                                                                        member_type->GetClangLayoutType(), 
                                                                                        accessibility, 
                                                                                        bit_size);
                                
                                GetClangASTContext().SetMetadataAsUserID ((uintptr_t)field_decl, MakeUserID(die->GetOffset()));
                            }
                            else
                            {
                                if (name)
                                    GetObjectFile()->GetModule()->ReportError ("0x%8.8llx: DW_TAG_member '%s' refers to type 0x%8.8llx which was unable to be parsed",
                                                                               MakeUserID(die->GetOffset()),
                                                                               name,
                                                                               encoding_uid);
                                else
                                    GetObjectFile()->GetModule()->ReportError ("0x%8.8llx: DW_TAG_member refers to type 0x%8.8llx which was unable to be parsed",
                                                                               MakeUserID(die->GetOffset()),
                                                                               encoding_uid);
                            }

                            if (member_byte_offset != UINT32_MAX || bit_size != 0)
                            {
                                /////////////////////////////////////////////////////////////
                                // How to locate a field given the DWARF debug information
                                //
                                // AT_byte_size indicates the size of the word in which the
                                // bit offset must be interpreted.
                                //
                                // AT_data_member_location indicates the byte offset of the
                                // word from the base address of the structure.
                                //
                                // AT_bit_offset indicates how many bits into the word
                                // (according to the host endianness) the low-order bit of
                                // the field starts.  AT_bit_offset can be negative.
                                //
                                // AT_bit_size indicates the size of the field in bits.
                                /////////////////////////////////////////////////////////////
                                                        
                                uint64_t total_bit_offset = 0;
                                
                                total_bit_offset += (member_byte_offset == UINT32_MAX ? 0 : (member_byte_offset * 8));
                                
                                if (GetObjectFile()->GetByteOrder() == eByteOrderLittle)
                                {  
                                    total_bit_offset += byte_size * 8;
                                    total_bit_offset -= (bit_offset + bit_size);
                                }
                                else
                                {
                                    total_bit_offset += bit_offset;
                                }
                                                            
                                layout_info.field_offsets.insert(std::make_pair(field_decl, total_bit_offset));
                            }
                        }
                        
                        if (prop_name != NULL)
                        {
                            clang::ObjCIvarDecl *ivar_decl = NULL;
                            
                            if (field_decl)
                            {
                                ivar_decl = clang::dyn_cast<clang::ObjCIvarDecl>(field_decl);
                                assert (ivar_decl != NULL);
                            }
                            
                            ClangASTMetadata metadata;
                            metadata.SetUserID (MakeUserID(die->GetOffset()));
                            delayed_properties.push_back(DelayedAddObjCClassProperty(GetClangASTContext().getASTContext(),
                                                                                     class_clang_type,
                                                                                     prop_name,
                                                                                     member_type->GetClangLayoutType(),
                                                                                     ivar_decl,
                                                                                     prop_setter_name,
                                                                                     prop_getter_name,
                                                                                     prop_attributes,
                                                                                     &metadata));
                            
                            if (ivar_decl)
                                GetClangASTContext().SetMetadataAsUserID ((uintptr_t)ivar_decl, MakeUserID(die->GetOffset()));
                        }
                    }
                }
                ++member_idx;
            }
            break;

        case DW_TAG_subprogram:
            // Let the type parsing code handle this one for us. 
            member_function_dies.Append (die);
            break;

        case DW_TAG_inheritance:
            {
                is_a_class = true;
                if (default_accessibility == eAccessNone)
                    default_accessibility = eAccessPrivate;
                // TODO: implement DW_TAG_inheritance type parsing
                DWARFDebugInfoEntry::Attributes attributes;
                const size_t num_attributes = die->GetAttributes (this, 
                                                                  dwarf_cu, 
                                                                  fixed_form_sizes, 
                                                                  attributes);
                if (num_attributes > 0)
                {
                    Declaration decl;
                    DWARFExpression location;
                    lldb::user_id_t encoding_uid = LLDB_INVALID_UID;
                    AccessType accessibility = default_accessibility;
                    bool is_virtual = false;
                    bool is_base_of_class = true;
                    off_t member_byte_offset = 0;
                    uint32_t i;
                    for (i=0; i<num_attributes; ++i)
                    {
                        const dw_attr_t attr = attributes.AttributeAtIndex(i);
                        DWARFFormValue form_value;
                        if (attributes.ExtractFormValueAtIndex(this, i, form_value))
                        {
                            switch (attr)
                            {
                            case DW_AT_decl_file:   decl.SetFile(sc.comp_unit->GetSupportFiles().GetFileSpecAtIndex(form_value.Unsigned())); break;
                            case DW_AT_decl_line:   decl.SetLine(form_value.Unsigned()); break;
                            case DW_AT_decl_column: decl.SetColumn(form_value.Unsigned()); break;
                            case DW_AT_type:        encoding_uid = form_value.Reference(dwarf_cu); break;
                            case DW_AT_data_member_location:
                                if (form_value.BlockData())
                                {
                                    Value initialValue(0);
                                    Value memberOffset(0);
                                    const DataExtractor& debug_info_data = get_debug_info_data();
                                    uint32_t block_length = form_value.Unsigned();
                                    uint32_t block_offset = form_value.BlockData() - debug_info_data.GetDataStart();
                                    if (DWARFExpression::Evaluate (NULL, 
                                                                   NULL, 
                                                                   NULL, 
                                                                   NULL, 
                                                                   NULL,
                                                                   debug_info_data, 
                                                                   block_offset, 
                                                                   block_length, 
                                                                   eRegisterKindDWARF, 
                                                                   &initialValue, 
                                                                   memberOffset, 
                                                                   NULL))
                                    {
                                        member_byte_offset = memberOffset.ResolveValue(NULL, NULL).UInt();
                                    }
                                }
                                break;

                            case DW_AT_accessibility:
                                accessibility = DW_ACCESS_to_AccessType(form_value.Unsigned());
                                break;

                            case DW_AT_virtuality: is_virtual = form_value.Unsigned() != 0; break;
                            default:
                            case DW_AT_sibling:
                                break;
                            }
                        }
                    }

                    Type *base_class_type = ResolveTypeUID(encoding_uid);
                    assert(base_class_type);
                    
                    clang_type_t base_class_clang_type = base_class_type->GetClangFullType();
                    assert (base_class_clang_type);
                    if (class_language == eLanguageTypeObjC)
                    {
                        GetClangASTContext().SetObjCSuperClass(class_clang_type, base_class_clang_type);
                    }
                    else
                    {
                        base_classes.push_back (GetClangASTContext().CreateBaseClassSpecifier (base_class_clang_type, 
                                                                                               accessibility, 
                                                                                               is_virtual, 
                                                                                               is_base_of_class));
                        
                        if (is_virtual)
                        {
                            layout_info.vbase_offsets.insert(std::make_pair(ClangASTType::GetAsCXXRecordDecl(class_clang_type),
                                                                            clang::CharUnits::fromQuantity(member_byte_offset)));
                        }
                        else
                        {
                            layout_info.base_offsets.insert(std::make_pair(ClangASTType::GetAsCXXRecordDecl(class_clang_type),
                                                                           clang::CharUnits::fromQuantity(member_byte_offset)));
                        }
                    }
                }
            }
            break;

        default:
            break;
        }
    }
    
    return count;
}


clang::DeclContext*
SymbolFileDWARF::GetClangDeclContextContainingTypeUID (lldb::user_id_t type_uid)
{
    DWARFDebugInfo* debug_info = DebugInfo();
    if (debug_info && UserIDMatches(type_uid))
    {
        DWARFCompileUnitSP cu_sp;
        const DWARFDebugInfoEntry* die = debug_info->GetDIEPtr(type_uid, &cu_sp);
        if (die)
            return GetClangDeclContextContainingDIE (cu_sp.get(), die, NULL);
    }
    return NULL;
}

clang::DeclContext*
SymbolFileDWARF::GetClangDeclContextForTypeUID (const lldb_private::SymbolContext &sc, lldb::user_id_t type_uid)
{
    if (UserIDMatches(type_uid))
        return GetClangDeclContextForDIEOffset (sc, type_uid);
    return NULL;
}

Type*
SymbolFileDWARF::ResolveTypeUID (lldb::user_id_t type_uid)
{
    if (UserIDMatches(type_uid))
    {
        DWARFDebugInfo* debug_info = DebugInfo();
        if (debug_info)
        {
            DWARFCompileUnitSP cu_sp;
            const DWARFDebugInfoEntry* type_die = debug_info->GetDIEPtr(type_uid, &cu_sp);
            const bool assert_not_being_parsed = true;
            return ResolveTypeUID (cu_sp.get(), type_die, assert_not_being_parsed);
        }
    }
    return NULL;
}

Type*
SymbolFileDWARF::ResolveTypeUID (DWARFCompileUnit* cu, const DWARFDebugInfoEntry* die, bool assert_not_being_parsed)
{    
    if (die != NULL)
    {
        LogSP log (LogChannelDWARF::GetLogIfAll(DWARF_LOG_DEBUG_INFO));
        if (log)
            GetObjectFile()->GetModule()->LogMessage (log.get(), 
                                                      "SymbolFileDWARF::ResolveTypeUID (die = 0x%8.8x) %s '%s'", 
                                                      die->GetOffset(), 
                                                      DW_TAG_value_to_name(die->Tag()), 
                                                      die->GetName(this, cu));

        // We might be coming in in the middle of a type tree (a class
        // withing a class, an enum within a class), so parse any needed
        // parent DIEs before we get to this one...
        const DWARFDebugInfoEntry *decl_ctx_die = GetDeclContextDIEContainingDIE (cu, die);
        switch (decl_ctx_die->Tag())
        {
            case DW_TAG_structure_type:
            case DW_TAG_union_type:
            case DW_TAG_class_type:
            {
                // Get the type, which could be a forward declaration
                if (log)
                    GetObjectFile()->GetModule()->LogMessage (log.get(), 
                                                              "SymbolFileDWARF::ResolveTypeUID (die = 0x%8.8x) %s '%s' resolve parent forward type for 0x%8.8x", 
                                                              die->GetOffset(), 
                                                              DW_TAG_value_to_name(die->Tag()), 
                                                              die->GetName(this, cu), 
                                                              decl_ctx_die->GetOffset());
//
//                Type *parent_type = ResolveTypeUID (cu, decl_ctx_die, assert_not_being_parsed);
//                if (child_requires_parent_class_union_or_struct_to_be_completed(die->Tag()))
//                {
//                    if (log)
//                        GetObjectFile()->GetModule()->LogMessage (log.get(), 
//                                                                  "SymbolFileDWARF::ResolveTypeUID (die = 0x%8.8x) %s '%s' resolve parent full type for 0x%8.8x since die is a function", 
//                                                                  die->GetOffset(), 
//                                                                  DW_TAG_value_to_name(die->Tag()), 
//                                                                  die->GetName(this, cu), 
//                                                                  decl_ctx_die->GetOffset());
//                    // Ask the type to complete itself if it already hasn't since if we
//                    // want a function (method or static) from a class, the class must 
//                    // create itself and add it's own methods and class functions.
//                    if (parent_type)
//                        parent_type->GetClangFullType();
//                }
            }
            break;

            default:
                break;
        }
        return ResolveType (cu, die);
    }
    return NULL;
}

// This function is used when SymbolFileDWARFDebugMap owns a bunch of
// SymbolFileDWARF objects to detect if this DWARF file is the one that
// can resolve a clang_type.
bool
SymbolFileDWARF::HasForwardDeclForClangType (lldb::clang_type_t clang_type)
{
    clang_type_t clang_type_no_qualifiers = ClangASTType::RemoveFastQualifiers(clang_type);
    const DWARFDebugInfoEntry* die = m_forward_decl_clang_type_to_die.lookup (clang_type_no_qualifiers);
    return die != NULL;
}


lldb::clang_type_t
SymbolFileDWARF::ResolveClangOpaqueTypeDefinition (lldb::clang_type_t clang_type)
{
    // We have a struct/union/class/enum that needs to be fully resolved.
    clang_type_t clang_type_no_qualifiers = ClangASTType::RemoveFastQualifiers(clang_type);
    const DWARFDebugInfoEntry* die = m_forward_decl_clang_type_to_die.lookup (clang_type_no_qualifiers);
    if (die == NULL)
    {
        // We have already resolved this type...
        return clang_type;
    }
    // Once we start resolving this type, remove it from the forward declaration
    // map in case anyone child members or other types require this type to get resolved.
    // The type will get resolved when all of the calls to SymbolFileDWARF::ResolveClangOpaqueTypeDefinition
    // are done.
    m_forward_decl_clang_type_to_die.erase (clang_type_no_qualifiers);
    

    // Disable external storage for this type so we don't get anymore 
    // clang::ExternalASTSource queries for this type.
    ClangASTContext::SetHasExternalStorage (clang_type, false);

    DWARFDebugInfo* debug_info = DebugInfo();

    DWARFCompileUnit *dwarf_cu = debug_info->GetCompileUnitContainingDIE (die->GetOffset()).get();
    Type *type = m_die_to_type.lookup (die);

    const dw_tag_t tag = die->Tag();

    LogSP log (LogChannelDWARF::GetLogIfAny(DWARF_LOG_DEBUG_INFO|DWARF_LOG_TYPE_COMPLETION));
    if (log)
    {
        GetObjectFile()->GetModule()->LogMessageVerboseBacktrace (log.get(),
                                                                  "0x%8.8llx: %s '%s' resolving forward declaration...",
                                                                  MakeUserID(die->GetOffset()),
                                                                  DW_TAG_value_to_name(tag),
                                                                  type->GetName().AsCString());
    
    }
    assert (clang_type);
    DWARFDebugInfoEntry::Attributes attributes;

    ClangASTContext &ast = GetClangASTContext();

    switch (tag)
    {
    case DW_TAG_structure_type:
    case DW_TAG_union_type:
    case DW_TAG_class_type:
        {
            LayoutInfo layout_info;
            
            {
                if (die->HasChildren())
                {
                    
                    LanguageType class_language = eLanguageTypeUnknown;
                    bool is_objc_class = ClangASTContext::IsObjCClassType (clang_type);
                    if (is_objc_class)
                    {
                        class_language = eLanguageTypeObjC;
                        // For objective C we don't start the definition when
                        // the class is created.
                        ast.StartTagDeclarationDefinition (clang_type);
                    }
                    
                    int tag_decl_kind = -1;
                    AccessType default_accessibility = eAccessNone;
                    if (tag == DW_TAG_structure_type)
                    {
                        tag_decl_kind = clang::TTK_Struct;
                        default_accessibility = eAccessPublic;
                    }
                    else if (tag == DW_TAG_union_type)
                    {
                        tag_decl_kind = clang::TTK_Union;
                        default_accessibility = eAccessPublic;
                    }
                    else if (tag == DW_TAG_class_type)
                    {
                        tag_decl_kind = clang::TTK_Class;
                        default_accessibility = eAccessPrivate;
                    }
                    
                    SymbolContext sc(GetCompUnitForDWARFCompUnit(dwarf_cu));
                    std::vector<clang::CXXBaseSpecifier *> base_classes;
                    std::vector<int> member_accessibilities;
                    bool is_a_class = false;
                    // Parse members and base classes first
                    DWARFDIECollection member_function_dies;
                                        
                    DelayedPropertyList delayed_properties;
                    BitfieldMap bitfield_map;
                    ParseChildMembers (sc,
                                       dwarf_cu,
                                       die, 
                                       clang_type,
                                       class_language,
                                       base_classes, 
                                       member_accessibilities,
                                       member_function_dies,
                                       bitfield_map,
                                       delayed_properties,
                                       default_accessibility, 
                                       is_a_class,
                                       layout_info);
                    
                    // Now parse any methods if there were any...
                    size_t num_functions = member_function_dies.Size();                
                    if (num_functions > 0)
                    {
                        for (size_t i=0; i<num_functions; ++i)
                        {
                            ResolveType(dwarf_cu, member_function_dies.GetDIEPtrAtIndex(i));
                        }
                    }
                    
                    if (class_language == eLanguageTypeObjC)
                    {
                        std::string class_str (ClangASTType::GetTypeNameForOpaqueQualType(ast.getASTContext(), clang_type));
                        if (!class_str.empty())
                        {
                            
                            DIEArray method_die_offsets;
                            if (m_using_apple_tables)
                            {
                                if (m_apple_objc_ap.get())
                                    m_apple_objc_ap->FindByName(class_str.c_str(), method_die_offsets);
                            }
                            else
                            {
                                if (!m_indexed)
                                    Index ();
                                
                                ConstString class_name (class_str.c_str());
                                m_objc_class_selectors_index.Find (class_name, method_die_offsets);
                            }
                            
                            if (!method_die_offsets.empty())
                            {
                                DWARFDebugInfo* debug_info = DebugInfo();
                                
                                DWARFCompileUnit* method_cu = NULL;
                                const size_t num_matches = method_die_offsets.size();
                                for (size_t i=0; i<num_matches; ++i)
                                {
                                    const dw_offset_t die_offset = method_die_offsets[i];
                                    DWARFDebugInfoEntry *method_die = debug_info->GetDIEPtrWithCompileUnitHint (die_offset, &method_cu);
                                    
                                    if (method_die)
                                        ResolveType (method_cu, method_die);
                                    else
                                    {
                                        if (m_using_apple_tables)
                                        {
                                            GetObjectFile()->GetModule()->ReportErrorIfModifyDetected ("the DWARF debug information has been modified (.apple_objc accelerator table had bad die 0x%8.8x for '%s')\n",
                                                                                                       die_offset, class_str.c_str());
                                        }
                                    }            
                                }
                            }
                            
                            for (DelayedPropertyList::const_iterator pi = delayed_properties.begin(), pe = delayed_properties.end();
                                 pi != pe;
                                 ++pi)
                                pi->Finalize();
                        }
                    }
                    
                    // If we have a DW_TAG_structure_type instead of a DW_TAG_class_type we
                    // need to tell the clang type it is actually a class.
                    if (class_language != eLanguageTypeObjC)
                    {
                        if (is_a_class && tag_decl_kind != clang::TTK_Class)
                            ast.SetTagTypeKind (clang_type, clang::TTK_Class);
                    }
                    
                    // Since DW_TAG_structure_type gets used for both classes
                    // and structures, we may need to set any DW_TAG_member
                    // fields to have a "private" access if none was specified.
                    // When we parsed the child members we tracked that actual
                    // accessibility value for each DW_TAG_member in the
                    // "member_accessibilities" array. If the value for the
                    // member is zero, then it was set to the "default_accessibility"
                    // which for structs was "public". Below we correct this
                    // by setting any fields to "private" that weren't correctly
                    // set.
                    if (is_a_class && !member_accessibilities.empty())
                    {
                        // This is a class and all members that didn't have
                        // their access specified are private.
                        ast.SetDefaultAccessForRecordFields (clang_type, 
                                                             eAccessPrivate, 
                                                             &member_accessibilities.front(), 
                                                             member_accessibilities.size());
                    }
                    
                    if (!base_classes.empty())
                    {
                        ast.SetBaseClassesForClassType (clang_type, 
                                                        &base_classes.front(), 
                                                        base_classes.size());
                        
                        // Clang will copy each CXXBaseSpecifier in "base_classes"
                        // so we have to free them all.
                        ClangASTContext::DeleteBaseClassSpecifiers (&base_classes.front(), 
                                                                    base_classes.size());
                    }
                }
            }
            
            ast.BuildIndirectFields (clang_type);
            
            ast.CompleteTagDeclarationDefinition (clang_type);
            
            if (!layout_info.field_offsets.empty() ||
                !layout_info.base_offsets.empty()  ||
                !layout_info.vbase_offsets.empty() )
            {
                if (type)
                    layout_info.bit_size = type->GetByteSize() * 8;
                if (layout_info.bit_size == 0)
                    layout_info.bit_size = die->GetAttributeValueAsUnsigned(this, dwarf_cu, DW_AT_byte_size, 0) * 8;
                
                clang::CXXRecordDecl *record_decl = ClangASTType::GetAsCXXRecordDecl(clang_type);
                if (record_decl)
                {
                    if (log)
                    {
                        GetObjectFile()->GetModule()->LogMessage (log.get(), 
                                                                  "SymbolFileDWARF::ResolveClangOpaqueTypeDefinition (clang_type = %p) caching layout info for record_decl = %p, bit_size = %llu, alignment = %llu, field_offsets[%u], base_offsets[%u], vbase_offsets[%u])",
                                                                  clang_type,
                                                                  record_decl,
                                                                  layout_info.bit_size,
                                                                  layout_info.alignment,
                                                                  (uint32_t)layout_info.field_offsets.size(),
                                                                  (uint32_t)layout_info.base_offsets.size(),
                                                                  (uint32_t)layout_info.vbase_offsets.size());
                        
                        uint32_t idx;
                        {
                        llvm::DenseMap <const clang::FieldDecl *, uint64_t>::const_iterator pos, end = layout_info.field_offsets.end();
                        for (idx = 0, pos = layout_info.field_offsets.begin(); pos != end; ++pos, ++idx)
                        {
                            GetObjectFile()->GetModule()->LogMessage (log.get(), 
                                                                      "SymbolFileDWARF::ResolveClangOpaqueTypeDefinition (clang_type = %p) field[%u] = { bit_offset=%u, name='%s' }",
                                                                      clang_type,
                                                                      idx,
                                                                      (uint32_t)pos->second,
                                                                      pos->first->getNameAsString().c_str());
                        }
                        }
                        
                        {
                        llvm::DenseMap <const clang::CXXRecordDecl *, clang::CharUnits>::const_iterator base_pos, base_end = layout_info.base_offsets.end();
                        for (idx = 0, base_pos = layout_info.base_offsets.begin(); base_pos != base_end; ++base_pos, ++idx)
                        {
                            GetObjectFile()->GetModule()->LogMessage (log.get(),
                                                                      "SymbolFileDWARF::ResolveClangOpaqueTypeDefinition (clang_type = %p) base[%u] = { byte_offset=%u, name='%s' }",
                                                                      clang_type,
                                                                      idx,
                                                                      (uint32_t)base_pos->second.getQuantity(),
                                                                      base_pos->first->getNameAsString().c_str());
                        }
                        }
                        {
                        llvm::DenseMap <const clang::CXXRecordDecl *, clang::CharUnits>::const_iterator vbase_pos, vbase_end = layout_info.vbase_offsets.end();
                        for (idx = 0, vbase_pos = layout_info.vbase_offsets.begin(); vbase_pos != vbase_end; ++vbase_pos, ++idx)
                        {
                            GetObjectFile()->GetModule()->LogMessage (log.get(),
                                                                      "SymbolFileDWARF::ResolveClangOpaqueTypeDefinition (clang_type = %p) vbase[%u] = { byte_offset=%u, name='%s' }",
                                                                      clang_type,
                                                                      idx,
                                                                      (uint32_t)vbase_pos->second.getQuantity(),
                                                                      vbase_pos->first->getNameAsString().c_str());
                        }
                        }
                    }
                    m_record_decl_to_layout_map.insert(std::make_pair(record_decl, layout_info));
                }
            }
        }

        return clang_type;

    case DW_TAG_enumeration_type:
        ast.StartTagDeclarationDefinition (clang_type);
        if (die->HasChildren())
        {
            SymbolContext sc(GetCompUnitForDWARFCompUnit(dwarf_cu));
            ParseChildEnumerators(sc, clang_type, type->GetByteSize(), dwarf_cu, die);
        }
        ast.CompleteTagDeclarationDefinition (clang_type);
        return clang_type;

    default:
        assert(false && "not a forward clang type decl!");
        break;
    }
    return NULL;
}

Type*
SymbolFileDWARF::ResolveType (DWARFCompileUnit* dwarf_cu, const DWARFDebugInfoEntry* type_die, bool assert_not_being_parsed)
{
    if (type_die != NULL)
    {
        Type *type = m_die_to_type.lookup (type_die);

        if (type == NULL)
            type = GetTypeForDIE (dwarf_cu, type_die).get();

        if (assert_not_being_parsed)
        { 
            if (type != DIE_IS_BEING_PARSED)
                return type;
            
            GetObjectFile()->GetModule()->ReportError ("Parsing a die that is being parsed die: 0x%8.8x: %s %s",
                                                       type_die->GetOffset(), 
                                                       DW_TAG_value_to_name(type_die->Tag()), 
                                                       type_die->GetName(this, dwarf_cu));

        }
        else
            return type;
    }
    return NULL;
}

CompileUnit*
SymbolFileDWARF::GetCompUnitForDWARFCompUnit (DWARFCompileUnit* dwarf_cu, uint32_t cu_idx)
{
    // Check if the symbol vendor already knows about this compile unit?
    if (dwarf_cu->GetUserData() == NULL)
    {
        // The symbol vendor doesn't know about this compile unit, we
        // need to parse and add it to the symbol vendor object.
        return ParseCompileUnit(dwarf_cu, cu_idx).get();
    }
    return (CompileUnit*)dwarf_cu->GetUserData();
}

bool
SymbolFileDWARF::GetFunction (DWARFCompileUnit* dwarf_cu, const DWARFDebugInfoEntry* func_die, SymbolContext& sc)
{
    sc.Clear();
    // Check if the symbol vendor already knows about this compile unit?
    sc.comp_unit = GetCompUnitForDWARFCompUnit(dwarf_cu, UINT32_MAX);

    sc.function = sc.comp_unit->FindFunctionByUID (MakeUserID(func_die->GetOffset())).get();
    if (sc.function == NULL)
        sc.function = ParseCompileUnitFunction(sc, dwarf_cu, func_die);
        
    if (sc.function)
    {        
        sc.module_sp = sc.function->CalculateSymbolContextModule();
        return true;
    }
    
    return false;
}

uint32_t
SymbolFileDWARF::ResolveSymbolContext (const Address& so_addr, uint32_t resolve_scope, SymbolContext& sc)
{
    Timer scoped_timer(__PRETTY_FUNCTION__,
                       "SymbolFileDWARF::ResolveSymbolContext (so_addr = { section = %p, offset = 0x%llx }, resolve_scope = 0x%8.8x)",
                       so_addr.GetSection().get(),
                       so_addr.GetOffset(),
                       resolve_scope);
    uint32_t resolved = 0;
    if (resolve_scope & (   eSymbolContextCompUnit |
                            eSymbolContextFunction |
                            eSymbolContextBlock |
                            eSymbolContextLineEntry))
    {
        lldb::addr_t file_vm_addr = so_addr.GetFileAddress();

        DWARFDebugInfo* debug_info = DebugInfo();
        if (debug_info)
        {
            const dw_offset_t cu_offset = debug_info->GetCompileUnitAranges().FindAddress(file_vm_addr);
            if (cu_offset != DW_INVALID_OFFSET)
            {
                uint32_t cu_idx = DW_INVALID_INDEX;
                DWARFCompileUnit* dwarf_cu = debug_info->GetCompileUnit(cu_offset, &cu_idx).get();
                if (dwarf_cu)
                {
                    sc.comp_unit = GetCompUnitForDWARFCompUnit(dwarf_cu, cu_idx);
                    if (sc.comp_unit)
                    {
                        resolved |= eSymbolContextCompUnit;

                        if (resolve_scope & eSymbolContextLineEntry)
                        {
                            LineTable *line_table = sc.comp_unit->GetLineTable();
                            if (line_table != NULL)
                            {
                                if (so_addr.IsLinkedAddress())
                                {
                                    Address linked_addr (so_addr);
                                    linked_addr.ResolveLinkedAddress();
                                    if (line_table->FindLineEntryByAddress (linked_addr, sc.line_entry))
                                    {
                                        resolved |= eSymbolContextLineEntry;
                                    }
                                }
                                else if (line_table->FindLineEntryByAddress (so_addr, sc.line_entry))
                                {
                                    resolved |= eSymbolContextLineEntry;
                                }
                            }
                        }

                        if (resolve_scope & (eSymbolContextFunction | eSymbolContextBlock))
                        {
                            DWARFDebugInfoEntry *function_die = NULL;
                            DWARFDebugInfoEntry *block_die = NULL;
                            if (resolve_scope & eSymbolContextBlock)
                            {
                                dwarf_cu->LookupAddress(file_vm_addr, &function_die, &block_die);
                            }
                            else
                            {
                                dwarf_cu->LookupAddress(file_vm_addr, &function_die, NULL);
                            }

                            if (function_die != NULL)
                            {
                                sc.function = sc.comp_unit->FindFunctionByUID (MakeUserID(function_die->GetOffset())).get();
                                if (sc.function == NULL)
                                    sc.function = ParseCompileUnitFunction(sc, dwarf_cu, function_die);
                            }
                            else
                            {
                                // We might have had a compile unit that had discontiguous
                                // address ranges where the gaps are symbols that don't have
                                // any debug info. Discontiguous compile unit address ranges
                                // should only happen when there aren't other functions from
                                // other compile units in these gaps. This helps keep the size
                                // of the aranges down.
                                sc.comp_unit = NULL;
                                resolved &= ~eSymbolContextCompUnit;
                            }

                            if (sc.function != NULL)
                            {
                                resolved |= eSymbolContextFunction;

                                if (resolve_scope & eSymbolContextBlock)
                                {
                                    Block& block = sc.function->GetBlock (true);

                                    if (block_die != NULL)
                                        sc.block = block.FindBlockByID (MakeUserID(block_die->GetOffset()));
                                    else
                                        sc.block = block.FindBlockByID (MakeUserID(function_die->GetOffset()));
                                    if (sc.block)
                                        resolved |= eSymbolContextBlock;
                                }
                            }
                        }
                    }
                    else
                    {
                        GetObjectFile()->GetModule()->ReportWarning ("0x%8.8x: compile unit %u failed to create a valid lldb_private::CompileUnit class.",
                                                                     cu_offset,
                                                                     cu_idx);
                    }
                }
            }
        }
    }
    return resolved;
}



uint32_t
SymbolFileDWARF::ResolveSymbolContext(const FileSpec& file_spec, uint32_t line, bool check_inlines, uint32_t resolve_scope, SymbolContextList& sc_list)
{
    const uint32_t prev_size = sc_list.GetSize();
    if (resolve_scope & eSymbolContextCompUnit)
    {
        DWARFDebugInfo* debug_info = DebugInfo();
        if (debug_info)
        {
            uint32_t cu_idx;
            DWARFCompileUnit* dwarf_cu = NULL;

            for (cu_idx = 0; (dwarf_cu = debug_info->GetCompileUnitAtIndex(cu_idx)) != NULL; ++cu_idx)
            {
                CompileUnit *dc_cu = GetCompUnitForDWARFCompUnit(dwarf_cu, cu_idx);
                const bool full_match = file_spec.GetDirectory();
                bool file_spec_matches_cu_file_spec = dc_cu != NULL && FileSpec::Equal(file_spec, *dc_cu, full_match);
                if (check_inlines || file_spec_matches_cu_file_spec)
                {
                    SymbolContext sc (m_obj_file->GetModule());
                    sc.comp_unit = GetCompUnitForDWARFCompUnit(dwarf_cu, cu_idx);
                    if (sc.comp_unit)
                    {
                        uint32_t file_idx = UINT32_MAX;

                        // If we are looking for inline functions only and we don't
                        // find it in the support files, we are done.
                        if (check_inlines)
                        {
                            file_idx = sc.comp_unit->GetSupportFiles().FindFileIndex (1, file_spec, true);
                            if (file_idx == UINT32_MAX)
                                continue;
                        }

                        if (line != 0)
                        {
                            LineTable *line_table = sc.comp_unit->GetLineTable();

                            if (line_table != NULL && line != 0)
                            {
                                // We will have already looked up the file index if
                                // we are searching for inline entries.
                                if (!check_inlines)
                                    file_idx = sc.comp_unit->GetSupportFiles().FindFileIndex (1, file_spec, true);

                                if (file_idx != UINT32_MAX)
                                {
                                    uint32_t found_line;
                                    uint32_t line_idx = line_table->FindLineEntryIndexByFileIndex (0, file_idx, line, false, &sc.line_entry);
                                    found_line = sc.line_entry.line;

                                    while (line_idx != UINT32_MAX)
                                    {
                                        sc.function = NULL;
                                        sc.block = NULL;
                                        if (resolve_scope & (eSymbolContextFunction | eSymbolContextBlock))
                                        {
                                            const lldb::addr_t file_vm_addr = sc.line_entry.range.GetBaseAddress().GetFileAddress();
                                            if (file_vm_addr != LLDB_INVALID_ADDRESS)
                                            {
                                                DWARFDebugInfoEntry *function_die = NULL;
                                                DWARFDebugInfoEntry *block_die = NULL;
                                                dwarf_cu->LookupAddress(file_vm_addr, &function_die, resolve_scope & eSymbolContextBlock ? &block_die : NULL);

                                                if (function_die != NULL)
                                                {
                                                    sc.function = sc.comp_unit->FindFunctionByUID (MakeUserID(function_die->GetOffset())).get();
                                                    if (sc.function == NULL)
                                                        sc.function = ParseCompileUnitFunction(sc, dwarf_cu, function_die);
                                                }

                                                if (sc.function != NULL)
                                                {
                                                    Block& block = sc.function->GetBlock (true);

                                                    if (block_die != NULL)
                                                        sc.block = block.FindBlockByID (MakeUserID(block_die->GetOffset()));
                                                    else
                                                        sc.block = block.FindBlockByID (MakeUserID(function_die->GetOffset()));
                                                }
                                            }
                                        }

                                        sc_list.Append(sc);
                                        line_idx = line_table->FindLineEntryIndexByFileIndex (line_idx + 1, file_idx, found_line, true, &sc.line_entry);
                                    }
                                }
                            }
                            else if (file_spec_matches_cu_file_spec && !check_inlines)
                            {
                                // only append the context if we aren't looking for inline call sites
                                // by file and line and if the file spec matches that of the compile unit
                                sc_list.Append(sc);
                            }
                        }
                        else if (file_spec_matches_cu_file_spec && !check_inlines)
                        {
                            // only append the context if we aren't looking for inline call sites
                            // by file and line and if the file spec matches that of the compile unit
                            sc_list.Append(sc);
                        }

                        if (!check_inlines)
                            break;
                    }
                }
            }
        }
    }
    return sc_list.GetSize() - prev_size;
}

void
SymbolFileDWARF::Index ()
{
    if (m_indexed)
        return;
    m_indexed = true;
    Timer scoped_timer (__PRETTY_FUNCTION__,
                        "SymbolFileDWARF::Index (%s)",
                        GetObjectFile()->GetFileSpec().GetFilename().AsCString());

    DWARFDebugInfo* debug_info = DebugInfo();
    if (debug_info)
    {
        uint32_t cu_idx = 0;
        const uint32_t num_compile_units = GetNumCompileUnits();
        for (cu_idx = 0; cu_idx < num_compile_units; ++cu_idx)
        {
            DWARFCompileUnit* dwarf_cu = debug_info->GetCompileUnitAtIndex(cu_idx);

            bool clear_dies = dwarf_cu->ExtractDIEsIfNeeded (false) > 1;

            dwarf_cu->Index (cu_idx,
                             m_function_basename_index,
                             m_function_fullname_index,
                             m_function_method_index,
                             m_function_selector_index,
                             m_objc_class_selectors_index,
                             m_global_index, 
                             m_type_index,
                             m_namespace_index);
            
            // Keep memory down by clearing DIEs if this generate function
            // caused them to be parsed
            if (clear_dies)
                dwarf_cu->ClearDIEs (true);
        }
        
        m_function_basename_index.Finalize();
        m_function_fullname_index.Finalize();
        m_function_method_index.Finalize();
        m_function_selector_index.Finalize();
        m_objc_class_selectors_index.Finalize();
        m_global_index.Finalize(); 
        m_type_index.Finalize();
        m_namespace_index.Finalize();

#if defined (ENABLE_DEBUG_PRINTF)
        StreamFile s(stdout, false);
        s.Printf ("DWARF index for '%s/%s':", 
                  GetObjectFile()->GetFileSpec().GetDirectory().AsCString(), 
                  GetObjectFile()->GetFileSpec().GetFilename().AsCString());
        s.Printf("\nFunction basenames:\n");    m_function_basename_index.Dump (&s);
        s.Printf("\nFunction fullnames:\n");    m_function_fullname_index.Dump (&s);
        s.Printf("\nFunction methods:\n");      m_function_method_index.Dump (&s);
        s.Printf("\nFunction selectors:\n");    m_function_selector_index.Dump (&s);
        s.Printf("\nObjective C class selectors:\n");    m_objc_class_selectors_index.Dump (&s);
        s.Printf("\nGlobals and statics:\n");   m_global_index.Dump (&s); 
        s.Printf("\nTypes:\n");                 m_type_index.Dump (&s);
        s.Printf("\nNamepaces:\n");             m_namespace_index.Dump (&s);
#endif
    }
}

bool
SymbolFileDWARF::NamespaceDeclMatchesThisSymbolFile (const ClangNamespaceDecl *namespace_decl)
{
    if (namespace_decl == NULL)
    {
        // Invalid namespace decl which means we aren't matching only things
        // in this symbol file, so return true to indicate it matches this
        // symbol file.
        return true;
    }
    
    clang::ASTContext *namespace_ast = namespace_decl->GetASTContext();

    if (namespace_ast == NULL)
        return true;    // No AST in the "namespace_decl", return true since it 
                        // could then match any symbol file, including this one

    if (namespace_ast == GetClangASTContext().getASTContext())
        return true;    // The ASTs match, return true
    
    // The namespace AST was valid, and it does not match...
    LogSP log (LogChannelDWARF::GetLogIfAll(DWARF_LOG_LOOKUPS));

    if (log)
        GetObjectFile()->GetModule()->LogMessage(log.get(), "Valid namespace does not match symbol file");
    
    return false;
}

bool
SymbolFileDWARF::DIEIsInNamespace (const ClangNamespaceDecl *namespace_decl, 
                                   DWARFCompileUnit* cu, 
                                   const DWARFDebugInfoEntry* die)
{
    // No namespace specified, so the answesr i
    if (namespace_decl == NULL)
        return true;
    
    LogSP log (LogChannelDWARF::GetLogIfAll(DWARF_LOG_LOOKUPS));

    const DWARFDebugInfoEntry *decl_ctx_die = NULL;
    clang::DeclContext *die_clang_decl_ctx = GetClangDeclContextContainingDIE (cu, die, &decl_ctx_die);
    if (decl_ctx_die)
    { 
        clang::NamespaceDecl *clang_namespace_decl = namespace_decl->GetNamespaceDecl();

        if (clang_namespace_decl)
        {
            if (decl_ctx_die->Tag() != DW_TAG_namespace)
            {
                if (log)
                    GetObjectFile()->GetModule()->LogMessage(log.get(), "Found a match, but its parent is not a namespace");
                return false;
            }
                
            if (clang_namespace_decl == die_clang_decl_ctx)
                return true;
            else
                return false;
        }
        else
        {
            // We have a namespace_decl that was not NULL but it contained
            // a NULL "clang::NamespaceDecl", so this means the global namespace
            // So as long the the contained decl context DIE isn't a namespace
            // we should be ok.
            if (decl_ctx_die->Tag() != DW_TAG_namespace)
                return true;
        }
    }
    
    if (log)
        GetObjectFile()->GetModule()->LogMessage(log.get(), "Found a match, but its parent doesn't exist");
    
    return false;
}
uint32_t
SymbolFileDWARF::FindGlobalVariables (const ConstString &name, const lldb_private::ClangNamespaceDecl *namespace_decl, bool append, uint32_t max_matches, VariableList& variables)
{
    LogSP log (LogChannelDWARF::GetLogIfAll(DWARF_LOG_LOOKUPS));

    if (log)
    {
        GetObjectFile()->GetModule()->LogMessage (log.get(), 
                                                  "SymbolFileDWARF::FindGlobalVariables (name=\"%s\", namespace_decl=%p, append=%u, max_matches=%u, variables)", 
                                                  name.GetCString(), 
                                                  namespace_decl,
                                                  append, 
                                                  max_matches);
    }
    
    if (!NamespaceDeclMatchesThisSymbolFile(namespace_decl))
		return 0;
    
    DWARFDebugInfo* info = DebugInfo();
    if (info == NULL)
        return 0;

    // If we aren't appending the results to this list, then clear the list
    if (!append)
        variables.Clear();

    // Remember how many variables are in the list before we search in case
    // we are appending the results to a variable list.
    const uint32_t original_size = variables.GetSize();

    DIEArray die_offsets;
    
    if (m_using_apple_tables)
    {
        if (m_apple_names_ap.get())
        {
            const char *name_cstr = name.GetCString();
            const char *base_name_start;
            const char *base_name_end = NULL;
            
            if (!CPPLanguageRuntime::StripNamespacesFromVariableName(name_cstr, base_name_start, base_name_end))
                base_name_start = name_cstr;
                
            m_apple_names_ap->FindByName (base_name_start, die_offsets);
        }
    }
    else
    {
        // Index the DWARF if we haven't already
        if (!m_indexed)
            Index ();

        m_global_index.Find (name, die_offsets);
    }
    
    const size_t num_die_matches = die_offsets.size();
    if (num_die_matches)
    {
        SymbolContext sc;
        sc.module_sp = m_obj_file->GetModule();
        assert (sc.module_sp);
        
        DWARFDebugInfo* debug_info = DebugInfo();
        DWARFCompileUnit* dwarf_cu = NULL;
        const DWARFDebugInfoEntry* die = NULL;
        bool done = false;
        for (size_t i=0; i<num_die_matches && !done; ++i)
        {
            const dw_offset_t die_offset = die_offsets[i];
            die = debug_info->GetDIEPtrWithCompileUnitHint (die_offset, &dwarf_cu);

            if (die)
            {
                switch (die->Tag())
                {
                    default:
                    case DW_TAG_subprogram:
                    case DW_TAG_inlined_subroutine:
                    case DW_TAG_try_block:
                    case DW_TAG_catch_block:
                        break;
                        
                    case DW_TAG_variable:
                        {
                            sc.comp_unit = GetCompUnitForDWARFCompUnit(dwarf_cu, UINT32_MAX);
                    
                            if (namespace_decl && !DIEIsInNamespace (namespace_decl, dwarf_cu, die))
                                continue;

                            ParseVariables(sc, dwarf_cu, LLDB_INVALID_ADDRESS, die, false, false, &variables);

                            if (variables.GetSize() - original_size >= max_matches)
                                done = true;
                        }
                        break;
                }
            }
            else
            {
                if (m_using_apple_tables)
                {
                    GetObjectFile()->GetModule()->ReportErrorIfModifyDetected ("the DWARF debug information has been modified (.apple_names accelerator table had bad die 0x%8.8x for '%s')\n",
                                                                               die_offset, name.GetCString());
                }
            }
        }
    }

    // Return the number of variable that were appended to the list
    const uint32_t num_matches = variables.GetSize() - original_size;
    if (log && num_matches > 0)
    {
        GetObjectFile()->GetModule()->LogMessage (log.get(), 
                                                  "SymbolFileDWARF::FindGlobalVariables (name=\"%s\", namespace_decl=%p, append=%u, max_matches=%u, variables) => %u",
                                                  name.GetCString(), 
                                                  namespace_decl,
                                                  append, 
                                                  max_matches,
                                                  num_matches);
    }
    return num_matches;
}

uint32_t
SymbolFileDWARF::FindGlobalVariables(const RegularExpression& regex, bool append, uint32_t max_matches, VariableList& variables)
{
    LogSP log (LogChannelDWARF::GetLogIfAll(DWARF_LOG_LOOKUPS));
    
    if (log)
    {
        GetObjectFile()->GetModule()->LogMessage (log.get(), 
                                                  "SymbolFileDWARF::FindGlobalVariables (regex=\"%s\", append=%u, max_matches=%u, variables)", 
                                                  regex.GetText(), 
                                                  append, 
                                                  max_matches);
    }

    DWARFDebugInfo* info = DebugInfo();
    if (info == NULL)
        return 0;

    // If we aren't appending the results to this list, then clear the list
    if (!append)
        variables.Clear();

    // Remember how many variables are in the list before we search in case
    // we are appending the results to a variable list.
    const uint32_t original_size = variables.GetSize();

    DIEArray die_offsets;
    
    if (m_using_apple_tables)
    {
        if (m_apple_names_ap.get())
        {
            DWARFMappedHash::DIEInfoArray hash_data_array;
            if (m_apple_names_ap->AppendAllDIEsThatMatchingRegex (regex, hash_data_array))
                DWARFMappedHash::ExtractDIEArray (hash_data_array, die_offsets);
        }
    }
    else
    {
        // Index the DWARF if we haven't already
        if (!m_indexed)
            Index ();
        
        m_global_index.Find (regex, die_offsets);
    }

    SymbolContext sc;
    sc.module_sp = m_obj_file->GetModule();
    assert (sc.module_sp);
    
    DWARFCompileUnit* dwarf_cu = NULL;
    const DWARFDebugInfoEntry* die = NULL;
    const size_t num_matches = die_offsets.size();
    if (num_matches)
    {
        DWARFDebugInfo* debug_info = DebugInfo();
        for (size_t i=0; i<num_matches; ++i)
        {
            const dw_offset_t die_offset = die_offsets[i];
            die = debug_info->GetDIEPtrWithCompileUnitHint (die_offset, &dwarf_cu);
            
            if (die)
            {
                sc.comp_unit = GetCompUnitForDWARFCompUnit(dwarf_cu, UINT32_MAX);

                ParseVariables(sc, dwarf_cu, LLDB_INVALID_ADDRESS, die, false, false, &variables);

                if (variables.GetSize() - original_size >= max_matches)
                    break;
            }
            else
            {
                if (m_using_apple_tables)
                {
                    GetObjectFile()->GetModule()->ReportErrorIfModifyDetected ("the DWARF debug information has been modified (.apple_names accelerator table had bad die 0x%8.8x for regex '%s')\n",
                                                                               die_offset, regex.GetText());
                }
            }            
        }
    }

    // Return the number of variable that were appended to the list
    return variables.GetSize() - original_size;
}


bool
SymbolFileDWARF::ResolveFunction (dw_offset_t die_offset,
                                  DWARFCompileUnit *&dwarf_cu,
                                  SymbolContextList& sc_list)
{
    const DWARFDebugInfoEntry *die = DebugInfo()->GetDIEPtrWithCompileUnitHint (die_offset, &dwarf_cu);
    return ResolveFunction (dwarf_cu, die, sc_list);
}
    

bool
SymbolFileDWARF::ResolveFunction (DWARFCompileUnit *cu,
                                  const DWARFDebugInfoEntry *die,
                                  SymbolContextList& sc_list)
{
    SymbolContext sc;

    if (die == NULL)
        return false;

    // If we were passed a die that is not a function, just return false...
    if (die->Tag() != DW_TAG_subprogram && die->Tag() != DW_TAG_inlined_subroutine)
        return false;
    
    const DWARFDebugInfoEntry* inlined_die = NULL;
    if (die->Tag() == DW_TAG_inlined_subroutine)
    {
        inlined_die = die;
        
        while ((die = die->GetParent()) != NULL)
        {
            if (die->Tag() == DW_TAG_subprogram)
                break;
        }
    }
    assert (die->Tag() == DW_TAG_subprogram);
    if (GetFunction (cu, die, sc))
    {
        Address addr;
        // Parse all blocks if needed
        if (inlined_die)
        {
            sc.block = sc.function->GetBlock (true).FindBlockByID (MakeUserID(inlined_die->GetOffset()));
            assert (sc.block != NULL);
            if (sc.block->GetStartAddress (addr) == false)
                addr.Clear();
        }
        else 
        {
            sc.block = NULL;
            addr = sc.function->GetAddressRange().GetBaseAddress();
        }

        if (addr.IsValid())
        {
            sc_list.Append(sc);
            return true;
        }
    }
    
    return false;
}

void
SymbolFileDWARF::FindFunctions (const ConstString &name, 
                                const NameToDIE &name_to_die,
                                SymbolContextList& sc_list)
{
    DIEArray die_offsets;
    if (name_to_die.Find (name, die_offsets))
    {
        ParseFunctions (die_offsets, sc_list);
    }
}


void
SymbolFileDWARF::FindFunctions (const RegularExpression &regex, 
                                const NameToDIE &name_to_die,
                                SymbolContextList& sc_list)
{
    DIEArray die_offsets;
    if (name_to_die.Find (regex, die_offsets))
    {
        ParseFunctions (die_offsets, sc_list);
    }
}


void
SymbolFileDWARF::FindFunctions (const RegularExpression &regex, 
                                const DWARFMappedHash::MemoryTable &memory_table,
                                SymbolContextList& sc_list)
{
    DIEArray die_offsets;
    DWARFMappedHash::DIEInfoArray hash_data_array;
    if (memory_table.AppendAllDIEsThatMatchingRegex (regex, hash_data_array))
    {
        DWARFMappedHash::ExtractDIEArray (hash_data_array, die_offsets);
        ParseFunctions (die_offsets, sc_list);
    }
}

void
SymbolFileDWARF::ParseFunctions (const DIEArray &die_offsets,
                                 SymbolContextList& sc_list)
{
    const size_t num_matches = die_offsets.size();
    if (num_matches)
    {
        SymbolContext sc;

        DWARFCompileUnit* dwarf_cu = NULL;
        for (size_t i=0; i<num_matches; ++i)
        {
            const dw_offset_t die_offset = die_offsets[i];
            ResolveFunction (die_offset, dwarf_cu, sc_list);
        }
    }
}

bool
SymbolFileDWARF::FunctionDieMatchesPartialName (const DWARFDebugInfoEntry* die,
                                                const DWARFCompileUnit *dwarf_cu,
                                                uint32_t name_type_mask, 
                                                const char *partial_name,
                                                const char *base_name_start,
                                                const char *base_name_end)
{
    // If we are looking only for methods, throw away all the ones that aren't in C++ classes:
    if (name_type_mask == eFunctionNameTypeMethod
        || name_type_mask == eFunctionNameTypeBase)
    {
        clang::DeclContext *containing_decl_ctx = GetClangDeclContextContainingDIEOffset(die->GetOffset());
        if (!containing_decl_ctx)
            return false;
        
        bool is_cxx_method = DeclKindIsCXXClass(containing_decl_ctx->getDeclKind());
        
        if (!is_cxx_method && name_type_mask == eFunctionNameTypeMethod)
            return false;
        if (is_cxx_method && name_type_mask == eFunctionNameTypeBase)
            return false;
    }

    // Now we need to check whether the name we got back for this type matches the extra specifications
    // that were in the name we're looking up:
    if (base_name_start != partial_name || *base_name_end != '\0')
    {
        // First see if the stuff to the left matches the full name.  To do that let's see if
        // we can pull out the mips linkage name attribute:
        
        Mangled best_name;

        DWARFDebugInfoEntry::Attributes attributes;
        die->GetAttributes(this, dwarf_cu, NULL, attributes);
        uint32_t idx = attributes.FindAttributeIndex(DW_AT_MIPS_linkage_name);
        if (idx != UINT32_MAX)
        {
            DWARFFormValue form_value;
            if (attributes.ExtractFormValueAtIndex(this, idx, form_value))
            {
                const char *name = form_value.AsCString(&get_debug_str_data());
                best_name.SetValue (ConstString(name), true);
            } 
        }
        if (best_name)
        {
            const char *demangled = best_name.GetDemangledName().GetCString();
            if (demangled)
            {
                std::string name_no_parens(partial_name, base_name_end - partial_name);
                const char *partial_in_demangled = strstr (demangled, name_no_parens.c_str());
                if (partial_in_demangled == NULL)
                    return false;
                else
                {
                    // Sort out the case where our name is something like "Process::Destroy" and the match is
                    // "SBProcess::Destroy" - that shouldn't be a match.  We should really always match on
                    // namespace boundaries...
                    
                    if (partial_name[0] == ':'  && partial_name[1] == ':')
                    {
                        // The partial name was already on a namespace boundary so all matches are good.
                        return true;
                    }
                    else if (partial_in_demangled == demangled)
                    {
                        // They both start the same, so this is an good match.
                        return true;
                    }
                    else
                    {
                        if (partial_in_demangled - demangled == 1)
                        {
                            // Only one character difference, can't be a namespace boundary...
                            return false;
                        }
                        else if (*(partial_in_demangled - 1) == ':' && *(partial_in_demangled - 2) == ':')
                        {
                            // We are on a namespace boundary, so this is also good.
                            return true;
                        }
                        else
                            return false;
                    }
                }
            }
        }
    }
    
    return true;
}

uint32_t
SymbolFileDWARF::FindFunctions (const ConstString &name, 
                                const lldb_private::ClangNamespaceDecl *namespace_decl, 
                                uint32_t name_type_mask,
                                bool include_inlines,
                                bool append, 
                                SymbolContextList& sc_list)
{
    Timer scoped_timer (__PRETTY_FUNCTION__,
                        "SymbolFileDWARF::FindFunctions (name = '%s')",
                        name.AsCString());

    LogSP log (LogChannelDWARF::GetLogIfAll(DWARF_LOG_LOOKUPS));
    
    if (log)
    {
        GetObjectFile()->GetModule()->LogMessage (log.get(), 
                                                  "SymbolFileDWARF::FindFunctions (name=\"%s\", name_type_mask=0x%x, append=%u, sc_list)", 
                                                  name.GetCString(), 
                                                  name_type_mask, 
                                                  append);
    }

    // If we aren't appending the results to this list, then clear the list
    if (!append)
        sc_list.Clear();
    
    if (!NamespaceDeclMatchesThisSymbolFile(namespace_decl))
		return 0;
        
    // If name is empty then we won't find anything.
    if (name.IsEmpty())
        return 0;

    // Remember how many sc_list are in the list before we search in case
    // we are appending the results to a variable list.

    const uint32_t original_size = sc_list.GetSize();

    const char *name_cstr = name.GetCString();
    uint32_t effective_name_type_mask = eFunctionNameTypeNone;
    const char *base_name_start = name_cstr;
    const char *base_name_end = name_cstr + strlen(name_cstr);
    
    if (name_type_mask & eFunctionNameTypeAuto)
    {
        if (CPPLanguageRuntime::IsCPPMangledName (name_cstr))
            effective_name_type_mask = eFunctionNameTypeFull;
        else if (ObjCLanguageRuntime::IsPossibleObjCMethodName (name_cstr))
            effective_name_type_mask = eFunctionNameTypeFull;
        else
        {
            if (ObjCLanguageRuntime::IsPossibleObjCSelector(name_cstr))
                effective_name_type_mask |= eFunctionNameTypeSelector;
                
            if (CPPLanguageRuntime::IsPossibleCPPCall(name_cstr, base_name_start, base_name_end))
                effective_name_type_mask |= (eFunctionNameTypeMethod | eFunctionNameTypeBase);
        }
    }
    else
    {
        effective_name_type_mask = name_type_mask;
        if (effective_name_type_mask & eFunctionNameTypeMethod || name_type_mask & eFunctionNameTypeBase)
        {
            // If they've asked for a CPP method or function name and it can't be that, we don't
            // even need to search for CPP methods or names.
            if (!CPPLanguageRuntime::IsPossibleCPPCall(name_cstr, base_name_start, base_name_end))
            {
                effective_name_type_mask &= ~(eFunctionNameTypeMethod | eFunctionNameTypeBase);
                if (effective_name_type_mask == eFunctionNameTypeNone)
                    return 0;
            }
        }
        
        if (effective_name_type_mask & eFunctionNameTypeSelector)
        {
            if (!ObjCLanguageRuntime::IsPossibleObjCSelector(name_cstr))
            {
                effective_name_type_mask &= ~(eFunctionNameTypeSelector);
                if (effective_name_type_mask == eFunctionNameTypeNone)
                    return 0;
            }
        }
    }
    
    DWARFDebugInfo* info = DebugInfo();
    if (info == NULL)
        return 0;

    DWARFCompileUnit *dwarf_cu = NULL;
    if (m_using_apple_tables)
    {
        if (m_apple_names_ap.get())
        {

            DIEArray die_offsets;

            uint32_t num_matches = 0;
                
            if (effective_name_type_mask & eFunctionNameTypeFull)
            {
                // If they asked for the full name, match what they typed.  At some point we may
                // want to canonicalize this (strip double spaces, etc.  For now, we just add all the
                // dies that we find by exact match.
                num_matches = m_apple_names_ap->FindByName (name_cstr, die_offsets);
                for (uint32_t i = 0; i < num_matches; i++)
                {
                    const dw_offset_t die_offset = die_offsets[i];
                    const DWARFDebugInfoEntry *die = info->GetDIEPtrWithCompileUnitHint (die_offset, &dwarf_cu);
                    if (die)
                    {
                        if (namespace_decl && !DIEIsInNamespace (namespace_decl, dwarf_cu, die))
                            continue;
                        
                        if (!include_inlines && die->Tag() == DW_TAG_inlined_subroutine)
                            continue;
                        
                        ResolveFunction (dwarf_cu, die, sc_list);
                    }
                    else
                    {
                        GetObjectFile()->GetModule()->ReportErrorIfModifyDetected ("the DWARF debug information has been modified (.apple_names accelerator table had bad die 0x%8.8x for '%s')", 
                                                                                   die_offset, name_cstr);
                    }                                    
                }
            }
            else
            {                
                if (effective_name_type_mask & eFunctionNameTypeSelector)
                {
                    if (namespace_decl && *namespace_decl)
                        return 0; // no selectors in namespaces
                        
                    num_matches = m_apple_names_ap->FindByName (name_cstr, die_offsets);
                    // Now make sure these are actually ObjC methods.  In this case we can simply look up the name,
                    // and if it is an ObjC method name, we're good.
                    
                    for (uint32_t i = 0; i < num_matches; i++)
                    {
                        const dw_offset_t die_offset = die_offsets[i];
                        const DWARFDebugInfoEntry* die = info->GetDIEPtrWithCompileUnitHint (die_offset, &dwarf_cu);
                        if (die)
                        {
                            const char *die_name = die->GetName(this, dwarf_cu);
                            if (ObjCLanguageRuntime::IsPossibleObjCMethodName(die_name))
                            {
                                if (!include_inlines && die->Tag() == DW_TAG_inlined_subroutine)
                                    continue;
                                
                                ResolveFunction (dwarf_cu, die, sc_list);
                            }
                        }
                        else
                        {
                            GetObjectFile()->GetModule()->ReportError ("the DWARF debug information has been modified (.apple_names accelerator table had bad die 0x%8.8x for '%s')",
                                                                       die_offset, name_cstr);
                        }                                    
                    }
                    die_offsets.clear();
                }
                
                if (effective_name_type_mask & eFunctionNameTypeMethod
                    || effective_name_type_mask & eFunctionNameTypeBase)
                {
                    if ((effective_name_type_mask & eFunctionNameTypeMethod) &&
                        (namespace_decl && *namespace_decl))
                        return 0; // no methods in namespaces
                    
                    // The apple_names table stores just the "base name" of C++ methods in the table.  So we have to 
                    // extract the base name, look that up, and if there is any other information in the name we were
                    // passed in we have to post-filter based on that.
                    
                    // FIXME: Arrange the logic above so that we don't calculate the base name twice:
                    std::string base_name(base_name_start, base_name_end - base_name_start);
                    num_matches = m_apple_names_ap->FindByName (base_name.c_str(), die_offsets);
                    
                    for (uint32_t i = 0; i < num_matches; i++)
                    {
                        const dw_offset_t die_offset = die_offsets[i];
                        const DWARFDebugInfoEntry* die = info->GetDIEPtrWithCompileUnitHint (die_offset, &dwarf_cu);
                        if (die)
                        {
                            if (namespace_decl && !DIEIsInNamespace (namespace_decl, dwarf_cu, die))
                                continue;
                            
                            if (!FunctionDieMatchesPartialName(die, 
                                                               dwarf_cu, 
                                                               effective_name_type_mask, 
                                                               name_cstr, 
                                                               base_name_start, 
                                                               base_name_end))
                                continue;
                            
                            if (!include_inlines && die->Tag() == DW_TAG_inlined_subroutine)
                                continue;
                                
                            // If we get to here, the die is good, and we should add it:
                            ResolveFunction (dwarf_cu, die, sc_list);
                        }
                        else
                        {
                            GetObjectFile()->GetModule()->ReportErrorIfModifyDetected ("the DWARF debug information has been modified (.apple_names accelerator table had bad die 0x%8.8x for '%s')",
                                                                                       die_offset, name_cstr);
                        }                                    
                    }
                    die_offsets.clear();
                }
            }
        }
    }
    else
    {

        // Index the DWARF if we haven't already
        if (!m_indexed)
            Index ();

        if (name_type_mask & eFunctionNameTypeFull)
            FindFunctions (name, m_function_fullname_index, sc_list);

        std::string base_name(base_name_start, base_name_end - base_name_start);
        ConstString base_name_const(base_name.c_str());
        DIEArray die_offsets;
        DWARFCompileUnit *dwarf_cu = NULL;
        
        if (effective_name_type_mask & eFunctionNameTypeBase)
        {
            uint32_t num_base = m_function_basename_index.Find(base_name_const, die_offsets);
            for (uint32_t i = 0; i < num_base; i++)
            {
                const DWARFDebugInfoEntry* die = info->GetDIEPtrWithCompileUnitHint (die_offsets[i], &dwarf_cu);
                if (die)
                {
                    if (namespace_decl && !DIEIsInNamespace (namespace_decl, dwarf_cu, die))
                        continue;
                    
                    if (!FunctionDieMatchesPartialName(die, 
                                                       dwarf_cu, 
                                                       effective_name_type_mask, 
                                                       name_cstr, 
                                                       base_name_start, 
                                                       base_name_end))
                        continue;
                    
                    if (!include_inlines && die->Tag() == DW_TAG_inlined_subroutine)
                        continue;
                    
                    // If we get to here, the die is good, and we should add it:
                    ResolveFunction (dwarf_cu, die, sc_list);
                }
            }
            die_offsets.clear();
        }
        
        if (effective_name_type_mask & eFunctionNameTypeMethod)
        {
            if (namespace_decl && *namespace_decl)
                return 0; // no methods in namespaces

            uint32_t num_base = m_function_method_index.Find(base_name_const, die_offsets);
            {
                for (uint32_t i = 0; i < num_base; i++)
                {
                    const DWARFDebugInfoEntry* die = info->GetDIEPtrWithCompileUnitHint (die_offsets[i], &dwarf_cu);
                    if (die)
                    {
                        if (!FunctionDieMatchesPartialName(die, 
                                                           dwarf_cu, 
                                                           effective_name_type_mask, 
                                                           name_cstr, 
                                                           base_name_start, 
                                                           base_name_end))
                            continue;
                        
                        if (!include_inlines && die->Tag() == DW_TAG_inlined_subroutine)
                            continue;
                        
                        // If we get to here, the die is good, and we should add it:
                        ResolveFunction (dwarf_cu, die, sc_list);
                    }
                }
            }
            die_offsets.clear();
        }

        if ((effective_name_type_mask & eFunctionNameTypeSelector) && (!namespace_decl || !*namespace_decl))
        {
            FindFunctions (name, m_function_selector_index, sc_list);
        }
        
    }

    // Return the number of variable that were appended to the list
    const uint32_t num_matches = sc_list.GetSize() - original_size;
    
    if (log && num_matches > 0)
    {
        GetObjectFile()->GetModule()->LogMessage (log.get(), 
                                                  "SymbolFileDWARF::FindFunctions (name=\"%s\", name_type_mask=0x%x, append=%u, sc_list) => %u",
                                                  name.GetCString(), 
                                                  name_type_mask, 
                                                  append,
                                                  num_matches);
    }
    return num_matches;
}

uint32_t
SymbolFileDWARF::FindFunctions(const RegularExpression& regex, bool include_inlines, bool append, SymbolContextList& sc_list)
{
    Timer scoped_timer (__PRETTY_FUNCTION__,
                        "SymbolFileDWARF::FindFunctions (regex = '%s')",
                        regex.GetText());

    LogSP log (LogChannelDWARF::GetLogIfAll(DWARF_LOG_LOOKUPS));
    
    if (log)
    {
        GetObjectFile()->GetModule()->LogMessage (log.get(), 
                                                  "SymbolFileDWARF::FindFunctions (regex=\"%s\", append=%u, sc_list)", 
                                                  regex.GetText(), 
                                                  append);
    }
    

    // If we aren't appending the results to this list, then clear the list
    if (!append)
        sc_list.Clear();

    // Remember how many sc_list are in the list before we search in case
    // we are appending the results to a variable list.
    uint32_t original_size = sc_list.GetSize();

    if (m_using_apple_tables)
    {
        if (m_apple_names_ap.get())
            FindFunctions (regex, *m_apple_names_ap, sc_list);
    }
    else
    {
        // Index the DWARF if we haven't already
        if (!m_indexed)
            Index ();

        FindFunctions (regex, m_function_basename_index, sc_list);

        FindFunctions (regex, m_function_fullname_index, sc_list);
    }

    // Return the number of variable that were appended to the list
    return sc_list.GetSize() - original_size;
}

uint32_t
SymbolFileDWARF::FindTypes (const SymbolContext& sc, 
                            const ConstString &name, 
                            const lldb_private::ClangNamespaceDecl *namespace_decl, 
                            bool append, 
                            uint32_t max_matches, 
                            TypeList& types)
{
    DWARFDebugInfo* info = DebugInfo();
    if (info == NULL)
        return 0;

    LogSP log (LogChannelDWARF::GetLogIfAll(DWARF_LOG_LOOKUPS));
    
    if (log)
    {
        if (namespace_decl)
        {
            GetObjectFile()->GetModule()->LogMessage (log.get(),
                                                      "SymbolFileDWARF::FindTypes (sc, name=\"%s\", clang::NamespaceDecl(%p) \"%s\", append=%u, max_matches=%u, type_list)", 
                                                      name.GetCString(),
                                                      namespace_decl->GetNamespaceDecl(),
                                                      namespace_decl->GetQualifiedName().c_str(),
                                                      append, 
                                                      max_matches);
        }
        else
        {
            GetObjectFile()->GetModule()->LogMessage (log.get(),
                                                      "SymbolFileDWARF::FindTypes (sc, name=\"%s\", clang::NamespaceDecl(NULL), append=%u, max_matches=%u, type_list)",
                                                      name.GetCString(), 
                                                      append, 
                                                      max_matches);
        }
    }

    // If we aren't appending the results to this list, then clear the list
    if (!append)
        types.Clear();
    
    if (!NamespaceDeclMatchesThisSymbolFile(namespace_decl))
		return 0;

    DIEArray die_offsets;
    
    if (m_using_apple_tables)
    {
        if (m_apple_types_ap.get())
        {
            const char *name_cstr = name.GetCString();
            m_apple_types_ap->FindByName (name_cstr, die_offsets);
        }
    }
    else
    {
        if (!m_indexed)
            Index ();
        
        m_type_index.Find (name, die_offsets);
    }
    
    const size_t num_die_matches = die_offsets.size();

    if (num_die_matches)
    {
        const uint32_t initial_types_size = types.GetSize();
        DWARFCompileUnit* dwarf_cu = NULL;
        const DWARFDebugInfoEntry* die = NULL;
        DWARFDebugInfo* debug_info = DebugInfo();
        for (size_t i=0; i<num_die_matches; ++i)
        {
            const dw_offset_t die_offset = die_offsets[i];
            die = debug_info->GetDIEPtrWithCompileUnitHint (die_offset, &dwarf_cu);

            if (die)
            {
                if (namespace_decl && !DIEIsInNamespace (namespace_decl, dwarf_cu, die))
                    continue;
                
                Type *matching_type = ResolveType (dwarf_cu, die);
                if (matching_type)
                {
                    // We found a type pointer, now find the shared pointer form our type list
                    types.InsertUnique (matching_type->shared_from_this());
                    if (types.GetSize() >= max_matches)
                        break;
                }
            }
            else
            {
                if (m_using_apple_tables)
                {
                    GetObjectFile()->GetModule()->ReportErrorIfModifyDetected ("the DWARF debug information has been modified (.apple_types accelerator table had bad die 0x%8.8x for '%s')\n",
                                                                               die_offset, name.GetCString());
                }
            }            

        }
        const uint32_t num_matches = types.GetSize() - initial_types_size;
        if (log && num_matches)
        {
            if (namespace_decl)
            {
                GetObjectFile()->GetModule()->LogMessage (log.get(),
                                                          "SymbolFileDWARF::FindTypes (sc, name=\"%s\", clang::NamespaceDecl(%p) \"%s\", append=%u, max_matches=%u, type_list) => %u", 
                                                          name.GetCString(),
                                                          namespace_decl->GetNamespaceDecl(),
                                                          namespace_decl->GetQualifiedName().c_str(),
                                                          append, 
                                                          max_matches,
                                                          num_matches);
            }
            else
            {
                GetObjectFile()->GetModule()->LogMessage (log.get(),
                                                          "SymbolFileDWARF::FindTypes (sc, name=\"%s\", clang::NamespaceDecl(NULL), append=%u, max_matches=%u, type_list) => %u",
                                                          name.GetCString(), 
                                                          append, 
                                                          max_matches,
                                                          num_matches);
            }
        }
        return num_matches;
    }
    return 0;
}


ClangNamespaceDecl
SymbolFileDWARF::FindNamespace (const SymbolContext& sc, 
                                const ConstString &name,
                                const lldb_private::ClangNamespaceDecl *parent_namespace_decl)
{
    LogSP log (LogChannelDWARF::GetLogIfAll(DWARF_LOG_LOOKUPS));
    
    if (log)
    {
        GetObjectFile()->GetModule()->LogMessage (log.get(), 
                                                  "SymbolFileDWARF::FindNamespace (sc, name=\"%s\")", 
                                                  name.GetCString());
    }
    
    if (!NamespaceDeclMatchesThisSymbolFile(parent_namespace_decl))
		return ClangNamespaceDecl();

    ClangNamespaceDecl namespace_decl;
    DWARFDebugInfo* info = DebugInfo();
    if (info)
    {
        DIEArray die_offsets;

        // Index if we already haven't to make sure the compile units
        // get indexed and make their global DIE index list
        if (m_using_apple_tables)
        {
            if (m_apple_namespaces_ap.get())
            {
                const char *name_cstr = name.GetCString();
                m_apple_namespaces_ap->FindByName (name_cstr, die_offsets);
            }
        }
        else
        {
            if (!m_indexed)
                Index ();

            m_namespace_index.Find (name, die_offsets);
        }
        
        DWARFCompileUnit* dwarf_cu = NULL;
        const DWARFDebugInfoEntry* die = NULL;
        const size_t num_matches = die_offsets.size();
        if (num_matches)
        {
            DWARFDebugInfo* debug_info = DebugInfo();
            for (size_t i=0; i<num_matches; ++i)
            {
                const dw_offset_t die_offset = die_offsets[i];
                die = debug_info->GetDIEPtrWithCompileUnitHint (die_offset, &dwarf_cu);
                
                if (die)
                {
                    if (parent_namespace_decl && !DIEIsInNamespace (parent_namespace_decl, dwarf_cu, die))
                        continue;

                    clang::NamespaceDecl *clang_namespace_decl = ResolveNamespaceDIE (dwarf_cu, die);
                    if (clang_namespace_decl)
                    {
                        namespace_decl.SetASTContext (GetClangASTContext().getASTContext());
                        namespace_decl.SetNamespaceDecl (clang_namespace_decl);
                        break;
                    }
                }
                else
                {
                    if (m_using_apple_tables)
                    {
                        GetObjectFile()->GetModule()->ReportErrorIfModifyDetected ("the DWARF debug information has been modified (.apple_namespaces accelerator table had bad die 0x%8.8x for '%s')\n",
                                                                   die_offset, name.GetCString());
                    }
                }            

            }
        }
    }
    if (log && namespace_decl.GetNamespaceDecl())
    {
        GetObjectFile()->GetModule()->LogMessage (log.get(), 
                                                  "SymbolFileDWARF::FindNamespace (sc, name=\"%s\") => clang::NamespaceDecl(%p) \"%s\"",
                                                  name.GetCString(),
                                                  namespace_decl.GetNamespaceDecl(),
                                                  namespace_decl.GetQualifiedName().c_str());
    }

    return namespace_decl;
}

uint32_t
SymbolFileDWARF::FindTypes(std::vector<dw_offset_t> die_offsets, uint32_t max_matches, TypeList& types)
{
    // Remember how many sc_list are in the list before we search in case
    // we are appending the results to a variable list.
    uint32_t original_size = types.GetSize();

    const uint32_t num_die_offsets = die_offsets.size();
    // Parse all of the types we found from the pubtypes matches
    uint32_t i;
    uint32_t num_matches = 0;
    for (i = 0; i < num_die_offsets; ++i)
    {
        Type *matching_type = ResolveTypeUID (die_offsets[i]);
        if (matching_type)
        {
            // We found a type pointer, now find the shared pointer form our type list
            types.InsertUnique (matching_type->shared_from_this());
            ++num_matches;
            if (num_matches >= max_matches)
                break;
        }
    }

    // Return the number of variable that were appended to the list
    return types.GetSize() - original_size;
}


size_t
SymbolFileDWARF::ParseChildParameters (const SymbolContext& sc,
                                       clang::DeclContext *containing_decl_ctx,
                                       DWARFCompileUnit* dwarf_cu,
                                       const DWARFDebugInfoEntry *parent_die,
                                       bool skip_artificial,
                                       bool &is_static,
                                       TypeList* type_list,
                                       std::vector<clang_type_t>& function_param_types,
                                       std::vector<clang::ParmVarDecl*>& function_param_decls,
                                       unsigned &type_quals,
                                       ClangASTContext::TemplateParameterInfos &template_param_infos)
{
    if (parent_die == NULL)
        return 0;

    const uint8_t *fixed_form_sizes = DWARFFormValue::GetFixedFormSizesForAddressSize (dwarf_cu->GetAddressByteSize());

    size_t arg_idx = 0;
    const DWARFDebugInfoEntry *die;
    for (die = parent_die->GetFirstChild(); die != NULL; die = die->GetSibling())
    {
        dw_tag_t tag = die->Tag();
        switch (tag)
        {
        case DW_TAG_formal_parameter:
            {
                DWARFDebugInfoEntry::Attributes attributes;
                const size_t num_attributes = die->GetAttributes(this, dwarf_cu, fixed_form_sizes, attributes);
                if (num_attributes > 0)
                {
                    const char *name = NULL;
                    Declaration decl;
                    dw_offset_t param_type_die_offset = DW_INVALID_OFFSET;
                    bool is_artificial = false;
                    // one of None, Auto, Register, Extern, Static, PrivateExtern

                    clang::StorageClass storage = clang::SC_None;
                    uint32_t i;
                    for (i=0; i<num_attributes; ++i)
                    {
                        const dw_attr_t attr = attributes.AttributeAtIndex(i);
                        DWARFFormValue form_value;
                        if (attributes.ExtractFormValueAtIndex(this, i, form_value))
                        {
                            switch (attr)
                            {
                            case DW_AT_decl_file:   decl.SetFile(sc.comp_unit->GetSupportFiles().GetFileSpecAtIndex(form_value.Unsigned())); break;
                            case DW_AT_decl_line:   decl.SetLine(form_value.Unsigned()); break;
                            case DW_AT_decl_column: decl.SetColumn(form_value.Unsigned()); break;
                            case DW_AT_name:        name = form_value.AsCString(&get_debug_str_data()); break;
                            case DW_AT_type:        param_type_die_offset = form_value.Reference(dwarf_cu); break;
                            case DW_AT_artificial:  is_artificial = form_value.Unsigned() != 0; break;
                            case DW_AT_location:
    //                          if (form_value.BlockData())
    //                          {
    //                              const DataExtractor& debug_info_data = debug_info();
    //                              uint32_t block_length = form_value.Unsigned();
    //                              DataExtractor location(debug_info_data, form_value.BlockData() - debug_info_data.GetDataStart(), block_length);
    //                          }
    //                          else
    //                          {
    //                          }
    //                          break;
                            case DW_AT_const_value:
                            case DW_AT_default_value:
                            case DW_AT_description:
                            case DW_AT_endianity:
                            case DW_AT_is_optional:
                            case DW_AT_segment:
                            case DW_AT_variable_parameter:
                            default:
                            case DW_AT_abstract_origin:
                            case DW_AT_sibling:
                                break;
                            }
                        }
                    }

                    bool skip = false;
                    if (skip_artificial)
                    {
                        if (is_artificial)
                        {
                            // In order to determine if a C++ member function is
                            // "const" we have to look at the const-ness of "this"...
                            // Ugly, but that
                            if (arg_idx == 0)
                            {
                                if (DeclKindIsCXXClass(containing_decl_ctx->getDeclKind()))
                                {                                    
                                    // Often times compilers omit the "this" name for the
                                    // specification DIEs, so we can't rely upon the name
                                    // being in the formal parameter DIE...
                                    if (name == NULL || ::strcmp(name, "this")==0)
                                    {
                                        Type *this_type = ResolveTypeUID (param_type_die_offset);
                                        if (this_type)
                                        {                              
                                            uint32_t encoding_mask = this_type->GetEncodingMask();
                                            if (encoding_mask & Type::eEncodingIsPointerUID)
                                            {
                                                is_static = false;
                                                
                                                if (encoding_mask & (1u << Type::eEncodingIsConstUID))
                                                    type_quals |= clang::Qualifiers::Const;
                                                if (encoding_mask & (1u << Type::eEncodingIsVolatileUID))
                                                    type_quals |= clang::Qualifiers::Volatile;
                                            }
                                        }
                                    }
                                }
                            }
                            skip = true;
                        }
                        else
                        {

                            // HACK: Objective C formal parameters "self" and "_cmd" 
                            // are not marked as artificial in the DWARF...
                            CompileUnit *comp_unit = GetCompUnitForDWARFCompUnit(dwarf_cu, UINT32_MAX);
                            if (comp_unit)
                            {
                                switch (comp_unit->GetLanguage())
                                {
                                    case eLanguageTypeObjC:
                                    case eLanguageTypeObjC_plus_plus:
                                        if (name && name[0] && (strcmp (name, "self") == 0 || strcmp (name, "_cmd") == 0))
                                            skip = true;
                                        break;
                                    default:
                                        break;
                                }
                            }
                        }
                    }

                    if (!skip)
                    {
                        Type *type = ResolveTypeUID(param_type_die_offset);
                        if (type)
                        {
                            function_param_types.push_back (type->GetClangForwardType());

                            clang::ParmVarDecl *param_var_decl = GetClangASTContext().CreateParameterDeclaration (name, 
                                                                                                                  type->GetClangForwardType(), 
                                                                                                                  storage);
                            assert(param_var_decl);
                            function_param_decls.push_back(param_var_decl);
                            
                            GetClangASTContext().SetMetadataAsUserID ((uintptr_t)param_var_decl, MakeUserID(die->GetOffset()));
                        }
                    }
                }
                arg_idx++;
            }
            break;

        case DW_TAG_template_type_parameter:
        case DW_TAG_template_value_parameter:
            ParseTemplateDIE (dwarf_cu, die,template_param_infos);
            break;

        default:
            break;
        }
    }
    return arg_idx;
}

size_t
SymbolFileDWARF::ParseChildEnumerators
(
    const SymbolContext& sc,
    clang_type_t  enumerator_clang_type,
    uint32_t enumerator_byte_size,
    DWARFCompileUnit* dwarf_cu,
    const DWARFDebugInfoEntry *parent_die
)
{
    if (parent_die == NULL)
        return 0;

    size_t enumerators_added = 0;
    const DWARFDebugInfoEntry *die;
    const uint8_t *fixed_form_sizes = DWARFFormValue::GetFixedFormSizesForAddressSize (dwarf_cu->GetAddressByteSize());

    for (die = parent_die->GetFirstChild(); die != NULL; die = die->GetSibling())
    {
        const dw_tag_t tag = die->Tag();
        if (tag == DW_TAG_enumerator)
        {
            DWARFDebugInfoEntry::Attributes attributes;
            const size_t num_child_attributes = die->GetAttributes(this, dwarf_cu, fixed_form_sizes, attributes);
            if (num_child_attributes > 0)
            {
                const char *name = NULL;
                bool got_value = false;
                int64_t enum_value = 0;
                Declaration decl;

                uint32_t i;
                for (i=0; i<num_child_attributes; ++i)
                {
                    const dw_attr_t attr = attributes.AttributeAtIndex(i);
                    DWARFFormValue form_value;
                    if (attributes.ExtractFormValueAtIndex(this, i, form_value))
                    {
                        switch (attr)
                        {
                        case DW_AT_const_value:
                            got_value = true;
                            enum_value = form_value.Unsigned();
                            break;

                        case DW_AT_name:
                            name = form_value.AsCString(&get_debug_str_data());
                            break;

                        case DW_AT_description:
                        default:
                        case DW_AT_decl_file:   decl.SetFile(sc.comp_unit->GetSupportFiles().GetFileSpecAtIndex(form_value.Unsigned())); break;
                        case DW_AT_decl_line:   decl.SetLine(form_value.Unsigned()); break;
                        case DW_AT_decl_column: decl.SetColumn(form_value.Unsigned()); break;
                        case DW_AT_sibling:
                            break;
                        }
                    }
                }

                if (name && name[0] && got_value)
                {
                    GetClangASTContext().AddEnumerationValueToEnumerationType (enumerator_clang_type, 
                                                                               enumerator_clang_type, 
                                                                               decl, 
                                                                               name, 
                                                                               enum_value, 
                                                                               enumerator_byte_size * 8);
                    ++enumerators_added;
                }
            }
        }
    }
    return enumerators_added;
}

void
SymbolFileDWARF::ParseChildArrayInfo
(
    const SymbolContext& sc,
    DWARFCompileUnit* dwarf_cu,
    const DWARFDebugInfoEntry *parent_die,
    int64_t& first_index,
    std::vector<uint64_t>& element_orders,
    uint32_t& byte_stride,
    uint32_t& bit_stride
)
{
    if (parent_die == NULL)
        return;

    const DWARFDebugInfoEntry *die;
    const uint8_t *fixed_form_sizes = DWARFFormValue::GetFixedFormSizesForAddressSize (dwarf_cu->GetAddressByteSize());
    for (die = parent_die->GetFirstChild(); die != NULL; die = die->GetSibling())
    {
        const dw_tag_t tag = die->Tag();
        switch (tag)
        {
        case DW_TAG_subrange_type:
            {
                DWARFDebugInfoEntry::Attributes attributes;
                const size_t num_child_attributes = die->GetAttributes(this, dwarf_cu, fixed_form_sizes, attributes);
                if (num_child_attributes > 0)
                {
                    uint64_t num_elements = 0;
                    uint64_t lower_bound = 0;
                    uint64_t upper_bound = 0;
                    uint32_t i;
                    for (i=0; i<num_child_attributes; ++i)
                    {
                        const dw_attr_t attr = attributes.AttributeAtIndex(i);
                        DWARFFormValue form_value;
                        if (attributes.ExtractFormValueAtIndex(this, i, form_value))
                        {
                            switch (attr)
                            {
                            case DW_AT_name:
                                break;

                            case DW_AT_count:
                                num_elements = form_value.Unsigned();
                                break;

                            case DW_AT_bit_stride:
                                bit_stride = form_value.Unsigned();
                                break;

                            case DW_AT_byte_stride:
                                byte_stride = form_value.Unsigned();
                                break;

                            case DW_AT_lower_bound:
                                lower_bound = form_value.Unsigned();
                                break;

                            case DW_AT_upper_bound:
                                upper_bound = form_value.Unsigned();
                                break;

                            default:
                            case DW_AT_abstract_origin:
                            case DW_AT_accessibility:
                            case DW_AT_allocated:
                            case DW_AT_associated:
                            case DW_AT_data_location:
                            case DW_AT_declaration:
                            case DW_AT_description:
                            case DW_AT_sibling:
                            case DW_AT_threads_scaled:
                            case DW_AT_type:
                            case DW_AT_visibility:
                                break;
                            }
                        }
                    }

                    if (upper_bound > lower_bound)
                        num_elements = upper_bound - lower_bound + 1;

                    element_orders.push_back (num_elements);
                }
            }
            break;
        }
    }
}

TypeSP
SymbolFileDWARF::GetTypeForDIE (DWARFCompileUnit *dwarf_cu, const DWARFDebugInfoEntry* die)
{
    TypeSP type_sp;
    if (die != NULL)
    {
        assert(dwarf_cu != NULL);
        Type *type_ptr = m_die_to_type.lookup (die);
        if (type_ptr == NULL)
        {
            CompileUnit* lldb_cu = GetCompUnitForDWARFCompUnit(dwarf_cu);
            assert (lldb_cu);
            SymbolContext sc(lldb_cu);
            type_sp = ParseType(sc, dwarf_cu, die, NULL);
        }
        else if (type_ptr != DIE_IS_BEING_PARSED)
        {
            // Grab the existing type from the master types lists
            type_sp = type_ptr->shared_from_this();
        }

    }
    return type_sp;
}

clang::DeclContext *
SymbolFileDWARF::GetClangDeclContextContainingDIEOffset (dw_offset_t die_offset)
{
    if (die_offset != DW_INVALID_OFFSET)
    {
        DWARFCompileUnitSP cu_sp;
        const DWARFDebugInfoEntry* die = DebugInfo()->GetDIEPtr(die_offset, &cu_sp);
        return GetClangDeclContextContainingDIE (cu_sp.get(), die, NULL);
    }
    return NULL;
}

clang::DeclContext *
SymbolFileDWARF::GetClangDeclContextForDIEOffset (const SymbolContext &sc, dw_offset_t die_offset)
{
    if (die_offset != DW_INVALID_OFFSET)
    {
        DWARFDebugInfo* debug_info = DebugInfo();
        if (debug_info)
        {
            DWARFCompileUnitSP cu_sp;
            const DWARFDebugInfoEntry* die = debug_info->GetDIEPtr(die_offset, &cu_sp);
            if (die)
                return GetClangDeclContextForDIE (sc, cu_sp.get(), die);
        }
    }
    return NULL;
}

clang::NamespaceDecl *
SymbolFileDWARF::ResolveNamespaceDIE (DWARFCompileUnit *dwarf_cu, const DWARFDebugInfoEntry *die)
{
    if (die && die->Tag() == DW_TAG_namespace)
    {
        // See if we already parsed this namespace DIE and associated it with a
        // uniqued namespace declaration
        clang::NamespaceDecl *namespace_decl = static_cast<clang::NamespaceDecl *>(m_die_to_decl_ctx[die]);
        if (namespace_decl)
            return namespace_decl;
        else
        {
            const char *namespace_name = die->GetAttributeValueAsString(this, dwarf_cu, DW_AT_name, NULL);
            clang::DeclContext *containing_decl_ctx = GetClangDeclContextContainingDIE (dwarf_cu, die, NULL);
            namespace_decl = GetClangASTContext().GetUniqueNamespaceDeclaration (namespace_name, containing_decl_ctx);
            LogSP log (LogChannelDWARF::GetLogIfAll(DWARF_LOG_DEBUG_INFO));
            if (log)
            {
                if (namespace_name)
                {
                    GetObjectFile()->GetModule()->LogMessage (log.get(), 
                                                              "ASTContext => %p: 0x%8.8llx: DW_TAG_namespace with DW_AT_name(\"%s\") => clang::NamespaceDecl *%p (original = %p)", 
                                                              GetClangASTContext().getASTContext(),
                                                              MakeUserID(die->GetOffset()),
                                                              namespace_name,
                                                              namespace_decl,
                                                              namespace_decl->getOriginalNamespace());
                }
                else
                {
                    GetObjectFile()->GetModule()->LogMessage (log.get(),
                                                              "ASTContext => %p: 0x%8.8llx: DW_TAG_namespace (anonymous) => clang::NamespaceDecl *%p (original = %p)", 
                                                              GetClangASTContext().getASTContext(),
                                                              MakeUserID(die->GetOffset()),
                                                              namespace_decl,
                                                              namespace_decl->getOriginalNamespace());
                }
            }

            if (namespace_decl)
                LinkDeclContextToDIE((clang::DeclContext*)namespace_decl, die);
            return namespace_decl;
        }
    }
    return NULL;
}

clang::DeclContext *
SymbolFileDWARF::GetClangDeclContextForDIE (const SymbolContext &sc, DWARFCompileUnit *cu, const DWARFDebugInfoEntry *die)
{
    clang::DeclContext *clang_decl_ctx = GetCachedClangDeclContextForDIE (die);
    if (clang_decl_ctx)
        return clang_decl_ctx;
    // If this DIE has a specification, or an abstract origin, then trace to those.
        
    dw_offset_t die_offset = die->GetAttributeValueAsReference(this, cu, DW_AT_specification, DW_INVALID_OFFSET);
    if (die_offset != DW_INVALID_OFFSET)
        return GetClangDeclContextForDIEOffset (sc, die_offset);
    
    die_offset = die->GetAttributeValueAsReference(this, cu, DW_AT_abstract_origin, DW_INVALID_OFFSET);
    if (die_offset != DW_INVALID_OFFSET)
        return GetClangDeclContextForDIEOffset (sc, die_offset);
    
    LogSP log (LogChannelDWARF::GetLogIfAll(DWARF_LOG_DEBUG_INFO));
    if (log)
        GetObjectFile()->GetModule()->LogMessage(log.get(), "SymbolFileDWARF::GetClangDeclContextForDIE (die = 0x%8.8x) %s '%s'", die->GetOffset(), DW_TAG_value_to_name(die->Tag()), die->GetName(this, cu));
    // This is the DIE we want.  Parse it, then query our map.
    bool assert_not_being_parsed = true;
    ResolveTypeUID (cu, die, assert_not_being_parsed);    

    clang_decl_ctx = GetCachedClangDeclContextForDIE (die);

    return clang_decl_ctx;
}

clang::DeclContext *
SymbolFileDWARF::GetClangDeclContextContainingDIE (DWARFCompileUnit *cu, const DWARFDebugInfoEntry *die, const DWARFDebugInfoEntry **decl_ctx_die_copy)
{
    if (m_clang_tu_decl == NULL)
        m_clang_tu_decl = GetClangASTContext().getASTContext()->getTranslationUnitDecl();

    const DWARFDebugInfoEntry *decl_ctx_die = GetDeclContextDIEContainingDIE (cu, die);

    if (decl_ctx_die_copy)
        *decl_ctx_die_copy = decl_ctx_die;
    
    if (decl_ctx_die)
    {

        DIEToDeclContextMap::iterator pos = m_die_to_decl_ctx.find (decl_ctx_die);
        if (pos != m_die_to_decl_ctx.end())
            return pos->second;

        switch (decl_ctx_die->Tag())
        {
        case DW_TAG_compile_unit:
            return m_clang_tu_decl;

        case DW_TAG_namespace:
            return ResolveNamespaceDIE (cu, decl_ctx_die);
            break;

        case DW_TAG_structure_type:
        case DW_TAG_union_type:
        case DW_TAG_class_type:
            {
                Type* type = ResolveType (cu, decl_ctx_die);
                if (type)
                {
                    clang::DeclContext *decl_ctx = ClangASTContext::GetDeclContextForType (type->GetClangForwardType ());
                    if (decl_ctx)
                    {
                        LinkDeclContextToDIE (decl_ctx, decl_ctx_die);
                        if (decl_ctx)
                            return decl_ctx;
                    }
                }
            }
            break;

        default:
            break;
        }
    }
    return m_clang_tu_decl;
}


const DWARFDebugInfoEntry *
SymbolFileDWARF::GetDeclContextDIEContainingDIE (DWARFCompileUnit *cu, const DWARFDebugInfoEntry *die)
{
    if (cu && die)
    {
        const DWARFDebugInfoEntry * const decl_die = die;
    
        while (die != NULL)
        {
            // If this is the original DIE that we are searching for a declaration 
            // for, then don't look in the cache as we don't want our own decl 
            // context to be our decl context...
            if (decl_die != die)
            {            
                switch (die->Tag())
                {
                    case DW_TAG_compile_unit:
                    case DW_TAG_namespace:
                    case DW_TAG_structure_type:
                    case DW_TAG_union_type:
                    case DW_TAG_class_type:
                        return die;
                        
                    default:
                        break;
                }
            }
            
            dw_offset_t die_offset = die->GetAttributeValueAsReference(this, cu, DW_AT_specification, DW_INVALID_OFFSET);
            if (die_offset != DW_INVALID_OFFSET)
            {
                DWARFCompileUnit *spec_cu = cu;
                const DWARFDebugInfoEntry *spec_die = DebugInfo()->GetDIEPtrWithCompileUnitHint (die_offset, &spec_cu);
                const DWARFDebugInfoEntry *spec_die_decl_ctx_die = GetDeclContextDIEContainingDIE (spec_cu, spec_die);
                if (spec_die_decl_ctx_die)
                    return spec_die_decl_ctx_die;
            }
            
            die_offset = die->GetAttributeValueAsReference(this, cu, DW_AT_abstract_origin, DW_INVALID_OFFSET);
            if (die_offset != DW_INVALID_OFFSET)
            {
                DWARFCompileUnit *abs_cu = cu;
                const DWARFDebugInfoEntry *abs_die = DebugInfo()->GetDIEPtrWithCompileUnitHint (die_offset, &abs_cu);
                const DWARFDebugInfoEntry *abs_die_decl_ctx_die = GetDeclContextDIEContainingDIE (abs_cu, abs_die);
                if (abs_die_decl_ctx_die)
                    return abs_die_decl_ctx_die;
            }
            
            die = die->GetParent();
        }
    }
    return NULL;
}


Symbol *
SymbolFileDWARF::GetObjCClassSymbol (const ConstString &objc_class_name)
{
    Symbol *objc_class_symbol = NULL;
    if (m_obj_file)
    {
        Symtab *symtab = m_obj_file->GetSymtab();
        if (symtab)
        {
            objc_class_symbol = symtab->FindFirstSymbolWithNameAndType (objc_class_name, 
                                                                        eSymbolTypeObjCClass, 
                                                                        Symtab::eDebugNo, 
                                                                        Symtab::eVisibilityAny);
        }
    }
    return objc_class_symbol;
}

// Some compilers don't emit the DW_AT_APPLE_objc_complete_type attribute. If they don't
// then we can end up looking through all class types for a complete type and never find
// the full definition. We need to know if this attribute is supported, so we determine
// this here and cache th result. We also need to worry about the debug map DWARF file
// if we are doing darwin DWARF in .o file debugging.
bool
SymbolFileDWARF::Supports_DW_AT_APPLE_objc_complete_type (DWARFCompileUnit *cu)
{
    if (m_supports_DW_AT_APPLE_objc_complete_type == eLazyBoolCalculate)
    {
        m_supports_DW_AT_APPLE_objc_complete_type = eLazyBoolNo;
        if (cu && cu->Supports_DW_AT_APPLE_objc_complete_type())
            m_supports_DW_AT_APPLE_objc_complete_type = eLazyBoolYes;
        else
        {
            DWARFDebugInfo* debug_info = DebugInfo();
            const uint32_t num_compile_units = GetNumCompileUnits();
            for (uint32_t cu_idx = 0; cu_idx < num_compile_units; ++cu_idx)
            {
                DWARFCompileUnit* dwarf_cu = debug_info->GetCompileUnitAtIndex(cu_idx);
                if (dwarf_cu != cu && dwarf_cu->Supports_DW_AT_APPLE_objc_complete_type())
                {
                    m_supports_DW_AT_APPLE_objc_complete_type = eLazyBoolYes;
                    break;
                }
            }
        }
        if (m_supports_DW_AT_APPLE_objc_complete_type == eLazyBoolNo && GetDebugMapSymfile ())
            return m_debug_map_symfile->Supports_DW_AT_APPLE_objc_complete_type (this);
    }
    return m_supports_DW_AT_APPLE_objc_complete_type == eLazyBoolYes;
}

// This function can be used when a DIE is found that is a forward declaration
// DIE and we want to try and find a type that has the complete definition.
TypeSP
SymbolFileDWARF::FindCompleteObjCDefinitionTypeForDIE (const DWARFDebugInfoEntry *die, 
                                                       const ConstString &type_name,
                                                       bool must_be_implementation)
{
    
    TypeSP type_sp;
    
    if (!type_name || (must_be_implementation && !GetObjCClassSymbol (type_name)))
        return type_sp;
    
    DIEArray die_offsets;
    
    if (m_using_apple_tables)
    {
        if (m_apple_types_ap.get())
        {
            const char *name_cstr = type_name.GetCString();
            m_apple_types_ap->FindCompleteObjCClassByName (name_cstr, die_offsets, must_be_implementation);
        }
    }
    else
    {
        if (!m_indexed)
            Index ();
        
        m_type_index.Find (type_name, die_offsets);
    }
    
    const size_t num_matches = die_offsets.size();
    
    DWARFCompileUnit* type_cu = NULL;
    const DWARFDebugInfoEntry* type_die = NULL;
    if (num_matches)
    {
        DWARFDebugInfo* debug_info = DebugInfo();
        for (size_t i=0; i<num_matches; ++i)
        {
            const dw_offset_t die_offset = die_offsets[i];
            type_die = debug_info->GetDIEPtrWithCompileUnitHint (die_offset, &type_cu);
            
            if (type_die)
            {
                bool try_resolving_type = false;
                
                // Don't try and resolve the DIE we are looking for with the DIE itself!
                if (type_die != die)
                {
                    switch (type_die->Tag())
                    {
                        case DW_TAG_class_type:
                        case DW_TAG_structure_type:
                            try_resolving_type = true;
                            break;
                        default:
                            break;
                    }
                }
                
                if (try_resolving_type)
                {
					if (must_be_implementation && type_cu->Supports_DW_AT_APPLE_objc_complete_type())
	                    try_resolving_type = type_die->GetAttributeValueAsUnsigned (this, type_cu, DW_AT_APPLE_objc_complete_type, 0);
                    
                    if (try_resolving_type)
                    {
                        Type *resolved_type = ResolveType (type_cu, type_die, false);
                        if (resolved_type && resolved_type != DIE_IS_BEING_PARSED)
                        {
                            DEBUG_PRINTF ("resolved 0x%8.8llx (cu 0x%8.8llx) from %s to 0x%8.8llx (cu 0x%8.8llx)\n",
                                          MakeUserID(die->GetOffset()), 
                                          MakeUserID(dwarf_cu->GetOffset()),
                                          m_obj_file->GetFileSpec().GetFilename().AsCString(),
                                          MakeUserID(type_die->GetOffset()), 
                                          MakeUserID(type_cu->GetOffset()));
                            
                            if (die)
                                m_die_to_type[die] = resolved_type;
                            type_sp = resolved_type->shared_from_this();
                            break;
                        }
                    }
                }
            }
            else
            {
                if (m_using_apple_tables)
                {
                    GetObjectFile()->GetModule()->ReportErrorIfModifyDetected ("the DWARF debug information has been modified (.apple_types accelerator table had bad die 0x%8.8x for '%s')\n",
                                                               die_offset, type_name.GetCString());
                }
            }            
            
        }
    }
    return type_sp;
}


//----------------------------------------------------------------------
// This function helps to ensure that the declaration contexts match for
// two different DIEs. Often times debug information will refer to a 
// forward declaration of a type (the equivalent of "struct my_struct;".
// There will often be a declaration of that type elsewhere that has the
// full definition. When we go looking for the full type "my_struct", we
// will find one or more matches in the accelerator tables and we will
// then need to make sure the type was in the same declaration context 
// as the original DIE. This function can efficiently compare two DIEs
// and will return true when the declaration context matches, and false
// when they don't. 
//----------------------------------------------------------------------
bool
SymbolFileDWARF::DIEDeclContextsMatch (DWARFCompileUnit* cu1, const DWARFDebugInfoEntry *die1,
                                       DWARFCompileUnit* cu2, const DWARFDebugInfoEntry *die2)
{
    if (die1 == die2)
        return true;

#if defined (LLDB_CONFIGURATION_DEBUG)
    // You can't and shouldn't call this function with a compile unit from
    // two different SymbolFileDWARF instances.
    assert (DebugInfo()->ContainsCompileUnit (cu1));
    assert (DebugInfo()->ContainsCompileUnit (cu2));
#endif

    DWARFDIECollection decl_ctx_1;
    DWARFDIECollection decl_ctx_2;
    //The declaration DIE stack is a stack of the declaration context 
    // DIEs all the way back to the compile unit. If a type "T" is
    // declared inside a class "B", and class "B" is declared inside
    // a class "A" and class "A" is in a namespace "lldb", and the
    // namespace is in a compile unit, there will be a stack of DIEs:
    //
    //   [0] DW_TAG_class_type for "B"
    //   [1] DW_TAG_class_type for "A"
    //   [2] DW_TAG_namespace  for "lldb"
    //   [3] DW_TAG_compile_unit for the source file.
    // 
    // We grab both contexts and make sure that everything matches 
    // all the way back to the compiler unit.
    
    // First lets grab the decl contexts for both DIEs
    die1->GetDeclContextDIEs (this, cu1, decl_ctx_1);
    die2->GetDeclContextDIEs (this, cu2, decl_ctx_2);
    // Make sure the context arrays have the same size, otherwise
    // we are done
    const size_t count1 = decl_ctx_1.Size();
    const size_t count2 = decl_ctx_2.Size();
    if (count1 != count2)
        return false;
    
    // Make sure the DW_TAG values match all the way back up the the
    // compile unit. If they don't, then we are done.
    const DWARFDebugInfoEntry *decl_ctx_die1;
    const DWARFDebugInfoEntry *decl_ctx_die2;
    size_t i;
    for (i=0; i<count1; i++)
    {
        decl_ctx_die1 = decl_ctx_1.GetDIEPtrAtIndex (i);
        decl_ctx_die2 = decl_ctx_2.GetDIEPtrAtIndex (i);
        if (decl_ctx_die1->Tag() != decl_ctx_die2->Tag())
            return false;
    }
#if defined LLDB_CONFIGURATION_DEBUG

    // Make sure the top item in the decl context die array is always 
    // DW_TAG_compile_unit. If it isn't then something went wrong in
    // the DWARFDebugInfoEntry::GetDeclContextDIEs() function...
    assert (decl_ctx_1.GetDIEPtrAtIndex (count1 - 1)->Tag() == DW_TAG_compile_unit);

#endif
    // Always skip the compile unit when comparing by only iterating up to
    // "count - 1". Here we compare the names as we go. 
    for (i=0; i<count1 - 1; i++)
    {
        decl_ctx_die1 = decl_ctx_1.GetDIEPtrAtIndex (i);
        decl_ctx_die2 = decl_ctx_2.GetDIEPtrAtIndex (i);
        const char *name1 = decl_ctx_die1->GetName(this, cu1);
        const char *name2 = decl_ctx_die2->GetName(this, cu2);
        // If the string was from a DW_FORM_strp, then the pointer will often
        // be the same!
        if (name1 == name2)
            continue;

        // Name pointers are not equal, so only compare the strings
        // if both are not NULL.
        if (name1 && name2)
        {
            // If the strings don't compare, we are done...
            if (strcmp(name1, name2) != 0)
                return false;
        }
        else
        {
            // One name was NULL while the other wasn't
            return false;
        }
    }
    // We made it through all of the checks and the declaration contexts
    // are equal.
    return true;
}
                                          
// This function can be used when a DIE is found that is a forward declaration
// DIE and we want to try and find a type that has the complete definition.
// "cu" and "die" must be from this SymbolFileDWARF
TypeSP
SymbolFileDWARF::FindDefinitionTypeForDIE (DWARFCompileUnit* cu,
                                           const DWARFDebugInfoEntry *die, 
                                           const ConstString &type_name)
{
    TypeSP type_sp;

#if defined (LLDB_CONFIGURATION_DEBUG)
    // You can't and shouldn't call this function with a compile unit from
    // another SymbolFileDWARF instance.
    assert (DebugInfo()->ContainsCompileUnit (cu));
#endif

    if (cu == NULL || die == NULL || !type_name)
        return type_sp;

    LogSP log (LogChannelDWARF::GetLogIfAny(DWARF_LOG_TYPE_COMPLETION|DWARF_LOG_LOOKUPS));
    if (log)
    {
        std::string qualified_name;
        die->GetQualifiedName(this, cu, qualified_name);
        GetObjectFile()->GetModule()->LogMessage (log.get(),
                                                  "SymbolFileDWARF::FindDefinitionTypeForDIE(die=0x%8.8x (%s), name='%s')",
                                                  die->GetOffset(),
                                                  qualified_name.c_str(),
                                                  type_name.GetCString());
    }

    DIEArray die_offsets;

    if (m_using_apple_tables)
    {
        if (m_apple_types_ap.get())
        {
            if (m_apple_types_ap->GetHeader().header_data.atoms.size() > 1)
            {
                m_apple_types_ap->FindByNameAndTag (type_name.GetCString(), die->Tag(), die_offsets);
            }
            else
            {
                m_apple_types_ap->FindByName (type_name.GetCString(), die_offsets);
            }
        }
    }
    else
    {
        if (!m_indexed)
            Index ();
        
        m_type_index.Find (type_name, die_offsets);
    }
    
    const size_t num_matches = die_offsets.size();

    const dw_tag_t die_tag = die->Tag();
    
    DWARFCompileUnit* type_cu = NULL;
    const DWARFDebugInfoEntry* type_die = NULL;
    if (num_matches)
    {
        DWARFDebugInfo* debug_info = DebugInfo();
        for (size_t i=0; i<num_matches; ++i)
        {
            const dw_offset_t die_offset = die_offsets[i];
            type_die = debug_info->GetDIEPtrWithCompileUnitHint (die_offset, &type_cu);
            
            if (type_die)
            {
                bool try_resolving_type = false;

                // Don't try and resolve the DIE we are looking for with the DIE itself!
                if (type_die != die)
                {
                    const dw_tag_t type_die_tag = type_die->Tag();
                    // Make sure the tags match
                    if (type_die_tag == die_tag)
                    {
                        // The tags match, lets try resolving this type
                        try_resolving_type = true;
                    }
                    else
                    {
                        // The tags don't match, but we need to watch our for a
                        // forward declaration for a struct and ("struct foo")
                        // ends up being a class ("class foo { ... };") or
                        // vice versa.
                        switch (type_die_tag)
                        {
                        case DW_TAG_class_type:
                            // We had a "class foo", see if we ended up with a "struct foo { ... };"
                            try_resolving_type = (die_tag == DW_TAG_structure_type);
                            break;
                        case DW_TAG_structure_type:
                            // We had a "struct foo", see if we ended up with a "class foo { ... };"
                            try_resolving_type = (die_tag == DW_TAG_class_type);
                            break;
                        default:
                            // Tags don't match, don't event try to resolve
                            // using this type whose name matches....
                            break;
                        }
                    }
                }
                        
                if (try_resolving_type)
                {
                    if (log)
                    {
                        std::string qualified_name;
                        type_die->GetQualifiedName(this, cu, qualified_name);
                        GetObjectFile()->GetModule()->LogMessage (log.get(),
                                                                  "SymbolFileDWARF::FindDefinitionTypeForDIE(die=0x%8.8x, name='%s') trying die=0x%8.8x (%s)",
                                                                  die->GetOffset(),
                                                                  type_name.GetCString(),
                                                                  type_die->GetOffset(),
                                                                  qualified_name.c_str());
                    }
                    
                    // Make sure the decl contexts match all the way up
                    if (DIEDeclContextsMatch(cu, die, type_cu, type_die))
                    {
                        Type *resolved_type = ResolveType (type_cu, type_die, false);
                        if (resolved_type && resolved_type != DIE_IS_BEING_PARSED)
                        {
                            DEBUG_PRINTF ("resolved 0x%8.8llx (cu 0x%8.8llx) from %s to 0x%8.8llx (cu 0x%8.8llx)\n",
                                          MakeUserID(die->GetOffset()), 
                                          MakeUserID(dwarf_cu->GetOffset()),
                                          m_obj_file->GetFileSpec().GetFilename().AsCString(),
                                          MakeUserID(type_die->GetOffset()), 
                                          MakeUserID(type_cu->GetOffset()));
                            
                            m_die_to_type[die] = resolved_type;
                            type_sp = resolved_type->shared_from_this();
                            break;
                        }
                    }
                }
                else
                {
                    if (log)
                    {
                        std::string qualified_name;
                        type_die->GetQualifiedName(this, cu, qualified_name);
                        GetObjectFile()->GetModule()->LogMessage (log.get(),
                                                                  "SymbolFileDWARF::FindDefinitionTypeForDIE(die=0x%8.8x, name='%s') ignoring die=0x%8.8x (%s)",
                                                                  die->GetOffset(),
                                                                  type_name.GetCString(),
                                                                  type_die->GetOffset(),
                                                                  qualified_name.c_str());
                    }
                }
            }
            else
            {
                if (m_using_apple_tables)
                {
                    GetObjectFile()->GetModule()->ReportErrorIfModifyDetected ("the DWARF debug information has been modified (.apple_types accelerator table had bad die 0x%8.8x for '%s')\n",
                                                                               die_offset, type_name.GetCString());
                }
            }            

        }
    }
    return type_sp;
}

TypeSP
SymbolFileDWARF::FindDefinitionTypeForDWARFDeclContext (const DWARFDeclContext &dwarf_decl_ctx)
{
    TypeSP type_sp;

    const uint32_t dwarf_decl_ctx_count = dwarf_decl_ctx.GetSize();
    if (dwarf_decl_ctx_count > 0)
    {
        const ConstString type_name(dwarf_decl_ctx[0].name);
        const dw_tag_t tag = dwarf_decl_ctx[0].tag;

        if (type_name)
        {
            LogSP log (LogChannelDWARF::GetLogIfAny(DWARF_LOG_TYPE_COMPLETION|DWARF_LOG_LOOKUPS));
            if (log)
            {
                GetObjectFile()->GetModule()->LogMessage (log.get(),
                                                          "SymbolFileDWARF::FindDefinitionTypeForDWARFDeclContext(tag=%s, qualified-name='%s')",
                                                          DW_TAG_value_to_name(dwarf_decl_ctx[0].tag),
                                                          dwarf_decl_ctx.GetQualifiedName());
            }
            
            DIEArray die_offsets;
            
            if (m_using_apple_tables)
            {
                if (m_apple_types_ap.get())
                {
                    if (m_apple_types_ap->GetHeader().header_data.atoms.size() > 1)
                    {
                        m_apple_types_ap->FindByNameAndTag (type_name.GetCString(), tag, die_offsets);
                    }
                    else
                    {
                        m_apple_types_ap->FindByName (type_name.GetCString(), die_offsets);
                    }
                }
            }
            else
            {
                if (!m_indexed)
                    Index ();
                
                m_type_index.Find (type_name, die_offsets);
            }
            
            const size_t num_matches = die_offsets.size();
            
            
            DWARFCompileUnit* type_cu = NULL;
            const DWARFDebugInfoEntry* type_die = NULL;
            if (num_matches)
            {
                DWARFDebugInfo* debug_info = DebugInfo();
                for (size_t i=0; i<num_matches; ++i)
                {
                    const dw_offset_t die_offset = die_offsets[i];
                    type_die = debug_info->GetDIEPtrWithCompileUnitHint (die_offset, &type_cu);
                    
                    if (type_die)
                    {
                        bool try_resolving_type = false;
                        
                        // Don't try and resolve the DIE we are looking for with the DIE itself!
                        const dw_tag_t type_tag = type_die->Tag();
                        // Make sure the tags match
                        if (type_tag == tag)
                        {
                            // The tags match, lets try resolving this type
                            try_resolving_type = true;
                        }
                        else
                        {
                            // The tags don't match, but we need to watch our for a
                            // forward declaration for a struct and ("struct foo")
                            // ends up being a class ("class foo { ... };") or
                            // vice versa.
                            switch (type_tag)
                            {
                                case DW_TAG_class_type:
                                    // We had a "class foo", see if we ended up with a "struct foo { ... };"
                                    try_resolving_type = (tag == DW_TAG_structure_type);
                                    break;
                                case DW_TAG_structure_type:
                                    // We had a "struct foo", see if we ended up with a "class foo { ... };"
                                    try_resolving_type = (tag == DW_TAG_class_type);
                                    break;
                                default:
                                    // Tags don't match, don't event try to resolve
                                    // using this type whose name matches....
                                    break;
                            }
                        }
                        
                        if (try_resolving_type)
                        {
                            DWARFDeclContext type_dwarf_decl_ctx;
                            type_die->GetDWARFDeclContext (this, type_cu, type_dwarf_decl_ctx);

                            if (log)
                            {
                                GetObjectFile()->GetModule()->LogMessage (log.get(),
                                                                          "SymbolFileDWARF::FindDefinitionTypeForDWARFDeclContext(tag=%s, qualified-name='%s') trying die=0x%8.8x (%s)",
                                                                          DW_TAG_value_to_name(dwarf_decl_ctx[0].tag),
                                                                          dwarf_decl_ctx.GetQualifiedName(),
                                                                          type_die->GetOffset(),
                                                                          type_dwarf_decl_ctx.GetQualifiedName());
                            }
                            
                            // Make sure the decl contexts match all the way up
                            if (dwarf_decl_ctx == type_dwarf_decl_ctx)
                            {
                                Type *resolved_type = ResolveType (type_cu, type_die, false);
                                if (resolved_type && resolved_type != DIE_IS_BEING_PARSED)
                                {
                                    type_sp = resolved_type->shared_from_this();
                                    break;
                                }
                            }
                        }
                        else
                        {
                            if (log)
                            {
                                std::string qualified_name;
                                type_die->GetQualifiedName(this, type_cu, qualified_name);
                                GetObjectFile()->GetModule()->LogMessage (log.get(),
                                                                          "SymbolFileDWARF::FindDefinitionTypeForDWARFDeclContext(tag=%s, qualified-name='%s') ignoring die=0x%8.8x (%s)",
                                                                          DW_TAG_value_to_name(dwarf_decl_ctx[0].tag),
                                                                          dwarf_decl_ctx.GetQualifiedName(),
                                                                          type_die->GetOffset(),
                                                                          qualified_name.c_str());
                            }
                        }
                    }
                    else
                    {
                        if (m_using_apple_tables)
                        {
                            GetObjectFile()->GetModule()->ReportErrorIfModifyDetected ("the DWARF debug information has been modified (.apple_types accelerator table had bad die 0x%8.8x for '%s')\n",
                                                                                       die_offset, type_name.GetCString());
                        }
                    }            
                    
                }
            }
        }
    }
    return type_sp;
}

bool
SymbolFileDWARF::CopyUniqueClassMethodTypes (Type *class_type,
                                             DWARFCompileUnit* src_cu,
                                             const DWARFDebugInfoEntry *src_class_die,
                                             DWARFCompileUnit* dst_cu,
                                             const DWARFDebugInfoEntry *dst_class_die)
{
    if (!class_type || !src_cu || !src_class_die || !dst_cu || !dst_class_die)
        return false;
    if (src_class_die->Tag() != dst_class_die->Tag())
        return false;
    
    // We need to complete the class type so we can get all of the method types
    // parsed so we can then unique those types to their equivalent counterparts
    // in "dst_cu" and "dst_class_die"
    class_type->GetClangFullType();

    const DWARFDebugInfoEntry *src_die;
    const DWARFDebugInfoEntry *dst_die;
    UniqueCStringMap<const DWARFDebugInfoEntry *> src_name_to_die;
    UniqueCStringMap<const DWARFDebugInfoEntry *> dst_name_to_die;
    for (src_die = src_class_die->GetFirstChild(); src_die != NULL; src_die = src_die->GetSibling())
    {
        if (src_die->Tag() == DW_TAG_subprogram)
        {
            // Make sure this is a declaration and not a concrete instance by looking
            // for DW_AT_declaration set to 1. Sometimes concrete function instances
            // are placed inside the class definitions and shouldn't be included in
            // the list of things are are tracking here.
            if (src_die->GetAttributeValueAsUnsigned(this, src_cu, DW_AT_declaration, 0) == 1)
            {
                const char *src_name = src_die->GetMangledName (this, src_cu);
                if (src_name)
                    src_name_to_die.Append(ConstString(src_name).GetCString(), src_die);
            }
        }
    }
    for (dst_die = dst_class_die->GetFirstChild(); dst_die != NULL; dst_die = dst_die->GetSibling())
    {
        if (dst_die->Tag() == DW_TAG_subprogram)
        {
            // Make sure this is a declaration and not a concrete instance by looking
            // for DW_AT_declaration set to 1. Sometimes concrete function instances
            // are placed inside the class definitions and shouldn't be included in
            // the list of things are are tracking here.
            if (dst_die->GetAttributeValueAsUnsigned(this, dst_cu, DW_AT_declaration, 0) == 1)
            {
                const char *dst_name = dst_die->GetMangledName (this, dst_cu);
                if (dst_name)
                    dst_name_to_die.Append(ConstString(dst_name).GetCString(), dst_die);
            }
        }
    }
    const uint32_t src_size = src_name_to_die.GetSize ();
    const uint32_t dst_size = dst_name_to_die.GetSize ();
    LogSP log (LogChannelDWARF::GetLogIfAny(DWARF_LOG_DEBUG_INFO | DWARF_LOG_TYPE_COMPLETION));
    
    if (src_size && dst_size)
    {
        if (src_size != dst_size)
        {
            if (log)
                log->Printf("warning: tried to unique class DIE 0x%8.8x to 0x%8.8x, but they didn't have the same size (src=%d, dst=%d)",
                            src_class_die->GetOffset(),
                            dst_class_die->GetOffset(),
                            src_size,
                            dst_size);
            
            return false;
        }
            
        uint32_t idx;
        for (idx = 0; idx < src_size; ++idx)
        {
            src_die = src_name_to_die.GetValueAtIndexUnchecked (idx);
            dst_die = dst_name_to_die.GetValueAtIndexUnchecked (idx);

            if (src_die->Tag() != dst_die->Tag())
            {
                if (log)
                    log->Printf("warning: tried to unique class DIE 0x%8.8x to 0x%8.8x, but 0x%8.8x (%s) tags didn't match 0x%8.8x (%s)",
                                src_class_die->GetOffset(),
                                dst_class_die->GetOffset(),
                                src_die->GetOffset(),
                                DW_TAG_value_to_name(src_die->Tag()),
                                dst_die->GetOffset(),
                                DW_TAG_value_to_name(src_die->Tag()));
                return false;
            }
            
            const char *src_name = src_die->GetMangledName (this, src_cu);
            const char *dst_name = dst_die->GetMangledName (this, dst_cu);
            
            // Make sure the names match
            if (src_name == dst_name || (strcmp (src_name, dst_name) == 0))
                continue;

            if (log)
                log->Printf("warning: tried to unique class DIE 0x%8.8x to 0x%8.8x, but 0x%8.8x (%s) names didn't match 0x%8.8x (%s)",
                            src_class_die->GetOffset(),
                            dst_class_die->GetOffset(),
                            src_die->GetOffset(),
                            src_name,
                            dst_die->GetOffset(),
                            dst_name);
            
            return false;
        }

        for (idx = 0; idx < src_size; ++idx)
        {
            src_die = src_name_to_die.GetValueAtIndexUnchecked (idx);
            dst_die = dst_name_to_die.GetValueAtIndexUnchecked (idx);
            
            clang::DeclContext *src_decl_ctx = m_die_to_decl_ctx[src_die];
            if (src_decl_ctx)
            {
                if (log)
                    log->Printf ("uniquing decl context %p from 0x%8.8x for 0x%8.8x\n", src_decl_ctx, src_die->GetOffset(), dst_die->GetOffset());
                LinkDeclContextToDIE (src_decl_ctx, dst_die);
            }
            else
            {
                if (log)
                    log->Printf ("warning: tried to unique decl context from 0x%8.8x for 0x%8.8x, but none was found\n", src_die->GetOffset(), dst_die->GetOffset());
            }
            
            Type *src_child_type = m_die_to_type[src_die];
            if (src_child_type)
            {
                if (log)
                    log->Printf ("uniquing type %p (uid=0x%llx) from 0x%8.8x for 0x%8.8x\n", src_child_type, src_child_type->GetID(), src_die->GetOffset(), dst_die->GetOffset());
                m_die_to_type[dst_die] = src_child_type;
            }
            else
            {
                if (log)
                    log->Printf ("warning: tried to unique lldb_private::Type from 0x%8.8x for 0x%8.8x, but none was found\n", src_die->GetOffset(), dst_die->GetOffset());
            }
        }
        return true;
    }
    else
    {
        if (log)
            log->Printf("warning: tried to unique class DIE 0x%8.8x to 0x%8.8x, but 0x%8.8x has %u methods and 0x%8.8x has %u",
                        src_class_die->GetOffset(),
                        dst_class_die->GetOffset(),
                        src_die->GetOffset(),
                        src_size,
                        dst_die->GetOffset(),
                        dst_size);
    }
    return false;
}

TypeSP
SymbolFileDWARF::ParseType (const SymbolContext& sc, DWARFCompileUnit* dwarf_cu, const DWARFDebugInfoEntry *die, bool *type_is_new_ptr)
{
    TypeSP type_sp;

    if (type_is_new_ptr)
        *type_is_new_ptr = false;

#if defined(LLDB_CONFIGURATION_DEBUG) or defined(LLDB_CONFIGURATION_RELEASE)
    static DIEStack g_die_stack;
    DIEStack::ScopedPopper scoped_die_logger(g_die_stack);
#endif

    AccessType accessibility = eAccessNone;
    if (die != NULL)
    {
        LogSP log (LogChannelDWARF::GetLogIfAll(DWARF_LOG_DEBUG_INFO));
        if (log)
        {
            const DWARFDebugInfoEntry *context_die;
            clang::DeclContext *context = GetClangDeclContextContainingDIE (dwarf_cu, die, &context_die);
            
            GetObjectFile()->GetModule()->LogMessage (log.get(), "SymbolFileDWARF::ParseType (die = 0x%8.8x, decl_ctx = %p (die 0x%8.8x)) %s name = '%s')", 
                        die->GetOffset(),
                        context,
                        context_die->GetOffset(),
                        DW_TAG_value_to_name(die->Tag()), 
                        die->GetName(this, dwarf_cu));
            
#if defined(LLDB_CONFIGURATION_DEBUG) or defined(LLDB_CONFIGURATION_RELEASE)
            scoped_die_logger.Push (dwarf_cu, die);
            g_die_stack.LogDIEs(log.get(), this);
#endif
        }
//
//        LogSP log (LogChannelDWARF::GetLogIfAll(DWARF_LOG_DEBUG_INFO));
//        if (log && dwarf_cu)
//        {
//            StreamString s;
//            die->DumpLocation (this, dwarf_cu, s);
//            GetObjectFile()->GetModule()->LogMessage (log.get(), "SymbolFileDwarf::%s %s", __FUNCTION__, s.GetData());
//            
//        }
        
        Type *type_ptr = m_die_to_type.lookup (die);
        TypeList* type_list = GetTypeList();
        if (type_ptr == NULL)
        {
            ClangASTContext &ast = GetClangASTContext();
            if (type_is_new_ptr)
                *type_is_new_ptr = true;

            const dw_tag_t tag = die->Tag();

            bool is_forward_declaration = false;
            DWARFDebugInfoEntry::Attributes attributes;
            const char *type_name_cstr = NULL;
            ConstString type_name_const_str;
            Type::ResolveState resolve_state = Type::eResolveStateUnresolved;
            size_t byte_size = 0;
            Declaration decl;

            Type::EncodingDataType encoding_data_type = Type::eEncodingIsUID;
            clang_type_t clang_type = NULL;

            dw_attr_t attr;

            switch (tag)
            {
            case DW_TAG_base_type:
            case DW_TAG_pointer_type:
            case DW_TAG_reference_type:
            case DW_TAG_rvalue_reference_type:
            case DW_TAG_typedef:
            case DW_TAG_const_type:
            case DW_TAG_restrict_type:
            case DW_TAG_volatile_type:
            case DW_TAG_unspecified_type:
                {
                    // Set a bit that lets us know that we are currently parsing this
                    m_die_to_type[die] = DIE_IS_BEING_PARSED;

                    const size_t num_attributes = die->GetAttributes(this, dwarf_cu, NULL, attributes);
                    uint32_t encoding = 0;
                    lldb::user_id_t encoding_uid = LLDB_INVALID_UID;

                    if (num_attributes > 0)
                    {
                        uint32_t i;
                        for (i=0; i<num_attributes; ++i)
                        {
                            attr = attributes.AttributeAtIndex(i);
                            DWARFFormValue form_value;
                            if (attributes.ExtractFormValueAtIndex(this, i, form_value))
                            {
                                switch (attr)
                                {
                                case DW_AT_decl_file:   decl.SetFile(sc.comp_unit->GetSupportFiles().GetFileSpecAtIndex(form_value.Unsigned())); break;
                                case DW_AT_decl_line:   decl.SetLine(form_value.Unsigned()); break;
                                case DW_AT_decl_column: decl.SetColumn(form_value.Unsigned()); break;
                                case DW_AT_name:
                                    
                                    type_name_cstr = form_value.AsCString(&get_debug_str_data());
                                    // Work around a bug in llvm-gcc where they give a name to a reference type which doesn't
                                    // include the "&"...
                                    if (tag == DW_TAG_reference_type)
                                    {
                                        if (strchr (type_name_cstr, '&') == NULL)
                                            type_name_cstr = NULL;
                                    }
                                    if (type_name_cstr)
                                        type_name_const_str.SetCString(type_name_cstr);
                                    break;
                                case DW_AT_byte_size:   byte_size = form_value.Unsigned(); break;
                                case DW_AT_encoding:    encoding = form_value.Unsigned(); break;
                                case DW_AT_type:        encoding_uid = form_value.Reference(dwarf_cu); break;
                                default:
                                case DW_AT_sibling:
                                    break;
                                }
                            }
                        }
                    }

                    DEBUG_PRINTF ("0x%8.8llx: %s (\"%s\") type => 0x%8.8x\n", MakeUserID(die->GetOffset()), DW_TAG_value_to_name(tag), type_name_cstr, encoding_uid);

                    switch (tag)
                    {
                    default:
                        break;

                    case DW_TAG_unspecified_type:
                        if (strcmp(type_name_cstr, "nullptr_t") == 0)
                        {
                            resolve_state = Type::eResolveStateFull;
                            clang_type = ast.getASTContext()->NullPtrTy.getAsOpaquePtr();
                            break;
                        }
                        // Fall through to base type below in case we can handle the type there...

                    case DW_TAG_base_type:
                        resolve_state = Type::eResolveStateFull;
                        clang_type = ast.GetBuiltinTypeForDWARFEncodingAndBitSize (type_name_cstr, 
                                                                                   encoding, 
                                                                                   byte_size * 8);
                        break;

                    case DW_TAG_pointer_type:           encoding_data_type = Type::eEncodingIsPointerUID;           break;
                    case DW_TAG_reference_type:         encoding_data_type = Type::eEncodingIsLValueReferenceUID;   break;
                    case DW_TAG_rvalue_reference_type:  encoding_data_type = Type::eEncodingIsRValueReferenceUID;   break;
                    case DW_TAG_typedef:                encoding_data_type = Type::eEncodingIsTypedefUID;           break;
                    case DW_TAG_const_type:             encoding_data_type = Type::eEncodingIsConstUID;             break;
                    case DW_TAG_restrict_type:          encoding_data_type = Type::eEncodingIsRestrictUID;          break;
                    case DW_TAG_volatile_type:          encoding_data_type = Type::eEncodingIsVolatileUID;          break;
                    }

                    if (clang_type == NULL && (encoding_data_type == Type::eEncodingIsPointerUID || encoding_data_type == Type::eEncodingIsTypedefUID))
                    {
                        if (type_name_cstr != NULL && sc.comp_unit != NULL && 
                            (sc.comp_unit->GetLanguage() == eLanguageTypeObjC || sc.comp_unit->GetLanguage() == eLanguageTypeObjC_plus_plus))
                        {
                            static ConstString g_objc_type_name_id("id");
                            static ConstString g_objc_type_name_Class("Class");
                            static ConstString g_objc_type_name_selector("SEL");
                            
                            if (type_name_const_str == g_objc_type_name_id)
                            {
                                if (log)
                                    GetObjectFile()->GetModule()->LogMessage (log.get(), "SymbolFileDWARF::ParseType (die = 0x%8.8x) %s '%s' is Objective C 'id' built-in type.", 
                                                                              die->GetOffset(), 
                                                                              DW_TAG_value_to_name(die->Tag()), 
                                                                              die->GetName(this, dwarf_cu));
                                clang_type = ast.GetBuiltInType_objc_id();
                                encoding_data_type = Type::eEncodingIsUID;
                                encoding_uid = LLDB_INVALID_UID;
                                resolve_state = Type::eResolveStateFull;

                            }
                            else if (type_name_const_str == g_objc_type_name_Class)
                            {
                                if (log)
                                    GetObjectFile()->GetModule()->LogMessage (log.get(), "SymbolFileDWARF::ParseType (die = 0x%8.8x) %s '%s' is Objective C 'Class' built-in type.", 
                                                                              die->GetOffset(), 
                                                                              DW_TAG_value_to_name(die->Tag()), 
                                                                              die->GetName(this, dwarf_cu));
                                clang_type = ast.GetBuiltInType_objc_Class();
                                encoding_data_type = Type::eEncodingIsUID;
                                encoding_uid = LLDB_INVALID_UID;
                                resolve_state = Type::eResolveStateFull;
                            }
                            else if (type_name_const_str == g_objc_type_name_selector)
                            {
                                if (log)
                                    GetObjectFile()->GetModule()->LogMessage (log.get(), "SymbolFileDWARF::ParseType (die = 0x%8.8x) %s '%s' is Objective C 'selector' built-in type.", 
                                                                              die->GetOffset(), 
                                                                              DW_TAG_value_to_name(die->Tag()), 
                                                                              die->GetName(this, dwarf_cu));
                                clang_type = ast.GetBuiltInType_objc_selector();
                                encoding_data_type = Type::eEncodingIsUID;
                                encoding_uid = LLDB_INVALID_UID;
                                resolve_state = Type::eResolveStateFull;
                            }
                        }
                    }
                        
                    type_sp.reset( new Type (MakeUserID(die->GetOffset()),
                                             this, 
                                             type_name_const_str, 
                                             byte_size, 
                                             NULL, 
                                             encoding_uid, 
                                             encoding_data_type, 
                                             &decl, 
                                             clang_type, 
                                             resolve_state));
                    
                    m_die_to_type[die] = type_sp.get();

//                  Type* encoding_type = GetUniquedTypeForDIEOffset(encoding_uid, type_sp, NULL, 0, 0, false);
//                  if (encoding_type != NULL)
//                  {
//                      if (encoding_type != DIE_IS_BEING_PARSED)
//                          type_sp->SetEncodingType(encoding_type);
//                      else
//                          m_indirect_fixups.push_back(type_sp.get());
//                  }
                }
                break;

            case DW_TAG_structure_type:
            case DW_TAG_union_type:
            case DW_TAG_class_type:
                {
                    // Set a bit that lets us know that we are currently parsing this
                    m_die_to_type[die] = DIE_IS_BEING_PARSED;
                    bool byte_size_valid = false;

                    LanguageType class_language = eLanguageTypeUnknown;
                    bool is_complete_objc_class = false;
                    //bool struct_is_class = false;
                    const size_t num_attributes = die->GetAttributes(this, dwarf_cu, NULL, attributes);
                    if (num_attributes > 0)
                    {
                        uint32_t i;
                        for (i=0; i<num_attributes; ++i)
                        {
                            attr = attributes.AttributeAtIndex(i);
                            DWARFFormValue form_value;
                            if (attributes.ExtractFormValueAtIndex(this, i, form_value))
                            {
                                switch (attr)
                                {
                                case DW_AT_decl_file:
                                    if (dwarf_cu->DW_AT_decl_file_attributes_are_invalid())
									{
										// llvm-gcc outputs invalid DW_AT_decl_file attributes that always
										// point to the compile unit file, so we clear this invalid value
										// so that we can still unique types efficiently.
                                        decl.SetFile(FileSpec ("<invalid>", false));
									}
                                    else
                                        decl.SetFile(sc.comp_unit->GetSupportFiles().GetFileSpecAtIndex(form_value.Unsigned())); 
                                    break;

                                case DW_AT_decl_line:
                                    decl.SetLine(form_value.Unsigned()); 
                                    break;

                                case DW_AT_decl_column: 
                                    decl.SetColumn(form_value.Unsigned()); 
                                    break;

                                case DW_AT_name:
                                    type_name_cstr = form_value.AsCString(&get_debug_str_data());
                                    type_name_const_str.SetCString(type_name_cstr);
                                    break;

                                case DW_AT_byte_size:   
                                    byte_size = form_value.Unsigned(); 
                                    byte_size_valid = true;
                                    break;

                                case DW_AT_accessibility: 
                                    accessibility = DW_ACCESS_to_AccessType(form_value.Unsigned()); 
                                    break;

                                case DW_AT_declaration: 
                                    is_forward_declaration = form_value.Unsigned() != 0; 
                                    break;

                                case DW_AT_APPLE_runtime_class: 
                                    class_language = (LanguageType)form_value.Signed(); 
                                    break;

                                case DW_AT_APPLE_objc_complete_type:
                                    is_complete_objc_class = form_value.Signed(); 
                                    break;
                                        
                                case DW_AT_allocated:
                                case DW_AT_associated:
                                case DW_AT_data_location:
                                case DW_AT_description:
                                case DW_AT_start_scope:
                                case DW_AT_visibility:
                                default:
                                case DW_AT_sibling:
                                    break;
                                }
                            }
                        }
                    }

                    UniqueDWARFASTType unique_ast_entry;

                    // Only try and unique the type if it has a name. 
                    if (type_name_const_str &&
                        GetUniqueDWARFASTTypeMap().Find (type_name_const_str,
                                                         this,
                                                         dwarf_cu,
                                                         die,
                                                         decl,
                                                         byte_size_valid ? byte_size : -1,
                                                         unique_ast_entry))
                    {
                        // We have already parsed this type or from another 
                        // compile unit. GCC loves to use the "one definition
                        // rule" which can result in multiple definitions
                        // of the same class over and over in each compile
                        // unit.
                        type_sp = unique_ast_entry.m_type_sp;
                        if (type_sp)
                        {
                            m_die_to_type[die] = type_sp.get();
                            return type_sp;
                        }
                    }
                    
                    DEBUG_PRINTF ("0x%8.8llx: %s (\"%s\")\n", MakeUserID(die->GetOffset()), DW_TAG_value_to_name(tag), type_name_cstr);

                    int tag_decl_kind = -1;
                    AccessType default_accessibility = eAccessNone;
                    if (tag == DW_TAG_structure_type)
                    {
                        tag_decl_kind = clang::TTK_Struct;
                        default_accessibility = eAccessPublic;
                    }
                    else if (tag == DW_TAG_union_type)
                    {
                        tag_decl_kind = clang::TTK_Union;
                        default_accessibility = eAccessPublic;
                    }
                    else if (tag == DW_TAG_class_type)
                    {
                        tag_decl_kind = clang::TTK_Class;
                        default_accessibility = eAccessPrivate;
                    }
                    
                    if (byte_size_valid && byte_size == 0 && type_name_cstr &&
                        die->HasChildren() == false && 
                        sc.comp_unit->GetLanguage() == eLanguageTypeObjC)
                    {
                        // Work around an issue with clang at the moment where
                        // forward declarations for objective C classes are emitted
                        // as:
                        //  DW_TAG_structure_type [2]  
                        //  DW_AT_name( "ForwardObjcClass" )
                        //  DW_AT_byte_size( 0x00 )
                        //  DW_AT_decl_file( "..." )
                        //  DW_AT_decl_line( 1 )
                        //
                        // Note that there is no DW_AT_declaration and there are
                        // no children, and the byte size is zero.
                        is_forward_declaration = true;
                    }

                    if (class_language == eLanguageTypeObjC ||
                        class_language == eLanguageTypeObjC_plus_plus)
                    {
                        if (!is_complete_objc_class && Supports_DW_AT_APPLE_objc_complete_type(dwarf_cu))
                        {
                            // We have a valid eSymbolTypeObjCClass class symbol whose
                            // name matches the current objective C class that we
                            // are trying to find and this DIE isn't the complete
                            // definition (we checked is_complete_objc_class above and
                            // know it is false), so the real definition is in here somewhere
                            type_sp = FindCompleteObjCDefinitionTypeForDIE (die, type_name_const_str, true);

                            if (!type_sp && GetDebugMapSymfile ())
                            {
                                // We weren't able to find a full declaration in
                                // this DWARF, see if we have a declaration anywhere    
                                // else...
                                type_sp = m_debug_map_symfile->FindCompleteObjCDefinitionTypeForDIE (die, type_name_const_str, true);
                            }
                            
                            if (type_sp)
                            {
                                if (log)
                                {
                                    GetObjectFile()->GetModule()->LogMessage (log.get(),
                                                                              "SymbolFileDWARF(%p) - 0x%8.8x: %s type \"%s\" is an incomplete objc type, complete type is 0x%8.8llx", 
                                                                              this,
                                                                              die->GetOffset(), 
                                                                              DW_TAG_value_to_name(tag),
                                                                              type_name_cstr,
                                                                              type_sp->GetID());
                                }
                                
                                // We found a real definition for this type elsewhere
                                // so lets use it and cache the fact that we found
                                // a complete type for this die
                                m_die_to_type[die] = type_sp.get();
                                return type_sp;
                            }
                        }
                    }
                    

                    if (is_forward_declaration)
                    {
                        // We have a forward declaration to a type and we need
                        // to try and find a full declaration. We look in the
                        // current type index just in case we have a forward
                        // declaration followed by an actual declarations in the
                        // DWARF. If this fails, we need to look elsewhere...
                        if (log)
                        {
                            GetObjectFile()->GetModule()->LogMessage (log.get(), 
                                                                      "SymbolFileDWARF(%p) - 0x%8.8x: %s type \"%s\" is a forward declaration, trying to find complete type", 
                                                                      this,
                                                                      die->GetOffset(), 
                                                                      DW_TAG_value_to_name(tag),
                                                                      type_name_cstr);
                        }
                    
                        DWARFDeclContext die_decl_ctx;
                        die->GetDWARFDeclContext(this, dwarf_cu, die_decl_ctx);

                        //type_sp = FindDefinitionTypeForDIE (dwarf_cu, die, type_name_const_str);
                        type_sp = FindDefinitionTypeForDWARFDeclContext (die_decl_ctx);

                        if (!type_sp && GetDebugMapSymfile ())
                        {
                            // We weren't able to find a full declaration in
                            // this DWARF, see if we have a declaration anywhere    
                            // else...
                            type_sp = m_debug_map_symfile->FindDefinitionTypeForDWARFDeclContext (die_decl_ctx);
                        }

                        if (type_sp)
                        {
                            if (log)
                            {
                                GetObjectFile()->GetModule()->LogMessage (log.get(),
                                                                          "SymbolFileDWARF(%p) - 0x%8.8x: %s type \"%s\" is a forward declaration, complete type is 0x%8.8llx", 
                                                                          this,
                                                                          die->GetOffset(), 
                                                                          DW_TAG_value_to_name(tag),
                                                                          type_name_cstr,
                                                                          type_sp->GetID());
                            }

                            // We found a real definition for this type elsewhere
                            // so lets use it and cache the fact that we found
                            // a complete type for this die
                            m_die_to_type[die] = type_sp.get();
                            return type_sp;
                        }
                    }
                    assert (tag_decl_kind != -1);
                    bool clang_type_was_created = false;
                    clang_type = m_forward_decl_die_to_clang_type.lookup (die);
                    if (clang_type == NULL)
                    {
                        const DWARFDebugInfoEntry *decl_ctx_die;
                        
                        clang::DeclContext *decl_ctx = GetClangDeclContextContainingDIE (dwarf_cu, die, &decl_ctx_die);
                        if (accessibility == eAccessNone && decl_ctx)
                        {
                            // Check the decl context that contains this class/struct/union.
                            // If it is a class we must give it an accessability.
                            const clang::Decl::Kind containing_decl_kind = decl_ctx->getDeclKind();
                            if (DeclKindIsCXXClass (containing_decl_kind))
                                accessibility = default_accessibility;
                        }

                        if (type_name_cstr && strchr (type_name_cstr, '<'))
                        {
                            ClangASTContext::TemplateParameterInfos template_param_infos;
                            if (ParseTemplateParameterInfos (dwarf_cu, die, template_param_infos))
                            {
                                clang::ClassTemplateDecl *class_template_decl = ParseClassTemplateDecl (decl_ctx,
                                                                                                        accessibility,
                                                                                                        type_name_cstr,
                                                                                                        tag_decl_kind,
                                                                                                        template_param_infos);
                            
                                clang::ClassTemplateSpecializationDecl *class_specialization_decl = ast.CreateClassTemplateSpecializationDecl (decl_ctx,
                                                                                                                                               class_template_decl,
                                                                                                                                               tag_decl_kind,
                                                                                                                                               template_param_infos);
                                clang_type = ast.CreateClassTemplateSpecializationType (class_specialization_decl);
                                clang_type_was_created = true;
                                
                                GetClangASTContext().SetMetadataAsUserID ((uintptr_t)class_template_decl, MakeUserID(die->GetOffset()));
                                GetClangASTContext().SetMetadataAsUserID ((uintptr_t)class_specialization_decl, MakeUserID(die->GetOffset()));
                            }
                        }

                        if (!clang_type_was_created)
                        {
                            clang_type_was_created = true;
                            ClangASTMetadata metadata;
                            metadata.SetUserID(MakeUserID(die->GetOffset()));
                            clang_type = ast.CreateRecordType (decl_ctx, 
                                                               accessibility, 
                                                               type_name_cstr, 
                                                               tag_decl_kind, 
                                                               class_language,
                                                               &metadata);
                        }
                    }

                    // Store a forward declaration to this class type in case any 
                    // parameters in any class methods need it for the clang 
                    // types for function prototypes.
                    LinkDeclContextToDIE(ClangASTContext::GetDeclContextForType(clang_type), die);
                    type_sp.reset (new Type (MakeUserID(die->GetOffset()), 
                                             this, 
                                             type_name_const_str, 
                                             byte_size, 
                                             NULL, 
                                             LLDB_INVALID_UID, 
                                             Type::eEncodingIsUID, 
                                             &decl, 
                                             clang_type, 
                                             Type::eResolveStateForward));
                    
                    type_sp->SetIsCompleteObjCClass(is_complete_objc_class);


                    // Add our type to the unique type map so we don't
                    // end up creating many copies of the same type over
                    // and over in the ASTContext for our module
                    unique_ast_entry.m_type_sp = type_sp;
                    unique_ast_entry.m_symfile = this;
                    unique_ast_entry.m_cu = dwarf_cu;
                    unique_ast_entry.m_die = die;
                    unique_ast_entry.m_declaration = decl;
                    unique_ast_entry.m_byte_size = byte_size;
                    GetUniqueDWARFASTTypeMap().Insert (type_name_const_str, 
                                                       unique_ast_entry);
                    
                    if (!is_forward_declaration)
                    {
                        // Always start the definition for a class type so that
                        // if the class has child classes or types that require
                        // the class to be created for use as their decl contexts
                        // the class will be ready to accept these child definitions.
                        if (die->HasChildren() == false)
                        {
                            // No children for this struct/union/class, lets finish it
                            ast.StartTagDeclarationDefinition (clang_type);
                            ast.CompleteTagDeclarationDefinition (clang_type);
                            
                            if (tag == DW_TAG_structure_type) // this only applies in C
                            {
                                clang::QualType qual_type = clang::QualType::getFromOpaquePtr (clang_type);
                                const clang::RecordType *record_type = qual_type->getAs<clang::RecordType> ();
                                
                                if (record_type)
                                {
                                    clang::RecordDecl *record_decl = record_type->getDecl();
                                    
                                    if (record_decl)
                                    {
                                        LayoutInfo layout_info;
                                        
                                        layout_info.alignment = 0;
                                        layout_info.bit_size = 0;
                                        
                                        m_record_decl_to_layout_map.insert(std::make_pair(record_decl, layout_info));
                                    }
                                }
                            }
                        }
                        else if (clang_type_was_created)
                        {
                            // Start the definition if the class is not objective C since
                            // the underlying decls respond to isCompleteDefinition(). Objective
                            // C decls dont' respond to isCompleteDefinition() so we can't
                            // start the declaration definition right away. For C++ classs/union/structs
                            // we want to start the definition in case the class is needed as the
                            // declaration context for a contained class or type without the need
                            // to complete that type..
                            
                            if (class_language != eLanguageTypeObjC &&
                                class_language != eLanguageTypeObjC_plus_plus)
                                ast.StartTagDeclarationDefinition (clang_type);

                            // Leave this as a forward declaration until we need
                            // to know the details of the type. lldb_private::Type
                            // will automatically call the SymbolFile virtual function
                            // "SymbolFileDWARF::ResolveClangOpaqueTypeDefinition(Type *)"
                            // When the definition needs to be defined.
                            m_forward_decl_die_to_clang_type[die] = clang_type;
                            m_forward_decl_clang_type_to_die[ClangASTType::RemoveFastQualifiers (clang_type)] = die;
                            ClangASTContext::SetHasExternalStorage (clang_type, true);
                        }
                    }
                    
                }
                break;

            case DW_TAG_enumeration_type:
                {
                    // Set a bit that lets us know that we are currently parsing this
                    m_die_to_type[die] = DIE_IS_BEING_PARSED;

                    lldb::user_id_t encoding_uid = DW_INVALID_OFFSET;

                    const size_t num_attributes = die->GetAttributes(this, dwarf_cu, NULL, attributes);
                    if (num_attributes > 0)
                    {
                        uint32_t i;

                        for (i=0; i<num_attributes; ++i)
                        {
                            attr = attributes.AttributeAtIndex(i);
                            DWARFFormValue form_value;
                            if (attributes.ExtractFormValueAtIndex(this, i, form_value))
                            {
                                switch (attr)
                                {
                                case DW_AT_decl_file:       decl.SetFile(sc.comp_unit->GetSupportFiles().GetFileSpecAtIndex(form_value.Unsigned())); break;
                                case DW_AT_decl_line:       decl.SetLine(form_value.Unsigned()); break;
                                case DW_AT_decl_column:     decl.SetColumn(form_value.Unsigned()); break;
                                case DW_AT_name:
                                    type_name_cstr = form_value.AsCString(&get_debug_str_data());
                                    type_name_const_str.SetCString(type_name_cstr);
                                    break;
                                case DW_AT_type:            encoding_uid = form_value.Reference(dwarf_cu); break;
                                case DW_AT_byte_size:       byte_size = form_value.Unsigned(); break;
                                case DW_AT_accessibility:   break; //accessibility = DW_ACCESS_to_AccessType(form_value.Unsigned()); break;
                                case DW_AT_declaration:     break; //is_forward_declaration = form_value.Unsigned() != 0; break;
                                case DW_AT_allocated:
                                case DW_AT_associated:
                                case DW_AT_bit_stride:
                                case DW_AT_byte_stride:
                                case DW_AT_data_location:
                                case DW_AT_description:
                                case DW_AT_start_scope:
                                case DW_AT_visibility:
                                case DW_AT_specification:
                                case DW_AT_abstract_origin:
                                case DW_AT_sibling:
                                    break;
                                }
                            }
                        }

                        DEBUG_PRINTF ("0x%8.8llx: %s (\"%s\")\n", MakeUserID(die->GetOffset()), DW_TAG_value_to_name(tag), type_name_cstr);

                        clang_type_t enumerator_clang_type = NULL;
                        clang_type = m_forward_decl_die_to_clang_type.lookup (die);
                        if (clang_type == NULL)
                        {
                            enumerator_clang_type = ast.GetBuiltinTypeForDWARFEncodingAndBitSize (NULL, 
                                                                                                  DW_ATE_signed, 
                                                                                                  byte_size * 8);
                            clang_type = ast.CreateEnumerationType (type_name_cstr, 
                                                                    GetClangDeclContextContainingDIE (dwarf_cu, die, NULL), 
                                                                    decl,
                                                                    enumerator_clang_type);
                        }
                        else
                        {
                            enumerator_clang_type = ClangASTContext::GetEnumerationIntegerType (clang_type);
                            assert (enumerator_clang_type != NULL);
                        }

                        LinkDeclContextToDIE(ClangASTContext::GetDeclContextForType(clang_type), die);
                        
                        type_sp.reset( new Type (MakeUserID(die->GetOffset()), 
                                                 this, 
                                                 type_name_const_str, 
                                                 byte_size, 
                                                 NULL, 
                                                 encoding_uid, 
                                                 Type::eEncodingIsUID,
                                                 &decl, 
                                                 clang_type, 
                                                 Type::eResolveStateForward));

                        ast.StartTagDeclarationDefinition (clang_type);
                        if (die->HasChildren())
                        {
                            SymbolContext cu_sc(GetCompUnitForDWARFCompUnit(dwarf_cu));
                            ParseChildEnumerators(cu_sc, clang_type, type_sp->GetByteSize(), dwarf_cu, die);
                        }
                        ast.CompleteTagDeclarationDefinition (clang_type);
                    }
                }
                break;

            case DW_TAG_inlined_subroutine:
            case DW_TAG_subprogram:
            case DW_TAG_subroutine_type:
                {
                    // Set a bit that lets us know that we are currently parsing this
                    m_die_to_type[die] = DIE_IS_BEING_PARSED;

                    //const char *mangled = NULL;
                    dw_offset_t type_die_offset = DW_INVALID_OFFSET;
                    bool is_variadic = false;
                    bool is_inline = false;
                    bool is_static = false;
                    bool is_virtual = false;
                    bool is_explicit = false;
                    bool is_artificial = false;
                    dw_offset_t specification_die_offset = DW_INVALID_OFFSET;
                    dw_offset_t abstract_origin_die_offset = DW_INVALID_OFFSET;
                    dw_offset_t object_pointer_die_offset = DW_INVALID_OFFSET;

                    unsigned type_quals = 0;
                    clang::StorageClass storage = clang::SC_None;//, Extern, Static, PrivateExtern


                    const size_t num_attributes = die->GetAttributes(this, dwarf_cu, NULL, attributes);
                    if (num_attributes > 0)
                    {
                        uint32_t i;
                        for (i=0; i<num_attributes; ++i)
                        {
                            attr = attributes.AttributeAtIndex(i);
                            DWARFFormValue form_value;
                            if (attributes.ExtractFormValueAtIndex(this, i, form_value))
                            {
                                switch (attr)
                                {
                                case DW_AT_decl_file:   decl.SetFile(sc.comp_unit->GetSupportFiles().GetFileSpecAtIndex(form_value.Unsigned())); break;
                                case DW_AT_decl_line:   decl.SetLine(form_value.Unsigned()); break;
                                case DW_AT_decl_column: decl.SetColumn(form_value.Unsigned()); break;
                                case DW_AT_name:
                                    type_name_cstr = form_value.AsCString(&get_debug_str_data());
                                    type_name_const_str.SetCString(type_name_cstr);
                                    break;

                                case DW_AT_MIPS_linkage_name:   break; // mangled = form_value.AsCString(&get_debug_str_data()); break;
                                case DW_AT_type:                type_die_offset = form_value.Reference(dwarf_cu); break;
                                case DW_AT_accessibility:       accessibility = DW_ACCESS_to_AccessType(form_value.Unsigned()); break;
                                case DW_AT_declaration:         break; // is_forward_declaration = form_value.Unsigned() != 0; break;
                                case DW_AT_inline:              is_inline = form_value.Unsigned() != 0; break;
                                case DW_AT_virtuality:          is_virtual = form_value.Unsigned() != 0;  break;
                                case DW_AT_explicit:            is_explicit = form_value.Unsigned() != 0;  break; 
                                case DW_AT_artificial:          is_artificial = form_value.Unsigned() != 0;  break; 
                                        

                                case DW_AT_external:
                                    if (form_value.Unsigned())
                                    {
                                        if (storage == clang::SC_None)
                                            storage = clang::SC_Extern;
                                        else
                                            storage = clang::SC_PrivateExtern;
                                    }
                                    break;

                                case DW_AT_specification:
                                    specification_die_offset = form_value.Reference(dwarf_cu);
                                    break;

                                case DW_AT_abstract_origin:
                                    abstract_origin_die_offset = form_value.Reference(dwarf_cu);
                                    break;

                                case DW_AT_object_pointer:
                                    object_pointer_die_offset = form_value.Reference(dwarf_cu);
                                    break;

                                case DW_AT_allocated:
                                case DW_AT_associated:
                                case DW_AT_address_class:
                                case DW_AT_calling_convention:
                                case DW_AT_data_location:
                                case DW_AT_elemental:
                                case DW_AT_entry_pc:
                                case DW_AT_frame_base:
                                case DW_AT_high_pc:
                                case DW_AT_low_pc:
                                case DW_AT_prototyped:
                                case DW_AT_pure:
                                case DW_AT_ranges:
                                case DW_AT_recursive:
                                case DW_AT_return_addr:
                                case DW_AT_segment:
                                case DW_AT_start_scope:
                                case DW_AT_static_link:
                                case DW_AT_trampoline:
                                case DW_AT_visibility:
                                case DW_AT_vtable_elem_location:
                                case DW_AT_description:
                                case DW_AT_sibling:
                                    break;
                                }
                            }
                        }
                    }

                    std::string object_pointer_name;
                    if (object_pointer_die_offset != DW_INVALID_OFFSET)
                    {
                        // Get the name from the object pointer die
                        StreamString s;
                        if (DWARFDebugInfoEntry::GetName (this, dwarf_cu, object_pointer_die_offset, s))
                        {
                            object_pointer_name.assign(s.GetData());
                        }
                    }
                    
                    DEBUG_PRINTF ("0x%8.8llx: %s (\"%s\")\n", MakeUserID(die->GetOffset()), DW_TAG_value_to_name(tag), type_name_cstr);

                    clang_type_t return_clang_type = NULL;
                    Type *func_type = NULL;
                    
                    if (type_die_offset != DW_INVALID_OFFSET)
                        func_type = ResolveTypeUID(type_die_offset);

                    if (func_type)
                        return_clang_type = func_type->GetClangForwardType();
                    else
                        return_clang_type = ast.GetBuiltInType_void();


                    std::vector<clang_type_t> function_param_types;
                    std::vector<clang::ParmVarDecl*> function_param_decls;

                    // Parse the function children for the parameters
                    
                    const DWARFDebugInfoEntry *decl_ctx_die = NULL;
                    clang::DeclContext *containing_decl_ctx = GetClangDeclContextContainingDIE (dwarf_cu, die, &decl_ctx_die);
                    const clang::Decl::Kind containing_decl_kind = containing_decl_ctx->getDeclKind();

                    const bool is_cxx_method = DeclKindIsCXXClass (containing_decl_kind);
                    // Start off static. This will be set to false in ParseChildParameters(...)
                    // if we find a "this" paramters as the first parameter
                    if (is_cxx_method)
                        is_static = true;
                    ClangASTContext::TemplateParameterInfos template_param_infos;

                    if (die->HasChildren())
                    {
                        bool skip_artificial = true;
                        ParseChildParameters (sc,
                                              containing_decl_ctx,
                                              dwarf_cu,
                                              die,
                                              skip_artificial,
                                              is_static,
                                              type_list,
                                              function_param_types,
                                              function_param_decls,
                                              type_quals,
                                              template_param_infos);
                    }

                    // clang_type will get the function prototype clang type after this call
                    clang_type = ast.CreateFunctionType (return_clang_type, 
                                                         function_param_types.data(),
                                                         function_param_types.size(), 
                                                         is_variadic, 
                                                         type_quals);
                    
                    if (type_name_cstr)
                    {
                        bool type_handled = false;
                        if (tag == DW_TAG_subprogram)
                        {
                            ConstString class_name;
                            ConstString class_name_no_category;
                            if (ObjCLanguageRuntime::ParseMethodName (type_name_cstr, &class_name, NULL, NULL, &class_name_no_category))
                            {
                                // Use the class name with no category if there is one
                                if (class_name_no_category)
                                    class_name = class_name_no_category;

                                SymbolContext empty_sc;
                                clang_type_t class_opaque_type = NULL;
                                if (class_name)
                                {
                                    TypeList types;
                                    TypeSP complete_objc_class_type_sp (FindCompleteObjCDefinitionTypeForDIE (NULL, class_name, false));

                                    if (complete_objc_class_type_sp)
                                    {
                                        clang_type_t type_clang_forward_type = complete_objc_class_type_sp->GetClangForwardType();
                                        if (ClangASTContext::IsObjCClassType (type_clang_forward_type))
                                            class_opaque_type = type_clang_forward_type;
                                    }
                                }

                                if (class_opaque_type)
                                {
                                    // If accessibility isn't set to anything valid, assume public for 
                                    // now...
                                    if (accessibility == eAccessNone)
                                        accessibility = eAccessPublic;

                                    clang::ObjCMethodDecl *objc_method_decl = ast.AddMethodToObjCObjectType (class_opaque_type, 
                                                                                                             type_name_cstr,
                                                                                                             clang_type,
                                                                                                             accessibility);
                                    type_handled = objc_method_decl != NULL;
                                    if (type_handled)
                                    {
                                        LinkDeclContextToDIE(ClangASTContext::GetAsDeclContext(objc_method_decl), die);
                                        GetClangASTContext().SetMetadataAsUserID ((uintptr_t)objc_method_decl, MakeUserID(die->GetOffset()));
                                    }
                                }
                            }
                            else if (is_cxx_method)
                            {
                                // Look at the parent of this DIE and see if is is
                                // a class or struct and see if this is actually a
                                // C++ method
                                Type *class_type = ResolveType (dwarf_cu, decl_ctx_die);
                                if (class_type)
                                {
                                    if (class_type->GetID() != MakeUserID(decl_ctx_die->GetOffset()))
                                    {
                                        // We uniqued the parent class of this function to another class
                                        // so we now need to associate all dies under "decl_ctx_die" to
                                        // DIEs in the DIE for "class_type"...
                                        DWARFCompileUnitSP class_type_cu_sp;
                                        const DWARFDebugInfoEntry *class_type_die = DebugInfo()->GetDIEPtr(class_type->GetID(), &class_type_cu_sp);
                                        if (class_type_die)
                                        {
                                            if (CopyUniqueClassMethodTypes (class_type,
                                                                            class_type_cu_sp.get(),
                                                                            class_type_die,
                                                                            dwarf_cu,
                                                                            decl_ctx_die))
                                            {
                                                type_ptr = m_die_to_type[die];
                                                if (type_ptr && type_ptr != DIE_IS_BEING_PARSED)
                                                {
                                                    type_sp = type_ptr->shared_from_this();
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                    
                                    if (specification_die_offset != DW_INVALID_OFFSET)
                                    {
                                        // We have a specification which we are going to base our function
                                        // prototype off of, so we need this type to be completed so that the
                                        // m_die_to_decl_ctx for the method in the specification has a valid
                                        // clang decl context.
                                        class_type->GetClangForwardType();
                                        // If we have a specification, then the function type should have been
                                        // made with the specification and not with this die.
                                        DWARFCompileUnitSP spec_cu_sp;
                                        const DWARFDebugInfoEntry* spec_die = DebugInfo()->GetDIEPtr(specification_die_offset, &spec_cu_sp);
                                        clang::DeclContext *spec_clang_decl_ctx = GetClangDeclContextForDIE (sc, dwarf_cu, spec_die);
                                        if (spec_clang_decl_ctx)
                                        {
                                            LinkDeclContextToDIE(spec_clang_decl_ctx, die);
                                        }
                                        else
                                        {
                                            GetObjectFile()->GetModule()->ReportWarning ("0x%8.8llx: DW_AT_specification(0x%8.8x) has no decl\n", 
                                                                                         MakeUserID(die->GetOffset()), 
                                                                                         specification_die_offset);
                                        }
                                        type_handled = true;
                                    }
                                    else if (abstract_origin_die_offset != DW_INVALID_OFFSET)
                                    {
                                        // We have a specification which we are going to base our function
                                        // prototype off of, so we need this type to be completed so that the
                                        // m_die_to_decl_ctx for the method in the abstract origin has a valid
                                        // clang decl context.
                                        class_type->GetClangForwardType();

                                        DWARFCompileUnitSP abs_cu_sp;
                                        const DWARFDebugInfoEntry* abs_die = DebugInfo()->GetDIEPtr(abstract_origin_die_offset, &abs_cu_sp);
                                        clang::DeclContext *abs_clang_decl_ctx = GetClangDeclContextForDIE (sc, dwarf_cu, abs_die);
                                        if (abs_clang_decl_ctx)
                                        {
                                            LinkDeclContextToDIE (abs_clang_decl_ctx, die);
                                        }
                                        else
                                        {
                                            GetObjectFile()->GetModule()->ReportWarning ("0x%8.8llx: DW_AT_abstract_origin(0x%8.8x) has no decl\n", 
                                                                                         MakeUserID(die->GetOffset()), 
                                                                                         abstract_origin_die_offset);
                                        }
                                        type_handled = true;
                                    }
                                    else
                                    {
                                        clang_type_t class_opaque_type = class_type->GetClangForwardType();
                                        if (ClangASTContext::IsCXXClassType (class_opaque_type))
                                        {
                                            if (ClangASTContext::IsBeingDefined (class_opaque_type))
                                            {
                                                // Neither GCC 4.2 nor clang++ currently set a valid accessibility
                                                // in the DWARF for C++ methods... Default to public for now...
                                                if (accessibility == eAccessNone)
                                                    accessibility = eAccessPublic;
                                                
                                                if (!is_static && !die->HasChildren())
                                                {
                                                    // We have a C++ member function with no children (this pointer!)
                                                    // and clang will get mad if we try and make a function that isn't
                                                    // well formed in the DWARF, so we will just skip it...
                                                    type_handled = true;
                                                }
                                                else
                                                {
                                                    clang::CXXMethodDecl *cxx_method_decl;
                                                    // REMOVE THE CRASH DESCRIPTION BELOW
                                                    Host::SetCrashDescriptionWithFormat ("SymbolFileDWARF::ParseType() is adding a method %s to class %s in DIE 0x%8.8llx from %s/%s", 
                                                                                         type_name_cstr, 
                                                                                         class_type->GetName().GetCString(),
                                                                                         MakeUserID(die->GetOffset()),
                                                                                         m_obj_file->GetFileSpec().GetDirectory().GetCString(),
                                                                                         m_obj_file->GetFileSpec().GetFilename().GetCString());

                                                    const bool is_attr_used = false;
                                                    
                                                    cxx_method_decl = ast.AddMethodToCXXRecordType (class_opaque_type, 
                                                                                                    type_name_cstr,
                                                                                                    clang_type,
                                                                                                    accessibility,
                                                                                                    is_virtual,
                                                                                                    is_static,
                                                                                                    is_inline,
                                                                                                    is_explicit,
                                                                                                    is_attr_used,
                                                                                                    is_artificial);
                                                    
                                                    type_handled = cxx_method_decl != NULL;

                                                    if (type_handled)
                                                    {
                                                        LinkDeclContextToDIE(ClangASTContext::GetAsDeclContext(cxx_method_decl), die);

                                                        Host::SetCrashDescription (NULL);

                                                        
                                                        ClangASTMetadata metadata;
                                                        metadata.SetUserID(MakeUserID(die->GetOffset()));
                                                        
                                                        if (!object_pointer_name.empty())
                                                        {
                                                            metadata.SetObjectPtrName(object_pointer_name.c_str());
                                                            if (log)
                                                                log->Printf ("Setting object pointer name: %s on method object 0x%ld.\n",
                                                                             object_pointer_name.c_str(),
                                                                             (uintptr_t) cxx_method_decl);
                                                        }
                                                        GetClangASTContext().SetMetadata ((uintptr_t)cxx_method_decl, metadata);
                                                    }
                                                }
                                            }
                                            else
                                            {
                                                // We were asked to parse the type for a method in a class, yet the
                                                // class hasn't been asked to complete itself through the 
                                                // clang::ExternalASTSource protocol, so we need to just have the
                                                // class complete itself and do things the right way, then our 
                                                // DIE should then have an entry in the m_die_to_type map. First 
                                                // we need to modify the m_die_to_type so it doesn't think we are 
                                                // trying to parse this DIE anymore...
                                                m_die_to_type[die] = NULL;
                                                
                                                // Now we get the full type to force our class type to complete itself 
                                                // using the clang::ExternalASTSource protocol which will parse all 
                                                // base classes and all methods (including the method for this DIE).
                                                class_type->GetClangFullType();

                                                // The type for this DIE should have been filled in the function call above
                                                type_ptr = m_die_to_type[die];
                                                if (type_ptr && type_ptr != DIE_IS_BEING_PARSED)
                                                {
                                                    type_sp = type_ptr->shared_from_this();
                                                    break;
                                                }
                                                
                                                // FIXME This is fixing some even uglier behavior but we really need to
                                                // uniq the methods of each class as well as the class itself.
                                                // <rdar://problem/11240464>
                                                type_handled = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                            
                        if (!type_handled)
                        {
                            // We just have a function that isn't part of a class
                            clang::FunctionDecl *function_decl = ast.CreateFunctionDeclaration (containing_decl_ctx,
                                                                                                type_name_cstr, 
                                                                                                clang_type, 
                                                                                                storage, 
                                                                                                is_inline);

//                            if (template_param_infos.GetSize() > 0)
//                            {
//                                clang::FunctionTemplateDecl *func_template_decl = ast.CreateFunctionTemplateDecl (containing_decl_ctx,
//                                                                                                                  function_decl,
//                                                                                                                  type_name_cstr, 
//                                                                                                                  template_param_infos);
//                                
//                                ast.CreateFunctionTemplateSpecializationInfo (function_decl,
//                                                                              func_template_decl,
//                                                                              template_param_infos);
//                            }
                            // Add the decl to our DIE to decl context map
                            assert (function_decl);
                            LinkDeclContextToDIE(function_decl, die);
                            if (!function_param_decls.empty())
                                ast.SetFunctionParameters (function_decl, 
                                                           &function_param_decls.front(), 
                                                           function_param_decls.size());
                            
                            ClangASTMetadata metadata;
                            metadata.SetUserID(MakeUserID(die->GetOffset()));
                            
                            if (!object_pointer_name.empty())
                            {
                                metadata.SetObjectPtrName(object_pointer_name.c_str());
                                if (log)
                                    log->Printf ("Setting object pointer name: %s on function object 0x%ld.\n",
                                                 object_pointer_name.c_str(),
                                                 (uintptr_t) function_decl);
                            }
                            GetClangASTContext().SetMetadata ((uintptr_t)function_decl, metadata);
                        }
                    }
                    type_sp.reset( new Type (MakeUserID(die->GetOffset()), 
                                             this, 
                                             type_name_const_str, 
                                             0, 
                                             NULL, 
                                             LLDB_INVALID_UID, 
                                             Type::eEncodingIsUID, 
                                             &decl, 
                                             clang_type, 
                                             Type::eResolveStateFull));                    
                    assert(type_sp.get());
                }
                break;

            case DW_TAG_array_type:
                {
                    // Set a bit that lets us know that we are currently parsing this
                    m_die_to_type[die] = DIE_IS_BEING_PARSED;

                    lldb::user_id_t type_die_offset = DW_INVALID_OFFSET;
                    int64_t first_index = 0;
                    uint32_t byte_stride = 0;
                    uint32_t bit_stride = 0;
                    const size_t num_attributes = die->GetAttributes(this, dwarf_cu, NULL, attributes);

                    if (num_attributes > 0)
                    {
                        uint32_t i;
                        for (i=0; i<num_attributes; ++i)
                        {
                            attr = attributes.AttributeAtIndex(i);
                            DWARFFormValue form_value;
                            if (attributes.ExtractFormValueAtIndex(this, i, form_value))
                            {
                                switch (attr)
                                {
                                case DW_AT_decl_file:   decl.SetFile(sc.comp_unit->GetSupportFiles().GetFileSpecAtIndex(form_value.Unsigned())); break;
                                case DW_AT_decl_line:   decl.SetLine(form_value.Unsigned()); break;
                                case DW_AT_decl_column: decl.SetColumn(form_value.Unsigned()); break;
                                case DW_AT_name:
                                    type_name_cstr = form_value.AsCString(&get_debug_str_data());
                                    type_name_const_str.SetCString(type_name_cstr);
                                    break;

                                case DW_AT_type:            type_die_offset = form_value.Reference(dwarf_cu); break;
                                case DW_AT_byte_size:       break; // byte_size = form_value.Unsigned(); break;
                                case DW_AT_byte_stride:     byte_stride = form_value.Unsigned(); break;
                                case DW_AT_bit_stride:      bit_stride = form_value.Unsigned(); break;
                                case DW_AT_accessibility:   break; // accessibility = DW_ACCESS_to_AccessType(form_value.Unsigned()); break;
                                case DW_AT_declaration:     break; // is_forward_declaration = form_value.Unsigned() != 0; break;
                                case DW_AT_allocated:
                                case DW_AT_associated:
                                case DW_AT_data_location:
                                case DW_AT_description:
                                case DW_AT_ordering:
                                case DW_AT_start_scope:
                                case DW_AT_visibility:
                                case DW_AT_specification:
                                case DW_AT_abstract_origin:
                                case DW_AT_sibling:
                                    break;
                                }
                            }
                        }

                        DEBUG_PRINTF ("0x%8.8llx: %s (\"%s\")\n", MakeUserID(die->GetOffset()), DW_TAG_value_to_name(tag), type_name_cstr);

                        Type *element_type = ResolveTypeUID(type_die_offset);

                        if (element_type)
                        {
                            std::vector<uint64_t> element_orders;
                            ParseChildArrayInfo(sc, dwarf_cu, die, first_index, element_orders, byte_stride, bit_stride);
                            if (byte_stride == 0 && bit_stride == 0)
                                byte_stride = element_type->GetByteSize();
                            clang_type_t array_element_type = element_type->GetClangForwardType();
                            uint64_t array_element_bit_stride = byte_stride * 8 + bit_stride;
                            uint64_t num_elements = 0;
                            std::vector<uint64_t>::const_reverse_iterator pos;
                            std::vector<uint64_t>::const_reverse_iterator end = element_orders.rend();
                            for (pos = element_orders.rbegin(); pos != end; ++pos)
                            {
                                num_elements = *pos;
                                clang_type = ast.CreateArrayType (array_element_type, 
                                                                  num_elements, 
                                                                  num_elements * array_element_bit_stride);
                                array_element_type = clang_type;
                                array_element_bit_stride = array_element_bit_stride * num_elements;
                            }
                            ConstString empty_name;
                            type_sp.reset( new Type (MakeUserID(die->GetOffset()), 
                                                     this, 
                                                     empty_name, 
                                                     array_element_bit_stride / 8, 
                                                     NULL, 
                                                     type_die_offset, 
                                                     Type::eEncodingIsUID, 
                                                     &decl, 
                                                     clang_type, 
                                                     Type::eResolveStateFull));
                            type_sp->SetEncodingType (element_type);
                        }
                    }
                }
                break;

            case DW_TAG_ptr_to_member_type:
                {
                    dw_offset_t type_die_offset = DW_INVALID_OFFSET;
                    dw_offset_t containing_type_die_offset = DW_INVALID_OFFSET;

                    const size_t num_attributes = die->GetAttributes(this, dwarf_cu, NULL, attributes);
                    
                    if (num_attributes > 0) {
                        uint32_t i;
                        for (i=0; i<num_attributes; ++i)
                        {
                            attr = attributes.AttributeAtIndex(i);
                            DWARFFormValue form_value;
                            if (attributes.ExtractFormValueAtIndex(this, i, form_value))
                            {
                                switch (attr)
                                {
                                    case DW_AT_type:
                                        type_die_offset = form_value.Reference(dwarf_cu); break;
                                    case DW_AT_containing_type:
                                        containing_type_die_offset = form_value.Reference(dwarf_cu); break;
                                }
                            }
                        }
                        
                        Type *pointee_type = ResolveTypeUID(type_die_offset);
                        Type *class_type = ResolveTypeUID(containing_type_die_offset);
                        
                        clang_type_t pointee_clang_type = pointee_type->GetClangForwardType();
                        clang_type_t class_clang_type = class_type->GetClangLayoutType();

                        clang_type = ast.CreateMemberPointerType(pointee_clang_type, 
                                                                 class_clang_type);

                        byte_size = ClangASTType::GetClangTypeBitWidth (ast.getASTContext(), 
                                                                       clang_type) / 8;

                        type_sp.reset( new Type (MakeUserID(die->GetOffset()), 
                                                 this, 
                                                 type_name_const_str, 
                                                 byte_size, 
                                                 NULL, 
                                                 LLDB_INVALID_UID, 
                                                 Type::eEncodingIsUID, 
                                                 NULL, 
                                                 clang_type, 
                                                 Type::eResolveStateForward));
                    }
                                            
                    break;
                }
            default:
                GetObjectFile()->GetModule()->ReportError ("{0x%8.8x}: unhandled type tag 0x%4.4x (%s), please file a bug and attach the file at the start of this error message",
                                                           die->GetOffset(),
                                                           tag,
                                                           DW_TAG_value_to_name(tag));
                break;
            }

            if (type_sp.get())
            {
                const DWARFDebugInfoEntry *sc_parent_die = GetParentSymbolContextDIE(die);
                dw_tag_t sc_parent_tag = sc_parent_die ? sc_parent_die->Tag() : 0;

                SymbolContextScope * symbol_context_scope = NULL;
                if (sc_parent_tag == DW_TAG_compile_unit)
                {
                    symbol_context_scope = sc.comp_unit;
                }
                else if (sc.function != NULL)
                {
                    symbol_context_scope = sc.function->GetBlock(true).FindBlockByID(MakeUserID(sc_parent_die->GetOffset()));
                    if (symbol_context_scope == NULL)
                        symbol_context_scope = sc.function;
                }

                if (symbol_context_scope != NULL)
                {
                    type_sp->SetSymbolContextScope(symbol_context_scope);
                }

                // We are ready to put this type into the uniqued list up at the module level
                type_list->Insert (type_sp);

                m_die_to_type[die] = type_sp.get();
            }
        }
        else if (type_ptr != DIE_IS_BEING_PARSED)
        {
            type_sp = type_ptr->shared_from_this();
        }
    }
    return type_sp;
}

size_t
SymbolFileDWARF::ParseTypes
(
    const SymbolContext& sc, 
    DWARFCompileUnit* dwarf_cu, 
    const DWARFDebugInfoEntry *die, 
    bool parse_siblings, 
    bool parse_children
)
{
    size_t types_added = 0;
    while (die != NULL)
    {
        bool type_is_new = false;
        if (ParseType(sc, dwarf_cu, die, &type_is_new).get())
        {
            if (type_is_new)
                ++types_added;
        }

        if (parse_children && die->HasChildren())
        {
            if (die->Tag() == DW_TAG_subprogram)
            {
                SymbolContext child_sc(sc);
                child_sc.function = sc.comp_unit->FindFunctionByUID(MakeUserID(die->GetOffset())).get();
                types_added += ParseTypes(child_sc, dwarf_cu, die->GetFirstChild(), true, true);
            }
            else
                types_added += ParseTypes(sc, dwarf_cu, die->GetFirstChild(), true, true);
        }

        if (parse_siblings)
            die = die->GetSibling();
        else
            die = NULL;
    }
    return types_added;
}


size_t
SymbolFileDWARF::ParseFunctionBlocks (const SymbolContext &sc)
{
    assert(sc.comp_unit && sc.function);
    size_t functions_added = 0;
    DWARFCompileUnit* dwarf_cu = GetDWARFCompileUnit(sc.comp_unit);
    if (dwarf_cu)
    {
        dw_offset_t function_die_offset = sc.function->GetID();
        const DWARFDebugInfoEntry *function_die = dwarf_cu->GetDIEPtr(function_die_offset);
        if (function_die)
        {
            ParseFunctionBlocks(sc, &sc.function->GetBlock (false), dwarf_cu, function_die, LLDB_INVALID_ADDRESS, 0);
        }
    }

    return functions_added;
}


size_t
SymbolFileDWARF::ParseTypes (const SymbolContext &sc)
{
    // At least a compile unit must be valid
    assert(sc.comp_unit);
    size_t types_added = 0;
    DWARFCompileUnit* dwarf_cu = GetDWARFCompileUnit(sc.comp_unit);
    if (dwarf_cu)
    {
        if (sc.function)
        {
            dw_offset_t function_die_offset = sc.function->GetID();
            const DWARFDebugInfoEntry *func_die = dwarf_cu->GetDIEPtr(function_die_offset);
            if (func_die && func_die->HasChildren())
            {
                types_added = ParseTypes(sc, dwarf_cu, func_die->GetFirstChild(), true, true);
            }
        }
        else
        {
            const DWARFDebugInfoEntry *dwarf_cu_die = dwarf_cu->DIE();
            if (dwarf_cu_die && dwarf_cu_die->HasChildren())
            {
                types_added = ParseTypes(sc, dwarf_cu, dwarf_cu_die->GetFirstChild(), true, true);
            }
        }
    }

    return types_added;
}

size_t
SymbolFileDWARF::ParseVariablesForContext (const SymbolContext& sc)
{
    if (sc.comp_unit != NULL)
    {
        DWARFDebugInfo* info = DebugInfo();
        if (info == NULL)
            return 0;
        
        uint32_t cu_idx = UINT32_MAX;
        DWARFCompileUnit* dwarf_cu = info->GetCompileUnit(sc.comp_unit->GetID(), &cu_idx).get();

        if (dwarf_cu == NULL)
            return 0;

        if (sc.function)
        {
            const DWARFDebugInfoEntry *function_die = dwarf_cu->GetDIEPtr(sc.function->GetID());
            
            dw_addr_t func_lo_pc = function_die->GetAttributeValueAsUnsigned (this, dwarf_cu, DW_AT_low_pc, DW_INVALID_ADDRESS);
            if (func_lo_pc != DW_INVALID_ADDRESS)
            {
                const size_t num_variables = ParseVariables(sc, dwarf_cu, func_lo_pc, function_die->GetFirstChild(), true, true);
            
                // Let all blocks know they have parse all their variables
                sc.function->GetBlock (false).SetDidParseVariables (true, true);
                return num_variables;
            }
        }
        else if (sc.comp_unit)
        {
            uint32_t vars_added = 0;
            VariableListSP variables (sc.comp_unit->GetVariableList(false));
            
            if (variables.get() == NULL)
            {
                variables.reset(new VariableList());
                sc.comp_unit->SetVariableList(variables);

                DWARFCompileUnit* match_dwarf_cu = NULL;
                const DWARFDebugInfoEntry* die = NULL;
                DIEArray die_offsets;
                if (m_using_apple_tables)
                {
                    if (m_apple_names_ap.get())
                    {
                        DWARFMappedHash::DIEInfoArray hash_data_array;
                        if (m_apple_names_ap->AppendAllDIEsInRange (dwarf_cu->GetOffset(), 
                                                                    dwarf_cu->GetNextCompileUnitOffset(), 
                                                                    hash_data_array))
                        {
                            DWARFMappedHash::ExtractDIEArray (hash_data_array, die_offsets);
                        }
                    }
                }
                else
                {
                    // Index if we already haven't to make sure the compile units
                    // get indexed and make their global DIE index list
                    if (!m_indexed)
                        Index ();

                    m_global_index.FindAllEntriesForCompileUnit (dwarf_cu->GetOffset(), 
                                                                 dwarf_cu->GetNextCompileUnitOffset(), 
                                                                 die_offsets);
                }

                const size_t num_matches = die_offsets.size();
                if (num_matches)
                {
                    DWARFDebugInfo* debug_info = DebugInfo();
                    for (size_t i=0; i<num_matches; ++i)
                    {
                        const dw_offset_t die_offset = die_offsets[i];
                        die = debug_info->GetDIEPtrWithCompileUnitHint (die_offset, &match_dwarf_cu);
                        if (die)
                        {
                            VariableSP var_sp (ParseVariableDIE(sc, dwarf_cu, die, LLDB_INVALID_ADDRESS));
                            if (var_sp)
                            {
                                variables->AddVariableIfUnique (var_sp);
                                ++vars_added;
                            }
                        }
                        else
                        {
                            if (m_using_apple_tables)
                            {
                                GetObjectFile()->GetModule()->ReportErrorIfModifyDetected ("the DWARF debug information has been modified (.apple_names accelerator table had bad die 0x%8.8x)\n", die_offset);
                            }
                        }            

                    }
                }
            }
            return vars_added;
        }
    }
    return 0;
}


VariableSP
SymbolFileDWARF::ParseVariableDIE
(
    const SymbolContext& sc,
    DWARFCompileUnit* dwarf_cu,
    const DWARFDebugInfoEntry *die,
    const lldb::addr_t func_low_pc
)
{

    VariableSP var_sp (m_die_to_variable_sp[die]);
    if (var_sp)
        return var_sp;  // Already been parsed!
    
    const dw_tag_t tag = die->Tag();
    
    if ((tag == DW_TAG_variable) ||
        (tag == DW_TAG_constant) ||
        (tag == DW_TAG_formal_parameter && sc.function))
    {
        DWARFDebugInfoEntry::Attributes attributes;
        const size_t num_attributes = die->GetAttributes(this, dwarf_cu, NULL, attributes);
        if (num_attributes > 0)
        {
            const char *name = NULL;
            const char *mangled = NULL;
            Declaration decl;
            uint32_t i;
            lldb::user_id_t type_uid = LLDB_INVALID_UID;
            DWARFExpression location;
            bool is_external = false;
            bool is_artificial = false;
            bool location_is_const_value_data = false;
            //AccessType accessibility = eAccessNone;

            for (i=0; i<num_attributes; ++i)
            {
                dw_attr_t attr = attributes.AttributeAtIndex(i);
                DWARFFormValue form_value;
                if (attributes.ExtractFormValueAtIndex(this, i, form_value))
                {
                    switch (attr)
                    {
                    case DW_AT_decl_file:   decl.SetFile(sc.comp_unit->GetSupportFiles().GetFileSpecAtIndex(form_value.Unsigned())); break;
                    case DW_AT_decl_line:   decl.SetLine(form_value.Unsigned()); break;
                    case DW_AT_decl_column: decl.SetColumn(form_value.Unsigned()); break;
                    case DW_AT_name:        name = form_value.AsCString(&get_debug_str_data()); break;
                    case DW_AT_MIPS_linkage_name: mangled = form_value.AsCString(&get_debug_str_data()); break;
                    case DW_AT_type:        type_uid = form_value.Reference(dwarf_cu); break;
                    case DW_AT_external:    is_external = form_value.Unsigned() != 0; break;
                    case DW_AT_const_value:
                        location_is_const_value_data = true;
                        // Fall through...
                    case DW_AT_location:
                        {
                            if (form_value.BlockData())
                            {
                                const DataExtractor& debug_info_data = get_debug_info_data();

                                uint32_t block_offset = form_value.BlockData() - debug_info_data.GetDataStart();
                                uint32_t block_length = form_value.Unsigned();
                                location.CopyOpcodeData(get_debug_info_data(), block_offset, block_length);
                            }
                            else
                            {
                                const DataExtractor&    debug_loc_data = get_debug_loc_data();
                                const dw_offset_t debug_loc_offset = form_value.Unsigned();

                                size_t loc_list_length = DWARFLocationList::Size(debug_loc_data, debug_loc_offset);
                                if (loc_list_length > 0)
                                {
                                    location.CopyOpcodeData(debug_loc_data, debug_loc_offset, loc_list_length);
                                    assert (func_low_pc != LLDB_INVALID_ADDRESS);
                                    location.SetLocationListSlide (func_low_pc - dwarf_cu->GetBaseAddress());
                                }
                            }
                        }
                        break;

                    case DW_AT_artificial:      is_artificial = form_value.Unsigned() != 0; break;
                    case DW_AT_accessibility:   break; //accessibility = DW_ACCESS_to_AccessType(form_value.Unsigned()); break;
                    case DW_AT_declaration:
                    case DW_AT_description:
                    case DW_AT_endianity:
                    case DW_AT_segment:
                    case DW_AT_start_scope:
                    case DW_AT_visibility:
                    default:
                    case DW_AT_abstract_origin:
                    case DW_AT_sibling:
                    case DW_AT_specification:
                        break;
                    }
                }
            }

            if (location.IsValid())
            {
                ValueType scope = eValueTypeInvalid;

                const DWARFDebugInfoEntry *sc_parent_die = GetParentSymbolContextDIE(die);
                dw_tag_t parent_tag = sc_parent_die ? sc_parent_die->Tag() : 0;
                SymbolContextScope * symbol_context_scope = NULL;

                // DWARF doesn't specify if a DW_TAG_variable is a local, global
                // or static variable, so we have to do a little digging by
                // looking at the location of a varaible to see if it contains
                // a DW_OP_addr opcode _somewhere_ in the definition. I say
                // somewhere because clang likes to combine small global variables
                // into the same symbol and have locations like:
                // DW_OP_addr(0x1000), DW_OP_constu(2), DW_OP_plus
                // So if we don't have a DW_TAG_formal_parameter, we can look at
                // the location to see if it contains a DW_OP_addr opcode, and
                // then we can correctly classify  our variables.
                if (tag == DW_TAG_formal_parameter)
                    scope = eValueTypeVariableArgument;
                else
                {
                    bool op_error = false;
                    // Check if the location has a DW_OP_addr with any address value...
                    addr_t location_has_op_addr = false;
                    if (!location_is_const_value_data)
                    {
                        location_has_op_addr = location.LocationContains_DW_OP_addr (LLDB_INVALID_ADDRESS, op_error);
                        if (op_error)
                        {
                            StreamString strm;
                            location.DumpLocationForAddress (&strm, eDescriptionLevelFull, 0, 0, NULL);
                            GetObjectFile()->GetModule()->ReportError ("0x%8.8x: %s has an invalid location: %s", die->GetOffset(), DW_TAG_value_to_name(die->Tag()), strm.GetString().c_str());
                        }
                    }

                    if (location_has_op_addr)
                    {
                        if (is_external)
                        {
                            scope = eValueTypeVariableGlobal;

                            if (GetDebugMapSymfile ())
                            {
                                // When leaving the DWARF in the .o files on darwin,
                                // when we have a global variable that wasn't initialized,
                                // the .o file might not have allocated a virtual
                                // address for the global variable. In this case it will
                                // have created a symbol for the global variable
                                // that is undefined and external and the value will
                                // be the byte size of the variable. When we do the
                                // address map in SymbolFileDWARFDebugMap we rely on
                                // having an address, we need to do some magic here
                                // so we can get the correct address for our global 
                                // variable. The address for all of these entries
                                // will be zero, and there will be an undefined symbol
                                // in this object file, and the executable will have
                                // a matching symbol with a good address. So here we
                                // dig up the correct address and replace it in the
                                // location for the variable, and set the variable's
                                // symbol context scope to be that of the main executable
                                // so the file address will resolve correctly.
                                if (location.LocationContains_DW_OP_addr (0, op_error))
                                {
                                    
                                    // we have a possible uninitialized extern global
                                    Symtab *symtab = m_obj_file->GetSymtab();
                                    if (symtab)
                                    {
                                        ConstString const_name(name);
                                        Symbol *undefined_symbol = symtab->FindFirstSymbolWithNameAndType (const_name,
                                                                                                           eSymbolTypeUndefined, 
                                                                                                           Symtab::eDebugNo, 
                                                                                                           Symtab::eVisibilityExtern);
                                        
                                        if (undefined_symbol)
                                        {
                                            ObjectFile *debug_map_objfile = m_debug_map_symfile->GetObjectFile();
                                            if (debug_map_objfile)
                                            {
                                                Symtab *debug_map_symtab = debug_map_objfile->GetSymtab();
                                                Symbol *defined_symbol = debug_map_symtab->FindFirstSymbolWithNameAndType (const_name,
                                                                                                                           eSymbolTypeData, 
                                                                                                                           Symtab::eDebugYes, 
                                                                                                                           Symtab::eVisibilityExtern);
                                                if (defined_symbol)
                                                {
                                                    if (defined_symbol->ValueIsAddress())
                                                    {
                                                        const addr_t defined_addr = defined_symbol->GetAddress().GetFileAddress();
                                                        if (defined_addr != LLDB_INVALID_ADDRESS)
                                                        {
                                                            if (location.Update_DW_OP_addr (defined_addr))
                                                            {
                                                                symbol_context_scope = defined_symbol;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else  
                        {
                            scope = eValueTypeVariableStatic;
                        }
                    }
                    else
                    {
                        scope = eValueTypeVariableLocal;
                    }
                }

                if (symbol_context_scope == NULL)
                {
                    switch (parent_tag)
                    {
                    case DW_TAG_subprogram:
                    case DW_TAG_inlined_subroutine:
                    case DW_TAG_lexical_block:
                        if (sc.function)
                        {
                            symbol_context_scope = sc.function->GetBlock(true).FindBlockByID(MakeUserID(sc_parent_die->GetOffset()));
                            if (symbol_context_scope == NULL)
                                symbol_context_scope = sc.function;
                        }
                        break;
                    
                    default:
                        symbol_context_scope = sc.comp_unit;
                        break;
                    }
                }

                if (symbol_context_scope)
                {
                    var_sp.reset (new Variable (MakeUserID(die->GetOffset()), 
                                                name, 
                                                mangled,
                                                SymbolFileTypeSP (new SymbolFileType(*this, type_uid)),
                                                scope, 
                                                symbol_context_scope, 
                                                &decl, 
                                                location, 
                                                is_external, 
                                                is_artificial));
                    
                    var_sp->SetLocationIsConstantValueData (location_is_const_value_data);
                }
                else
                {
                    // Not ready to parse this variable yet. It might be a global
                    // or static variable that is in a function scope and the function
                    // in the symbol context wasn't filled in yet
                    return var_sp;
                }
            }
        }
        // Cache var_sp even if NULL (the variable was just a specification or
        // was missing vital information to be able to be displayed in the debugger
        // (missing location due to optimization, etc)) so we don't re-parse
        // this DIE over and over later...
        m_die_to_variable_sp[die] = var_sp;
    }
    return var_sp;
}


const DWARFDebugInfoEntry *
SymbolFileDWARF::FindBlockContainingSpecification (dw_offset_t func_die_offset, 
                                                   dw_offset_t spec_block_die_offset,
                                                   DWARFCompileUnit **result_die_cu_handle)
{
    // Give the concrete function die specified by "func_die_offset", find the 
    // concrete block whose DW_AT_specification or DW_AT_abstract_origin points
    // to "spec_block_die_offset"
    DWARFDebugInfo* info = DebugInfo();

    const DWARFDebugInfoEntry *die = info->GetDIEPtrWithCompileUnitHint(func_die_offset, result_die_cu_handle);
    if (die)
    {
        assert (*result_die_cu_handle);
        return FindBlockContainingSpecification (*result_die_cu_handle, die, spec_block_die_offset, result_die_cu_handle);
    }
    return NULL;
}


const DWARFDebugInfoEntry *
SymbolFileDWARF::FindBlockContainingSpecification(DWARFCompileUnit* dwarf_cu,
                                                  const DWARFDebugInfoEntry *die,
                                                  dw_offset_t spec_block_die_offset,
                                                  DWARFCompileUnit **result_die_cu_handle)
{
    if (die)
    {
        switch (die->Tag())
        {
        case DW_TAG_subprogram:
        case DW_TAG_inlined_subroutine:
        case DW_TAG_lexical_block:
            {
                if (die->GetAttributeValueAsReference (this, dwarf_cu, DW_AT_specification, DW_INVALID_OFFSET) == spec_block_die_offset)
                {
                    *result_die_cu_handle = dwarf_cu;
                    return die;
                }

                if (die->GetAttributeValueAsReference (this, dwarf_cu, DW_AT_abstract_origin, DW_INVALID_OFFSET) == spec_block_die_offset)
                {
                    *result_die_cu_handle = dwarf_cu;
                    return die;
                }
            }
            break;
        }

        // Give the concrete function die specified by "func_die_offset", find the 
        // concrete block whose DW_AT_specification or DW_AT_abstract_origin points
        // to "spec_block_die_offset"
        for (const DWARFDebugInfoEntry *child_die = die->GetFirstChild(); child_die != NULL; child_die = child_die->GetSibling())
        {
            const DWARFDebugInfoEntry *result_die = FindBlockContainingSpecification (dwarf_cu,
                                                                                      child_die,
                                                                                      spec_block_die_offset,
                                                                                      result_die_cu_handle);
            if (result_die)
                return result_die;
        }
    }
    
    *result_die_cu_handle = NULL;
    return NULL;
}

size_t
SymbolFileDWARF::ParseVariables
(
    const SymbolContext& sc,
    DWARFCompileUnit* dwarf_cu,
    const lldb::addr_t func_low_pc,
    const DWARFDebugInfoEntry *orig_die,
    bool parse_siblings,
    bool parse_children,
    VariableList* cc_variable_list
)
{
    if (orig_die == NULL)
        return 0;

    VariableListSP variable_list_sp;

    size_t vars_added = 0;
    const DWARFDebugInfoEntry *die = orig_die;
    while (die != NULL)
    {
        dw_tag_t tag = die->Tag();

        // Check to see if we have already parsed this variable or constant?
        if (m_die_to_variable_sp[die])
        {
            if (cc_variable_list)
                cc_variable_list->AddVariableIfUnique (m_die_to_variable_sp[die]);
        }
        else
        {
            // We haven't already parsed it, lets do that now.
            if ((tag == DW_TAG_variable) ||
                (tag == DW_TAG_constant) ||
                (tag == DW_TAG_formal_parameter && sc.function))
            {
                if (variable_list_sp.get() == NULL)
                {
                    const DWARFDebugInfoEntry *sc_parent_die = GetParentSymbolContextDIE(orig_die);
                    dw_tag_t parent_tag = sc_parent_die ? sc_parent_die->Tag() : 0;
                    switch (parent_tag)
                    {
                        case DW_TAG_compile_unit:
                            if (sc.comp_unit != NULL)
                            {
                                variable_list_sp = sc.comp_unit->GetVariableList(false);
                                if (variable_list_sp.get() == NULL)
                                {
                                    variable_list_sp.reset(new VariableList());
                                    sc.comp_unit->SetVariableList(variable_list_sp);
                                }
                            }
                            else
                            {
                                GetObjectFile()->GetModule()->ReportError ("parent 0x%8.8llx %s with no valid compile unit in symbol context for 0x%8.8llx %s.\n",
                                                                           MakeUserID(sc_parent_die->GetOffset()),
                                                                           DW_TAG_value_to_name (parent_tag),
                                                                           MakeUserID(orig_die->GetOffset()),
                                                                           DW_TAG_value_to_name (orig_die->Tag()));
                            }
                            break;
                            
                        case DW_TAG_subprogram:
                        case DW_TAG_inlined_subroutine:
                        case DW_TAG_lexical_block:
                            if (sc.function != NULL)
                            {
                                // Check to see if we already have parsed the variables for the given scope
                                
                                Block *block = sc.function->GetBlock(true).FindBlockByID(MakeUserID(sc_parent_die->GetOffset()));
                                if (block == NULL)
                                {
                                    // This must be a specification or abstract origin with 
                                    // a concrete block couterpart in the current function. We need
                                    // to find the concrete block so we can correctly add the 
                                    // variable to it
                                    DWARFCompileUnit *concrete_block_die_cu = dwarf_cu;
                                    const DWARFDebugInfoEntry *concrete_block_die = FindBlockContainingSpecification (sc.function->GetID(), 
                                                                                                                      sc_parent_die->GetOffset(), 
                                                                                                                      &concrete_block_die_cu);
                                    if (concrete_block_die)
                                        block = sc.function->GetBlock(true).FindBlockByID(MakeUserID(concrete_block_die->GetOffset()));
                                }
                                
                                if (block != NULL)
                                {
                                    const bool can_create = false;
                                    variable_list_sp = block->GetBlockVariableList (can_create);
                                    if (variable_list_sp.get() == NULL)
                                    {
                                        variable_list_sp.reset(new VariableList());
                                        block->SetVariableList(variable_list_sp);
                                    }
                                }
                            }
                            break;
                            
                        default:
                             GetObjectFile()->GetModule()->ReportError ("didn't find appropriate parent DIE for variable list for 0x%8.8llx %s.\n",
                                                                        MakeUserID(orig_die->GetOffset()),
                                                                        DW_TAG_value_to_name (orig_die->Tag()));
                            break;
                    }
                }
                
                if (variable_list_sp)
                {
                    VariableSP var_sp (ParseVariableDIE(sc, dwarf_cu, die, func_low_pc));
                    if (var_sp)
                    {
                        variable_list_sp->AddVariableIfUnique (var_sp);
                        if (cc_variable_list)
                            cc_variable_list->AddVariableIfUnique (var_sp);
                        ++vars_added;
                    }
                }
            }
        }

        bool skip_children = (sc.function == NULL && tag == DW_TAG_subprogram);

        if (!skip_children && parse_children && die->HasChildren())
        {
            vars_added += ParseVariables(sc, dwarf_cu, func_low_pc, die->GetFirstChild(), true, true, cc_variable_list);
        }

        if (parse_siblings)
            die = die->GetSibling();
        else
            die = NULL;
    }
    return vars_added;
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
SymbolFileDWARF::GetPluginName()
{
    return "SymbolFileDWARF";
}

const char *
SymbolFileDWARF::GetShortPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
SymbolFileDWARF::GetPluginVersion()
{
    return 1;
}

void
SymbolFileDWARF::CompleteTagDecl (void *baton, clang::TagDecl *decl)
{
    SymbolFileDWARF *symbol_file_dwarf = (SymbolFileDWARF *)baton;
    clang_type_t clang_type = symbol_file_dwarf->GetClangASTContext().GetTypeForDecl (decl);
    if (clang_type)
        symbol_file_dwarf->ResolveClangOpaqueTypeDefinition (clang_type);
}

void
SymbolFileDWARF::CompleteObjCInterfaceDecl (void *baton, clang::ObjCInterfaceDecl *decl)
{
    SymbolFileDWARF *symbol_file_dwarf = (SymbolFileDWARF *)baton;
    clang_type_t clang_type = symbol_file_dwarf->GetClangASTContext().GetTypeForDecl (decl);
    if (clang_type)
        symbol_file_dwarf->ResolveClangOpaqueTypeDefinition (clang_type);
}

void
SymbolFileDWARF::DumpIndexes ()
{
    StreamFile s(stdout, false);
    
    s.Printf ("DWARF index for (%s) '%s/%s':", 
              GetObjectFile()->GetModule()->GetArchitecture().GetArchitectureName(),
              GetObjectFile()->GetFileSpec().GetDirectory().AsCString(), 
              GetObjectFile()->GetFileSpec().GetFilename().AsCString());
    s.Printf("\nFunction basenames:\n");    m_function_basename_index.Dump (&s);
    s.Printf("\nFunction fullnames:\n");    m_function_fullname_index.Dump (&s);
    s.Printf("\nFunction methods:\n");      m_function_method_index.Dump (&s);
    s.Printf("\nFunction selectors:\n");    m_function_selector_index.Dump (&s);
    s.Printf("\nObjective C class selectors:\n");    m_objc_class_selectors_index.Dump (&s);
    s.Printf("\nGlobals and statics:\n");   m_global_index.Dump (&s); 
    s.Printf("\nTypes:\n");                 m_type_index.Dump (&s);
    s.Printf("\nNamepaces:\n");             m_namespace_index.Dump (&s);
}

void
SymbolFileDWARF::SearchDeclContext (const clang::DeclContext *decl_context, 
                                    const char *name, 
                                    llvm::SmallVectorImpl <clang::NamedDecl *> *results)
{    
    DeclContextToDIEMap::iterator iter = m_decl_ctx_to_die.find(decl_context);
    
    if (iter == m_decl_ctx_to_die.end())
        return;
    
    for (DIEPointerSet::iterator pos = iter->second.begin(), end = iter->second.end(); pos != end; ++pos)
    {
        const DWARFDebugInfoEntry *context_die = *pos;
    
        if (!results)
            return;
        
        DWARFDebugInfo* info = DebugInfo();
        
        DIEArray die_offsets;
        
        DWARFCompileUnit* dwarf_cu = NULL;
        const DWARFDebugInfoEntry* die = NULL;
        
        if (m_using_apple_tables)
        {
            if (m_apple_types_ap.get())
                m_apple_types_ap->FindByName (name, die_offsets);
        }
        else
        {
            if (!m_indexed)
                Index ();
            
            m_type_index.Find (ConstString(name), die_offsets);
        }
        
        const size_t num_matches = die_offsets.size();
        
        if (num_matches)
        {
            for (size_t i = 0; i < num_matches; ++i)
            {
                const dw_offset_t die_offset = die_offsets[i];
                die = info->GetDIEPtrWithCompileUnitHint (die_offset, &dwarf_cu);

                if (die->GetParent() != context_die)
                    continue;
                
                Type *matching_type = ResolveType (dwarf_cu, die);
                
                lldb::clang_type_t type = matching_type->GetClangForwardType();
                clang::QualType qual_type = clang::QualType::getFromOpaquePtr(type);
                
                if (const clang::TagType *tag_type = llvm::dyn_cast<clang::TagType>(qual_type.getTypePtr()))
                {
                    clang::TagDecl *tag_decl = tag_type->getDecl();
                    results->push_back(tag_decl);
                }
                else if (const clang::TypedefType *typedef_type = llvm::dyn_cast<clang::TypedefType>(qual_type.getTypePtr()))
                {
                    clang::TypedefNameDecl *typedef_decl = typedef_type->getDecl();
                    results->push_back(typedef_decl); 
                }
            }
        }
    }
}

void
SymbolFileDWARF::FindExternalVisibleDeclsByName (void *baton,
                                                 const clang::DeclContext *decl_context,
                                                 clang::DeclarationName decl_name,
                                                 llvm::SmallVectorImpl <clang::NamedDecl *> *results)
{
    
    switch (decl_context->getDeclKind())
    {
    case clang::Decl::Namespace:
    case clang::Decl::TranslationUnit:
        {
            SymbolFileDWARF *symbol_file_dwarf = (SymbolFileDWARF *)baton;
            symbol_file_dwarf->SearchDeclContext (decl_context, decl_name.getAsString().c_str(), results);
        }
        break;
    default:
        break;
    }
}

bool 
SymbolFileDWARF::LayoutRecordType (void *baton, 
                                   const clang::RecordDecl *record_decl,
                                   uint64_t &size, 
                                   uint64_t &alignment,
                                   llvm::DenseMap <const clang::FieldDecl *, uint64_t> &field_offsets,
                                   llvm::DenseMap <const clang::CXXRecordDecl *, clang::CharUnits> &base_offsets,
                                   llvm::DenseMap <const clang::CXXRecordDecl *, clang::CharUnits> &vbase_offsets)
{
    SymbolFileDWARF *symbol_file_dwarf = (SymbolFileDWARF *)baton;
    return symbol_file_dwarf->LayoutRecordType (record_decl, size, alignment, field_offsets, base_offsets, vbase_offsets);
}


bool 
SymbolFileDWARF::LayoutRecordType (const clang::RecordDecl *record_decl,
                                   uint64_t &bit_size, 
                                   uint64_t &alignment,
                                   llvm::DenseMap <const clang::FieldDecl *, uint64_t> &field_offsets,
                                   llvm::DenseMap <const clang::CXXRecordDecl *, clang::CharUnits> &base_offsets,
                                   llvm::DenseMap <const clang::CXXRecordDecl *, clang::CharUnits> &vbase_offsets)
{
    LogSP log (LogChannelDWARF::GetLogIfAll(DWARF_LOG_DEBUG_INFO));
    RecordDeclToLayoutMap::iterator pos = m_record_decl_to_layout_map.find (record_decl);
    bool success = false;
    base_offsets.clear();
    vbase_offsets.clear();
    if (pos != m_record_decl_to_layout_map.end())
    {
        bit_size = pos->second.bit_size;
        alignment = pos->second.alignment;
        field_offsets.swap(pos->second.field_offsets);
        base_offsets.swap (pos->second.base_offsets);
        vbase_offsets.swap (pos->second.vbase_offsets);
        m_record_decl_to_layout_map.erase(pos);
        success = true;
    }
    else
    {
        bit_size = 0;
        alignment = 0;
        field_offsets.clear();
    }
    
    if (log)
        GetObjectFile()->GetModule()->LogMessage (log.get(), 
                                                  "SymbolFileDWARF::LayoutRecordType (record_decl = %p, bit_size = %llu, alignment = %llu, field_offsets[%u],base_offsets[%u], vbase_offsets[%u]) success = %i",
                                                  record_decl,
                                                  bit_size,
                                                  alignment,
                                                  (uint32_t)field_offsets.size(),
                                                  (uint32_t)base_offsets.size(),
                                                  (uint32_t)vbase_offsets.size(),
                                                  success);
    return success;
}


SymbolFileDWARFDebugMap *
SymbolFileDWARF::GetDebugMapSymfile ()
{
    if (m_debug_map_symfile == NULL && !m_debug_map_module_wp.expired())
    {
        lldb::ModuleSP module_sp (m_debug_map_module_wp.lock());
        if (module_sp)
        {
            SymbolVendor *sym_vendor = module_sp->GetSymbolVendor();
            if (sym_vendor)
                m_debug_map_symfile = (SymbolFileDWARFDebugMap *)sym_vendor->GetSymbolFile();
        }
    }
    return m_debug_map_symfile;
}


