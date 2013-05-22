//===-- SymbolFileDWARFDebugMap.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SymbolFileDWARFDebugMap.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/Section.h"

//#define DEBUG_OSO_DMAP // DO NOT CHECKIN WITH THIS NOT COMMENTED OUT
#if defined(DEBUG_OSO_DMAP)
#include "lldb/Core/StreamFile.h"
#endif
#include "lldb/Core/Timer.h"

#include "lldb/Symbol/ClangExternalASTSourceCallbacks.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/LineTable.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Symbol/VariableList.h"

#include "LogChannelDWARF.h"
#include "SymbolFileDWARF.h"

using namespace lldb;
using namespace lldb_private;

// Subclass lldb_private::Module so we can intercept the "Module::GetObjectFile()" 
// (so we can fixup the object file sections) and also for "Module::GetSymbolVendor()"
// (so we can fixup the symbol file id.




const SymbolFileDWARFDebugMap::FileRangeMap &
SymbolFileDWARFDebugMap::CompileUnitInfo::GetFileRangeMap(SymbolFileDWARFDebugMap *exe_symfile)
{
    if (file_range_map_valid)
        return file_range_map;

    file_range_map_valid = true;

    Module *oso_module = exe_symfile->GetModuleByCompUnitInfo (this);
    if (!oso_module)
        return file_range_map;
    
    ObjectFile *oso_objfile = oso_module->GetObjectFile();
    if (!oso_objfile)
        return file_range_map;
    
    Log *log (LogChannelDWARF::GetLogIfAll(DWARF_LOG_DEBUG_MAP));
    if (log)
    {
        ConstString object_name (oso_module->GetObjectName());
        log->Printf("%p: SymbolFileDWARFDebugMap::CompileUnitInfo::GetFileRangeMap ('%s')",
                    this,
                    oso_module->GetSpecificationDescription().c_str());
    }
    

    std::vector<SymbolFileDWARFDebugMap::CompileUnitInfo *> cu_infos;
    if (exe_symfile->GetCompUnitInfosForModule(oso_module, cu_infos))
    {
        for (auto comp_unit_info : cu_infos)
        {
            Symtab *exe_symtab = exe_symfile->GetObjectFile()->GetSymtab();
            ModuleSP oso_module_sp (oso_objfile->GetModule());
            Symtab *oso_symtab = oso_objfile->GetSymtab();
            
            ///const uint32_t fun_resolve_flags = SymbolContext::Module | eSymbolContextCompUnit | eSymbolContextFunction;
            //SectionList *oso_sections = oso_objfile->Sections();
            // Now we need to make sections that map from zero based object
            // file addresses to where things eneded up in the main executable.
            
            assert (comp_unit_info->first_symbol_index != UINT32_MAX);
            // End index is one past the last valid symbol index
            const uint32_t oso_end_idx = comp_unit_info->last_symbol_index + 1;
            for (uint32_t idx = comp_unit_info->first_symbol_index + 2; // Skip the N_SO and N_OSO
                 idx < oso_end_idx;
                 ++idx)
            {
                Symbol *exe_symbol = exe_symtab->SymbolAtIndex(idx);
                if (exe_symbol)
                {
                    if (exe_symbol->IsDebug() == false)
                        continue;
                    
                    switch (exe_symbol->GetType())
                    {
                    default:
                        break;
                        
                    case eSymbolTypeCode:
                        {
                            // For each N_FUN, or function that we run into in the debug map
                            // we make a new section that we add to the sections found in the
                            // .o file. This new section has the file address set to what the
                            // addresses are in the .o file, and the load address is adjusted
                            // to match where it ended up in the final executable! We do this
                            // before we parse any dwarf info so that when it goes get parsed
                            // all section/offset addresses that get registered will resolve
                            // correctly to the new addresses in the main executable.
                            
                            // First we find the original symbol in the .o file's symbol table
                            Symbol *oso_fun_symbol = oso_symtab->FindFirstSymbolWithNameAndType (exe_symbol->GetMangled().GetName(Mangled::ePreferMangled),
                                                                                                 eSymbolTypeCode,
                                                                                                 Symtab::eDebugNo,
                                                                                                 Symtab::eVisibilityAny);
                            if (oso_fun_symbol)
                            {
                                // Add the inverse OSO file address to debug map entry mapping
                                exe_symfile->AddOSOFileRange (this,
                                                              exe_symbol->GetAddress().GetFileAddress(),
                                                              oso_fun_symbol->GetAddress().GetFileAddress(),
                                                              std::min<addr_t>(exe_symbol->GetByteSize(), oso_fun_symbol->GetByteSize()));
                                
                            }
                        }
                        break;
                        
                    case eSymbolTypeData:
                        {
                            // For each N_GSYM we remap the address for the global by making
                            // a new section that we add to the sections found in the .o file.
                            // This new section has the file address set to what the
                            // addresses are in the .o file, and the load address is adjusted
                            // to match where it ended up in the final executable! We do this
                            // before we parse any dwarf info so that when it goes get parsed
                            // all section/offset addresses that get registered will resolve
                            // correctly to the new addresses in the main executable. We
                            // initially set the section size to be 1 byte, but will need to
                            // fix up these addresses further after all globals have been
                            // parsed to span the gaps, or we can find the global variable
                            // sizes from the DWARF info as we are parsing.
                            
                            // Next we find the non-stab entry that corresponds to the N_GSYM in the .o file
                            Symbol *oso_gsym_symbol = oso_symtab->FindFirstSymbolWithNameAndType (exe_symbol->GetMangled().GetName(Mangled::ePreferMangled),
                                                                                                  eSymbolTypeData,
                                                                                                  Symtab::eDebugNo,
                                                                                                  Symtab::eVisibilityAny);
                            
                            if (exe_symbol && oso_gsym_symbol &&
                                exe_symbol->ValueIsAddress() &&
                                oso_gsym_symbol->ValueIsAddress())
                            {
                                // Add the inverse OSO file address to debug map entry mapping
                                exe_symfile->AddOSOFileRange (this,
                                                              exe_symbol->GetAddress().GetFileAddress(),
                                                              oso_gsym_symbol->GetAddress().GetFileAddress(),
                                                              std::min<addr_t>(exe_symbol->GetByteSize(), oso_gsym_symbol->GetByteSize()));
                            }
                        }
                        break;
                    }
                }
            }
            
            exe_symfile->FinalizeOSOFileRanges (this);
            // We don't need the symbols anymore for the .o files
            oso_objfile->ClearSymtab();
        }
    }
    return file_range_map;
}


class DebugMapModule : public Module
{
public:
    DebugMapModule (const ModuleSP &exe_module_sp,
                    uint32_t cu_idx,
                    const FileSpec& file_spec,
                    const ArchSpec& arch,
                    const ConstString *object_name,
                    off_t object_offset,
                    const TimeValue *object_mod_time_ptr) :
        Module (file_spec, arch, object_name, object_offset, object_mod_time_ptr),
        m_exe_module_wp (exe_module_sp),
        m_cu_idx (cu_idx)
    {
    }

    virtual
    ~DebugMapModule ()
    {
    }

    
    virtual SymbolVendor*
    GetSymbolVendor(bool can_create = true, lldb_private::Stream *feedback_strm = NULL)
    {
        // Scope for locker
        if (m_symfile_ap.get() || can_create == false)
            return m_symfile_ap.get();

        ModuleSP exe_module_sp (m_exe_module_wp.lock());
        if (exe_module_sp)
        {
            // Now get the object file outside of a locking scope
            ObjectFile *oso_objfile = GetObjectFile ();
            if (oso_objfile)
            {
                Mutex::Locker locker (m_mutex);
                SymbolVendor* symbol_vendor = Module::GetSymbolVendor(can_create, feedback_strm);
                if (symbol_vendor)
                {
                    // Set a a pointer to this class to set our OSO DWARF file know
                    // that the DWARF is being used along with a debug map and that
                    // it will have the remapped sections that we do below.
                    SymbolFileDWARF *oso_symfile = SymbolFileDWARFDebugMap::GetSymbolFileAsSymbolFileDWARF(symbol_vendor->GetSymbolFile());
                    
                    if (!oso_symfile)
                        return NULL;

                    ObjectFile *exe_objfile = exe_module_sp->GetObjectFile();
                    SymbolVendor *exe_sym_vendor = exe_module_sp->GetSymbolVendor();
                    
                    if (exe_objfile && exe_sym_vendor)
                    {
                        if (oso_symfile->GetNumCompileUnits() == 1)
                        {
                            oso_symfile->SetDebugMapModule(exe_module_sp);
                            // Set the ID of the symbol file DWARF to the index of the OSO
                            // shifted left by 32 bits to provide a unique prefix for any
                            // UserID's that get created in the symbol file.
                            oso_symfile->SetID (((uint64_t)m_cu_idx + 1ull) << 32ull);
                        }
                        else
                        {
                            oso_symfile->SetID (UINT64_MAX);
                        }
                    }
                    return symbol_vendor;
                }
            }
        }
        return NULL;
    }

protected:
    ModuleWP m_exe_module_wp;
    const uint32_t m_cu_idx;
};

void
SymbolFileDWARFDebugMap::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   GetPluginDescriptionStatic(),
                                   CreateInstance);
}

void
SymbolFileDWARFDebugMap::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}


lldb_private::ConstString
SymbolFileDWARFDebugMap::GetPluginNameStatic()
{
    static ConstString g_name("dwarf-debugmap");
    return g_name;
}

const char *
SymbolFileDWARFDebugMap::GetPluginDescriptionStatic()
{
    return "DWARF and DWARF3 debug symbol file reader (debug map).";
}

SymbolFile*
SymbolFileDWARFDebugMap::CreateInstance (ObjectFile* obj_file)
{
    return new SymbolFileDWARFDebugMap (obj_file);
}


SymbolFileDWARFDebugMap::SymbolFileDWARFDebugMap (ObjectFile* ofile) :
    SymbolFile(ofile),
    m_flags(),
    m_compile_unit_infos(),
    m_func_indexes(),
    m_glob_indexes(),
    m_supports_DW_AT_APPLE_objc_complete_type (eLazyBoolCalculate)
{
}


SymbolFileDWARFDebugMap::~SymbolFileDWARFDebugMap()
{
}

void
SymbolFileDWARFDebugMap::InitializeObject()
{
    // Install our external AST source callbacks so we can complete Clang types.
    llvm::OwningPtr<clang::ExternalASTSource> ast_source_ap (
        new ClangExternalASTSourceCallbacks (SymbolFileDWARFDebugMap::CompleteTagDecl,
                                             SymbolFileDWARFDebugMap::CompleteObjCInterfaceDecl,
                                             NULL,
                                             SymbolFileDWARFDebugMap::LayoutRecordType,
                                             this));

    GetClangASTContext().SetExternalSource (ast_source_ap);
}

void
SymbolFileDWARFDebugMap::InitOSO()
{
    if (m_flags.test(kHaveInitializedOSOs))
        return;
    
    m_flags.set(kHaveInitializedOSOs);
    // In order to get the abilities of this plug-in, we look at the list of
    // N_OSO entries (object files) from the symbol table and make sure that
    // these files exist and also contain valid DWARF. If we get any of that
    // then we return the abilities of the first N_OSO's DWARF.

    Symtab* symtab = m_obj_file->GetSymtab();
    if (symtab)
    {
        Log *log (LogChannelDWARF::GetLogIfAll(DWARF_LOG_DEBUG_MAP));

        std::vector<uint32_t> oso_indexes;
        // When a mach-o symbol is encoded, the n_type field is encoded in bits
        // 23:16, and the n_desc field is encoded in bits 15:0.
        // 
        // To find all N_OSO entries that are part of the DWARF + debug map
        // we find only object file symbols with the flags value as follows:
        // bits 23:16 == 0x66 (N_OSO)
        // bits 15: 0 == 0x0001 (specifies this is a debug map object file)
        const uint32_t k_oso_symbol_flags_value = 0x660001u;

        const uint32_t oso_index_count = symtab->AppendSymbolIndexesWithTypeAndFlagsValue(eSymbolTypeObjectFile, k_oso_symbol_flags_value, oso_indexes);

        if (oso_index_count > 0)
        {
            symtab->AppendSymbolIndexesWithType (eSymbolTypeCode, Symtab::eDebugYes, Symtab::eVisibilityAny, m_func_indexes);
            symtab->AppendSymbolIndexesWithType (eSymbolTypeData, Symtab::eDebugYes, Symtab::eVisibilityAny, m_glob_indexes);

            symtab->SortSymbolIndexesByValue(m_func_indexes, true);
            symtab->SortSymbolIndexesByValue(m_glob_indexes, true);

            for (uint32_t sym_idx : m_func_indexes)
            {
                const Symbol *symbol = symtab->SymbolAtIndex(sym_idx);
                lldb::addr_t file_addr = symbol->GetAddress().GetFileAddress();
                lldb::addr_t byte_size = symbol->GetByteSize();
                DebugMap::Entry debug_map_entry(file_addr, byte_size, OSOEntry(sym_idx, LLDB_INVALID_ADDRESS));
                m_debug_map.Append(debug_map_entry);
            }
            for (uint32_t sym_idx : m_glob_indexes)
            {
                const Symbol *symbol = symtab->SymbolAtIndex(sym_idx);
                lldb::addr_t file_addr = symbol->GetAddress().GetFileAddress();
                lldb::addr_t byte_size = symbol->GetByteSize();
                DebugMap::Entry debug_map_entry(file_addr, byte_size, OSOEntry(sym_idx, LLDB_INVALID_ADDRESS));
                m_debug_map.Append(debug_map_entry);
            }
            m_debug_map.Sort();

            m_compile_unit_infos.resize(oso_index_count);

            for (uint32_t i=0; i<oso_index_count; ++i)
            {
                const uint32_t so_idx = oso_indexes[i] - 1;
                const uint32_t oso_idx = oso_indexes[i];
                const Symbol *so_symbol = symtab->SymbolAtIndex(so_idx);
                const Symbol *oso_symbol = symtab->SymbolAtIndex(oso_idx);
                if (so_symbol &&
                    oso_symbol &&
                    so_symbol->GetType() == eSymbolTypeSourceFile &&
                    oso_symbol->GetType() == eSymbolTypeObjectFile)
                {
                    m_compile_unit_infos[i].so_file.SetFile(so_symbol->GetName().AsCString(), false);
                    m_compile_unit_infos[i].oso_path = oso_symbol->GetName();
                    TimeValue oso_mod_time;
                    oso_mod_time.OffsetWithSeconds(oso_symbol->GetAddress().GetOffset());
                    m_compile_unit_infos[i].oso_mod_time = oso_mod_time;
                    uint32_t sibling_idx = so_symbol->GetSiblingIndex();
                    // The sibling index can't be less that or equal to the current index "i"
                    if (sibling_idx <= i)
                    {
                        m_obj_file->GetModule()->ReportError ("N_SO in symbol with UID %u has invalid sibling in debug map, please file a bug and attach the binary listed in this error", so_symbol->GetID());
                    }
                    else
                    {
                        const Symbol* last_symbol = symtab->SymbolAtIndex (sibling_idx - 1);
                        m_compile_unit_infos[i].first_symbol_index = so_idx;
                        m_compile_unit_infos[i].last_symbol_index = sibling_idx - 1;
                        m_compile_unit_infos[i].first_symbol_id = so_symbol->GetID();
                        m_compile_unit_infos[i].last_symbol_id = last_symbol->GetID();
                        
                        if (log)
                            log->Printf("Initialized OSO 0x%8.8x: file=%s", i, oso_symbol->GetName().GetCString());
                    }
                }
                else
                {
                    if (oso_symbol == NULL)
                        m_obj_file->GetModule()->ReportError ("N_OSO symbol[%u] can't be found, please file a bug and attach the binary listed in this error", oso_idx);
                    else if (so_symbol == NULL)
                        m_obj_file->GetModule()->ReportError ("N_SO not found for N_OSO symbol[%u], please file a bug and attach the binary listed in this error", oso_idx);
                    else if (so_symbol->GetType() != eSymbolTypeSourceFile)
                        m_obj_file->GetModule()->ReportError ("N_SO has incorrect symbol type (%u) for N_OSO symbol[%u], please file a bug and attach the binary listed in this error", so_symbol->GetType(), oso_idx);
                    else if (oso_symbol->GetType() != eSymbolTypeSourceFile)
                        m_obj_file->GetModule()->ReportError ("N_OSO has incorrect symbol type (%u) for N_OSO symbol[%u], please file a bug and attach the binary listed in this error", oso_symbol->GetType(), oso_idx);
                }
            }
        }
    }
}

Module *
SymbolFileDWARFDebugMap::GetModuleByOSOIndex (uint32_t oso_idx)
{
    const uint32_t cu_count = GetNumCompileUnits();
    if (oso_idx < cu_count)
        return GetModuleByCompUnitInfo (&m_compile_unit_infos[oso_idx]);
    return NULL;
}

Module *
SymbolFileDWARFDebugMap::GetModuleByCompUnitInfo (CompileUnitInfo *comp_unit_info)
{
    if (!comp_unit_info->oso_sp)
    {
        auto pos = m_oso_map.find (comp_unit_info->oso_path);
        if (pos != m_oso_map.end())
        {
            comp_unit_info->oso_sp = pos->second;
        }
        else
        {
            ObjectFile *obj_file = GetObjectFile();
            comp_unit_info->oso_sp.reset (new OSOInfo());
            m_oso_map[comp_unit_info->oso_path] = comp_unit_info->oso_sp;
            const char *oso_path = comp_unit_info->oso_path.GetCString();
            FileSpec oso_file (oso_path, false);
            ConstString oso_object;
            if (oso_file.Exists())
            {
                TimeValue oso_mod_time (oso_file.GetModificationTime());
                if (oso_mod_time != comp_unit_info->oso_mod_time)
                {
                    obj_file->GetModule()->ReportError ("debug map object file '%s' has changed (actual time is 0x%" PRIx64 ", debug map time is 0x%" PRIx64 ") since this executable was linked, file will be ignored",
                                                        oso_file.GetPath().c_str(),
                                                        oso_mod_time.GetAsSecondsSinceJan1_1970(),
                                                        comp_unit_info->oso_mod_time.GetAsSecondsSinceJan1_1970());
                    return NULL;
                }

            }
            else
            {
                const bool must_exist = true;

                if (!ObjectFile::SplitArchivePathWithObject (oso_path,
                                                             oso_file,
                                                             oso_object,
                                                             must_exist))
                {
                    return NULL;
                }
            }
            // Always create a new module for .o files. Why? Because we
            // use the debug map, to add new sections to each .o file and
            // even though a .o file might not have changed, the sections
            // that get added to the .o file can change.
            comp_unit_info->oso_sp->module_sp.reset (new DebugMapModule (obj_file->GetModule(),
                                                                         GetCompUnitInfoIndex(comp_unit_info),
                                                                         oso_file,
                                                                         m_obj_file->GetModule()->GetArchitecture(),
                                                                         oso_object ? &oso_object : NULL,
                                                                         0,
                                                                         oso_object ? &comp_unit_info->oso_mod_time : NULL));
        }
    }
    if (comp_unit_info->oso_sp)
        return comp_unit_info->oso_sp->module_sp.get();
    return NULL;
}


bool
SymbolFileDWARFDebugMap::GetFileSpecForSO (uint32_t oso_idx, FileSpec &file_spec)
{
    if (oso_idx < m_compile_unit_infos.size())
    {
        if (m_compile_unit_infos[oso_idx].so_file)
        {
            file_spec = m_compile_unit_infos[oso_idx].so_file;
            return true;
        }
    }
    return false;
}



ObjectFile *
SymbolFileDWARFDebugMap::GetObjectFileByOSOIndex (uint32_t oso_idx)
{
    Module *oso_module = GetModuleByOSOIndex (oso_idx);
    if (oso_module)
        return oso_module->GetObjectFile();
    return NULL;
}

SymbolFileDWARF *
SymbolFileDWARFDebugMap::GetSymbolFile (const SymbolContext& sc)
{
    CompileUnitInfo *comp_unit_info = GetCompUnitInfo (sc);
    if (comp_unit_info)
        return GetSymbolFileByCompUnitInfo (comp_unit_info);
    return NULL;
}

ObjectFile *
SymbolFileDWARFDebugMap::GetObjectFileByCompUnitInfo (CompileUnitInfo *comp_unit_info)
{
    Module *oso_module = GetModuleByCompUnitInfo (comp_unit_info);
    if (oso_module)
        return oso_module->GetObjectFile();
    return NULL;
}


uint32_t
SymbolFileDWARFDebugMap::GetCompUnitInfoIndex (const CompileUnitInfo *comp_unit_info)
{
    if (!m_compile_unit_infos.empty())
    {
        const CompileUnitInfo *first_comp_unit_info = &m_compile_unit_infos.front();
        const CompileUnitInfo *last_comp_unit_info = &m_compile_unit_infos.back();
        if (first_comp_unit_info <= comp_unit_info && comp_unit_info <= last_comp_unit_info)
            return comp_unit_info - first_comp_unit_info;
    }
    return UINT32_MAX;
}

SymbolFileDWARF *
SymbolFileDWARFDebugMap::GetSymbolFileByOSOIndex (uint32_t oso_idx)
{
    if (oso_idx < m_compile_unit_infos.size())
        return GetSymbolFileByCompUnitInfo (&m_compile_unit_infos[oso_idx]);
    return NULL;
}

SymbolFileDWARF *
SymbolFileDWARFDebugMap::GetSymbolFileAsSymbolFileDWARF (SymbolFile *sym_file)
{
    if (sym_file && sym_file->GetPluginName() == SymbolFileDWARF::GetPluginNameStatic())
        return (SymbolFileDWARF *)sym_file;
    return NULL;
}

SymbolFileDWARF *
SymbolFileDWARFDebugMap::GetSymbolFileByCompUnitInfo (CompileUnitInfo *comp_unit_info)
{
    Module *oso_module = GetModuleByCompUnitInfo (comp_unit_info);
    if (oso_module)
    {
        SymbolVendor *sym_vendor = oso_module->GetSymbolVendor();
        if (sym_vendor)
            return GetSymbolFileAsSymbolFileDWARF (sym_vendor->GetSymbolFile());
    }
    return NULL;
}

uint32_t
SymbolFileDWARFDebugMap::CalculateAbilities ()
{
    // In order to get the abilities of this plug-in, we look at the list of
    // N_OSO entries (object files) from the symbol table and make sure that
    // these files exist and also contain valid DWARF. If we get any of that
    // then we return the abilities of the first N_OSO's DWARF.

    const uint32_t oso_index_count = GetNumCompileUnits();
    if (oso_index_count > 0)
    {
        const uint32_t dwarf_abilities = SymbolFile::CompileUnits |
                                         SymbolFile::Functions |
                                         SymbolFile::Blocks |
                                         SymbolFile::GlobalVariables |
                                         SymbolFile::LocalVariables |
                                         SymbolFile::VariableTypes |
                                         SymbolFile::LineTables;

        InitOSO();
        if (!m_compile_unit_infos.empty())
            return dwarf_abilities;
//        for (uint32_t oso_idx=0; oso_idx<oso_index_count; ++oso_idx)
//        {
//            SymbolFileDWARF *oso_dwarf = GetSymbolFileByOSOIndex (oso_idx);
//            if (oso_dwarf)
//            {
//                uint32_t oso_abilities = oso_dwarf->GetAbilities();
//                if ((oso_abilities & dwarf_abilities) == dwarf_abilities)
//                    return oso_abilities;
//            }
//        }
    }
    return 0;
}

uint32_t
SymbolFileDWARFDebugMap::GetNumCompileUnits()
{
    InitOSO ();
    return m_compile_unit_infos.size();
}


CompUnitSP
SymbolFileDWARFDebugMap::ParseCompileUnitAtIndex(uint32_t cu_idx)
{
    CompUnitSP comp_unit_sp;
    const uint32_t cu_count = GetNumCompileUnits();

    if (cu_idx < cu_count)
    {
        Module *oso_module = GetModuleByCompUnitInfo (&m_compile_unit_infos[cu_idx]);
        if (oso_module)
        {
            FileSpec so_file_spec;
            if (GetFileSpecForSO (cu_idx, so_file_spec))
            {
                // User zero as the ID to match the compile unit at offset
                // zero in each .o file since each .o file can only have
                // one compile unit for now.
                lldb::user_id_t cu_id = 0;
                m_compile_unit_infos[cu_idx].compile_unit_sp.reset(new CompileUnit (m_obj_file->GetModule(),
                                                                                    NULL,
                                                                                    so_file_spec,
                                                                                    cu_id,
                                                                                    eLanguageTypeUnknown));
            
                if (m_compile_unit_infos[cu_idx].compile_unit_sp)
                {
                    // Let our symbol vendor know about this compile unit
                    m_obj_file->GetModule()->GetSymbolVendor()->SetCompileUnitAtIndex (cu_idx, m_compile_unit_infos[cu_idx].compile_unit_sp);
                }
            }
        }
        comp_unit_sp = m_compile_unit_infos[cu_idx].compile_unit_sp;
    }

    return comp_unit_sp;
}

SymbolFileDWARFDebugMap::CompileUnitInfo *
SymbolFileDWARFDebugMap::GetCompUnitInfo (const SymbolContext& sc)
{
    const uint32_t cu_count = GetNumCompileUnits();
    for (uint32_t i=0; i<cu_count; ++i)
    {
        if (sc.comp_unit == m_compile_unit_infos[i].compile_unit_sp.get())
            return &m_compile_unit_infos[i];
    }
    return NULL;
}


size_t
SymbolFileDWARFDebugMap::GetCompUnitInfosForModule (const lldb_private::Module *module, std::vector<CompileUnitInfo *>& cu_infos)
{
    const uint32_t cu_count = GetNumCompileUnits();
    for (uint32_t i=0; i<cu_count; ++i)
    {
        if (module == GetModuleByCompUnitInfo (&m_compile_unit_infos[i]))
            cu_infos.push_back (&m_compile_unit_infos[i]);
    }
    return cu_infos.size();
}

lldb::LanguageType
SymbolFileDWARFDebugMap::ParseCompileUnitLanguage (const SymbolContext& sc)
{
    SymbolFileDWARF *oso_dwarf = GetSymbolFile (sc);
    if (oso_dwarf)
        return oso_dwarf->ParseCompileUnitLanguage (sc);
    return eLanguageTypeUnknown;
}

size_t
SymbolFileDWARFDebugMap::ParseCompileUnitFunctions (const SymbolContext& sc)
{
    SymbolFileDWARF *oso_dwarf = GetSymbolFile (sc);
    if (oso_dwarf)
        return oso_dwarf->ParseCompileUnitFunctions (sc);
    return 0;
}

bool
SymbolFileDWARFDebugMap::ParseCompileUnitLineTable (const SymbolContext& sc)
{
    SymbolFileDWARF *oso_dwarf = GetSymbolFile (sc);
    if (oso_dwarf)
        return oso_dwarf->ParseCompileUnitLineTable (sc);
    return false;
}

bool
SymbolFileDWARFDebugMap::ParseCompileUnitSupportFiles (const SymbolContext& sc, FileSpecList &support_files)
{
    SymbolFileDWARF *oso_dwarf = GetSymbolFile (sc);
    if (oso_dwarf)
        return oso_dwarf->ParseCompileUnitSupportFiles (sc, support_files);
    return false;
}


size_t
SymbolFileDWARFDebugMap::ParseFunctionBlocks (const SymbolContext& sc)
{
    SymbolFileDWARF *oso_dwarf = GetSymbolFile (sc);
    if (oso_dwarf)
        return oso_dwarf->ParseFunctionBlocks (sc);
    return 0;
}


size_t
SymbolFileDWARFDebugMap::ParseTypes (const SymbolContext& sc)
{
    SymbolFileDWARF *oso_dwarf = GetSymbolFile (sc);
    if (oso_dwarf)
        return oso_dwarf->ParseTypes (sc);
    return 0;
}


size_t
SymbolFileDWARFDebugMap::ParseVariablesForContext (const SymbolContext& sc)
{
    SymbolFileDWARF *oso_dwarf = GetSymbolFile (sc);
    if (oso_dwarf)
        return oso_dwarf->ParseVariablesForContext (sc);
    return 0;
}



Type*
SymbolFileDWARFDebugMap::ResolveTypeUID(lldb::user_id_t type_uid)
{
    const uint64_t oso_idx = GetOSOIndexFromUserID (type_uid);
    SymbolFileDWARF *oso_dwarf = GetSymbolFileByOSOIndex (oso_idx);
    if (oso_dwarf)
        return oso_dwarf->ResolveTypeUID (type_uid);
    return NULL;
}

lldb::clang_type_t
SymbolFileDWARFDebugMap::ResolveClangOpaqueTypeDefinition (lldb::clang_type_t clang_type)
{
    // We have a struct/union/class/enum that needs to be fully resolved.
    return NULL;
}

uint32_t
SymbolFileDWARFDebugMap::ResolveSymbolContext (const Address& exe_so_addr, uint32_t resolve_scope, SymbolContext& sc)
{
    uint32_t resolved_flags = 0;
    Symtab* symtab = m_obj_file->GetSymtab();
    if (symtab)
    {
        const addr_t exe_file_addr = exe_so_addr.GetFileAddress();

        const DebugMap::Entry *debug_map_entry = m_debug_map.FindEntryThatContains (exe_file_addr);
        if (debug_map_entry)
        {

            sc.symbol = symtab->SymbolAtIndex(debug_map_entry->data.GetExeSymbolIndex());

            if (sc.symbol != NULL)
            {
                resolved_flags |= eSymbolContextSymbol;

                uint32_t oso_idx = 0;
                CompileUnitInfo* comp_unit_info = GetCompileUnitInfoForSymbolWithID (sc.symbol->GetID(), &oso_idx);
                if (comp_unit_info)
                {
                    comp_unit_info->GetFileRangeMap(this);
                    Module *oso_module = GetModuleByCompUnitInfo (comp_unit_info);
                    if (oso_module)
                    {
                        lldb::addr_t oso_file_addr = exe_file_addr - debug_map_entry->GetRangeBase() + debug_map_entry->data.GetOSOFileAddress();
                        Address oso_so_addr;
                        if (oso_module->ResolveFileAddress(oso_file_addr, oso_so_addr))
                        {
                            resolved_flags |= oso_module->GetSymbolVendor()->ResolveSymbolContext (oso_so_addr, resolve_scope, sc);
                        }
                    }
                }
            }
        }
    }
    return resolved_flags;
}


uint32_t
SymbolFileDWARFDebugMap::ResolveSymbolContext (const FileSpec& file_spec, uint32_t line, bool check_inlines, uint32_t resolve_scope, SymbolContextList& sc_list)
{
    const uint32_t initial = sc_list.GetSize();
    const uint32_t cu_count = GetNumCompileUnits();

    for (uint32_t i=0; i<cu_count; ++i)
    {
        // If we are checking for inlines, then we need to look through all
        // compile units no matter if "file_spec" matches.
        bool resolve = check_inlines;
        
        if (!resolve)
        {
            FileSpec so_file_spec;
            if (GetFileSpecForSO (i, so_file_spec))
            {
                // Match the full path if the incoming file_spec has a directory (not just a basename)
                const bool full_match = file_spec.GetDirectory();
                resolve = FileSpec::Equal (file_spec, so_file_spec, full_match);
            }
        }
        if (resolve)
        {
            SymbolFileDWARF *oso_dwarf = GetSymbolFileByOSOIndex (i);
            if (oso_dwarf)
                oso_dwarf->ResolveSymbolContext(file_spec, line, check_inlines, resolve_scope, sc_list);
        }
    }
    return sc_list.GetSize() - initial;
}

uint32_t
SymbolFileDWARFDebugMap::PrivateFindGlobalVariables
(
    const ConstString &name,
    const ClangNamespaceDecl *namespace_decl,
    const std::vector<uint32_t> &indexes,   // Indexes into the symbol table that match "name"
    uint32_t max_matches,
    VariableList& variables
)
{
    const uint32_t original_size = variables.GetSize();
    const size_t match_count = indexes.size();
    for (size_t i=0; i<match_count; ++i)
    {
        uint32_t oso_idx;
        CompileUnitInfo* comp_unit_info = GetCompileUnitInfoForSymbolWithIndex (indexes[i], &oso_idx);
        if (comp_unit_info)
        {
            SymbolFileDWARF *oso_dwarf = GetSymbolFileByOSOIndex (oso_idx);
            if (oso_dwarf)
            {
                if (oso_dwarf->FindGlobalVariables(name, namespace_decl, true, max_matches, variables))
                    if (variables.GetSize() > max_matches)
                        break;
            }
        }
    }
    return variables.GetSize() - original_size;
}

uint32_t
SymbolFileDWARFDebugMap::FindGlobalVariables (const ConstString &name, const ClangNamespaceDecl *namespace_decl, bool append, uint32_t max_matches, VariableList& variables)
{

    // If we aren't appending the results to this list, then clear the list
    if (!append)
        variables.Clear();

    // Remember how many variables are in the list before we search in case
    // we are appending the results to a variable list.
    const uint32_t original_size = variables.GetSize();

    uint32_t total_matches = 0;
    SymbolFileDWARF *oso_dwarf;
    for (uint32_t oso_idx = 0; ((oso_dwarf = GetSymbolFileByOSOIndex (oso_idx)) != NULL); ++oso_idx)
    {
        const uint32_t oso_matches = oso_dwarf->FindGlobalVariables (name,
                                                                     namespace_decl,
                                                                     true, 
                                                                     max_matches, 
                                                                     variables);
        if (oso_matches > 0)
        {
            total_matches += oso_matches;

            // Are we getting all matches?
            if (max_matches == UINT32_MAX)
                continue;   // Yep, continue getting everything

            // If we have found enough matches, lets get out
            if (max_matches >= total_matches)
                break;

            // Update the max matches for any subsequent calls to find globals
            // in any other object files with DWARF
            max_matches -= oso_matches;
        }
    }
    // Return the number of variable that were appended to the list
    return variables.GetSize() - original_size;
}


uint32_t
SymbolFileDWARFDebugMap::FindGlobalVariables (const RegularExpression& regex, bool append, uint32_t max_matches, VariableList& variables)
{
    // If we aren't appending the results to this list, then clear the list
    if (!append)
        variables.Clear();

    // Remember how many variables are in the list before we search in case
    // we are appending the results to a variable list.
    const uint32_t original_size = variables.GetSize();

    uint32_t total_matches = 0;
    SymbolFileDWARF *oso_dwarf;
    for (uint32_t oso_idx = 0; ((oso_dwarf = GetSymbolFileByOSOIndex (oso_idx)) != NULL); ++oso_idx)
    {
        const uint32_t oso_matches = oso_dwarf->FindGlobalVariables (regex, 
                                                                     true, 
                                                                     max_matches, 
                                                                     variables);
        if (oso_matches > 0)
        {
            total_matches += oso_matches;

            // Are we getting all matches?
            if (max_matches == UINT32_MAX)
                continue;   // Yep, continue getting everything

            // If we have found enough matches, lets get out
            if (max_matches >= total_matches)
                break;

            // Update the max matches for any subsequent calls to find globals
            // in any other object files with DWARF
            max_matches -= oso_matches;
        }
    }
    // Return the number of variable that were appended to the list
    return variables.GetSize() - original_size;
}


int
SymbolFileDWARFDebugMap::SymbolContainsSymbolWithIndex (uint32_t *symbol_idx_ptr, const CompileUnitInfo *comp_unit_info)
{
    const uint32_t symbol_idx = *symbol_idx_ptr;

    if (symbol_idx < comp_unit_info->first_symbol_index)
        return -1;

    if (symbol_idx <= comp_unit_info->last_symbol_index)
        return 0;

    return 1;
}


int
SymbolFileDWARFDebugMap::SymbolContainsSymbolWithID (user_id_t *symbol_idx_ptr, const CompileUnitInfo *comp_unit_info)
{
    const user_id_t symbol_id = *symbol_idx_ptr;

    if (symbol_id < comp_unit_info->first_symbol_id)
        return -1;

    if (symbol_id <= comp_unit_info->last_symbol_id)
        return 0;

    return 1;
}


SymbolFileDWARFDebugMap::CompileUnitInfo*
SymbolFileDWARFDebugMap::GetCompileUnitInfoForSymbolWithIndex (uint32_t symbol_idx, uint32_t *oso_idx_ptr)
{
    const uint32_t oso_index_count = m_compile_unit_infos.size();
    CompileUnitInfo *comp_unit_info = NULL;
    if (oso_index_count)
    {
        comp_unit_info = (CompileUnitInfo*)bsearch(&symbol_idx, 
                                                   &m_compile_unit_infos[0], 
                                                   m_compile_unit_infos.size(), 
                                                   sizeof(CompileUnitInfo), 
                                                   (ComparisonFunction)SymbolContainsSymbolWithIndex);
    }

    if (oso_idx_ptr)
    {
        if (comp_unit_info != NULL)
            *oso_idx_ptr = comp_unit_info - &m_compile_unit_infos[0];
        else
            *oso_idx_ptr = UINT32_MAX;
    }
    return comp_unit_info;
}

SymbolFileDWARFDebugMap::CompileUnitInfo*
SymbolFileDWARFDebugMap::GetCompileUnitInfoForSymbolWithID (user_id_t symbol_id, uint32_t *oso_idx_ptr)
{
    const uint32_t oso_index_count = m_compile_unit_infos.size();
    CompileUnitInfo *comp_unit_info = NULL;
    if (oso_index_count)
    {
        comp_unit_info = (CompileUnitInfo*)::bsearch (&symbol_id, 
                                                      &m_compile_unit_infos[0], 
                                                      m_compile_unit_infos.size(), 
                                                      sizeof(CompileUnitInfo), 
                                                      (ComparisonFunction)SymbolContainsSymbolWithID);
    }

    if (oso_idx_ptr)
    {
        if (comp_unit_info != NULL)
            *oso_idx_ptr = comp_unit_info - &m_compile_unit_infos[0];
        else
            *oso_idx_ptr = UINT32_MAX;
    }
    return comp_unit_info;
}


static void
RemoveFunctionsWithModuleNotEqualTo (const ModuleSP &module_sp, SymbolContextList &sc_list, uint32_t start_idx)
{
    // We found functions in .o files. Not all functions in the .o files
    // will have made it into the final output file. The ones that did
    // make it into the final output file will have a section whose module
    // matches the module from the ObjectFile for this SymbolFile. When
    // the modules don't match, then we have something that was in a
    // .o file, but doesn't map to anything in the final executable.
    uint32_t i=start_idx;
    while (i < sc_list.GetSize())
    {
        SymbolContext sc;
        sc_list.GetContextAtIndex(i, sc);
        if (sc.function)
        {
            const SectionSP section_sp (sc.function->GetAddressRange().GetBaseAddress().GetSection());
            if (section_sp->GetModule() != module_sp)
            {
                sc_list.RemoveContextAtIndex(i);
                continue;
            }
        }
        ++i;
    }
}

uint32_t
SymbolFileDWARFDebugMap::FindFunctions(const ConstString &name, const ClangNamespaceDecl *namespace_decl, uint32_t name_type_mask, bool include_inlines, bool append, SymbolContextList& sc_list)
{
    Timer scoped_timer (__PRETTY_FUNCTION__,
                        "SymbolFileDWARFDebugMap::FindFunctions (name = %s)",
                        name.GetCString());

    uint32_t initial_size = 0;
    if (append)
        initial_size = sc_list.GetSize();
    else
        sc_list.Clear();

    uint32_t oso_idx = 0;
    SymbolFileDWARF *oso_dwarf;
    while ((oso_dwarf = GetSymbolFileByOSOIndex (oso_idx++)) != NULL)
    {
        uint32_t sc_idx = sc_list.GetSize();
        if (oso_dwarf->FindFunctions(name, namespace_decl, name_type_mask, include_inlines, true, sc_list))
        {
            RemoveFunctionsWithModuleNotEqualTo (m_obj_file->GetModule(), sc_list, sc_idx);
        }
    }

    return sc_list.GetSize() - initial_size;
}


uint32_t
SymbolFileDWARFDebugMap::FindFunctions (const RegularExpression& regex, bool include_inlines, bool append, SymbolContextList& sc_list)
{
    Timer scoped_timer (__PRETTY_FUNCTION__,
                        "SymbolFileDWARFDebugMap::FindFunctions (regex = '%s')",
                        regex.GetText());

    uint32_t initial_size = 0;
    if (append)
        initial_size = sc_list.GetSize();
    else
        sc_list.Clear();

    uint32_t oso_idx = 0;
    SymbolFileDWARF *oso_dwarf;
    while ((oso_dwarf = GetSymbolFileByOSOIndex (oso_idx++)) != NULL)
    {
        uint32_t sc_idx = sc_list.GetSize();
        
        if (oso_dwarf->FindFunctions(regex, include_inlines, true, sc_list))
        {
            RemoveFunctionsWithModuleNotEqualTo (m_obj_file->GetModule(), sc_list, sc_idx);
        }
    }

    return sc_list.GetSize() - initial_size;
}

TypeSP
SymbolFileDWARFDebugMap::FindDefinitionTypeForDWARFDeclContext (const DWARFDeclContext &die_decl_ctx)
{
    TypeSP type_sp;
    SymbolFileDWARF *oso_dwarf;
    for (uint32_t oso_idx = 0; ((oso_dwarf = GetSymbolFileByOSOIndex (oso_idx)) != NULL); ++oso_idx)
    {
        type_sp = oso_dwarf->FindDefinitionTypeForDWARFDeclContext (die_decl_ctx);
        if (type_sp)
            break;
    }
    return type_sp;
}



bool
SymbolFileDWARFDebugMap::Supports_DW_AT_APPLE_objc_complete_type (SymbolFileDWARF *skip_dwarf_oso)
{
    if (m_supports_DW_AT_APPLE_objc_complete_type == eLazyBoolCalculate)
    {
        m_supports_DW_AT_APPLE_objc_complete_type = eLazyBoolNo;
        SymbolFileDWARF *oso_dwarf;
        for (uint32_t oso_idx = 0; ((oso_dwarf = GetSymbolFileByOSOIndex (oso_idx)) != NULL); ++oso_idx)
        {
            if (skip_dwarf_oso != oso_dwarf && oso_dwarf->Supports_DW_AT_APPLE_objc_complete_type(NULL))
            {
                m_supports_DW_AT_APPLE_objc_complete_type = eLazyBoolYes;
                break;
            }
        }
    }
    return m_supports_DW_AT_APPLE_objc_complete_type == eLazyBoolYes;
}

TypeSP
SymbolFileDWARFDebugMap::FindCompleteObjCDefinitionTypeForDIE (const DWARFDebugInfoEntry *die, 
                                                               const ConstString &type_name,
                                                               bool must_be_implementation)
{
    TypeSP type_sp;
    SymbolFileDWARF *oso_dwarf;
    for (uint32_t oso_idx = 0; ((oso_dwarf = GetSymbolFileByOSOIndex (oso_idx)) != NULL); ++oso_idx)
    {
        type_sp = oso_dwarf->FindCompleteObjCDefinitionTypeForDIE (die, type_name, must_be_implementation);
        if (type_sp)
            break;
    }
    return type_sp;
}

uint32_t
SymbolFileDWARFDebugMap::FindTypes 
(
    const SymbolContext& sc, 
    const ConstString &name,
    const ClangNamespaceDecl *namespace_decl,
    bool append, 
    uint32_t max_matches, 
    TypeList& types
)
{
    if (!append)
        types.Clear();

    const uint32_t initial_types_size = types.GetSize();
    SymbolFileDWARF *oso_dwarf;

    if (sc.comp_unit)
    {
        oso_dwarf = GetSymbolFile (sc);
        if (oso_dwarf)
            return oso_dwarf->FindTypes (sc, name, namespace_decl, append, max_matches, types);
    }
    else
    {
        uint32_t oso_idx = 0;
        while ((oso_dwarf = GetSymbolFileByOSOIndex (oso_idx++)) != NULL)
            oso_dwarf->FindTypes (sc, name, namespace_decl, append, max_matches, types);
    }

    return types.GetSize() - initial_types_size;
}

//
//uint32_t
//SymbolFileDWARFDebugMap::FindTypes (const SymbolContext& sc, const RegularExpression& regex, bool append, uint32_t max_matches, Type::Encoding encoding, lldb::user_id_t udt_uid, TypeList& types)
//{
//  SymbolFileDWARF *oso_dwarf = GetSymbolFile (sc);
//  if (oso_dwarf)
//      return oso_dwarf->FindTypes (sc, regex, append, max_matches, encoding, udt_uid, types);
//  return 0;
//}


ClangNamespaceDecl
SymbolFileDWARFDebugMap::FindNamespace (const lldb_private::SymbolContext& sc, 
                                        const lldb_private::ConstString &name,
                                        const ClangNamespaceDecl *parent_namespace_decl)
{
    ClangNamespaceDecl matching_namespace;
    SymbolFileDWARF *oso_dwarf;

    if (sc.comp_unit)
    {
        oso_dwarf = GetSymbolFile (sc);
        if (oso_dwarf)
            matching_namespace = oso_dwarf->FindNamespace (sc, name, parent_namespace_decl);
    }
    else
    {
        for (uint32_t oso_idx = 0; 
             ((oso_dwarf = GetSymbolFileByOSOIndex (oso_idx)) != NULL); 
             ++oso_idx)
        {
            matching_namespace = oso_dwarf->FindNamespace (sc, name, parent_namespace_decl);

            if (matching_namespace)
                break;
        }
    }

    return matching_namespace;
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
lldb_private::ConstString
SymbolFileDWARFDebugMap::GetPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
SymbolFileDWARFDebugMap::GetPluginVersion()
{
    return 1;
}

lldb::CompUnitSP
SymbolFileDWARFDebugMap::GetCompileUnit (SymbolFileDWARF *oso_dwarf)
{
    if (oso_dwarf)
    {
        const uint32_t cu_count = GetNumCompileUnits();
        for (uint32_t cu_idx=0; cu_idx<cu_count; ++cu_idx)
        {
            SymbolFileDWARF *oso_symfile = GetSymbolFileByCompUnitInfo (&m_compile_unit_infos[cu_idx]);
            if (oso_symfile == oso_dwarf)
            {
                if (!m_compile_unit_infos[cu_idx].compile_unit_sp)
                    m_compile_unit_infos[cu_idx].compile_unit_sp = ParseCompileUnitAtIndex (cu_idx);

                return m_compile_unit_infos[cu_idx].compile_unit_sp;
            }
        }
    }
    assert(!"this shouldn't happen");
    return lldb::CompUnitSP();
}

SymbolFileDWARFDebugMap::CompileUnitInfo *
SymbolFileDWARFDebugMap::GetCompileUnitInfo (SymbolFileDWARF *oso_dwarf)
{
    if (oso_dwarf)
    {
        const uint32_t cu_count = GetNumCompileUnits();
        for (uint32_t cu_idx=0; cu_idx<cu_count; ++cu_idx)
        {
            SymbolFileDWARF *oso_symfile = GetSymbolFileByCompUnitInfo (&m_compile_unit_infos[cu_idx]);
            if (oso_symfile == oso_dwarf)
            {
                return &m_compile_unit_infos[cu_idx];
            }
        }
    }
    return NULL;
}


void
SymbolFileDWARFDebugMap::SetCompileUnit (SymbolFileDWARF *oso_dwarf, const CompUnitSP &cu_sp)
{
    if (oso_dwarf)
    {
        const uint32_t cu_count = GetNumCompileUnits();
        for (uint32_t cu_idx=0; cu_idx<cu_count; ++cu_idx)
        {
            SymbolFileDWARF *oso_symfile = GetSymbolFileByCompUnitInfo (&m_compile_unit_infos[cu_idx]);
            if (oso_symfile == oso_dwarf)
            {
                if (m_compile_unit_infos[cu_idx].compile_unit_sp)
                {
                    assert (m_compile_unit_infos[cu_idx].compile_unit_sp.get() == cu_sp.get());
                }
                else
                {
                    m_compile_unit_infos[cu_idx].compile_unit_sp = cu_sp;
                    m_obj_file->GetModule()->GetSymbolVendor()->SetCompileUnitAtIndex(cu_idx, cu_sp);
                }
            }
        }
    }
}


void
SymbolFileDWARFDebugMap::CompleteTagDecl (void *baton, clang::TagDecl *decl)
{
    SymbolFileDWARFDebugMap *symbol_file_dwarf = (SymbolFileDWARFDebugMap *)baton;
    clang_type_t clang_type = symbol_file_dwarf->GetClangASTContext().GetTypeForDecl (decl);
    if (clang_type)
    {
        SymbolFileDWARF *oso_dwarf;

        for (uint32_t oso_idx = 0; ((oso_dwarf = symbol_file_dwarf->GetSymbolFileByOSOIndex (oso_idx)) != NULL); ++oso_idx)
        {
            if (oso_dwarf->HasForwardDeclForClangType (clang_type))
            {
                oso_dwarf->ResolveClangOpaqueTypeDefinition (clang_type);
                return;
            }
        }
    }
}

void
SymbolFileDWARFDebugMap::CompleteObjCInterfaceDecl (void *baton, clang::ObjCInterfaceDecl *decl)
{
    SymbolFileDWARFDebugMap *symbol_file_dwarf = (SymbolFileDWARFDebugMap *)baton;
    clang_type_t clang_type = symbol_file_dwarf->GetClangASTContext().GetTypeForDecl (decl);
    if (clang_type)
    {
        SymbolFileDWARF *oso_dwarf;

        for (uint32_t oso_idx = 0; ((oso_dwarf = symbol_file_dwarf->GetSymbolFileByOSOIndex (oso_idx)) != NULL); ++oso_idx)
        {
            if (oso_dwarf->HasForwardDeclForClangType (clang_type))
            {
                oso_dwarf->ResolveClangOpaqueTypeDefinition (clang_type);
                return;
            }
        }
    }
}

bool 
SymbolFileDWARFDebugMap::LayoutRecordType (void *baton, 
                                           const clang::RecordDecl *record_decl,
                                           uint64_t &size, 
                                           uint64_t &alignment,
                                           llvm::DenseMap <const clang::FieldDecl *, uint64_t> &field_offsets,
                                           llvm::DenseMap <const clang::CXXRecordDecl *, clang::CharUnits> &base_offsets,
                                           llvm::DenseMap <const clang::CXXRecordDecl *, clang::CharUnits> &vbase_offsets)
{
    SymbolFileDWARFDebugMap *symbol_file_dwarf = (SymbolFileDWARFDebugMap *)baton;
    SymbolFileDWARF *oso_dwarf;
    for (uint32_t oso_idx = 0; ((oso_dwarf = symbol_file_dwarf->GetSymbolFileByOSOIndex (oso_idx)) != NULL); ++oso_idx)
    {
        if (oso_dwarf->LayoutRecordType (record_decl, size, alignment, field_offsets, base_offsets, vbase_offsets))
            return true;
    }
    return false;
}



clang::DeclContext*
SymbolFileDWARFDebugMap::GetClangDeclContextContainingTypeUID (lldb::user_id_t type_uid)
{
    const uint64_t oso_idx = GetOSOIndexFromUserID (type_uid);
    SymbolFileDWARF *oso_dwarf = GetSymbolFileByOSOIndex (oso_idx);
    if (oso_dwarf)
        return oso_dwarf->GetClangDeclContextContainingTypeUID (type_uid);
    return NULL;
}

clang::DeclContext*
SymbolFileDWARFDebugMap::GetClangDeclContextForTypeUID (const lldb_private::SymbolContext &sc, lldb::user_id_t type_uid)
{
    const uint64_t oso_idx = GetOSOIndexFromUserID (type_uid);
    SymbolFileDWARF *oso_dwarf = GetSymbolFileByOSOIndex (oso_idx);
    if (oso_dwarf)
        return oso_dwarf->GetClangDeclContextForTypeUID (sc, type_uid);
    return NULL;
}

bool
SymbolFileDWARFDebugMap::AddOSOFileRange (CompileUnitInfo *cu_info,
                                          lldb::addr_t exe_file_addr,
                                          lldb::addr_t oso_file_addr,
                                          lldb::addr_t oso_byte_size)
{
    assert (cu_info);// REMOVE THIS PRIOR TO CHECKIN
    const uint32_t debug_map_idx = m_debug_map.FindEntryIndexThatContains(exe_file_addr);
    if (debug_map_idx != UINT32_MAX)
    {
        DebugMap::Entry *debug_map_entry = m_debug_map.FindEntryThatContains(exe_file_addr);
        assert (debug_map_entry);// REMOVE THIS PRIOR TO CHECKIN
        debug_map_entry->data.SetOSOFileAddress(oso_file_addr);
        cu_info->file_range_map.Append(FileRangeMap::Entry(oso_file_addr, oso_byte_size, exe_file_addr));
        return true;
    }
    return false;
}

void
SymbolFileDWARFDebugMap::FinalizeOSOFileRanges (CompileUnitInfo *cu_info)
{
    cu_info->file_range_map.Sort();
#if defined(DEBUG_OSO_DMAP)
    const FileRangeMap &oso_file_range_map = cu_info->GetFileRangeMap(this);
    const size_t n = oso_file_range_map.GetSize();
    printf ("SymbolFileDWARFDebugMap::FinalizeOSOFileRanges (cu_info = %p) %s\n",
            cu_info,
            cu_info->oso_sp->module_sp->GetFileSpec().GetPath().c_str());
    for (size_t i=0; i<n; ++i)
    {
        const FileRangeMap::Entry &entry = oso_file_range_map.GetEntryRef(i);
        printf ("oso [0x%16.16" PRIx64 " - 0x%16.16" PRIx64 ") ==> exe [0x%16.16" PRIx64 " - 0x%16.16" PRIx64 ")\n",
                entry.GetRangeBase(), entry.GetRangeEnd(),
                entry.data, entry.data + entry.GetByteSize());
    }
#endif
}

lldb::addr_t
SymbolFileDWARFDebugMap::LinkOSOFileAddress (SymbolFileDWARF *oso_symfile, lldb::addr_t oso_file_addr)
{
    CompileUnitInfo *cu_info = GetCompileUnitInfo (oso_symfile);
    if (cu_info)
    {
        const FileRangeMap::Entry *oso_range_entry = cu_info->GetFileRangeMap(this).FindEntryThatContains(oso_file_addr);
        if (oso_range_entry)
        {
            const DebugMap::Entry *debug_map_entry = m_debug_map.FindEntryThatContains(oso_range_entry->data);
            if (debug_map_entry)
            {
                const lldb::addr_t offset = oso_file_addr - oso_range_entry->GetRangeBase();
                const lldb::addr_t exe_file_addr = debug_map_entry->GetRangeBase() + offset;
                return exe_file_addr;
            }
        }
    }
    return LLDB_INVALID_ADDRESS;
}

bool
SymbolFileDWARFDebugMap::LinkOSOAddress (Address &addr)
{
    // Make sure this address hasn't been fixed already
    Module *exe_module = GetObjectFile()->GetModule().get();
    Module *addr_module = addr.GetModule().get();
    if (addr_module == exe_module)
        return true; // Address is already in terms of the main executable module

    CompileUnitInfo *cu_info = GetCompileUnitInfo (GetSymbolFileAsSymbolFileDWARF(addr_module->GetSymbolVendor()->GetSymbolFile()));
    if (cu_info)
    {
        const lldb::addr_t oso_file_addr = addr.GetFileAddress();
        const FileRangeMap::Entry *oso_range_entry = cu_info->GetFileRangeMap(this).FindEntryThatContains(oso_file_addr);
        if (oso_range_entry)
        {
            const DebugMap::Entry *debug_map_entry = m_debug_map.FindEntryThatContains(oso_range_entry->data);
            if (debug_map_entry)
            {
                const lldb::addr_t offset = oso_file_addr - oso_range_entry->GetRangeBase();
                const lldb::addr_t exe_file_addr = debug_map_entry->GetRangeBase() + offset;
                return exe_module->ResolveFileAddress(exe_file_addr, addr);
            }
        }
    }
    return true;
}

LineTable *
SymbolFileDWARFDebugMap::LinkOSOLineTable (SymbolFileDWARF *oso_dwarf, LineTable *line_table)
{
    CompileUnitInfo *cu_info = GetCompileUnitInfo (oso_dwarf);
    if (cu_info)
        return line_table->LinkLineTable(cu_info->GetFileRangeMap(this));
    return NULL;
}


