//===-- SymbolVendorMacOSX.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SymbolVendorMacOSX.h"

#include <libxml/parser.h>
#include <libxml/tree.h>
#include <string.h>

#include <AvailabilityMacros.h>

#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/Timer.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/Symbols.h"
#include "lldb/Symbol/ObjectFile.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// SymbolVendorMacOSX constructor
//----------------------------------------------------------------------
SymbolVendorMacOSX::SymbolVendorMacOSX(const lldb::ModuleSP &module_sp) :
    SymbolVendor (module_sp)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
SymbolVendorMacOSX::~SymbolVendorMacOSX()
{
}


static bool
UUIDsMatch(Module *module, ObjectFile *ofile)
{
    if (module && ofile)
    {
        // Make sure the UUIDs match
        lldb_private::UUID dsym_uuid;

        if (!ofile->GetUUID(&dsym_uuid))
        {
            Host::SystemLog (Host::eSystemLogWarning, 
                             "warning: failed to get the uuid for object file: '%s'\n", 
                             ofile->GetFileSpec().GetFilename().GetCString());
            return false;
        }

        if (dsym_uuid == module->GetUUID())
            return true;

        // Emit some warning messages since the UUIDs do not match!
        const FileSpec &m_file_spec = module->GetFileSpec();
        const FileSpec &o_file_spec = ofile->GetFileSpec();
        StreamString ss_m_path, ss_o_path;
        m_file_spec.Dump(&ss_m_path);
        o_file_spec.Dump(&ss_o_path);

        StreamString ss_m_uuid, ss_o_uuid;
        module->GetUUID().Dump(&ss_m_uuid);
        dsym_uuid.Dump(&ss_o_uuid);
        Host::SystemLog (Host::eSystemLogWarning, 
                         "warning: UUID mismatch detected between module '%s' (%s) and:\n\t'%s' (%s)\n", 
                         ss_m_path.GetData(), ss_m_uuid.GetData(), ss_o_path.GetData(), ss_o_uuid.GetData());
    }
    return false;
}

static void
ReplaceDSYMSectionsWithExecutableSections (ObjectFile *exec_objfile, ObjectFile *dsym_objfile)
{
    // We need both the executable and the dSYM to live off of the
    // same section lists. So we take all of the sections from the
    // executable, and replace them in the dSYM. This allows section
    // offset addresses that come from the dSYM to automatically
    // get updated as images (shared libraries) get loaded and
    // unloaded.
    SectionList *exec_section_list = exec_objfile->GetSectionList();
    SectionList *dsym_section_list = dsym_objfile->GetSectionList();
    if (exec_section_list && dsym_section_list)
    {
        const uint32_t num_exec_sections = dsym_section_list->GetSize();
        uint32_t exec_sect_idx;
        for (exec_sect_idx = 0; exec_sect_idx < num_exec_sections; ++exec_sect_idx)
        {
            SectionSP exec_sect_sp(exec_section_list->GetSectionAtIndex(exec_sect_idx));
            if (exec_sect_sp.get())
            {
                // Try and replace any sections that exist in both the executable
                // and in the dSYM with those from the executable. If we fail to
                // replace the one in the dSYM, then add the executable section to
                // the dSYM.
                if (dsym_section_list->ReplaceSection(exec_sect_sp->GetID(), exec_sect_sp, 0) == false)
                    dsym_section_list->AddSection(exec_sect_sp);
            }
        }
        
        dsym_section_list->Finalize(); // Now that we're done adding sections, finalize to build fast-lookup caches
    }
}

void
SymbolVendorMacOSX::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   GetPluginDescriptionStatic(),
                                   CreateInstance);
}

void
SymbolVendorMacOSX::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}


const char *
SymbolVendorMacOSX::GetPluginNameStatic()
{
    return "symbol-vendor.macosx";
}

const char *
SymbolVendorMacOSX::GetPluginDescriptionStatic()
{
    return "Symbol vendor for MacOSX that looks for dSYM files that match executables.";
}



//----------------------------------------------------------------------
// CreateInstance
//
// Platforms can register a callback to use when creating symbol
// vendors to allow for complex debug information file setups, and to
// also allow for finding separate debug information files.
//----------------------------------------------------------------------
SymbolVendor*
SymbolVendorMacOSX::CreateInstance (const lldb::ModuleSP &module_sp)
{
    if (!module_sp)
        return NULL;

    Timer scoped_timer (__PRETTY_FUNCTION__,
                        "SymbolVendorMacOSX::CreateInstance (module = %s/%s)",
                        module_sp->GetFileSpec().GetDirectory().AsCString(),
                        module_sp->GetFileSpec().GetFilename().AsCString());
    SymbolVendorMacOSX* symbol_vendor = new SymbolVendorMacOSX(module_sp);
    if (symbol_vendor)
    {
        char path[PATH_MAX];
        path[0] = '\0';

        // Try and locate the dSYM file on Mac OS X
        ObjectFile * obj_file = module_sp->GetObjectFile();
        if (obj_file)
        {
            Timer scoped_timer2 ("SymbolVendorMacOSX::CreateInstance () locate dSYM",
                                 "SymbolVendorMacOSX::CreateInstance (module = %s/%s) locate dSYM",
                                 module_sp->GetFileSpec().GetDirectory().AsCString(),
                                 module_sp->GetFileSpec().GetFilename().AsCString());

            // First check to see if the module has a symbol file in mind already.
            // If it does, then we MUST use that.
            FileSpec dsym_fspec (module_sp->GetSymbolFileFileSpec());
            
            ObjectFileSP dsym_objfile_sp;
            if (!dsym_fspec)
            {
                // No symbol file was specified in the module, lets try and find
                // one ourselves.
                const FileSpec &file_spec = obj_file->GetFileSpec();
                if (file_spec)
                {
                    ModuleSpec module_spec(file_spec, module_sp->GetArchitecture());
                    module_spec.GetUUID() = module_sp->GetUUID();
                    dsym_fspec = Symbols::LocateExecutableSymbolFile (module_spec);
                    if (module_spec.GetSourceMappingList().GetSize())
                    {
                        module_sp->GetSourceMappingList().Append (module_spec.GetSourceMappingList (), true);
                    }
                }
            }
            
            if (dsym_fspec)
            {
                DataBufferSP dsym_file_data_sp;
                dsym_objfile_sp = ObjectFile::FindPlugin(module_sp, &dsym_fspec, 0, dsym_fspec.GetByteSize(), dsym_file_data_sp);
                if (UUIDsMatch(module_sp.get(), dsym_objfile_sp.get()))
                {
                    char dsym_path[PATH_MAX];
                    if (module_sp->GetSourceMappingList().IsEmpty() && dsym_fspec.GetPath(dsym_path, sizeof(dsym_path)))
                    {
                        lldb_private::UUID dsym_uuid;
                        if (dsym_objfile_sp->GetUUID(&dsym_uuid))
                        {
                            char uuid_cstr_buf[64];
                            const char *uuid_cstr = dsym_uuid.GetAsCString (uuid_cstr_buf, sizeof(uuid_cstr_buf));
                            if (uuid_cstr)
                            {
                                char *resources = strstr (dsym_path, "/Contents/Resources/");
                                if (resources)
                                {
                                    char dsym_uuid_plist_path[PATH_MAX];
                                    resources[strlen("/Contents/Resources/")] = '\0';
                                    snprintf(dsym_uuid_plist_path, sizeof(dsym_uuid_plist_path), "%s%s.plist", dsym_path, uuid_cstr);
                                    FileSpec dsym_uuid_plist_spec(dsym_uuid_plist_path, false);
                                    if (dsym_uuid_plist_spec.Exists())
                                    {
                                        xmlDoc *doc = ::xmlReadFile (dsym_uuid_plist_path, NULL, 0);
                                        if (doc)
                                        {
                                            char DBGBuildSourcePath[PATH_MAX];
                                            char DBGSourcePath[PATH_MAX];
                                            DBGBuildSourcePath[0] = '\0';
                                            DBGSourcePath[0] = '\0';
                                            for (xmlNode *node = doc->children; node; node = node ? node->next : NULL)
                                            {
                                                if (node->type == XML_ELEMENT_NODE)
                                                {
                                                    if (node->name && strcmp((const char*)node->name, "plist") == 0)
                                                    {
                                                        xmlNode *dict_node = node->children;
                                                        while (dict_node && dict_node->type != XML_ELEMENT_NODE)
                                                            dict_node = dict_node->next;
                                                        if (dict_node && dict_node->name && strcmp((const char *)dict_node->name, "dict") == 0)
                                                        {
                                                            for (xmlNode *key_node = dict_node->children; key_node; key_node = key_node->next)
                                                            {
                                                                if (key_node && key_node->type == XML_ELEMENT_NODE && key_node->name)
                                                                {
                                                                    if (strcmp((const char *)key_node->name, "key") == 0)
                                                                    {
                                                                        const char *key_name = (const char *)::xmlNodeGetContent(key_node);
                                                                        if (strcmp(key_name, "DBGBuildSourcePath") == 0)
                                                                        {
                                                                            xmlNode *value_node = key_node->next;
                                                                            while (value_node && value_node->type != XML_ELEMENT_NODE)
                                                                                value_node = value_node->next;
                                                                            if (value_node && value_node->name)
                                                                            {
                                                                                if (strcmp((const char *)value_node->name, "string") == 0)
                                                                                {
                                                                                    const char *node_content = (const char *)::xmlNodeGetContent(value_node);
                                                                                    if (node_content)
                                                                                    {
                                                                                        strncpy(DBGBuildSourcePath, node_content, sizeof(DBGBuildSourcePath));
                                                                                    }
                                                                                }
                                                                                key_node = value_node;
                                                                            }
                                                                        }
                                                                        else if (strcmp(key_name, "DBGSourcePath") == 0)
                                                                        {
                                                                            xmlNode *value_node = key_node->next;
                                                                            while (value_node && value_node->type != XML_ELEMENT_NODE)
                                                                                value_node = value_node->next;
                                                                            if (value_node && value_node->name)
                                                                            {
                                                                                if (strcmp((const char *)value_node->name, "string") == 0)
                                                                                {
                                                                                    const char *node_content = (const char *)::xmlNodeGetContent(value_node);
                                                                                    if (node_content)
                                                                                    {
                                                                                        FileSpec resolved_source_path(node_content, true);
                                                                                        resolved_source_path.GetPath(DBGSourcePath, sizeof(DBGSourcePath));
                                                                                    }
                                                                                }
                                                                                key_node = value_node;
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            ::xmlFreeDoc (doc);
                                            
                                            if (DBGBuildSourcePath[0] && DBGSourcePath[0])
                                            {
                                                module_sp->GetSourceMappingList().Append (ConstString(DBGBuildSourcePath), ConstString(DBGSourcePath), true);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    ReplaceDSYMSectionsWithExecutableSections (obj_file, dsym_objfile_sp.get());
                    symbol_vendor->AddSymbolFileRepresentation(dsym_objfile_sp);
                    return symbol_vendor;
                }
            }

            // Just create our symbol vendor using the current objfile as this is either
            // an executable with no dSYM (that we could locate), an executable with
            // a dSYM that has a UUID that doesn't match.
            symbol_vendor->AddSymbolFileRepresentation(obj_file->shared_from_this());
        }
    }
    return symbol_vendor;
}



//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
SymbolVendorMacOSX::GetPluginName()
{
    return "SymbolVendorMacOSX";
}

const char *
SymbolVendorMacOSX::GetShortPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
SymbolVendorMacOSX::GetPluginVersion()
{
    return 1;
}

