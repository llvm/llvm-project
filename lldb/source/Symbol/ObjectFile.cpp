//===-- ObjectFile.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-private.h"
#include "lldb/lldb-private-log.h"
#include "lldb/Core/DataBuffer.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/Timer.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/ObjectContainer.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Target/Process.h"
#include "Plugins/ObjectContainer/BSD-Archive/ObjectContainerBSDArchive.h"

using namespace lldb;
using namespace lldb_private;

ObjectFileSP
ObjectFile::FindPlugin (const lldb::ModuleSP &module_sp, const FileSpec* file, addr_t file_offset, addr_t file_size, DataBufferSP &file_data_sp)
{
    ObjectFileSP object_file_sp;

    if (module_sp)
    {
        Timer scoped_timer (__PRETTY_FUNCTION__,
                            "ObjectFile::FindPlugin (module = %s/%s, file = %p, file_offset = 0x%8.8llx, file_size = 0x%8.8llx)",
                            module_sp->GetFileSpec().GetDirectory().AsCString(),
                            module_sp->GetFileSpec().GetFilename().AsCString(),
                            file, (uint64_t) file_offset, (uint64_t) file_size);
        if (file)
        {
            // Memory map the entire file contents
            if (!file_data_sp && file_size > 0)
            {
                assert (file_offset == 0);
                file_data_sp = file->MemoryMapFileContents(file_offset, file_size);
            }

            if (!file_data_sp || file_data_sp->GetByteSize() == 0)
            {
                // Check for archive file with format "/path/to/archive.a(object.o)"
                char path_with_object[PATH_MAX*2];
                module_sp->GetFileSpec().GetPath(path_with_object, sizeof(path_with_object));

                FileSpec archive_file;
                ConstString archive_object;
                if (ObjectFile::SplitArchivePathWithObject (path_with_object, archive_file, archive_object))
                {
                    file_size = archive_file.GetByteSize();
                    if (file_size > 0)
                    {
                        module_sp->SetFileSpecAndObjectName (archive_file, archive_object);
                        file_data_sp = archive_file.MemoryMapFileContents(file_offset, file_size);
                    }
                }
            }

            if (file_data_sp && file_data_sp->GetByteSize() > 0)
            {
                uint32_t idx;

                // Check if this is a normal object file by iterating through
                // all object file plugin instances.
                ObjectFileCreateInstance create_object_file_callback;
                for (idx = 0; (create_object_file_callback = PluginManager::GetObjectFileCreateCallbackAtIndex(idx)) != NULL; ++idx)
                {
                    object_file_sp.reset (create_object_file_callback(module_sp, file_data_sp, file, file_offset, file_size));
                    if (object_file_sp.get())
                        return object_file_sp;
                }

                // Check if this is a object container by iterating through
                // all object container plugin instances and then trying to get
                // an object file from the container.
                ObjectContainerCreateInstance create_object_container_callback;
                for (idx = 0; (create_object_container_callback = PluginManager::GetObjectContainerCreateCallbackAtIndex(idx)) != NULL; ++idx)
                {
                    std::auto_ptr<ObjectContainer> object_container_ap(create_object_container_callback(module_sp, file_data_sp, file, file_offset, file_size));

                    if (object_container_ap.get())
                        object_file_sp = object_container_ap->GetObjectFile(file);

                    if (object_file_sp.get())
                        return object_file_sp;
                }
            }
        }
    }
    // We didn't find it, so clear our shared pointer in case it
    // contains anything and return an empty shared pointer
    object_file_sp.reset();
    return object_file_sp;
}

ObjectFileSP
ObjectFile::FindPlugin (const lldb::ModuleSP &module_sp, 
                        const ProcessSP &process_sp,
                        lldb::addr_t header_addr,
                        DataBufferSP &file_data_sp)
{
    ObjectFileSP object_file_sp;
    
    if (module_sp)
    {
        Timer scoped_timer (__PRETTY_FUNCTION__,
                            "ObjectFile::FindPlugin (module = %s/%s, process = %p, header_addr = 0x%llx)",
                            module_sp->GetFileSpec().GetDirectory().AsCString(),
                            module_sp->GetFileSpec().GetFilename().AsCString(),
                            process_sp.get(), header_addr);
        uint32_t idx;
        
        // Check if this is a normal object file by iterating through
        // all object file plugin instances.
        ObjectFileCreateMemoryInstance create_callback;
        for (idx = 0; (create_callback = PluginManager::GetObjectFileCreateMemoryCallbackAtIndex(idx)) != NULL; ++idx)
        {
            object_file_sp.reset (create_callback(module_sp, file_data_sp, process_sp, header_addr));
            if (object_file_sp.get())
                return object_file_sp;
        }
        
    }
    // We didn't find it, so clear our shared pointer in case it
    // contains anything and return an empty shared pointer
    object_file_sp.reset();
    return object_file_sp;
}

ObjectFile::ObjectFile (const lldb::ModuleSP &module_sp, 
                        const FileSpec *file_spec_ptr, 
                        addr_t file_offset, 
                        addr_t file_size, 
                        DataBufferSP& file_data_sp) :
    ModuleChild (module_sp),
    m_file (),  // This file could be different from the original module's file
    m_type (eTypeInvalid),
    m_strata (eStrataInvalid),
    m_offset (file_offset),
    m_length (file_size),
    m_data (),
    m_unwind_table (*this),
    m_process_wp(),
    m_memory_addr (LLDB_INVALID_ADDRESS)
{    
    if (file_spec_ptr)
        m_file = *file_spec_ptr;
    if (file_data_sp)
        m_data.SetData (file_data_sp, file_offset, file_size);
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT));
    if (log)
    {
        if (m_file)
        {
            log->Printf ("%p ObjectFile::ObjectFile () module = %s/%s, file = %s/%s, offset = 0x%8.8llx, size = %llu\n",
                         this,
                         module_sp->GetFileSpec().GetDirectory().AsCString(),
                         module_sp->GetFileSpec().GetFilename().AsCString(),
                         m_file.GetDirectory().AsCString(),
                         m_file.GetFilename().AsCString(),
                         m_offset,
                         m_length);
        }
        else
        {
            log->Printf ("%p ObjectFile::ObjectFile () module = %s/%s, file = <NULL>, offset = 0x%8.8llx, size = %llu\n",
                         this,
                         module_sp->GetFileSpec().GetDirectory().AsCString(),
                         module_sp->GetFileSpec().GetFilename().AsCString(),
                         m_offset,
                         m_length);
        }
    }
}


ObjectFile::ObjectFile (const lldb::ModuleSP &module_sp, 
                        const ProcessSP &process_sp,
                        lldb::addr_t header_addr, 
                        DataBufferSP& header_data_sp) :
    ModuleChild (module_sp),
    m_file (),
    m_type (eTypeInvalid),
    m_strata (eStrataInvalid),
    m_offset (header_addr),
    m_length (0),
    m_data (),
    m_unwind_table (*this),
    m_process_wp (process_sp),
    m_memory_addr (header_addr)
{    
    if (header_data_sp)
        m_data.SetData (header_data_sp, 0, header_data_sp->GetByteSize());
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT));
    if (log)
    {
        log->Printf ("%p ObjectFile::ObjectFile () module = %s/%s, process = %p, header_addr = 0x%llx\n",
                     this,
                     module_sp->GetFileSpec().GetDirectory().AsCString(),
                     module_sp->GetFileSpec().GetFilename().AsCString(),
                     process_sp.get(),
                     m_offset);
    }
}


ObjectFile::~ObjectFile()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT));
    if (log)
    {
        ModuleSP module_sp (GetModule());
        if (m_file)
        {
            log->Printf ("%p ObjectFile::~ObjectFile () module = %s/%s, file = %s/%s, offset = 0x%8.8llx, size = %llu\n",
                         this,
                         module_sp->GetFileSpec().GetDirectory().AsCString(),
                         module_sp->GetFileSpec().GetFilename().AsCString(),
                         m_file.GetDirectory().AsCString(),
                         m_file.GetFilename().AsCString(),
                         m_offset,
                         m_length);
        }
        else
        {
            log->Printf ("%p ObjectFile::~ObjectFile () module = %s/%s, file = <NULL>, offset = 0x%8.8llx, size = %llu\n",
                         this,
                         module_sp->GetFileSpec().GetDirectory().AsCString(),
                         module_sp->GetFileSpec().GetFilename().AsCString(),
                         m_offset,
                         m_length);
        }
    }
}

bool 
ObjectFile::SetModulesArchitecture (const ArchSpec &new_arch)
{
    ModuleSP module_sp (GetModule());
    if (module_sp)
        return module_sp->SetArchitecture (new_arch);
    return false;
}

AddressClass
ObjectFile::GetAddressClass (addr_t file_addr)
{
    Symtab *symtab = GetSymtab();
    if (symtab)
    {
        Symbol *symbol = symtab->FindSymbolContainingFileAddress(file_addr);
        if (symbol)
        {
            if (symbol->ValueIsAddress())
            {
                const SectionSP section_sp (symbol->GetAddress().GetSection());
                if (section_sp)
                {
                    const SectionType section_type = section_sp->GetType();
                    switch (section_type)
                    {
                    case eSectionTypeInvalid:               return eAddressClassUnknown;
                    case eSectionTypeCode:                  return eAddressClassCode;
                    case eSectionTypeContainer:             return eAddressClassUnknown;
                    case eSectionTypeData:
                    case eSectionTypeDataCString:
                    case eSectionTypeDataCStringPointers:
                    case eSectionTypeDataSymbolAddress:
                    case eSectionTypeData4:
                    case eSectionTypeData8:
                    case eSectionTypeData16:
                    case eSectionTypeDataPointers:
                    case eSectionTypeZeroFill:
                    case eSectionTypeDataObjCMessageRefs:
                    case eSectionTypeDataObjCCFStrings:
                        return eAddressClassData;
                    case eSectionTypeDebug:
                    case eSectionTypeDWARFDebugAbbrev:
                    case eSectionTypeDWARFDebugAranges:
                    case eSectionTypeDWARFDebugFrame:
                    case eSectionTypeDWARFDebugInfo:
                    case eSectionTypeDWARFDebugLine:
                    case eSectionTypeDWARFDebugLoc:
                    case eSectionTypeDWARFDebugMacInfo:
                    case eSectionTypeDWARFDebugPubNames:
                    case eSectionTypeDWARFDebugPubTypes:
                    case eSectionTypeDWARFDebugRanges:
                    case eSectionTypeDWARFDebugStr:
                    case eSectionTypeDWARFAppleNames:
                    case eSectionTypeDWARFAppleTypes:
                    case eSectionTypeDWARFAppleNamespaces:
                    case eSectionTypeDWARFAppleObjC:
                        return eAddressClassDebug;
                    case eSectionTypeEHFrame:               return eAddressClassRuntime;
                    case eSectionTypeOther:                 return eAddressClassUnknown;
                    }
                }
            }
            
            const SymbolType symbol_type = symbol->GetType();
            switch (symbol_type)
            {
            case eSymbolTypeAny:            return eAddressClassUnknown;
            case eSymbolTypeAbsolute:       return eAddressClassUnknown;
            case eSymbolTypeCode:           return eAddressClassCode;
            case eSymbolTypeTrampoline:     return eAddressClassCode;
            case eSymbolTypeData:           return eAddressClassData;
            case eSymbolTypeRuntime:        return eAddressClassRuntime;
            case eSymbolTypeException:      return eAddressClassRuntime;
            case eSymbolTypeSourceFile:     return eAddressClassDebug;
            case eSymbolTypeHeaderFile:     return eAddressClassDebug;
            case eSymbolTypeObjectFile:     return eAddressClassDebug;
            case eSymbolTypeCommonBlock:    return eAddressClassDebug;
            case eSymbolTypeBlock:          return eAddressClassDebug;
            case eSymbolTypeLocal:          return eAddressClassData;
            case eSymbolTypeParam:          return eAddressClassData;
            case eSymbolTypeVariable:       return eAddressClassData;
            case eSymbolTypeVariableType:   return eAddressClassDebug;
            case eSymbolTypeLineEntry:      return eAddressClassDebug;
            case eSymbolTypeLineHeader:     return eAddressClassDebug;
            case eSymbolTypeScopeBegin:     return eAddressClassDebug;
            case eSymbolTypeScopeEnd:       return eAddressClassDebug;
            case eSymbolTypeAdditional:     return eAddressClassUnknown;
            case eSymbolTypeCompiler:       return eAddressClassDebug;
            case eSymbolTypeInstrumentation:return eAddressClassDebug;
            case eSymbolTypeUndefined:      return eAddressClassUnknown;
            case eSymbolTypeObjCClass:      return eAddressClassRuntime;
            case eSymbolTypeObjCMetaClass:  return eAddressClassRuntime;
            case eSymbolTypeObjCIVar:       return eAddressClassRuntime;
            }
        }
    }
    return eAddressClassUnknown;
}

DataBufferSP
ObjectFile::ReadMemory (const ProcessSP &process_sp, lldb::addr_t addr, size_t byte_size)
{
    DataBufferSP data_sp;
    if (process_sp)
    {
        std::auto_ptr<DataBufferHeap> data_ap (new DataBufferHeap (byte_size, 0));
        Error error;
        const size_t bytes_read = process_sp->ReadMemory (addr, 
                                                          data_ap->GetBytes(), 
                                                          data_ap->GetByteSize(), 
                                                          error);
        if (bytes_read == byte_size)
            data_sp.reset (data_ap.release());
    }
    return data_sp;
}

size_t
ObjectFile::GetData (off_t offset, size_t length, DataExtractor &data) const
{
    // The entire file has already been mmap'ed into m_data, so just copy from there
    // as the back mmap buffer will be shared with shared pointers.
    return data.SetData (m_data, offset, length);
}

size_t
ObjectFile::CopyData (off_t offset, size_t length, void *dst) const
{
    // The entire file has already been mmap'ed into m_data, so just copy from there
    return m_data.CopyByteOrderedData (offset, length, dst, length, lldb::endian::InlHostByteOrder());
}


size_t
ObjectFile::ReadSectionData (const Section *section, off_t section_offset, void *dst, size_t dst_len) const
{
    if (IsInMemory())
    {
        ProcessSP process_sp (m_process_wp.lock());
        if (process_sp)
        {
            Error error;
            return process_sp->ReadMemory (section->GetLoadBaseAddress (&process_sp->GetTarget()) + section_offset, dst, dst_len, error);
        }
    }
    else
    {
        const uint64_t section_file_size = section->GetFileSize();
        if (section_offset < section_file_size)
        {
            const uint64_t section_bytes_left = section_file_size - section_offset;
            uint64_t section_dst_len = dst_len;
            if (section_dst_len > section_bytes_left)
                section_dst_len = section_bytes_left;
            return CopyData (section->GetFileOffset() + section_offset, section_dst_len, dst);
        }
    }
    return 0;
}

//----------------------------------------------------------------------
// Get the section data the file on disk
//----------------------------------------------------------------------
size_t
ObjectFile::ReadSectionData (const Section *section, DataExtractor& section_data) const
{
    if (IsInMemory())
    {
        ProcessSP process_sp (m_process_wp.lock());
        if (process_sp)
        {
            DataBufferSP data_sp (ReadMemory (process_sp, section->GetLoadBaseAddress (&process_sp->GetTarget()), section->GetByteSize()));
            if (data_sp)
            {
                section_data.SetData (data_sp, 0, data_sp->GetByteSize());
                section_data.SetByteOrder (process_sp->GetByteOrder());
                section_data.SetAddressByteSize (process_sp->GetAddressByteSize());
                return section_data.GetByteSize();
            }
        }
    }
    else
    {
        // The object file now contains a full mmap'ed copy of the object file data, so just use this
        return MemoryMapSectionData (section, section_data);
    }
    section_data.Clear();
    return 0;
}

size_t
ObjectFile::MemoryMapSectionData (const Section *section, DataExtractor& section_data) const
{
    if (IsInMemory())
    {
        return ReadSectionData (section, section_data);
    }
    else
    {
        // The object file now contains a full mmap'ed copy of the object file data, so just use this
        return GetData(section->GetFileOffset(), section->GetFileSize(), section_data);
    }
    section_data.Clear();
    return 0;
}


bool
ObjectFile::SplitArchivePathWithObject (const char *path_with_object, FileSpec &archive_file, ConstString &archive_object)
{
    RegularExpression g_object_regex("(.*)\\(([^\\)]+)\\)$");
    if (g_object_regex.Execute (path_with_object, 2))
    {
        std::string path;
        std::string obj;
        if (g_object_regex.GetMatchAtIndex (path_with_object, 1, path) &&
            g_object_regex.GetMatchAtIndex (path_with_object, 2, obj))
        {
            archive_file.SetFile (path.c_str(), false);
            archive_object.SetCString(obj.c_str());
            return true;
        }
    }
    return false;
}

