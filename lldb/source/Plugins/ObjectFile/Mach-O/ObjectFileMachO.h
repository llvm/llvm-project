//===-- ObjectFileMachO.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ObjectFileMachO_h_
#define liblldb_ObjectFileMachO_h_

#include "llvm/Support/MachO.h"

#include "lldb/Core/Address.h"
#include "lldb/Core/RangeMap.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Symbol/ObjectFile.h"

//----------------------------------------------------------------------
// This class needs to be hidden as eventually belongs in a plugin that
// will export the ObjectFile protocol
//----------------------------------------------------------------------
class ObjectFileMachO :
    public lldb_private::ObjectFile
{
public:
    //------------------------------------------------------------------
    // Static Functions
    //------------------------------------------------------------------
    static void
    Initialize();

    static void
    Terminate();

    static const char *
    GetPluginNameStatic();

    static const char *
    GetPluginDescriptionStatic();

    static lldb_private::ObjectFile *
    CreateInstance (const lldb::ModuleSP &module_sp,
                    lldb::DataBufferSP& dataSP,
                    const lldb_private::FileSpec* file,
                    lldb::addr_t offset,
                    lldb::addr_t length);

    static lldb_private::ObjectFile *
    CreateMemoryInstance (const lldb::ModuleSP &module_sp, 
                          lldb::DataBufferSP& data_sp, 
                          const lldb::ProcessSP &process_sp, 
                          lldb::addr_t header_addr);

    static bool
    MagicBytesMatch (lldb::DataBufferSP& dataSP, 
                     lldb::addr_t offset, 
                     lldb::addr_t length);

    //------------------------------------------------------------------
    // Member Functions
    //------------------------------------------------------------------
    ObjectFileMachO (const lldb::ModuleSP &module_sp,
                     lldb::DataBufferSP& dataSP,
                     const lldb_private::FileSpec* file,
                     lldb::addr_t offset,
                     lldb::addr_t length);

    ObjectFileMachO (const lldb::ModuleSP &module_sp,
                     lldb::DataBufferSP& dataSP,
                     const lldb::ProcessSP &process_sp,
                     lldb::addr_t header_addr);

    virtual
    ~ObjectFileMachO();

    virtual bool
    ParseHeader ();

    virtual lldb::ByteOrder
    GetByteOrder () const;
    
    virtual bool
    IsExecutable () const;

    virtual size_t
    GetAddressByteSize ()  const;

    virtual lldb::AddressClass
    GetAddressClass (lldb::addr_t file_addr);

    virtual lldb_private::Symtab *
    GetSymtab();

    virtual lldb_private::SectionList *
    GetSectionList();

    virtual void
    Dump (lldb_private::Stream *s);

    virtual bool
    GetArchitecture (lldb_private::ArchSpec &arch);

    virtual bool
    GetUUID (lldb_private::UUID* uuid);

    virtual uint32_t
    GetDependentModules (lldb_private::FileSpecList& files);

    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual const char *
    GetPluginName();

    virtual const char *
    GetShortPluginName();

    virtual uint32_t
    GetPluginVersion();

    virtual lldb_private::Address
    GetEntryPointAddress ();
    
    virtual lldb_private::Address
    GetHeaderAddress ();
    
    virtual uint32_t
    GetNumThreadContexts ();
    
    virtual lldb::RegisterContextSP
    GetThreadContextAtIndex (uint32_t idx, lldb_private::Thread &thread);

    virtual ObjectFile::Type
    CalculateType();
    
    virtual ObjectFile::Strata
    CalculateStrata();

    virtual uint32_t
    GetVersion (uint32_t *versions, uint32_t num_versions);

protected:
    llvm::MachO::mach_header m_header;
    mutable std::auto_ptr<lldb_private::SectionList> m_sections_ap;
    mutable std::auto_ptr<lldb_private::Symtab> m_symtab_ap;
    static const lldb_private::ConstString &GetSegmentNameTEXT();
    static const lldb_private::ConstString &GetSegmentNameDATA();
    static const lldb_private::ConstString &GetSegmentNameOBJC();
    static const lldb_private::ConstString &GetSegmentNameLINKEDIT();
    static const lldb_private::ConstString &GetSectionNameEHFrame();

    llvm::MachO::dysymtab_command m_dysymtab;
    std::vector<llvm::MachO::segment_command_64> m_mach_segments;
    std::vector<llvm::MachO::section_64> m_mach_sections;
    typedef lldb_private::RangeArray<uint32_t, uint32_t, 1> FileRangeArray;
    lldb_private::Address  m_entry_point_address;
    FileRangeArray m_thread_context_offsets;
    bool m_thread_context_offsets_valid;

    size_t
    ParseSections ();

    size_t
    ParseSymtab (bool minimize);

};

#endif  // liblldb_ObjectFileMachO_h_
