//===-- ObjectFileMachO.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MachO.h"

#include "ObjectFileMachO.h"

#include "lldb/lldb-private-log.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/DataBuffer.h"
#include "lldb/Core/FileSpecList.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/RangeMap.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/Timer.h"
#include "lldb/Core/UUID.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Symbol/ClangNamespaceDecl.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "Plugins/Process/Utility/RegisterContextDarwin_arm.h"
#include "Plugins/Process/Utility/RegisterContextDarwin_i386.h"
#include "Plugins/Process/Utility/RegisterContextDarwin_x86_64.h"

using namespace lldb;
using namespace lldb_private;
using namespace llvm::MachO;

class RegisterContextDarwin_x86_64_Mach : public RegisterContextDarwin_x86_64 
{
public:
    RegisterContextDarwin_x86_64_Mach (lldb_private::Thread &thread, const DataExtractor &data) :
        RegisterContextDarwin_x86_64 (thread, 0)
    {
        SetRegisterDataFrom_LC_THREAD (data);
    }

    virtual void
    InvalidateAllRegisters ()
    {
        // Do nothing... registers are always valid...
    }

    void
    SetRegisterDataFrom_LC_THREAD (const DataExtractor &data)
    {
        uint32_t offset = 0;
        SetError (GPRRegSet, Read, -1);
        SetError (FPURegSet, Read, -1);
        SetError (EXCRegSet, Read, -1);
        bool done = false;
        
        while (!done)
        {
            int flavor = data.GetU32 (&offset);
            if (flavor == 0)
                done = true;
            else
            {
                uint32_t i;
                uint32_t count = data.GetU32 (&offset);
                switch (flavor)
                {
                    case GPRRegSet:
                        for (i=0; i<count; ++i)
                            (&gpr.rax)[i] = data.GetU64(&offset);
                        SetError (GPRRegSet, Read, 0);
                        done = true;
                        
                        break;
                    case FPURegSet:
                        // TODO: fill in FPU regs....
                        //SetError (FPURegSet, Read, -1);
                        done = true;
                        
                        break;
                    case EXCRegSet:
                        exc.trapno = data.GetU32(&offset);
                        exc.err = data.GetU32(&offset);
                        exc.faultvaddr = data.GetU64(&offset);
                        SetError (EXCRegSet, Read, 0);
                        done = true;
                        break;
                    case 7:
                    case 8:
                    case 9:
                        // fancy flavors that encapsulate of the the above
                        // falvors...
                        break;
                        
                    default:
                        done = true;
                        break;
                }
            }
        }
    }
protected:
    virtual int
    DoReadGPR (lldb::tid_t tid, int flavor, GPR &gpr)
    {
        return 0;
    }
    
    virtual int
    DoReadFPU (lldb::tid_t tid, int flavor, FPU &fpu)
    {
        return 0;
    }
    
    virtual int
    DoReadEXC (lldb::tid_t tid, int flavor, EXC &exc)
    {
        return 0;
    }
    
    virtual int
    DoWriteGPR (lldb::tid_t tid, int flavor, const GPR &gpr)
    {
        return 0;
    }
    
    virtual int
    DoWriteFPU (lldb::tid_t tid, int flavor, const FPU &fpu)
    {
        return 0;
    }
    
    virtual int
    DoWriteEXC (lldb::tid_t tid, int flavor, const EXC &exc)
    {
        return 0;
    }
};


class RegisterContextDarwin_i386_Mach : public RegisterContextDarwin_i386 
{
public:
    RegisterContextDarwin_i386_Mach (lldb_private::Thread &thread, const DataExtractor &data) :
    RegisterContextDarwin_i386 (thread, 0)
    {
        SetRegisterDataFrom_LC_THREAD (data);
    }
    
    virtual void
    InvalidateAllRegisters ()
    {
        // Do nothing... registers are always valid...
    }
    
    void
    SetRegisterDataFrom_LC_THREAD (const DataExtractor &data)
    {
        uint32_t offset = 0;
        SetError (GPRRegSet, Read, -1);
        SetError (FPURegSet, Read, -1);
        SetError (EXCRegSet, Read, -1);
        bool done = false;
        
        while (!done)
        {
            int flavor = data.GetU32 (&offset);
            if (flavor == 0)
                done = true;
            else
            {
                uint32_t i;
                uint32_t count = data.GetU32 (&offset);
                switch (flavor)
                {
                    case GPRRegSet:
                        for (i=0; i<count; ++i)
                            (&gpr.eax)[i] = data.GetU32(&offset);
                        SetError (GPRRegSet, Read, 0);
                        done = true;

                        break;
                    case FPURegSet:
                        // TODO: fill in FPU regs....
                        //SetError (FPURegSet, Read, -1);
                        done = true;

                        break;
                    case EXCRegSet:
                        exc.trapno = data.GetU32(&offset);
                        exc.err = data.GetU32(&offset);
                        exc.faultvaddr = data.GetU32(&offset);
                        SetError (EXCRegSet, Read, 0);
                        done = true;
                        break;
                    case 7:
                    case 8:
                    case 9:
                        // fancy flavors that encapsulate of the the above
                        // falvors...
                        break;
                        
                    default:
                        done = true;
                        break;
                }
            }
        }
    }
protected:
    virtual int
    DoReadGPR (lldb::tid_t tid, int flavor, GPR &gpr)
    {
        return 0;
    }
    
    virtual int
    DoReadFPU (lldb::tid_t tid, int flavor, FPU &fpu)
    {
        return 0;
    }
    
    virtual int
    DoReadEXC (lldb::tid_t tid, int flavor, EXC &exc)
    {
        return 0;
    }
    
    virtual int
    DoWriteGPR (lldb::tid_t tid, int flavor, const GPR &gpr)
    {
        return 0;
    }
    
    virtual int
    DoWriteFPU (lldb::tid_t tid, int flavor, const FPU &fpu)
    {
        return 0;
    }
    
    virtual int
    DoWriteEXC (lldb::tid_t tid, int flavor, const EXC &exc)
    {
        return 0;
    }
};

class RegisterContextDarwin_arm_Mach : public RegisterContextDarwin_arm 
{
public:
    RegisterContextDarwin_arm_Mach (lldb_private::Thread &thread, const DataExtractor &data) :
        RegisterContextDarwin_arm (thread, 0)
    {
        SetRegisterDataFrom_LC_THREAD (data);
    }
    
    virtual void
    InvalidateAllRegisters ()
    {
        // Do nothing... registers are always valid...
    }
    
    void
    SetRegisterDataFrom_LC_THREAD (const DataExtractor &data)
    {
        uint32_t offset = 0;
        SetError (GPRRegSet, Read, -1);
        SetError (FPURegSet, Read, -1);
        SetError (EXCRegSet, Read, -1);
        int flavor = data.GetU32 (&offset);
        uint32_t count = data.GetU32 (&offset);
        switch (flavor)
        {
            case GPRRegSet:
                for (uint32_t i=0; i<count; ++i)
                    gpr.r[i] = data.GetU32(&offset);
                SetError (GPRRegSet, Read, 0);
                break;
            case FPURegSet:
                // TODO: fill in FPU regs....
                //SetError (FPURegSet, Read, -1);
                break;
            case EXCRegSet:
                exc.exception = data.GetU32(&offset);
                exc.fsr = data.GetU32(&offset);
                exc.far = data.GetU32(&offset);
                SetError (EXCRegSet, Read, 0);
                break;
        }
    }
protected:
    virtual int
    DoReadGPR (lldb::tid_t tid, int flavor, GPR &gpr)
    {
        return 0;
    }
    
    virtual int
    DoReadFPU (lldb::tid_t tid, int flavor, FPU &fpu)
    {
        return 0;
    }
    
    virtual int
    DoReadEXC (lldb::tid_t tid, int flavor, EXC &exc)
    {
        return 0;
    }

    virtual int
    DoReadDBG (lldb::tid_t tid, int flavor, DBG &dbg)
    {
        return -1;
    }
    
    virtual int
    DoWriteGPR (lldb::tid_t tid, int flavor, const GPR &gpr)
    {
        return 0;
    }
    
    virtual int
    DoWriteFPU (lldb::tid_t tid, int flavor, const FPU &fpu)
    {
        return 0;
    }
    
    virtual int
    DoWriteEXC (lldb::tid_t tid, int flavor, const EXC &exc)
    {
        return 0;
    }
    
    virtual int
    DoWriteDBG (lldb::tid_t tid, int flavor, const DBG &dbg)
    {
        return -1;
    }
};

#define MACHO_NLIST_ARM_SYMBOL_IS_THUMB 0x0008

void
ObjectFileMachO::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   GetPluginDescriptionStatic(),
                                   CreateInstance,
                                   CreateMemoryInstance);
}

void
ObjectFileMachO::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}


const char *
ObjectFileMachO::GetPluginNameStatic()
{
    return "object-file.mach-o";
}

const char *
ObjectFileMachO::GetPluginDescriptionStatic()
{
    return "Mach-o object file reader (32 and 64 bit)";
}


ObjectFile *
ObjectFileMachO::CreateInstance (const lldb::ModuleSP &module_sp, DataBufferSP& data_sp, const FileSpec* file, addr_t offset, addr_t length)
{
    if (ObjectFileMachO::MagicBytesMatch(data_sp, offset, length))
    {
        std::auto_ptr<ObjectFile> objfile_ap(new ObjectFileMachO (module_sp, data_sp, file, offset, length));
        if (objfile_ap.get() && objfile_ap->ParseHeader())
            return objfile_ap.release();
    }
    return NULL;
}

ObjectFile *
ObjectFileMachO::CreateMemoryInstance (const lldb::ModuleSP &module_sp, 
                                       DataBufferSP& data_sp, 
                                       const ProcessSP &process_sp, 
                                       lldb::addr_t header_addr)
{
    if (ObjectFileMachO::MagicBytesMatch(data_sp, 0, data_sp->GetByteSize()))
    {
        std::auto_ptr<ObjectFile> objfile_ap(new ObjectFileMachO (module_sp, data_sp, process_sp, header_addr));
        if (objfile_ap.get() && objfile_ap->ParseHeader())
            return objfile_ap.release();
    }
    return NULL;    
}


const ConstString &
ObjectFileMachO::GetSegmentNameTEXT()
{
    static ConstString g_segment_name_TEXT ("__TEXT");
    return g_segment_name_TEXT;
}

const ConstString &
ObjectFileMachO::GetSegmentNameDATA()
{
    static ConstString g_segment_name_DATA ("__DATA");
    return g_segment_name_DATA;
}

const ConstString &
ObjectFileMachO::GetSegmentNameOBJC()
{
    static ConstString g_segment_name_OBJC ("__OBJC");
    return g_segment_name_OBJC;
}

const ConstString &
ObjectFileMachO::GetSegmentNameLINKEDIT()
{
    static ConstString g_section_name_LINKEDIT ("__LINKEDIT");
    return g_section_name_LINKEDIT;
}

const ConstString &
ObjectFileMachO::GetSectionNameEHFrame()
{
    static ConstString g_section_name_eh_frame ("__eh_frame");
    return g_section_name_eh_frame;
}



static uint32_t
MachHeaderSizeFromMagic(uint32_t magic)
{
    switch (magic)
    {
    case HeaderMagic32:
    case HeaderMagic32Swapped:
        return sizeof(struct mach_header);

    case HeaderMagic64:
    case HeaderMagic64Swapped:
        return sizeof(struct mach_header_64);
        break;

    default:
        break;
    }
    return 0;
}


bool
ObjectFileMachO::MagicBytesMatch (DataBufferSP& data_sp, 
                                  lldb::addr_t data_offset, 
                                  lldb::addr_t data_length)
{
    DataExtractor data;
    data.SetData (data_sp, data_offset, data_length);
    uint32_t offset = 0;
    uint32_t magic = data.GetU32(&offset);
    return MachHeaderSizeFromMagic(magic) != 0;
}


ObjectFileMachO::ObjectFileMachO(const lldb::ModuleSP &module_sp, DataBufferSP& data_sp, const FileSpec* file, addr_t offset, addr_t length) :
    ObjectFile(module_sp, file, offset, length, data_sp),
    m_sections_ap(),
    m_symtab_ap(),
    m_mach_segments(),
    m_mach_sections(),
    m_entry_point_address(),
    m_thread_context_offsets(),
    m_thread_context_offsets_valid(false)
{
    ::memset (&m_header, 0, sizeof(m_header));
    ::memset (&m_dysymtab, 0, sizeof(m_dysymtab));
}

ObjectFileMachO::ObjectFileMachO (const lldb::ModuleSP &module_sp,
                                  lldb::DataBufferSP& header_data_sp,
                                  const lldb::ProcessSP &process_sp,
                                  lldb::addr_t header_addr) :
    ObjectFile(module_sp, process_sp, header_addr, header_data_sp),
    m_sections_ap(),
    m_symtab_ap(),
    m_mach_segments(),
    m_mach_sections(),
    m_entry_point_address(),
    m_thread_context_offsets(),
    m_thread_context_offsets_valid(false)
{
    ::memset (&m_header, 0, sizeof(m_header));
    ::memset (&m_dysymtab, 0, sizeof(m_dysymtab));
}

ObjectFileMachO::~ObjectFileMachO()
{
}


bool
ObjectFileMachO::ParseHeader ()
{
    ModuleSP module_sp(GetModule());
    if (module_sp)
    {
        lldb_private::Mutex::Locker locker(module_sp->GetMutex());
        bool can_parse = false;
        uint32_t offset = 0;
        m_data.SetByteOrder (lldb::endian::InlHostByteOrder());
        // Leave magic in the original byte order
        m_header.magic = m_data.GetU32(&offset);
        switch (m_header.magic)
        {
        case HeaderMagic32:
            m_data.SetByteOrder (lldb::endian::InlHostByteOrder());
            m_data.SetAddressByteSize(4);
            can_parse = true;
            break;

        case HeaderMagic64:
            m_data.SetByteOrder (lldb::endian::InlHostByteOrder());
            m_data.SetAddressByteSize(8);
            can_parse = true;
            break;

        case HeaderMagic32Swapped:
            m_data.SetByteOrder(lldb::endian::InlHostByteOrder() == eByteOrderBig ? eByteOrderLittle : eByteOrderBig);
            m_data.SetAddressByteSize(4);
            can_parse = true;
            break;

        case HeaderMagic64Swapped:
            m_data.SetByteOrder(lldb::endian::InlHostByteOrder() == eByteOrderBig ? eByteOrderLittle : eByteOrderBig);
            m_data.SetAddressByteSize(8);
            can_parse = true;
            break;

        default:
            break;
        }

        if (can_parse)
        {
            m_data.GetU32(&offset, &m_header.cputype, 6);

            ArchSpec mach_arch(eArchTypeMachO, m_header.cputype, m_header.cpusubtype);
            
            if (SetModulesArchitecture (mach_arch))
            {
                const size_t header_and_lc_size = m_header.sizeofcmds + MachHeaderSizeFromMagic(m_header.magic);
                if (m_data.GetByteSize() < header_and_lc_size)
                {
                    DataBufferSP data_sp;
                    ProcessSP process_sp (m_process_wp.lock());
                    if (process_sp)
                    {
                        data_sp = ReadMemory (process_sp, m_offset, header_and_lc_size);
                    }
                    else
                    {
                        // Read in all only the load command data from the file on disk
                        data_sp = m_file.ReadFileContents(m_offset, header_and_lc_size);
                        if (data_sp->GetByteSize() != header_and_lc_size)
                            return false;
                    }
                    if (data_sp)
                        m_data.SetData (data_sp);
                }
            }
            return true;
        }
        else
        {
            memset(&m_header, 0, sizeof(struct mach_header));
        }
    }
    return false;
}


ByteOrder
ObjectFileMachO::GetByteOrder () const
{
    return m_data.GetByteOrder ();
}

bool
ObjectFileMachO::IsExecutable() const
{
    return m_header.filetype == HeaderFileTypeExecutable;
}

size_t
ObjectFileMachO::GetAddressByteSize () const
{
    return m_data.GetAddressByteSize ();
}

AddressClass
ObjectFileMachO::GetAddressClass (lldb::addr_t file_addr)
{
    Symtab *symtab = GetSymtab();
    if (symtab)
    {
        Symbol *symbol = symtab->FindSymbolContainingFileAddress(file_addr);
        if (symbol)
        {
            if (symbol->ValueIsAddress())
            {
                SectionSP section_sp (symbol->GetAddress().GetSection());
                if (section_sp)
                {
                    const SectionType section_type = section_sp->GetType();
                    switch (section_type)
                    {
                    case eSectionTypeInvalid:               return eAddressClassUnknown;
                    case eSectionTypeCode:
                        if (m_header.cputype == llvm::MachO::CPUTypeARM)
                        {
                            // For ARM we have a bit in the n_desc field of the symbol
                            // that tells us ARM/Thumb which is bit 0x0008.
                            if (symbol->GetFlags() & MACHO_NLIST_ARM_SYMBOL_IS_THUMB)
                                return eAddressClassCodeAlternateISA;
                        }
                        return eAddressClassCode;

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
                    
            case eSymbolTypeCode:
            case eSymbolTypeTrampoline:
                if (m_header.cputype == llvm::MachO::CPUTypeARM)
                {
                    // For ARM we have a bit in the n_desc field of the symbol
                    // that tells us ARM/Thumb which is bit 0x0008.
                    if (symbol->GetFlags() & MACHO_NLIST_ARM_SYMBOL_IS_THUMB)
                        return eAddressClassCodeAlternateISA;
                }
                return eAddressClassCode;

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

Symtab *
ObjectFileMachO::GetSymtab()
{
    ModuleSP module_sp(GetModule());
    if (module_sp)
    {
        lldb_private::Mutex::Locker locker(module_sp->GetMutex());
        if (m_symtab_ap.get() == NULL)
        {
            m_symtab_ap.reset(new Symtab(this));
            Mutex::Locker symtab_locker (m_symtab_ap->GetMutex());
            ParseSymtab (true);
            m_symtab_ap->Finalize ();
        }
    }
    return m_symtab_ap.get();
}


SectionList *
ObjectFileMachO::GetSectionList()
{
    ModuleSP module_sp(GetModule());
    if (module_sp)
    {
        lldb_private::Mutex::Locker locker(module_sp->GetMutex());
        if (m_sections_ap.get() == NULL)
        {
            m_sections_ap.reset(new SectionList());
            ParseSections();
        }
    }
    return m_sections_ap.get();
}


size_t
ObjectFileMachO::ParseSections ()
{
    lldb::user_id_t segID = 0;
    lldb::user_id_t sectID = 0;
    uint32_t offset = MachHeaderSizeFromMagic(m_header.magic);
    uint32_t i;
    const bool is_core = GetType() == eTypeCoreFile;
    //bool dump_sections = false;
    ModuleSP module_sp (GetModule());
    // First look up any LC_ENCRYPTION_INFO load commands
    typedef RangeArray<uint32_t, uint32_t, 8> EncryptedFileRanges;
    EncryptedFileRanges encrypted_file_ranges;
    encryption_info_command encryption_cmd;
    for (i=0; i<m_header.ncmds; ++i)
    {
        const uint32_t load_cmd_offset = offset;
        if (m_data.GetU32(&offset, &encryption_cmd, 2) == NULL)
            break;
        
        if (encryption_cmd.cmd == LoadCommandEncryptionInfo)
        {
            if (m_data.GetU32(&offset, &encryption_cmd.cryptoff, 3))
            {
                if (encryption_cmd.cryptid != 0)
                {
                    EncryptedFileRanges::Entry entry;
                    entry.SetRangeBase(encryption_cmd.cryptoff);
                    entry.SetByteSize(encryption_cmd.cryptsize);
                    encrypted_file_ranges.Append(entry);
                }
            }
        }
        offset = load_cmd_offset + encryption_cmd.cmdsize;
    }

    offset = MachHeaderSizeFromMagic(m_header.magic);

    struct segment_command_64 load_cmd;
    for (i=0; i<m_header.ncmds; ++i)
    {
        const uint32_t load_cmd_offset = offset;
        if (m_data.GetU32(&offset, &load_cmd, 2) == NULL)
            break;

        if (load_cmd.cmd == LoadCommandSegment32 || load_cmd.cmd == LoadCommandSegment64)
        {
            if (m_data.GetU8(&offset, (uint8_t*)load_cmd.segname, 16))
            {
                load_cmd.vmaddr = m_data.GetAddress(&offset);
                load_cmd.vmsize = m_data.GetAddress(&offset);
                load_cmd.fileoff = m_data.GetAddress(&offset);
                load_cmd.filesize = m_data.GetAddress(&offset);
                if (m_data.GetU32(&offset, &load_cmd.maxprot, 4))
                {
                    
                    const bool segment_is_encrypted = (load_cmd.flags & SegmentCommandFlagBitProtectedVersion1) != 0;

                    // Keep a list of mach segments around in case we need to
                    // get at data that isn't stored in the abstracted Sections.
                    m_mach_segments.push_back (load_cmd);

                    ConstString segment_name (load_cmd.segname, std::min<int>(strlen(load_cmd.segname), sizeof(load_cmd.segname)));
                    // Use a segment ID of the segment index shifted left by 8 so they
                    // never conflict with any of the sections.
                    SectionSP segment_sp;
                    if (segment_name || is_core)
                    {
                        segment_sp.reset(new Section (module_sp,              // Module to which this section belongs
                                                      ++segID << 8,           // Section ID is the 1 based segment index shifted right by 8 bits as not to collide with any of the 256 section IDs that are possible
                                                      segment_name,           // Name of this section
                                                      eSectionTypeContainer,  // This section is a container of other sections.
                                                      load_cmd.vmaddr,        // File VM address == addresses as they are found in the object file
                                                      load_cmd.vmsize,        // VM size in bytes of this section
                                                      load_cmd.fileoff,       // Offset to the data for this section in the file
                                                      load_cmd.filesize,      // Size in bytes of this section as found in the the file
                                                      load_cmd.flags));       // Flags for this section

                        segment_sp->SetIsEncrypted (segment_is_encrypted);
                        m_sections_ap->AddSection(segment_sp);
                    }

                    struct section_64 sect64;
                    ::memset (&sect64, 0, sizeof(sect64));
                    // Push a section into our mach sections for the section at
                    // index zero (NListSectionNoSection) if we don't have any 
                    // mach sections yet...
                    if (m_mach_sections.empty())
                        m_mach_sections.push_back(sect64);
                    uint32_t segment_sect_idx;
                    const lldb::user_id_t first_segment_sectID = sectID + 1;


                    const uint32_t num_u32s = load_cmd.cmd == LoadCommandSegment32 ? 7 : 8;
                    for (segment_sect_idx=0; segment_sect_idx<load_cmd.nsects; ++segment_sect_idx)
                    {
                        if (m_data.GetU8(&offset, (uint8_t*)sect64.sectname, sizeof(sect64.sectname)) == NULL)
                            break;
                        if (m_data.GetU8(&offset, (uint8_t*)sect64.segname, sizeof(sect64.segname)) == NULL)
                            break;
                        sect64.addr = m_data.GetAddress(&offset);
                        sect64.size = m_data.GetAddress(&offset);

                        if (m_data.GetU32(&offset, &sect64.offset, num_u32s) == NULL)
                            break;

                        // Keep a list of mach sections around in case we need to
                        // get at data that isn't stored in the abstracted Sections.
                        m_mach_sections.push_back (sect64);

                        ConstString section_name (sect64.sectname, std::min<size_t>(strlen(sect64.sectname), sizeof(sect64.sectname)));
                        if (!segment_name)
                        {
                            // We have a segment with no name so we need to conjure up
                            // segments that correspond to the section's segname if there
                            // isn't already such a section. If there is such a section,
                            // we resize the section so that it spans all sections.
                            // We also mark these sections as fake so address matches don't
                            // hit if they land in the gaps between the child sections.
                            segment_name.SetTrimmedCStringWithLength(sect64.segname, sizeof(sect64.segname));
                            segment_sp = m_sections_ap->FindSectionByName (segment_name);
                            if (segment_sp.get())
                            {
                                Section *segment = segment_sp.get();
                                // Grow the section size as needed.
                                const lldb::addr_t sect64_min_addr = sect64.addr;
                                const lldb::addr_t sect64_max_addr = sect64_min_addr + sect64.size;
                                const lldb::addr_t curr_seg_byte_size = segment->GetByteSize();
                                const lldb::addr_t curr_seg_min_addr = segment->GetFileAddress();
                                const lldb::addr_t curr_seg_max_addr = curr_seg_min_addr + curr_seg_byte_size;
                                if (sect64_min_addr >= curr_seg_min_addr)
                                {
                                    const lldb::addr_t new_seg_byte_size = sect64_max_addr - curr_seg_min_addr;
                                    // Only grow the section size if needed
                                    if (new_seg_byte_size > curr_seg_byte_size)
                                        segment->SetByteSize (new_seg_byte_size);
                                }
                                else
                                {
                                    // We need to change the base address of the segment and
                                    // adjust the child section offsets for all existing children.
                                    const lldb::addr_t slide_amount = sect64_min_addr - curr_seg_min_addr;
                                    segment->Slide(slide_amount, false);
                                    segment->GetChildren().Slide(-slide_amount, false);
                                    segment->SetByteSize (curr_seg_max_addr - sect64_min_addr);
                                }

                                // Grow the section size as needed.
                                if (sect64.offset)
                                {
                                    const lldb::addr_t segment_min_file_offset = segment->GetFileOffset();
                                    const lldb::addr_t segment_max_file_offset = segment_min_file_offset + segment->GetFileSize();

                                    const lldb::addr_t section_min_file_offset = sect64.offset;
                                    const lldb::addr_t section_max_file_offset = section_min_file_offset + sect64.size;
                                    const lldb::addr_t new_file_offset = std::min (section_min_file_offset, segment_min_file_offset);
                                    const lldb::addr_t new_file_size = std::max (section_max_file_offset, segment_max_file_offset) - new_file_offset;
                                    segment->SetFileOffset (new_file_offset);
                                    segment->SetFileSize (new_file_size);
                                }
                            }
                            else
                            {
                                // Create a fake section for the section's named segment
                                segment_sp.reset(new Section (segment_sp,            // Parent section
                                                              module_sp,           // Module to which this section belongs
                                                              ++segID << 8,          // Section ID is the 1 based segment index shifted right by 8 bits as not to collide with any of the 256 section IDs that are possible
                                                              segment_name,          // Name of this section
                                                              eSectionTypeContainer, // This section is a container of other sections.
                                                              sect64.addr,           // File VM address == addresses as they are found in the object file
                                                              sect64.size,           // VM size in bytes of this section
                                                              sect64.offset,         // Offset to the data for this section in the file
                                                              sect64.offset ? sect64.size : 0,        // Size in bytes of this section as found in the the file
                                                              load_cmd.flags));      // Flags for this section
                                segment_sp->SetIsFake(true);
                                m_sections_ap->AddSection(segment_sp);
                                segment_sp->SetIsEncrypted (segment_is_encrypted);
                            }
                        }
                        assert (segment_sp.get());

                        uint32_t mach_sect_type = sect64.flags & SectionFlagMaskSectionType;
                        static ConstString g_sect_name_objc_data ("__objc_data");
                        static ConstString g_sect_name_objc_msgrefs ("__objc_msgrefs");
                        static ConstString g_sect_name_objc_selrefs ("__objc_selrefs");
                        static ConstString g_sect_name_objc_classrefs ("__objc_classrefs");
                        static ConstString g_sect_name_objc_superrefs ("__objc_superrefs");
                        static ConstString g_sect_name_objc_const ("__objc_const");
                        static ConstString g_sect_name_objc_classlist ("__objc_classlist");
                        static ConstString g_sect_name_cfstring ("__cfstring");

                        static ConstString g_sect_name_dwarf_debug_abbrev ("__debug_abbrev");
                        static ConstString g_sect_name_dwarf_debug_aranges ("__debug_aranges");
                        static ConstString g_sect_name_dwarf_debug_frame ("__debug_frame");
                        static ConstString g_sect_name_dwarf_debug_info ("__debug_info");
                        static ConstString g_sect_name_dwarf_debug_line ("__debug_line");
                        static ConstString g_sect_name_dwarf_debug_loc ("__debug_loc");
                        static ConstString g_sect_name_dwarf_debug_macinfo ("__debug_macinfo");
                        static ConstString g_sect_name_dwarf_debug_pubnames ("__debug_pubnames");
                        static ConstString g_sect_name_dwarf_debug_pubtypes ("__debug_pubtypes");
                        static ConstString g_sect_name_dwarf_debug_ranges ("__debug_ranges");
                        static ConstString g_sect_name_dwarf_debug_str ("__debug_str");
                        static ConstString g_sect_name_dwarf_apple_names ("__apple_names");
                        static ConstString g_sect_name_dwarf_apple_types ("__apple_types");
                        static ConstString g_sect_name_dwarf_apple_namespaces ("__apple_namespac");
                        static ConstString g_sect_name_dwarf_apple_objc ("__apple_objc");
                        static ConstString g_sect_name_eh_frame ("__eh_frame");
                        static ConstString g_sect_name_DATA ("__DATA");
                        static ConstString g_sect_name_TEXT ("__TEXT");

                        SectionType sect_type = eSectionTypeOther;

                        if (section_name == g_sect_name_dwarf_debug_abbrev)
                            sect_type = eSectionTypeDWARFDebugAbbrev;
                        else if (section_name == g_sect_name_dwarf_debug_aranges)
                            sect_type = eSectionTypeDWARFDebugAranges;
                        else if (section_name == g_sect_name_dwarf_debug_frame)
                            sect_type = eSectionTypeDWARFDebugFrame;
                        else if (section_name == g_sect_name_dwarf_debug_info)
                            sect_type = eSectionTypeDWARFDebugInfo;
                        else if (section_name == g_sect_name_dwarf_debug_line)
                            sect_type = eSectionTypeDWARFDebugLine;
                        else if (section_name == g_sect_name_dwarf_debug_loc)
                            sect_type = eSectionTypeDWARFDebugLoc;
                        else if (section_name == g_sect_name_dwarf_debug_macinfo)
                            sect_type = eSectionTypeDWARFDebugMacInfo;
                        else if (section_name == g_sect_name_dwarf_debug_pubnames)
                            sect_type = eSectionTypeDWARFDebugPubNames;
                        else if (section_name == g_sect_name_dwarf_debug_pubtypes)
                            sect_type = eSectionTypeDWARFDebugPubTypes;
                        else if (section_name == g_sect_name_dwarf_debug_ranges)
                            sect_type = eSectionTypeDWARFDebugRanges;
                        else if (section_name == g_sect_name_dwarf_debug_str)
                            sect_type = eSectionTypeDWARFDebugStr;
                        else if (section_name == g_sect_name_dwarf_apple_names)
                            sect_type = eSectionTypeDWARFAppleNames;
                        else if (section_name == g_sect_name_dwarf_apple_types)
                            sect_type = eSectionTypeDWARFAppleTypes;
                        else if (section_name == g_sect_name_dwarf_apple_namespaces)
                            sect_type = eSectionTypeDWARFAppleNamespaces;
                        else if (section_name == g_sect_name_dwarf_apple_objc)
                            sect_type = eSectionTypeDWARFAppleObjC;
                        else if (section_name == g_sect_name_objc_selrefs)
                            sect_type = eSectionTypeDataCStringPointers;
                        else if (section_name == g_sect_name_objc_msgrefs)
                            sect_type = eSectionTypeDataObjCMessageRefs;
                        else if (section_name == g_sect_name_eh_frame)
                            sect_type = eSectionTypeEHFrame;
                        else if (section_name == g_sect_name_cfstring)
                            sect_type = eSectionTypeDataObjCCFStrings;
                        else if (section_name == g_sect_name_objc_data ||
                                 section_name == g_sect_name_objc_classrefs ||
                                 section_name == g_sect_name_objc_superrefs ||
                                 section_name == g_sect_name_objc_const ||
                                 section_name == g_sect_name_objc_classlist)
                        {
                            sect_type = eSectionTypeDataPointers;
                        }

                        if (sect_type == eSectionTypeOther)
                        {
                            switch (mach_sect_type)
                            {
                            // TODO: categorize sections by other flags for regular sections
                            case SectionTypeRegular:
                                if (segment_sp->GetName() == g_sect_name_TEXT)
                                    sect_type = eSectionTypeCode; 
                                else if (segment_sp->GetName() == g_sect_name_DATA)
                                    sect_type = eSectionTypeData; 
                                else
                                    sect_type = eSectionTypeOther; 
                                break;
                            case SectionTypeZeroFill:                   sect_type = eSectionTypeZeroFill; break;
                            case SectionTypeCStringLiterals:            sect_type = eSectionTypeDataCString;    break; // section with only literal C strings
                            case SectionType4ByteLiterals:              sect_type = eSectionTypeData4;    break; // section with only 4 byte literals
                            case SectionType8ByteLiterals:              sect_type = eSectionTypeData8;    break; // section with only 8 byte literals
                            case SectionTypeLiteralPointers:            sect_type = eSectionTypeDataPointers;  break; // section with only pointers to literals
                            case SectionTypeNonLazySymbolPointers:      sect_type = eSectionTypeDataPointers;  break; // section with only non-lazy symbol pointers
                            case SectionTypeLazySymbolPointers:         sect_type = eSectionTypeDataPointers;  break; // section with only lazy symbol pointers
                            case SectionTypeSymbolStubs:                sect_type = eSectionTypeCode;  break; // section with only symbol stubs, byte size of stub in the reserved2 field
                            case SectionTypeModuleInitFunctionPointers: sect_type = eSectionTypeDataPointers;    break; // section with only function pointers for initialization
                            case SectionTypeModuleTermFunctionPointers: sect_type = eSectionTypeDataPointers; break; // section with only function pointers for termination
                            case SectionTypeCoalesced:                  sect_type = eSectionTypeOther; break;
                            case SectionTypeZeroFillLarge:              sect_type = eSectionTypeZeroFill; break;
                            case SectionTypeInterposing:                sect_type = eSectionTypeCode;  break; // section with only pairs of function pointers for interposing
                            case SectionType16ByteLiterals:             sect_type = eSectionTypeData16; break; // section with only 16 byte literals
                            case SectionTypeDTraceObjectFormat:         sect_type = eSectionTypeDebug; break;
                            case SectionTypeLazyDylibSymbolPointers:    sect_type = eSectionTypeDataPointers;  break;
                            default: break;
                            }
                        }

                        SectionSP section_sp(new Section (segment_sp,
                                                          module_sp,
                                                          ++sectID,
                                                          section_name,
                                                          sect_type,
                                                          sect64.addr - segment_sp->GetFileAddress(),
                                                          sect64.size,
                                                          sect64.offset,
                                                          sect64.offset == 0 ? 0 : sect64.size,
                                                          sect64.flags));
                        // Set the section to be encrypted to match the segment
                        
                        bool section_is_encrypted = false;
                        if (!segment_is_encrypted && load_cmd.filesize != 0)
                            section_is_encrypted = encrypted_file_ranges.FindEntryThatContains(sect64.offset) != NULL;

                        section_sp->SetIsEncrypted (segment_is_encrypted || section_is_encrypted);
                        segment_sp->GetChildren().AddSection(section_sp);

                        if (segment_sp->IsFake())
                        {
                            segment_sp.reset();
                            segment_name.Clear();
                        }
                    }
                    if (segment_sp && m_header.filetype == HeaderFileTypeDSYM)
                    {
                        if (first_segment_sectID <= sectID)
                        {
                            lldb::user_id_t sect_uid;
                            for (sect_uid = first_segment_sectID; sect_uid <= sectID; ++sect_uid)
                            {
                                SectionSP curr_section_sp(segment_sp->GetChildren().FindSectionByID (sect_uid));
                                SectionSP next_section_sp;
                                if (sect_uid + 1 <= sectID)
                                    next_section_sp = segment_sp->GetChildren().FindSectionByID (sect_uid+1);

                                if (curr_section_sp.get())
                                {
                                    if (curr_section_sp->GetByteSize() == 0)
                                    {
                                        if (next_section_sp.get() != NULL)
                                            curr_section_sp->SetByteSize ( next_section_sp->GetFileAddress() - curr_section_sp->GetFileAddress() );
                                        else
                                            curr_section_sp->SetByteSize ( load_cmd.vmsize );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        else if (load_cmd.cmd == LoadCommandDynamicSymtabInfo)
        {
            m_dysymtab.cmd = load_cmd.cmd;
            m_dysymtab.cmdsize = load_cmd.cmdsize;
            m_data.GetU32 (&offset, &m_dysymtab.ilocalsym, (sizeof(m_dysymtab) / sizeof(uint32_t)) - 2);
        }

        offset = load_cmd_offset + load_cmd.cmdsize;
    }
//    if (dump_sections)
//    {
//        StreamFile s(stdout);
//        m_sections_ap->Dump(&s, true);
//    }
    return sectID;  // Return the number of sections we registered with the module
}

class MachSymtabSectionInfo
{
public:

    MachSymtabSectionInfo (SectionList *section_list) :
        m_section_list (section_list),
        m_section_infos()
    {
        // Get the number of sections down to a depth of 1 to include
        // all segments and their sections, but no other sections that
        // may be added for debug map or
        m_section_infos.resize(section_list->GetNumSections(1));
    }


    SectionSP
    GetSection (uint8_t n_sect, addr_t file_addr)
    {
        if (n_sect == 0)
            return SectionSP();
        if (n_sect < m_section_infos.size())
        {
            if (!m_section_infos[n_sect].section_sp)
            {
                SectionSP section_sp (m_section_list->FindSectionByID (n_sect));
                m_section_infos[n_sect].section_sp = section_sp;
                if (section_sp)
                {
                    m_section_infos[n_sect].vm_range.SetBaseAddress (section_sp->GetFileAddress());
                    m_section_infos[n_sect].vm_range.SetByteSize (section_sp->GetByteSize());
                }
                else
                {
                    Host::SystemLog (Host::eSystemLogError, "error: unable to find section for section %u\n", n_sect);
                }
            }
            if (m_section_infos[n_sect].vm_range.Contains(file_addr))
            {
                // Symbol is in section.
                return m_section_infos[n_sect].section_sp;
            }
            else if (m_section_infos[n_sect].vm_range.GetByteSize () == 0 &&
                     m_section_infos[n_sect].vm_range.GetBaseAddress() == file_addr)
            {
                // Symbol is in section with zero size, but has the same start
                // address as the section. This can happen with linker symbols
                // (symbols that start with the letter 'l' or 'L'.
                return m_section_infos[n_sect].section_sp;
            }
        }
        return m_section_list->FindSectionContainingFileAddress(file_addr);
    }

protected:
    struct SectionInfo
    {
        SectionInfo () :
            vm_range(),
            section_sp ()
        {
        }

        VMRange vm_range;
        SectionSP section_sp;
    };
    SectionList *m_section_list;
    std::vector<SectionInfo> m_section_infos;
};

size_t
ObjectFileMachO::ParseSymtab (bool minimize)
{
    Timer scoped_timer(__PRETTY_FUNCTION__,
                       "ObjectFileMachO::ParseSymtab () module = %s",
                       m_file.GetFilename().AsCString(""));
    ModuleSP module_sp (GetModule());
    if (!module_sp)
        return 0;

    struct symtab_command symtab_load_command = { 0, 0, 0, 0, 0, 0 };
    struct linkedit_data_command function_starts_load_command = { 0, 0, 0, 0 };
    typedef AddressDataArray<lldb::addr_t, bool, 100> FunctionStarts;
    FunctionStarts function_starts;
    uint32_t offset = MachHeaderSizeFromMagic(m_header.magic);
    uint32_t i;

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_SYMBOLS));

    for (i=0; i<m_header.ncmds; ++i)
    {
        const uint32_t cmd_offset = offset;
        // Read in the load command and load command size
        struct load_command lc;
        if (m_data.GetU32(&offset, &lc, 2) == NULL)
            break;
        // Watch for the symbol table load command
        switch (lc.cmd)
        {
        case LoadCommandSymtab:
            symtab_load_command.cmd = lc.cmd;
            symtab_load_command.cmdsize = lc.cmdsize;
            // Read in the rest of the symtab load command
            if (m_data.GetU32(&offset, &symtab_load_command.symoff, 4) == 0) // fill in symoff, nsyms, stroff, strsize fields
                return 0;
            if (symtab_load_command.symoff == 0)
            {
                if (log)
                    module_sp->LogMessage(log.get(), "LC_SYMTAB.symoff == 0");
                return 0;
            }

            if (symtab_load_command.stroff == 0)
            {
                if (log)
                    module_sp->LogMessage(log.get(), "LC_SYMTAB.stroff == 0");
                return 0;
            }
            
            if (symtab_load_command.nsyms == 0)
            {
                if (log)
                    module_sp->LogMessage(log.get(), "LC_SYMTAB.nsyms == 0");
                return 0;
            }
            
            if (symtab_load_command.strsize == 0)
            {
                if (log)
                    module_sp->LogMessage(log.get(), "LC_SYMTAB.strsize == 0");
                return 0;
            }
            break;

        case LoadCommandFunctionStarts:
            function_starts_load_command.cmd = lc.cmd;
            function_starts_load_command.cmdsize = lc.cmdsize;
            if (m_data.GetU32(&offset, &function_starts_load_command.dataoff, 2) == NULL) // fill in symoff, nsyms, stroff, strsize fields
                bzero (&function_starts_load_command, sizeof(function_starts_load_command));
            break;

        default:
            break;
        }
        offset = cmd_offset + lc.cmdsize;
    }

    if (symtab_load_command.cmd)
    {
        Symtab *symtab = m_symtab_ap.get();
        SectionList *section_list = GetSectionList();
        if (section_list == NULL)
            return 0;

        ProcessSP process_sp (m_process_wp.lock());
        Process *process = process_sp.get();

        const size_t addr_byte_size = m_data.GetAddressByteSize();
        bool bit_width_32 = addr_byte_size == 4;
        const size_t nlist_byte_size = bit_width_32 ? sizeof(struct nlist) : sizeof(struct nlist_64);

        DataExtractor nlist_data (NULL, 0, m_data.GetByteOrder(), m_data.GetAddressByteSize());
        DataExtractor strtab_data (NULL, 0, m_data.GetByteOrder(), m_data.GetAddressByteSize());
        DataExtractor function_starts_data (NULL, 0, m_data.GetByteOrder(), m_data.GetAddressByteSize());
        
        const addr_t nlist_data_byte_size = symtab_load_command.nsyms * nlist_byte_size;
        const addr_t strtab_data_byte_size = symtab_load_command.strsize;
        addr_t strtab_addr = LLDB_INVALID_ADDRESS;
        if (process)
        {
            Target &target = process->GetTarget();
            SectionSP linkedit_section_sp(section_list->FindSectionByName(GetSegmentNameLINKEDIT()));
            // Reading mach file from memory in a process or core file...

            if (linkedit_section_sp)
            {
                const addr_t linkedit_load_addr = linkedit_section_sp->GetLoadBaseAddress(&target);
                const addr_t linkedit_file_offset = linkedit_section_sp->GetFileOffset();
                const addr_t symoff_addr = linkedit_load_addr + symtab_load_command.symoff - linkedit_file_offset;
                strtab_addr = linkedit_load_addr + symtab_load_command.stroff - linkedit_file_offset;

                bool data_was_read = false;

#if defined (__APPLE__) && defined (__arm__)
                if (m_header.flags & 0x80000000u)
                {
                    // This mach-o memory file is in the dyld shared cache. If this
                    // program is not remote and this is iOS, then this process will
                    // share the same shared cache as the process we are debugging and
                    // we can read the entire __LINKEDIT from the address space in this
                    // process. This is a needed optimization that is used for local iOS
                    // debugging only since all shared libraries in the shared cache do
                    // not have corresponding files that exist in the file system of the
                    // device. They have been combined into a single file. This means we
                    // always have to load these files from memory. All of the symbol and
                    // string tables from all of the __LINKEDIT sections from the shared
                    // libraries in the shared cache have been merged into a single large
                    // symbol and string table. Reading all of this symbol and string table
                    // data across can slow down debug launch times, so we optimize this by
                    // reading the memory for the __LINKEDIT section from this process.
                    PlatformSP platform_sp (target.GetPlatform());
                    if (platform_sp && platform_sp->IsHost())
                    {
                        data_was_read = true;
                        nlist_data.SetData((void *)symoff_addr, nlist_data_byte_size, eByteOrderLittle);
                        strtab_data.SetData((void *)strtab_addr, strtab_data_byte_size, eByteOrderLittle);
                        if (function_starts_load_command.cmd)
                        {
                            const addr_t func_start_addr = linkedit_load_addr + function_starts_load_command.dataoff - linkedit_file_offset;
                            function_starts_data.SetData ((void *)func_start_addr, function_starts_load_command.datasize, eByteOrderLittle);
                        }
                    }
                }
#endif

                if (!data_was_read)
                {
                    DataBufferSP nlist_data_sp (ReadMemory (process_sp, symoff_addr, nlist_data_byte_size));
                    if (nlist_data_sp)
                        nlist_data.SetData (nlist_data_sp, 0, nlist_data_sp->GetByteSize());
                    //DataBufferSP strtab_data_sp (ReadMemory (process_sp, strtab_addr, strtab_data_byte_size));
                    //if (strtab_data_sp)
                    //    strtab_data.SetData (strtab_data_sp, 0, strtab_data_sp->GetByteSize());
                    if (function_starts_load_command.cmd)
                    {
                        const addr_t func_start_addr = linkedit_load_addr + function_starts_load_command.dataoff - linkedit_file_offset;
                        DataBufferSP func_start_data_sp (ReadMemory (process_sp, func_start_addr, function_starts_load_command.datasize));
                        if (func_start_data_sp)
                            function_starts_data.SetData (func_start_data_sp, 0, func_start_data_sp->GetByteSize());
                    }
                }
            }
        }
        else
        {
            nlist_data.SetData (m_data, 
                                symtab_load_command.symoff, 
                                nlist_data_byte_size);
            strtab_data.SetData (m_data,
                                 symtab_load_command.stroff, 
                                 strtab_data_byte_size);
            if (function_starts_load_command.cmd)
            {
                function_starts_data.SetData (m_data,
                                              function_starts_load_command.dataoff,
                                              function_starts_load_command.datasize);
            }
        }

        if (nlist_data.GetByteSize() == 0)
        {
            if (log)
                module_sp->LogMessage(log.get(), "failed to read nlist data");
            return 0;
        }


        const bool have_strtab_data = strtab_data.GetByteSize() > 0;
        if (!have_strtab_data)
        {
            if (process)
            {
                if (strtab_addr == LLDB_INVALID_ADDRESS)
                {
                    if (log)
                        module_sp->LogMessage(log.get(), "failed to locate the strtab in memory");
                    return 0;
                }
            }
            else
            {
                if (log)
                    module_sp->LogMessage(log.get(), "failed to read strtab data");
                return 0;
            }
        }

        const ConstString &g_segment_name_TEXT = GetSegmentNameTEXT();
        const ConstString &g_segment_name_DATA = GetSegmentNameDATA();
        const ConstString &g_segment_name_OBJC = GetSegmentNameOBJC();
        const ConstString &g_section_name_eh_frame = GetSectionNameEHFrame();
        SectionSP text_section_sp(section_list->FindSectionByName(g_segment_name_TEXT));
        SectionSP data_section_sp(section_list->FindSectionByName(g_segment_name_DATA));
        SectionSP objc_section_sp(section_list->FindSectionByName(g_segment_name_OBJC));
        SectionSP eh_frame_section_sp;
        if (text_section_sp.get())
            eh_frame_section_sp = text_section_sp->GetChildren().FindSectionByName (g_section_name_eh_frame);
        else
            eh_frame_section_sp = section_list->FindSectionByName (g_section_name_eh_frame);

        const bool is_arm = (m_header.cputype == llvm::MachO::CPUTypeARM);
        if (text_section_sp && function_starts_data.GetByteSize())
        {
            FunctionStarts::Entry function_start_entry;
            function_start_entry.data = false;
            uint32_t function_start_offset = 0;
            function_start_entry.addr = text_section_sp->GetFileAddress();
            uint64_t delta;
            while ((delta = function_starts_data.GetULEB128(&function_start_offset)) > 0)
            {
                // Now append the current entry
                function_start_entry.addr += delta;
                function_starts.Append(function_start_entry);
            }
        }
        
        const uint32_t function_starts_count = function_starts.GetSize();

        uint8_t TEXT_eh_frame_sectID = eh_frame_section_sp.get() ? eh_frame_section_sp->GetID() : NListSectionNoSection;

        uint32_t nlist_data_offset = 0;

        uint32_t N_SO_index = UINT32_MAX;

        MachSymtabSectionInfo section_info (section_list);
        std::vector<uint32_t> N_FUN_indexes;
        std::vector<uint32_t> N_NSYM_indexes;
        std::vector<uint32_t> N_INCL_indexes;
        std::vector<uint32_t> N_BRAC_indexes;
        std::vector<uint32_t> N_COMM_indexes;
        typedef std::map <uint64_t, uint32_t> ValueToSymbolIndexMap;
        typedef std::map <uint32_t, uint32_t> NListIndexToSymbolIndexMap;
        ValueToSymbolIndexMap N_FUN_addr_to_sym_idx;
        ValueToSymbolIndexMap N_STSYM_addr_to_sym_idx;
        // Any symbols that get merged into another will get an entry
        // in this map so we know
        NListIndexToSymbolIndexMap m_nlist_idx_to_sym_idx;
        uint32_t nlist_idx = 0;
        Symbol *symbol_ptr = NULL;

        uint32_t sym_idx = 0;
        Symbol *sym = NULL;
        uint32_t num_syms = 0;
        std::string memory_symbol_name;
        uint32_t unmapped_local_symbols_found = 0;

#if defined (__APPLE__) && defined (__arm__)

        // Some recent builds of the dyld_shared_cache (hereafter: DSC) have been optimized by moving LOCAL
        // symbols out of the memory mapped portion of the DSC. The symbol information has all been retained,
        // but it isn't available in the normal nlist data. However, there *are* duplicate entries of *some*
        // LOCAL symbols in the normal nlist data. To handle this situation correctly, we must first attempt
        // to parse any DSC unmapped symbol information. If we find any, we set a flag that tells the normal
        // nlist parser to ignore all LOCAL symbols.

        if (m_header.flags & 0x80000000u)
        {
            // Before we can start mapping the DSC, we need to make certain the target process is actually
            // using the cache we can find.

            /*
             * TODO (FIXME!)
             *
             * Consider the case of testing with a separate DSC file.
             * If we go through the normal code paths, we will give symbols for the wrong DSC, and
             * that is bad.  We need to read the target process' all_image_infos struct, and look
             * at the values of the processDetachedFromSharedRegion field. If that is set, we should skip
             * this code section.
             */

            // Next we need to determine the correct path for the dyld shared cache.

            ArchSpec header_arch(eArchTypeMachO, m_header.cputype, m_header.cpusubtype);
            char dsc_path[PATH_MAX];

            snprintf(dsc_path, sizeof(dsc_path), "%s%s%s",
                     "/System/Library/Caches/com.apple.dyld/",	/* IPHONE_DYLD_SHARED_CACHE_DIR */
                     "dyld_shared_cache_",			/* DYLD_SHARED_CACHE_BASE_NAME */
                     header_arch.GetArchitectureName());

            FileSpec dsc_filespec(dsc_path, false);

            // We need definitions of two structures in the on-disk DSC, copy them here manually
            struct lldb_copy_dyld_cache_header
            {
                char		magic[16];
                uint32_t	mappingOffset;
                uint32_t	mappingCount;
                uint32_t	imagesOffset;
                uint32_t	imagesCount;
                uint64_t	dyldBaseAddress;
                uint64_t	codeSignatureOffset;
                uint64_t	codeSignatureSize;
                uint64_t	slideInfoOffset;
                uint64_t	slideInfoSize;
                uint64_t	localSymbolsOffset;
                uint64_t	localSymbolsSize;
            };
            struct lldb_copy_dyld_cache_local_symbols_info
            {
                    uint32_t        nlistOffset;
                    uint32_t        nlistCount;
                    uint32_t        stringsOffset;
                    uint32_t        stringsSize;
                    uint32_t        entriesOffset;
                    uint32_t        entriesCount;
            };
            struct lldb_copy_dyld_cache_local_symbols_entry
            {
                    uint32_t        dylibOffset;
                    uint32_t        nlistStartIndex;
                    uint32_t        nlistCount;
            };

            /* The dyld_cache_header has a pointer to the dyld_cache_local_symbols_info structure (localSymbolsOffset).
               The dyld_cache_local_symbols_info structure gives us three things:
                 1. The start and count of the nlist records in the dyld_shared_cache file
                 2. The start and size of the strings for these nlist records
                 3. The start and count of dyld_cache_local_symbols_entry entries

               There is one dyld_cache_local_symbols_entry per dylib/framework in the dyld shared cache.
               The "dylibOffset" field is the Mach-O header of this dylib/framework in the dyld shared cache.
               The dyld_cache_local_symbols_entry also lists the start of this dylib/framework's nlist records 
               and the count of how many nlist records there are for this dylib/framework.
            */

            // Process the dsc header to find the unmapped symbols
            //
            // Save some VM space, do not map the entire cache in one shot.

            if (DataBufferSP dsc_data_sp = dsc_filespec.MemoryMapFileContents(0, sizeof(struct lldb_copy_dyld_cache_header))) 
            {
                DataExtractor dsc_header_data(dsc_data_sp, m_data.GetByteOrder(), m_data.GetAddressByteSize());

                uint32_t offset = offsetof (struct lldb_copy_dyld_cache_header, mappingOffset); 
                uint32_t mappingOffset = dsc_header_data.GetU32(&offset);

                // If the mappingOffset points to a location inside the header, we've
                // opened an old dyld shared cache, and should not proceed further.
                if (mappingOffset >= sizeof(struct lldb_copy_dyld_cache_header)) 
                {

                    offset = offsetof (struct lldb_copy_dyld_cache_header, localSymbolsOffset);
                    uint64_t localSymbolsOffset = dsc_header_data.GetU64(&offset);
                    uint64_t localSymbolsSize = dsc_header_data.GetU64(&offset);

                    if (localSymbolsOffset && localSymbolsSize) 
                    {
                        // Map the local symbols
                        if (DataBufferSP dsc_local_symbols_data_sp = dsc_filespec.MemoryMapFileContents(localSymbolsOffset, localSymbolsSize)) 
                        {
                            DataExtractor dsc_local_symbols_data(dsc_local_symbols_data_sp, m_data.GetByteOrder(), m_data.GetAddressByteSize());

                            offset = 0;

                            // Read the local_symbols_infos struct in one shot
                            struct lldb_copy_dyld_cache_local_symbols_info local_symbols_info;
                            dsc_local_symbols_data.GetU32(&offset, &local_symbols_info.nlistOffset, 6);

                            // The local_symbols_infos offsets are offsets into local symbols memory, NOT file offsets!
                            // We first need to identify the local "entry" that matches the current header.
                            // The "entry" is stored as a file offset in the dyld_shared_cache, so we need to
                            // adjust the raw m_header value by slide and 0x30000000.

                            SectionSP text_section_sp(section_list->FindSectionByName(GetSegmentNameTEXT()));

                            uint32_t header_file_offset = (text_section_sp->GetFileAddress() - 0x30000000);

                            offset = local_symbols_info.entriesOffset;
                            for (uint32_t entry_index = 0; entry_index < local_symbols_info.entriesCount; entry_index++)
                            {
                                struct lldb_copy_dyld_cache_local_symbols_entry local_symbols_entry;
                                local_symbols_entry.dylibOffset = dsc_local_symbols_data.GetU32(&offset);
                                local_symbols_entry.nlistStartIndex = dsc_local_symbols_data.GetU32(&offset);
                                local_symbols_entry.nlistCount = dsc_local_symbols_data.GetU32(&offset);

                                if (header_file_offset == local_symbols_entry.dylibOffset) 
                                {
                                    unmapped_local_symbols_found = local_symbols_entry.nlistCount;

                                    // The normal nlist code cannot correctly size the Symbols array, we need to allocate it here.
                                    sym = symtab->Resize (symtab_load_command.nsyms + m_dysymtab.nindirectsyms + unmapped_local_symbols_found - m_dysymtab.nlocalsym);
                                    num_syms = symtab->GetNumSymbols();

                                    nlist_data_offset = local_symbols_info.nlistOffset + (nlist_byte_size * local_symbols_entry.nlistStartIndex);
                                    uint32_t string_table_offset = local_symbols_info.stringsOffset;

                                    for (uint32_t nlist_index = 0; nlist_index < local_symbols_entry.nlistCount; nlist_index++) 
                                    {
                                        /////////////////////////////
                                        {
                                            struct nlist_64 nlist;
                                            if (!dsc_local_symbols_data.ValidOffsetForDataOfSize(nlist_data_offset, nlist_byte_size))
                                                break;

                                            nlist.n_strx  = dsc_local_symbols_data.GetU32_unchecked(&nlist_data_offset);
                                            nlist.n_type  = dsc_local_symbols_data.GetU8_unchecked (&nlist_data_offset);
                                            nlist.n_sect  = dsc_local_symbols_data.GetU8_unchecked (&nlist_data_offset);
                                            nlist.n_desc  = dsc_local_symbols_data.GetU16_unchecked (&nlist_data_offset);
                                            nlist.n_value = dsc_local_symbols_data.GetAddress_unchecked (&nlist_data_offset);

                                            SymbolType type = eSymbolTypeInvalid;
                                            const char *symbol_name = dsc_local_symbols_data.PeekCStr(string_table_offset + nlist.n_strx);

                                            if (symbol_name == NULL)
                                            {
                                                // No symbol should be NULL, even the symbols with no
                                                // string values should have an offset zero which points
                                                // to an empty C-string
                                                Host::SystemLog (Host::eSystemLogError,
                                                                 "error: DSC unmapped local symbol[%u] has invalid string table offset 0x%x in %s/%s, ignoring symbol\n",
                                                                 entry_index,
                                                                 nlist.n_strx,
                                                                 module_sp->GetFileSpec().GetDirectory().GetCString(),
                                                                 module_sp->GetFileSpec().GetFilename().GetCString());
                                                continue;
                                            }
                                            if (symbol_name[0] == '\0')
                                                symbol_name = NULL;

                                            const char *symbol_name_non_abi_mangled = NULL;

                                            SectionSP symbol_section;
                                            uint32_t symbol_byte_size = 0;
                                            bool add_nlist = true;
                                            bool is_debug = ((nlist.n_type & NlistMaskStab) != 0);

                                            assert (sym_idx < num_syms);

                                            sym[sym_idx].SetDebug (is_debug);

                                            if (is_debug)
                                            {
                                                switch (nlist.n_type)
                                                {
                                                    case StabGlobalSymbol:
                                                        // N_GSYM -- global symbol: name,,NO_SECT,type,0
                                                        // Sometimes the N_GSYM value contains the address.

                                                        // FIXME: In the .o files, we have a GSYM and a debug symbol for all the ObjC data.  They
                                                        // have the same address, but we want to ensure that we always find only the real symbol,
                                                        // 'cause we don't currently correctly attribute the GSYM one to the ObjCClass/Ivar/MetaClass
                                                        // symbol type.  This is a temporary hack to make sure the ObjectiveC symbols get treated
                                                        // correctly.  To do this right, we should coalesce all the GSYM & global symbols that have the
                                                        // same address.

                                                        if (symbol_name && symbol_name[0] == '_' && symbol_name[1] ==  'O'
                                                            && (strncmp (symbol_name, "_OBJC_IVAR_$_", strlen ("_OBJC_IVAR_$_")) == 0
                                                                || strncmp (symbol_name, "_OBJC_CLASS_$_", strlen ("_OBJC_CLASS_$_")) == 0
                                                                || strncmp (symbol_name, "_OBJC_METACLASS_$_", strlen ("_OBJC_METACLASS_$_")) == 0))
                                                            add_nlist = false;
                                                        else
                                                        {
                                                            sym[sym_idx].SetExternal(true);
                                                            if (nlist.n_value != 0)
                                                                symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);
                                                            type = eSymbolTypeData;
                                                        }
                                                        break;

                                                    case StabFunctionName:
                                                        // N_FNAME -- procedure name (f77 kludge): name,,NO_SECT,0,0
                                                        type = eSymbolTypeCompiler;
                                                        break;

                                                    case StabFunction:
                                                        // N_FUN -- procedure: name,,n_sect,linenumber,address
                                                        if (symbol_name)
                                                        {
                                                            type = eSymbolTypeCode;
                                                            symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);

                                                            N_FUN_addr_to_sym_idx[nlist.n_value] = sym_idx;
                                                            // We use the current number of symbols in the symbol table in lieu of
                                                            // using nlist_idx in case we ever start trimming entries out
                                                            N_FUN_indexes.push_back(sym_idx);
                                                        }
                                                        else
                                                        {
                                                            type = eSymbolTypeCompiler;

                                                            if ( !N_FUN_indexes.empty() )
                                                            {
                                                                // Copy the size of the function into the original STAB entry so we don't have
                                                                // to hunt for it later
                                                                symtab->SymbolAtIndex(N_FUN_indexes.back())->SetByteSize(nlist.n_value);
                                                                N_FUN_indexes.pop_back();
                                                                // We don't really need the end function STAB as it contains the size which
                                                                // we already placed with the original symbol, so don't add it if we want a
                                                                // minimal symbol table
                                                                if (minimize)
                                                                    add_nlist = false;
                                                            }
                                                        }
                                                        break;

                                                    case StabStaticSymbol:
                                                        // N_STSYM -- static symbol: name,,n_sect,type,address
                                                        N_STSYM_addr_to_sym_idx[nlist.n_value] = sym_idx;
                                                        symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);
                                                        type = eSymbolTypeData;
                                                        break;

                                                    case StabLocalCommon:
                                                        // N_LCSYM -- .lcomm symbol: name,,n_sect,type,address
                                                        symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);
                                                        type = eSymbolTypeCommonBlock;
                                                        break;

                                                    case StabBeginSymbol:
                                                        // N_BNSYM
                                                        // We use the current number of symbols in the symbol table in lieu of
                                                        // using nlist_idx in case we ever start trimming entries out
                                                        if (minimize)
                                                        {
                                                            // Skip these if we want minimal symbol tables
                                                            add_nlist = false;
                                                        }
                                                        else
                                                        {
                                                            symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);
                                                            N_NSYM_indexes.push_back(sym_idx);
                                                            type = eSymbolTypeScopeBegin;
                                                        }
                                                        break;

                                                    case StabEndSymbol:
                                                        // N_ENSYM
                                                        // Set the size of the N_BNSYM to the terminating index of this N_ENSYM
                                                        // so that we can always skip the entire symbol if we need to navigate
                                                        // more quickly at the source level when parsing STABS
                                                        if (minimize)
                                                        {
                                                            // Skip these if we want minimal symbol tables
                                                            add_nlist = false;
                                                        }
                                                        else
                                                        {
                                                            if ( !N_NSYM_indexes.empty() )
                                                            {
                                                                symbol_ptr = symtab->SymbolAtIndex(N_NSYM_indexes.back());
                                                                symbol_ptr->SetByteSize(sym_idx + 1);
                                                                symbol_ptr->SetSizeIsSibling(true);
                                                                N_NSYM_indexes.pop_back();
                                                            }
                                                            type = eSymbolTypeScopeEnd;
                                                        }
                                                        break;


                                                    case StabSourceFileOptions:
                                                        // N_OPT - emitted with gcc2_compiled and in gcc source
                                                        type = eSymbolTypeCompiler;
                                                        break;

                                                    case StabRegisterSymbol:
                                                        // N_RSYM - register sym: name,,NO_SECT,type,register
                                                        type = eSymbolTypeVariable;
                                                        break;

                                                    case StabSourceLine:
                                                        // N_SLINE - src line: 0,,n_sect,linenumber,address
                                                        symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);
                                                        type = eSymbolTypeLineEntry;
                                                        break;

                                                    case StabStructureType:
                                                        // N_SSYM - structure elt: name,,NO_SECT,type,struct_offset
                                                        type = eSymbolTypeVariableType;
                                                        break;

                                                    case StabSourceFileName:
                                                        // N_SO - source file name
                                                        type = eSymbolTypeSourceFile;
                                                        if (symbol_name == NULL)
                                                        {
                                                            if (minimize)
                                                                add_nlist = false;
                                                            if (N_SO_index != UINT32_MAX)
                                                            {
                                                                // Set the size of the N_SO to the terminating index of this N_SO
                                                                // so that we can always skip the entire N_SO if we need to navigate
                                                                // more quickly at the source level when parsing STABS
                                                                symbol_ptr = symtab->SymbolAtIndex(N_SO_index);
                                                                symbol_ptr->SetByteSize(sym_idx + (minimize ? 0 : 1));
                                                                symbol_ptr->SetSizeIsSibling(true);
                                                            }
                                                            N_NSYM_indexes.clear();
                                                            N_INCL_indexes.clear();
                                                            N_BRAC_indexes.clear();
                                                            N_COMM_indexes.clear();
                                                            N_FUN_indexes.clear();
                                                            N_SO_index = UINT32_MAX;
                                                        }
                                                        else
                                                        {
                                                            // We use the current number of symbols in the symbol table in lieu of
                                                            // using nlist_idx in case we ever start trimming entries out
                                                            const bool N_SO_has_full_path = symbol_name[0] == '/';
                                                            if (N_SO_has_full_path)
                                                            {
                                                                if (minimize && (N_SO_index == sym_idx - 1) && ((sym_idx - 1) < num_syms))
                                                                {
                                                                    // We have two consecutive N_SO entries where the first contains a directory
                                                                    // and the second contains a full path.
                                                                    sym[sym_idx - 1].GetMangled().SetValue(ConstString(symbol_name), false);
                                                                    m_nlist_idx_to_sym_idx[nlist_idx] = sym_idx - 1;
                                                                    add_nlist = false;
                                                                }
                                                                else
                                                                {
                                                                    // This is the first entry in a N_SO that contains a directory or
                                                                    // a full path to the source file
                                                                    N_SO_index = sym_idx;
                                                                }
                                                            }
                                                            else if (minimize && (N_SO_index == sym_idx - 1) && ((sym_idx - 1) < num_syms))
                                                            {
                                                                // This is usually the second N_SO entry that contains just the filename,
                                                                // so here we combine it with the first one if we are minimizing the symbol table
                                                                const char *so_path = sym[sym_idx - 1].GetMangled().GetDemangledName().AsCString();
                                                                if (so_path && so_path[0])
                                                                {
                                                                    std::string full_so_path (so_path);
                                                                    const size_t double_slash_pos = full_so_path.find("//");
                                                                    if (double_slash_pos != std::string::npos)
                                                                    {
                                                                        // The linker has been generating bad N_SO entries with doubled up paths
                                                                        // in the format "%s%s" where the first stirng in the DW_AT_comp_dir,
                                                                        // and the second is the directory for the source file so you end up with
                                                                        // a path that looks like "/tmp/src//tmp/src/"
                                                                        FileSpec so_dir(so_path, false);
                                                                        if (!so_dir.Exists())
                                                                        {
                                                                            so_dir.SetFile(&full_so_path[double_slash_pos + 1], false);
                                                                            if (so_dir.Exists())
                                                                            {
                                                                                // Trim off the incorrect path
                                                                                full_so_path.erase(0, double_slash_pos + 1);
                                                                            }
                                                                        }
                                                                    }
                                                                    if (*full_so_path.rbegin() != '/')
                                                                        full_so_path += '/';
                                                                    full_so_path += symbol_name;
                                                                    sym[sym_idx - 1].GetMangled().SetValue(ConstString(full_so_path.c_str()), false);
                                                                    add_nlist = false;
                                                                    m_nlist_idx_to_sym_idx[nlist_idx] = sym_idx - 1;
                                                                }
                                                            }
                                                            else
                                                            {
                                                                // This could be a relative path to a N_SO
                                                                N_SO_index = sym_idx;
                                                            }
                                                        }
                                                        break;

                                                    case StabObjectFileName:
                                                        // N_OSO - object file name: name,,0,0,st_mtime
                                                        type = eSymbolTypeObjectFile;
                                                        break;

                                                    case StabLocalSymbol:
                                                        // N_LSYM - local sym: name,,NO_SECT,type,offset
                                                        type = eSymbolTypeLocal;
                                                        break;

                                                        //----------------------------------------------------------------------
                                                        // INCL scopes
                                                        //----------------------------------------------------------------------
                                                    case StabBeginIncludeFileName:
                                                        // N_BINCL - include file beginning: name,,NO_SECT,0,sum
                                                        // We use the current number of symbols in the symbol table in lieu of
                                                        // using nlist_idx in case we ever start trimming entries out
                                                        N_INCL_indexes.push_back(sym_idx);
                                                        type = eSymbolTypeScopeBegin;
                                                        break;

                                                    case StabEndIncludeFile:
                                                        // N_EINCL - include file end: name,,NO_SECT,0,0
                                                        // Set the size of the N_BINCL to the terminating index of this N_EINCL
                                                        // so that we can always skip the entire symbol if we need to navigate
                                                        // more quickly at the source level when parsing STABS
                                                        if ( !N_INCL_indexes.empty() )
                                                        {
                                                            symbol_ptr = symtab->SymbolAtIndex(N_INCL_indexes.back());
                                                            symbol_ptr->SetByteSize(sym_idx + 1);
                                                            symbol_ptr->SetSizeIsSibling(true);
                                                            N_INCL_indexes.pop_back();
                                                        }
                                                        type = eSymbolTypeScopeEnd;
                                                        break;

                                                    case StabIncludeFileName:
                                                        // N_SOL - #included file name: name,,n_sect,0,address
                                                        type = eSymbolTypeHeaderFile;

                                                        // We currently don't use the header files on darwin
                                                        if (minimize)
                                                            add_nlist = false;
                                                        break;

                                                    case StabCompilerParameters:
                                                        // N_PARAMS - compiler parameters: name,,NO_SECT,0,0
                                                        type = eSymbolTypeCompiler;
                                                        break;

                                                    case StabCompilerVersion:
                                                        // N_VERSION - compiler version: name,,NO_SECT,0,0
                                                        type = eSymbolTypeCompiler;
                                                        break;

                                                    case StabCompilerOptLevel:
                                                        // N_OLEVEL - compiler -O level: name,,NO_SECT,0,0
                                                        type = eSymbolTypeCompiler;
                                                        break;

                                                    case StabParameter:
                                                        // N_PSYM - parameter: name,,NO_SECT,type,offset
                                                        type = eSymbolTypeVariable;
                                                        break;

                                                    case StabAlternateEntry:
                                                        // N_ENTRY - alternate entry: name,,n_sect,linenumber,address
                                                        symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);
                                                        type = eSymbolTypeLineEntry;
                                                        break;

                                                        //----------------------------------------------------------------------
                                                        // Left and Right Braces
                                                        //----------------------------------------------------------------------
                                                    case StabLeftBracket:
                                                        // N_LBRAC - left bracket: 0,,NO_SECT,nesting level,address
                                                        // We use the current number of symbols in the symbol table in lieu of
                                                        // using nlist_idx in case we ever start trimming entries out
                                                        symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);
                                                        N_BRAC_indexes.push_back(sym_idx);
                                                        type = eSymbolTypeScopeBegin;
                                                        break;

                                                    case StabRightBracket:
                                                        // N_RBRAC - right bracket: 0,,NO_SECT,nesting level,address
                                                        // Set the size of the N_LBRAC to the terminating index of this N_RBRAC
                                                        // so that we can always skip the entire symbol if we need to navigate
                                                        // more quickly at the source level when parsing STABS
                                                        symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);
                                                        if ( !N_BRAC_indexes.empty() )
                                                        {
                                                            symbol_ptr = symtab->SymbolAtIndex(N_BRAC_indexes.back());
                                                            symbol_ptr->SetByteSize(sym_idx + 1);
                                                            symbol_ptr->SetSizeIsSibling(true);
                                                            N_BRAC_indexes.pop_back();
                                                        }
                                                        type = eSymbolTypeScopeEnd;
                                                        break;

                                                    case StabDeletedIncludeFile:
                                                        // N_EXCL - deleted include file: name,,NO_SECT,0,sum
                                                        type = eSymbolTypeHeaderFile;
                                                        break;

                                                        //----------------------------------------------------------------------
                                                        // COMM scopes
                                                        //----------------------------------------------------------------------
                                                    case StabBeginCommon:
                                                        // N_BCOMM - begin common: name,,NO_SECT,0,0
                                                        // We use the current number of symbols in the symbol table in lieu of
                                                        // using nlist_idx in case we ever start trimming entries out
                                                        type = eSymbolTypeScopeBegin;
                                                        N_COMM_indexes.push_back(sym_idx);
                                                        break;

                                                    case StabEndCommonLocal:
                                                        // N_ECOML - end common (local name): 0,,n_sect,0,address
                                                        symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);
                                                        // Fall through

                                                    case StabEndCommon:
                                                        // N_ECOMM - end common: name,,n_sect,0,0
                                                        // Set the size of the N_BCOMM to the terminating index of this N_ECOMM/N_ECOML
                                                        // so that we can always skip the entire symbol if we need to navigate
                                                        // more quickly at the source level when parsing STABS
                                                        if ( !N_COMM_indexes.empty() )
                                                        {
                                                            symbol_ptr = symtab->SymbolAtIndex(N_COMM_indexes.back());
                                                            symbol_ptr->SetByteSize(sym_idx + 1);
                                                            symbol_ptr->SetSizeIsSibling(true);
                                                            N_COMM_indexes.pop_back();
                                                        }
                                                        type = eSymbolTypeScopeEnd;
                                                        break;

                                                    case StabLength:
                                                        // N_LENG - second stab entry with length information
                                                        type = eSymbolTypeAdditional;
                                                        break;

                                                    default: break;
                                                }
                                            }
                                            else
                                            {
                                                //uint8_t n_pext    = NlistMaskPrivateExternal & nlist.n_type;
                                                uint8_t n_type  = NlistMaskType & nlist.n_type;
                                                sym[sym_idx].SetExternal((NlistMaskExternal & nlist.n_type) != 0);

                                                switch (n_type)
                                                {
                                                    case NListTypeIndirect:         // N_INDR - Fall through
                                                    case NListTypePreboundUndefined:// N_PBUD - Fall through
                                                    case NListTypeUndefined:        // N_UNDF
                                                        type = eSymbolTypeUndefined;
                                                        break;

                                                    case NListTypeAbsolute:         // N_ABS
                                                        type = eSymbolTypeAbsolute;
                                                        break;

                                                    case NListTypeSection:          // N_SECT
                                                    {
                                                        symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);

                                                        if (symbol_section == NULL)
                                                        {
                                                            // TODO: warn about this?
                                                            add_nlist = false;
                                                            break;
                                                        }

                                                        if (TEXT_eh_frame_sectID == nlist.n_sect)
                                                        {
                                                            type = eSymbolTypeException;
                                                        }
                                                        else
                                                        {
                                                            uint32_t section_type = symbol_section->Get() & SectionFlagMaskSectionType;
                                                            
                                                            switch (section_type)
                                                            {
                                                                case SectionTypeRegular:                     break; // regular section
                                                                                                                    //case SectionTypeZeroFill:                 type = eSymbolTypeData;    break; // zero fill on demand section
                                                                case SectionTypeCStringLiterals:            type = eSymbolTypeData;    break; // section with only literal C strings
                                                                case SectionType4ByteLiterals:              type = eSymbolTypeData;    break; // section with only 4 byte literals
                                                                case SectionType8ByteLiterals:              type = eSymbolTypeData;    break; // section with only 8 byte literals
                                                                case SectionTypeLiteralPointers:            type = eSymbolTypeTrampoline; break; // section with only pointers to literals
                                                                case SectionTypeNonLazySymbolPointers:      type = eSymbolTypeTrampoline; break; // section with only non-lazy symbol pointers
                                                                case SectionTypeLazySymbolPointers:         type = eSymbolTypeTrampoline; break; // section with only lazy symbol pointers
                                                                case SectionTypeSymbolStubs:                type = eSymbolTypeTrampoline; break; // section with only symbol stubs, byte size of stub in the reserved2 field
                                                                case SectionTypeModuleInitFunctionPointers: type = eSymbolTypeCode;    break; // section with only function pointers for initialization
                                                                case SectionTypeModuleTermFunctionPointers: type = eSymbolTypeCode;    break; // section with only function pointers for termination
                                                                                                                                              //case SectionTypeCoalesced:                type = eSymbolType;    break; // section contains symbols that are to be coalesced
                                                                                                                                              //case SectionTypeZeroFillLarge:            type = eSymbolTypeData;    break; // zero fill on demand section (that can be larger than 4 gigabytes)
                                                                case SectionTypeInterposing:                type = eSymbolTypeTrampoline;  break; // section with only pairs of function pointers for interposing
                                                                case SectionType16ByteLiterals:             type = eSymbolTypeData;    break; // section with only 16 byte literals
                                                                case SectionTypeDTraceObjectFormat:         type = eSymbolTypeInstrumentation; break;
                                                                case SectionTypeLazyDylibSymbolPointers:    type = eSymbolTypeTrampoline; break;
                                                                default: break;
                                                            }
                                                            
                                                            if (type == eSymbolTypeInvalid)
                                                            {
                                                                const char *symbol_sect_name = symbol_section->GetName().AsCString();
                                                                if (symbol_section->IsDescendant (text_section_sp.get()))
                                                                {
                                                                    if (symbol_section->IsClear(SectionAttrUserPureInstructions | 
                                                                                                SectionAttrUserSelfModifyingCode | 
                                                                                                SectionAttrSytemSomeInstructions))
                                                                        type = eSymbolTypeData;
                                                                    else
                                                                        type = eSymbolTypeCode;
                                                                }
                                                                else
                                                                    if (symbol_section->IsDescendant(data_section_sp.get()))
                                                                    {
                                                                        if (symbol_sect_name && ::strstr (symbol_sect_name, "__objc") == symbol_sect_name)
                                                                        {
                                                                            type = eSymbolTypeRuntime;
                                                                            
                                                                            if (symbol_name && 
                                                                                symbol_name[0] == '_' && 
                                                                                symbol_name[1] == 'O' && 
                                                                                symbol_name[2] == 'B')
                                                                            {
                                                                                llvm::StringRef symbol_name_ref(symbol_name);
                                                                                static const llvm::StringRef g_objc_v2_prefix_class ("_OBJC_CLASS_$_");
                                                                                static const llvm::StringRef g_objc_v2_prefix_metaclass ("_OBJC_METACLASS_$_");
                                                                                static const llvm::StringRef g_objc_v2_prefix_ivar ("_OBJC_IVAR_$_");
                                                                                if (symbol_name_ref.startswith(g_objc_v2_prefix_class))
                                                                                {
                                                                                    symbol_name_non_abi_mangled = symbol_name + 1;
                                                                                    symbol_name = symbol_name + g_objc_v2_prefix_class.size();
                                                                                    type = eSymbolTypeObjCClass;
                                                                                }
                                                                                else if (symbol_name_ref.startswith(g_objc_v2_prefix_metaclass))
                                                                                {
                                                                                    symbol_name_non_abi_mangled = symbol_name + 1;
                                                                                    symbol_name = symbol_name + g_objc_v2_prefix_metaclass.size();
                                                                                    type = eSymbolTypeObjCMetaClass;
                                                                                }
                                                                                else if (symbol_name_ref.startswith(g_objc_v2_prefix_ivar))
                                                                                {
                                                                                    symbol_name_non_abi_mangled = symbol_name + 1;
                                                                                    symbol_name = symbol_name + g_objc_v2_prefix_ivar.size();
                                                                                    type = eSymbolTypeObjCIVar;
                                                                                }
                                                                            }
                                                                        }
                                                                        else
                                                                            if (symbol_sect_name && ::strstr (symbol_sect_name, "__gcc_except_tab") == symbol_sect_name)
                                                                            {
                                                                                type = eSymbolTypeException;
                                                                            }
                                                                            else
                                                                            {
                                                                                type = eSymbolTypeData;
                                                                            }
                                                                    }
                                                                    else
                                                                        if (symbol_sect_name && ::strstr (symbol_sect_name, "__IMPORT") == symbol_sect_name)
                                                                        {
                                                                            type = eSymbolTypeTrampoline;
                                                                        }
                                                                        else
                                                                            if (symbol_section->IsDescendant(objc_section_sp.get()))
                                                                            {
                                                                                type = eSymbolTypeRuntime;
                                                                                if (symbol_name && symbol_name[0] == '.')
                                                                                {
                                                                                    llvm::StringRef symbol_name_ref(symbol_name);
                                                                                    static const llvm::StringRef g_objc_v1_prefix_class (".objc_class_name_");
                                                                                    if (symbol_name_ref.startswith(g_objc_v1_prefix_class))
                                                                                    {
                                                                                        symbol_name_non_abi_mangled = symbol_name;
                                                                                        symbol_name = symbol_name + g_objc_v1_prefix_class.size();
                                                                                        type = eSymbolTypeObjCClass;
                                                                                    }
                                                                                }
                                                                            }
                                                            }
                                                        }
                                                    }
                                                        break;
                                                }                            
                                            }

                                            if (add_nlist)
                                            {
                                                uint64_t symbol_value = nlist.n_value;
                                                bool symbol_name_is_mangled = false;
                                                
                                                if (symbol_name_non_abi_mangled)
                                                {
                                                    sym[sym_idx].GetMangled().SetMangledName (ConstString(symbol_name_non_abi_mangled));
                                                    sym[sym_idx].GetMangled().SetDemangledName (ConstString(symbol_name));
                                                }
                                                else
                                                {
                                                    if (symbol_name && symbol_name[0] == '_')
                                                    {
                                                        symbol_name_is_mangled = symbol_name[1] == '_';
                                                        symbol_name++;  // Skip the leading underscore
                                                    }
                                                    
                                                    if (symbol_name)
                                                    {
                                                        sym[sym_idx].GetMangled().SetValue(ConstString(symbol_name), symbol_name_is_mangled);
                                                    }
                                                }
                                                
                                                if (is_debug == false)
                                                {
                                                    if (type == eSymbolTypeCode)
                                                    {
                                                        // See if we can find a N_FUN entry for any code symbols.
                                                        // If we do find a match, and the name matches, then we
                                                        // can merge the two into just the function symbol to avoid
                                                        // duplicate entries in the symbol table
                                                        ValueToSymbolIndexMap::const_iterator pos = N_FUN_addr_to_sym_idx.find (nlist.n_value);
                                                        if (pos != N_FUN_addr_to_sym_idx.end())
                                                        {
                                                            if ((symbol_name_is_mangled == true && sym[sym_idx].GetMangled().GetMangledName() == sym[pos->second].GetMangled().GetMangledName()) ||
                                                                (symbol_name_is_mangled == false && sym[sym_idx].GetMangled().GetDemangledName() == sym[pos->second].GetMangled().GetDemangledName()))
                                                            {
                                                                m_nlist_idx_to_sym_idx[nlist_idx] = pos->second;
                                                                // We just need the flags from the linker symbol, so put these flags
                                                                // into the N_FUN flags to avoid duplicate symbols in the symbol table
                                                                sym[pos->second].SetFlags (nlist.n_type << 16 | nlist.n_desc);
                                                                sym[sym_idx].Clear();
                                                                continue;
                                                            }
                                                        }
                                                    }
                                                    else if (type == eSymbolTypeData)
                                                    {
                                                        // See if we can find a N_STSYM entry for any data symbols.
                                                        // If we do find a match, and the name matches, then we
                                                        // can merge the two into just the Static symbol to avoid
                                                        // duplicate entries in the symbol table
                                                        ValueToSymbolIndexMap::const_iterator pos = N_STSYM_addr_to_sym_idx.find (nlist.n_value);
                                                        if (pos != N_STSYM_addr_to_sym_idx.end())
                                                        {
                                                            if ((symbol_name_is_mangled == true && sym[sym_idx].GetMangled().GetMangledName() == sym[pos->second].GetMangled().GetMangledName()) ||
                                                                (symbol_name_is_mangled == false && sym[sym_idx].GetMangled().GetDemangledName() == sym[pos->second].GetMangled().GetDemangledName()))
                                                            {
                                                                m_nlist_idx_to_sym_idx[nlist_idx] = pos->second;
                                                                // We just need the flags from the linker symbol, so put these flags
                                                                // into the N_STSYM flags to avoid duplicate symbols in the symbol table
                                                                sym[pos->second].SetFlags (nlist.n_type << 16 | nlist.n_desc);
                                                                sym[sym_idx].Clear();
                                                                continue;
                                                            }
                                                        }
                                                    }
                                                }
                                                if (symbol_section)
                                                {
                                                    const addr_t section_file_addr = symbol_section->GetFileAddress();
                                                    if (symbol_byte_size == 0 && function_starts_count > 0)
                                                    {
                                                        addr_t symbol_lookup_file_addr = nlist.n_value;
                                                        // Do an exact address match for non-ARM addresses, else get the closest since
                                                        // the symbol might be a thumb symbol which has an address with bit zero set
                                                        FunctionStarts::Entry *func_start_entry = function_starts.FindEntry (symbol_lookup_file_addr, !is_arm);
                                                        if (is_arm && func_start_entry)
                                                        {
                                                            // Verify that the function start address is the symbol address (ARM)
                                                            // or the symbol address + 1 (thumb)
                                                            if (func_start_entry->addr != symbol_lookup_file_addr &&
                                                                func_start_entry->addr != (symbol_lookup_file_addr + 1))
                                                            {
                                                                // Not the right entry, NULL it out...
                                                                func_start_entry = NULL;
                                                            }
                                                        }
                                                        if (func_start_entry)
                                                        {
                                                            func_start_entry->data = true;
                                                            
                                                            addr_t symbol_file_addr = func_start_entry->addr;
                                                            uint32_t symbol_flags = 0;
                                                            if (is_arm)
                                                            {
                                                                if (symbol_file_addr & 1)
                                                                    symbol_flags = MACHO_NLIST_ARM_SYMBOL_IS_THUMB;
                                                                symbol_file_addr &= 0xfffffffffffffffeull;
                                                            }
                                                            
                                                            const FunctionStarts::Entry *next_func_start_entry = function_starts.FindNextEntry (func_start_entry);
                                                            const addr_t section_end_file_addr = section_file_addr + symbol_section->GetByteSize();
                                                            if (next_func_start_entry)
                                                            {
                                                                addr_t next_symbol_file_addr = next_func_start_entry->addr;
                                                                // Be sure the clear the Thumb address bit when we calculate the size
                                                                // from the current and next address
                                                                if (is_arm)
                                                                    next_symbol_file_addr &= 0xfffffffffffffffeull;
                                                                symbol_byte_size = std::min<lldb::addr_t>(next_symbol_file_addr - symbol_file_addr, section_end_file_addr - symbol_file_addr);
                                                            }
                                                            else
                                                            {
                                                                symbol_byte_size = section_end_file_addr - symbol_file_addr;
                                                            }
                                                        }
                                                    }
                                                    symbol_value -= section_file_addr;
                                                }
                                                
                                                sym[sym_idx].SetID (nlist_idx);
                                                sym[sym_idx].SetType (type);
                                                sym[sym_idx].GetAddress().SetSection (symbol_section);
                                                sym[sym_idx].GetAddress().SetOffset (symbol_value);
                                                sym[sym_idx].SetFlags (nlist.n_type << 16 | nlist.n_desc);
                                                
                                                if (symbol_byte_size > 0)
                                                    sym[sym_idx].SetByteSize(symbol_byte_size);

                                                ++sym_idx;
                                            }
                                            else
                                            {
                                                sym[sym_idx].Clear();
                                            }
                                            
                                        }
                                        /////////////////////////////
                                    }
                                    break; // No more entries to consider
                                }
                            }
                        }
                    }
                }
            }
        }

        // Must reset this in case it was mutated above!
        nlist_data_offset = 0;
#endif

        // If the sym array was not created while parsing the DSC unmapped
        // symbols, create it now.
        if (sym == NULL)
        {
            sym = symtab->Resize (symtab_load_command.nsyms + m_dysymtab.nindirectsyms);
            num_syms = symtab->GetNumSymbols();
        }

        if (unmapped_local_symbols_found)
        {
            assert(m_dysymtab.ilocalsym == 0);
            nlist_data_offset += (m_dysymtab.nlocalsym * nlist_byte_size);
            nlist_idx = m_dysymtab.nlocalsym;
        }
        else
        {
            nlist_idx = 0;
        }

        for (; nlist_idx < symtab_load_command.nsyms; ++nlist_idx)
        {
            struct nlist_64 nlist;
            if (!nlist_data.ValidOffsetForDataOfSize(nlist_data_offset, nlist_byte_size))
                break;

            nlist.n_strx  = nlist_data.GetU32_unchecked(&nlist_data_offset);
            nlist.n_type  = nlist_data.GetU8_unchecked (&nlist_data_offset);
            nlist.n_sect  = nlist_data.GetU8_unchecked (&nlist_data_offset);
            nlist.n_desc  = nlist_data.GetU16_unchecked (&nlist_data_offset);
            nlist.n_value = nlist_data.GetAddress_unchecked (&nlist_data_offset);

            SymbolType type = eSymbolTypeInvalid;
            const char *symbol_name = NULL;
            
            if (have_strtab_data)
            {
                symbol_name = strtab_data.PeekCStr(nlist.n_strx);
                
                if (symbol_name == NULL)
                {
                    // No symbol should be NULL, even the symbols with no
                    // string values should have an offset zero which points
                    // to an empty C-string
                    Host::SystemLog (Host::eSystemLogError,
                                     "error: symbol[%u] has invalid string table offset 0x%x in %s/%s, ignoring symbol\n", 
                                     nlist_idx,
                                     nlist.n_strx,
                                     module_sp->GetFileSpec().GetDirectory().GetCString(),
                                     module_sp->GetFileSpec().GetFilename().GetCString());
                    continue;
                }
                if (symbol_name[0] == '\0')
                    symbol_name = NULL;
            }
            else
            {
                const addr_t str_addr = strtab_addr + nlist.n_strx;
                Error str_error;
                if (process->ReadCStringFromMemory(str_addr, memory_symbol_name, str_error))
                    symbol_name = memory_symbol_name.c_str();
            }
            const char *symbol_name_non_abi_mangled = NULL;

            SectionSP symbol_section;
            uint32_t symbol_byte_size = 0;
            bool add_nlist = true;
            bool is_debug = ((nlist.n_type & NlistMaskStab) != 0);

            assert (sym_idx < num_syms);

            sym[sym_idx].SetDebug (is_debug);

            if (is_debug)
            {
                switch (nlist.n_type)
                {
                case StabGlobalSymbol:    
                    // N_GSYM -- global symbol: name,,NO_SECT,type,0
                    // Sometimes the N_GSYM value contains the address.
                    
                    // FIXME: In the .o files, we have a GSYM and a debug symbol for all the ObjC data.  They
                    // have the same address, but we want to ensure that we always find only the real symbol,
                    // 'cause we don't currently correctly attribute the GSYM one to the ObjCClass/Ivar/MetaClass
                    // symbol type.  This is a temporary hack to make sure the ObjectiveC symbols get treated
                    // correctly.  To do this right, we should coalesce all the GSYM & global symbols that have the
                    // same address.
                    
                    if (symbol_name && symbol_name[0] == '_' && symbol_name[1] ==  'O' 
                        && (strncmp (symbol_name, "_OBJC_IVAR_$_", strlen ("_OBJC_IVAR_$_")) == 0
                            || strncmp (symbol_name, "_OBJC_CLASS_$_", strlen ("_OBJC_CLASS_$_")) == 0
                            || strncmp (symbol_name, "_OBJC_METACLASS_$_", strlen ("_OBJC_METACLASS_$_")) == 0))
                        add_nlist = false;
                    else
                    {
                        sym[sym_idx].SetExternal(true);
                        if (nlist.n_value != 0)
                            symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);
                        type = eSymbolTypeData;
                    }
                    break;

                case StabFunctionName:
                    // N_FNAME -- procedure name (f77 kludge): name,,NO_SECT,0,0
                    type = eSymbolTypeCompiler;
                    break;

                case StabFunction:       
                    // N_FUN -- procedure: name,,n_sect,linenumber,address
                    if (symbol_name)
                    {
                        type = eSymbolTypeCode;
                        symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);
                        
                        N_FUN_addr_to_sym_idx[nlist.n_value] = sym_idx;
                        // We use the current number of symbols in the symbol table in lieu of
                        // using nlist_idx in case we ever start trimming entries out
                        N_FUN_indexes.push_back(sym_idx);
                    }
                    else
                    {
                        type = eSymbolTypeCompiler;

                        if ( !N_FUN_indexes.empty() )
                        {
                            // Copy the size of the function into the original STAB entry so we don't have
                            // to hunt for it later
                            symtab->SymbolAtIndex(N_FUN_indexes.back())->SetByteSize(nlist.n_value);
                            N_FUN_indexes.pop_back();
                            // We don't really need the end function STAB as it contains the size which
                            // we already placed with the original symbol, so don't add it if we want a
                            // minimal symbol table
                            if (minimize)
                                add_nlist = false;
                        }
                    }
                    break;

                case StabStaticSymbol:   
                    // N_STSYM -- static symbol: name,,n_sect,type,address
                    N_STSYM_addr_to_sym_idx[nlist.n_value] = sym_idx;
                    symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);
                    type = eSymbolTypeData;
                    break;

                case StabLocalCommon:
                    // N_LCSYM -- .lcomm symbol: name,,n_sect,type,address
                    symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);
                    type = eSymbolTypeCommonBlock;
                    break;

                case StabBeginSymbol:
                    // N_BNSYM
                    // We use the current number of symbols in the symbol table in lieu of
                    // using nlist_idx in case we ever start trimming entries out
                    if (minimize)
                    {
                        // Skip these if we want minimal symbol tables
                        add_nlist = false;
                    }
                    else
                    {
                        symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);
                        N_NSYM_indexes.push_back(sym_idx);
                        type = eSymbolTypeScopeBegin;
                    }
                    break;

                case StabEndSymbol:
                    // N_ENSYM
                    // Set the size of the N_BNSYM to the terminating index of this N_ENSYM
                    // so that we can always skip the entire symbol if we need to navigate
                    // more quickly at the source level when parsing STABS
                    if (minimize)
                    {
                        // Skip these if we want minimal symbol tables
                        add_nlist = false;
                    }
                    else
                    {
                        if ( !N_NSYM_indexes.empty() )
                        {
                            symbol_ptr = symtab->SymbolAtIndex(N_NSYM_indexes.back());
                            symbol_ptr->SetByteSize(sym_idx + 1);
                            symbol_ptr->SetSizeIsSibling(true);
                            N_NSYM_indexes.pop_back();
                        }
                        type = eSymbolTypeScopeEnd;
                    }
                    break;


                case StabSourceFileOptions:
                    // N_OPT - emitted with gcc2_compiled and in gcc source
                    type = eSymbolTypeCompiler;
                    break;

                case StabRegisterSymbol:
                    // N_RSYM - register sym: name,,NO_SECT,type,register
                    type = eSymbolTypeVariable;
                    break;

                case StabSourceLine:
                    // N_SLINE - src line: 0,,n_sect,linenumber,address
                    symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);
                    type = eSymbolTypeLineEntry;
                    break;

                case StabStructureType:
                    // N_SSYM - structure elt: name,,NO_SECT,type,struct_offset
                    type = eSymbolTypeVariableType;
                    break;

                case StabSourceFileName:
                    // N_SO - source file name
                    type = eSymbolTypeSourceFile;
                    if (symbol_name == NULL)
                    {
                        if (minimize)
                            add_nlist = false;
                        if (N_SO_index != UINT32_MAX)
                        {
                            // Set the size of the N_SO to the terminating index of this N_SO
                            // so that we can always skip the entire N_SO if we need to navigate
                            // more quickly at the source level when parsing STABS
                            symbol_ptr = symtab->SymbolAtIndex(N_SO_index);
                            symbol_ptr->SetByteSize(sym_idx + (minimize ? 0 : 1));
                            symbol_ptr->SetSizeIsSibling(true);
                        }
                        N_NSYM_indexes.clear();
                        N_INCL_indexes.clear();
                        N_BRAC_indexes.clear();
                        N_COMM_indexes.clear();
                        N_FUN_indexes.clear();
                        N_SO_index = UINT32_MAX;
                    }
                    else
                    {
                        // We use the current number of symbols in the symbol table in lieu of
                        // using nlist_idx in case we ever start trimming entries out
                        const bool N_SO_has_full_path = symbol_name[0] == '/';
                        if (N_SO_has_full_path)
                        {
                            if (minimize && (N_SO_index == sym_idx - 1) && ((sym_idx - 1) < num_syms))
                            {
                                // We have two consecutive N_SO entries where the first contains a directory
                                // and the second contains a full path.
                                sym[sym_idx - 1].GetMangled().SetValue(ConstString(symbol_name), false);
                                m_nlist_idx_to_sym_idx[nlist_idx] = sym_idx - 1;
                                add_nlist = false;
                            }
                            else
                            {
                                // This is the first entry in a N_SO that contains a directory or
                                // a full path to the source file
                                N_SO_index = sym_idx;
                            }
                        }
                        else if (minimize && (N_SO_index == sym_idx - 1) && ((sym_idx - 1) < num_syms))
                        {
                            // This is usually the second N_SO entry that contains just the filename,
                            // so here we combine it with the first one if we are minimizing the symbol table
                            const char *so_path = sym[sym_idx - 1].GetMangled().GetDemangledName().AsCString();
                            if (so_path && so_path[0])
                            {
                                std::string full_so_path (so_path);
                                const size_t double_slash_pos = full_so_path.find("//");
                                if (double_slash_pos != std::string::npos)
                                {
                                    // The linker has been generating bad N_SO entries with doubled up paths
                                    // in the format "%s%s" where the first stirng in the DW_AT_comp_dir,
                                    // and the second is the directory for the source file so you end up with
                                    // a path that looks like "/tmp/src//tmp/src/"
                                    FileSpec so_dir(so_path, false);
                                    if (!so_dir.Exists())
                                    {
                                        so_dir.SetFile(&full_so_path[double_slash_pos + 1], false);
                                        if (so_dir.Exists())
                                        {
                                            // Trim off the incorrect path
                                            full_so_path.erase(0, double_slash_pos + 1);
                                        }
                                    }
                                }
                                if (*full_so_path.rbegin() != '/')
                                    full_so_path += '/';
                                full_so_path += symbol_name;
                                sym[sym_idx - 1].GetMangled().SetValue(ConstString(full_so_path.c_str()), false);
                                add_nlist = false;
                                m_nlist_idx_to_sym_idx[nlist_idx] = sym_idx - 1;
                            }
                        }
                        else
                        {
                            // This could be a relative path to a N_SO
                            N_SO_index = sym_idx;
                        }
                    }
                    
                    break;

                case StabObjectFileName:
                    // N_OSO - object file name: name,,0,0,st_mtime
                    type = eSymbolTypeObjectFile;
                    break;

                case StabLocalSymbol:
                    // N_LSYM - local sym: name,,NO_SECT,type,offset
                    type = eSymbolTypeLocal;
                    break;

                //----------------------------------------------------------------------
                // INCL scopes
                //----------------------------------------------------------------------
                case StabBeginIncludeFileName:
                    // N_BINCL - include file beginning: name,,NO_SECT,0,sum
                    // We use the current number of symbols in the symbol table in lieu of
                    // using nlist_idx in case we ever start trimming entries out
                    N_INCL_indexes.push_back(sym_idx);
                    type = eSymbolTypeScopeBegin;
                    break;

                case StabEndIncludeFile:
                    // N_EINCL - include file end: name,,NO_SECT,0,0
                    // Set the size of the N_BINCL to the terminating index of this N_EINCL
                    // so that we can always skip the entire symbol if we need to navigate
                    // more quickly at the source level when parsing STABS
                    if ( !N_INCL_indexes.empty() )
                    {
                        symbol_ptr = symtab->SymbolAtIndex(N_INCL_indexes.back());
                        symbol_ptr->SetByteSize(sym_idx + 1);
                        symbol_ptr->SetSizeIsSibling(true);
                        N_INCL_indexes.pop_back();
                    }
                    type = eSymbolTypeScopeEnd;
                    break;

                case StabIncludeFileName:
                    // N_SOL - #included file name: name,,n_sect,0,address
                    type = eSymbolTypeHeaderFile;

                    // We currently don't use the header files on darwin
                    if (minimize)
                        add_nlist = false;
                    break;

                case StabCompilerParameters:  
                    // N_PARAMS - compiler parameters: name,,NO_SECT,0,0
                    type = eSymbolTypeCompiler;
                    break;

                case StabCompilerVersion:
                    // N_VERSION - compiler version: name,,NO_SECT,0,0
                    type = eSymbolTypeCompiler;
                    break;

                case StabCompilerOptLevel:
                    // N_OLEVEL - compiler -O level: name,,NO_SECT,0,0
                    type = eSymbolTypeCompiler;
                    break;

                case StabParameter:
                    // N_PSYM - parameter: name,,NO_SECT,type,offset
                    type = eSymbolTypeVariable;
                    break;

                case StabAlternateEntry:
                    // N_ENTRY - alternate entry: name,,n_sect,linenumber,address
                    symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);
                    type = eSymbolTypeLineEntry;
                    break;

                //----------------------------------------------------------------------
                // Left and Right Braces
                //----------------------------------------------------------------------
                case StabLeftBracket:
                    // N_LBRAC - left bracket: 0,,NO_SECT,nesting level,address
                    // We use the current number of symbols in the symbol table in lieu of
                    // using nlist_idx in case we ever start trimming entries out
                    symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);
                    N_BRAC_indexes.push_back(sym_idx);
                    type = eSymbolTypeScopeBegin;
                    break;

                case StabRightBracket:
                    // N_RBRAC - right bracket: 0,,NO_SECT,nesting level,address
                    // Set the size of the N_LBRAC to the terminating index of this N_RBRAC
                    // so that we can always skip the entire symbol if we need to navigate
                    // more quickly at the source level when parsing STABS
                    symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);
                    if ( !N_BRAC_indexes.empty() )
                    {
                        symbol_ptr = symtab->SymbolAtIndex(N_BRAC_indexes.back());
                        symbol_ptr->SetByteSize(sym_idx + 1);
                        symbol_ptr->SetSizeIsSibling(true);
                        N_BRAC_indexes.pop_back();
                    }
                    type = eSymbolTypeScopeEnd;
                    break;

                case StabDeletedIncludeFile:
                    // N_EXCL - deleted include file: name,,NO_SECT,0,sum
                    type = eSymbolTypeHeaderFile;
                    break;

                //----------------------------------------------------------------------
                // COMM scopes
                //----------------------------------------------------------------------
                case StabBeginCommon:
                    // N_BCOMM - begin common: name,,NO_SECT,0,0
                    // We use the current number of symbols in the symbol table in lieu of
                    // using nlist_idx in case we ever start trimming entries out
                    type = eSymbolTypeScopeBegin;
                    N_COMM_indexes.push_back(sym_idx);
                    break;

                case StabEndCommonLocal:
                    // N_ECOML - end common (local name): 0,,n_sect,0,address
                    symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);
                    // Fall through

                case StabEndCommon:
                    // N_ECOMM - end common: name,,n_sect,0,0
                    // Set the size of the N_BCOMM to the terminating index of this N_ECOMM/N_ECOML
                    // so that we can always skip the entire symbol if we need to navigate
                    // more quickly at the source level when parsing STABS
                    if ( !N_COMM_indexes.empty() )
                    {
                        symbol_ptr = symtab->SymbolAtIndex(N_COMM_indexes.back());
                        symbol_ptr->SetByteSize(sym_idx + 1);
                        symbol_ptr->SetSizeIsSibling(true);
                        N_COMM_indexes.pop_back();
                    }
                    type = eSymbolTypeScopeEnd;
                    break;

                case StabLength:
                    // N_LENG - second stab entry with length information
                    type = eSymbolTypeAdditional;
                    break;

                default: break;
                }
            }
            else
            {
                //uint8_t n_pext    = NlistMaskPrivateExternal & nlist.n_type;
                uint8_t n_type  = NlistMaskType & nlist.n_type;
                sym[sym_idx].SetExternal((NlistMaskExternal & nlist.n_type) != 0);

                switch (n_type)
                {
                case NListTypeIndirect:         // N_INDR - Fall through
                case NListTypePreboundUndefined:// N_PBUD - Fall through
                case NListTypeUndefined:        // N_UNDF
                    type = eSymbolTypeUndefined;
                    break;

                case NListTypeAbsolute:         // N_ABS
                    type = eSymbolTypeAbsolute;
                    break;

                case NListTypeSection:          // N_SECT
                    {
                        symbol_section = section_info.GetSection (nlist.n_sect, nlist.n_value);

                        if (!symbol_section)
                        {
                            // TODO: warn about this?
                            add_nlist = false;
                            break;
                        }

                        if (TEXT_eh_frame_sectID == nlist.n_sect)
                        {
                            type = eSymbolTypeException;
                        }
                        else
                        {
                            uint32_t section_type = symbol_section->Get() & SectionFlagMaskSectionType;

                            switch (section_type)
                            {
                            case SectionTypeRegular:                     break; // regular section
                            //case SectionTypeZeroFill:                 type = eSymbolTypeData;    break; // zero fill on demand section
                            case SectionTypeCStringLiterals:            type = eSymbolTypeData;    break; // section with only literal C strings
                            case SectionType4ByteLiterals:              type = eSymbolTypeData;    break; // section with only 4 byte literals
                            case SectionType8ByteLiterals:              type = eSymbolTypeData;    break; // section with only 8 byte literals
                            case SectionTypeLiteralPointers:            type = eSymbolTypeTrampoline; break; // section with only pointers to literals
                            case SectionTypeNonLazySymbolPointers:      type = eSymbolTypeTrampoline; break; // section with only non-lazy symbol pointers
                            case SectionTypeLazySymbolPointers:         type = eSymbolTypeTrampoline; break; // section with only lazy symbol pointers
                            case SectionTypeSymbolStubs:                type = eSymbolTypeTrampoline; break; // section with only symbol stubs, byte size of stub in the reserved2 field
                            case SectionTypeModuleInitFunctionPointers: type = eSymbolTypeCode;    break; // section with only function pointers for initialization
                            case SectionTypeModuleTermFunctionPointers: type = eSymbolTypeCode;    break; // section with only function pointers for termination
                            //case SectionTypeCoalesced:                type = eSymbolType;    break; // section contains symbols that are to be coalesced
                            //case SectionTypeZeroFillLarge:            type = eSymbolTypeData;    break; // zero fill on demand section (that can be larger than 4 gigabytes)
                            case SectionTypeInterposing:                type = eSymbolTypeTrampoline;  break; // section with only pairs of function pointers for interposing
                            case SectionType16ByteLiterals:             type = eSymbolTypeData;    break; // section with only 16 byte literals
                            case SectionTypeDTraceObjectFormat:         type = eSymbolTypeInstrumentation; break;
                            case SectionTypeLazyDylibSymbolPointers:    type = eSymbolTypeTrampoline; break;
                            default: break;
                            }

                            if (type == eSymbolTypeInvalid)
                            {
                                const char *symbol_sect_name = symbol_section->GetName().AsCString();
                                if (symbol_section->IsDescendant (text_section_sp.get()))
                                {
                                    if (symbol_section->IsClear(SectionAttrUserPureInstructions | 
                                                                SectionAttrUserSelfModifyingCode | 
                                                                SectionAttrSytemSomeInstructions))
                                        type = eSymbolTypeData;
                                    else
                                        type = eSymbolTypeCode;
                                }
                                else
                                if (symbol_section->IsDescendant(data_section_sp.get()))
                                {
                                    if (symbol_sect_name && ::strstr (symbol_sect_name, "__objc") == symbol_sect_name)
                                    {
                                        type = eSymbolTypeRuntime;

                                        if (symbol_name && 
                                            symbol_name[0] == '_' && 
                                            symbol_name[1] == 'O' && 
                                            symbol_name[2] == 'B')
                                        {
                                            llvm::StringRef symbol_name_ref(symbol_name);
                                            static const llvm::StringRef g_objc_v2_prefix_class ("_OBJC_CLASS_$_");
                                            static const llvm::StringRef g_objc_v2_prefix_metaclass ("_OBJC_METACLASS_$_");
                                            static const llvm::StringRef g_objc_v2_prefix_ivar ("_OBJC_IVAR_$_");
                                            if (symbol_name_ref.startswith(g_objc_v2_prefix_class))
                                            {
                                                symbol_name_non_abi_mangled = symbol_name + 1;
                                                symbol_name = symbol_name + g_objc_v2_prefix_class.size();
                                                type = eSymbolTypeObjCClass;
                                            }
                                            else if (symbol_name_ref.startswith(g_objc_v2_prefix_metaclass))
                                            {
                                                symbol_name_non_abi_mangled = symbol_name + 1;
                                                symbol_name = symbol_name + g_objc_v2_prefix_metaclass.size();
                                                type = eSymbolTypeObjCMetaClass;
                                            }
                                            else if (symbol_name_ref.startswith(g_objc_v2_prefix_ivar))
                                            {
                                                symbol_name_non_abi_mangled = symbol_name + 1;
                                                symbol_name = symbol_name + g_objc_v2_prefix_ivar.size();
                                                type = eSymbolTypeObjCIVar;
                                            }
                                        }
                                    }
                                    else
                                    if (symbol_sect_name && ::strstr (symbol_sect_name, "__gcc_except_tab") == symbol_sect_name)
                                    {
                                        type = eSymbolTypeException;
                                    }
                                    else
                                    {
                                        type = eSymbolTypeData;
                                    }
                                }
                                else
                                if (symbol_sect_name && ::strstr (symbol_sect_name, "__IMPORT") == symbol_sect_name)
                                {
                                    type = eSymbolTypeTrampoline;
                                }
                                else
                                if (symbol_section->IsDescendant(objc_section_sp.get()))
                                {
                                    type = eSymbolTypeRuntime;
                                    if (symbol_name && symbol_name[0] == '.')
                                    {
                                        llvm::StringRef symbol_name_ref(symbol_name);
                                        static const llvm::StringRef g_objc_v1_prefix_class (".objc_class_name_");
                                        if (symbol_name_ref.startswith(g_objc_v1_prefix_class))
                                        {
                                            symbol_name_non_abi_mangled = symbol_name;
                                            symbol_name = symbol_name + g_objc_v1_prefix_class.size();
                                            type = eSymbolTypeObjCClass;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    break;
                }                            
            }

            if (add_nlist)
            {
                uint64_t symbol_value = nlist.n_value;
                bool symbol_name_is_mangled = false;

                if (symbol_name_non_abi_mangled)
                {
                    sym[sym_idx].GetMangled().SetMangledName (ConstString(symbol_name_non_abi_mangled));
                    sym[sym_idx].GetMangled().SetDemangledName (ConstString(symbol_name));
                }
                else
                {
                    if (symbol_name && symbol_name[0] == '_')
                    {
                        symbol_name_is_mangled = symbol_name[1] == '_';
                        symbol_name++;  // Skip the leading underscore
                    }

                    if (symbol_name)
                    {
                        sym[sym_idx].GetMangled().SetValue(ConstString(symbol_name), symbol_name_is_mangled);
                    }
                }

                if (is_debug == false)
                {
                    if (type == eSymbolTypeCode)
                    {
                        // See if we can find a N_FUN entry for any code symbols.
                        // If we do find a match, and the name matches, then we
                        // can merge the two into just the function symbol to avoid
                        // duplicate entries in the symbol table
                        ValueToSymbolIndexMap::const_iterator pos = N_FUN_addr_to_sym_idx.find (nlist.n_value);
                        if (pos != N_FUN_addr_to_sym_idx.end())
                        {
                            if ((symbol_name_is_mangled == true && sym[sym_idx].GetMangled().GetMangledName() == sym[pos->second].GetMangled().GetMangledName()) ||
                                (symbol_name_is_mangled == false && sym[sym_idx].GetMangled().GetDemangledName() == sym[pos->second].GetMangled().GetDemangledName()))
                            {
                                m_nlist_idx_to_sym_idx[nlist_idx] = pos->second;
                                // We just need the flags from the linker symbol, so put these flags
                                // into the N_FUN flags to avoid duplicate symbols in the symbol table
                                sym[pos->second].SetFlags (nlist.n_type << 16 | nlist.n_desc);
                                sym[sym_idx].Clear();
                                continue;
                            }
                        }
                    }
                    else if (type == eSymbolTypeData)
                    {
                        // See if we can find a N_STSYM entry for any data symbols.
                        // If we do find a match, and the name matches, then we
                        // can merge the two into just the Static symbol to avoid
                        // duplicate entries in the symbol table
                        ValueToSymbolIndexMap::const_iterator pos = N_STSYM_addr_to_sym_idx.find (nlist.n_value);
                        if (pos != N_STSYM_addr_to_sym_idx.end())
                        {
                            if ((symbol_name_is_mangled == true && sym[sym_idx].GetMangled().GetMangledName() == sym[pos->second].GetMangled().GetMangledName()) ||
                                (symbol_name_is_mangled == false && sym[sym_idx].GetMangled().GetDemangledName() == sym[pos->second].GetMangled().GetDemangledName()))
                            {
                                m_nlist_idx_to_sym_idx[nlist_idx] = pos->second;
                                // We just need the flags from the linker symbol, so put these flags
                                // into the N_STSYM flags to avoid duplicate symbols in the symbol table
                                sym[pos->second].SetFlags (nlist.n_type << 16 | nlist.n_desc);
                                sym[sym_idx].Clear();
                                continue;
                            }
                        }
                    }
                }
                if (symbol_section)
                {
                    const addr_t section_file_addr = symbol_section->GetFileAddress();
                    if (symbol_byte_size == 0 && function_starts_count > 0)
                    {
                        addr_t symbol_lookup_file_addr = nlist.n_value;
                        // Do an exact address match for non-ARM addresses, else get the closest since
                        // the symbol might be a thumb symbol which has an address with bit zero set
                        FunctionStarts::Entry *func_start_entry = function_starts.FindEntry (symbol_lookup_file_addr, !is_arm);
                        if (is_arm && func_start_entry)
                        {
                            // Verify that the function start address is the symbol address (ARM)
                            // or the symbol address + 1 (thumb)
                            if (func_start_entry->addr != symbol_lookup_file_addr &&
                                func_start_entry->addr != (symbol_lookup_file_addr + 1))
                            {
                                // Not the right entry, NULL it out...
                                func_start_entry = NULL;
                            }
                        }
                        if (func_start_entry)
                        {
                            func_start_entry->data = true;
                            
                            addr_t symbol_file_addr = func_start_entry->addr;
                            if (is_arm)
                                symbol_file_addr &= 0xfffffffffffffffeull;

                            const FunctionStarts::Entry *next_func_start_entry = function_starts.FindNextEntry (func_start_entry);
                            const addr_t section_end_file_addr = section_file_addr + symbol_section->GetByteSize();
                            if (next_func_start_entry)
                            {
                                addr_t next_symbol_file_addr = next_func_start_entry->addr;
                                // Be sure the clear the Thumb address bit when we calculate the size
                                // from the current and next address
                                if (is_arm)
                                    next_symbol_file_addr &= 0xfffffffffffffffeull;
                                symbol_byte_size = std::min<lldb::addr_t>(next_symbol_file_addr - symbol_file_addr, section_end_file_addr - symbol_file_addr);
                            }
                            else
                            {
                                symbol_byte_size = section_end_file_addr - symbol_file_addr;
                            }
                        }
                    }
                    symbol_value -= section_file_addr;
                }

                sym[sym_idx].SetID (nlist_idx);
                sym[sym_idx].SetType (type);
                sym[sym_idx].GetAddress().SetSection (symbol_section);
                sym[sym_idx].GetAddress().SetOffset (symbol_value);
                sym[sym_idx].SetFlags (nlist.n_type << 16 | nlist.n_desc);

                if (symbol_byte_size > 0)
                    sym[sym_idx].SetByteSize(symbol_byte_size);

                ++sym_idx;
            }
            else
            {
                sym[sym_idx].Clear();
            }

        }

        // STAB N_GSYM entries end up having a symbol type eSymbolTypeGlobal and when the symbol value
        // is zero, the address of the global ends up being in a non-STAB entry. Try and fix up all
        // such entries by figuring out what the address for the global is by looking up this non-STAB
        // entry and copying the value into the debug symbol's value to save us the hassle in the
        // debug symbol parser.

        Symbol *global_symbol = NULL;
        for (nlist_idx = 0;
             nlist_idx < symtab_load_command.nsyms && (global_symbol = symtab->FindSymbolWithType (eSymbolTypeData, Symtab::eDebugYes, Symtab::eVisibilityAny, nlist_idx)) != NULL;
             nlist_idx++)
        {
            if (global_symbol->GetAddress().GetFileAddress() == 0)
            {
                std::vector<uint32_t> indexes;
                if (symtab->AppendSymbolIndexesWithName (global_symbol->GetMangled().GetName(), indexes) > 0)
                {
                    std::vector<uint32_t>::const_iterator pos;
                    std::vector<uint32_t>::const_iterator end = indexes.end();
                    for (pos = indexes.begin(); pos != end; ++pos)
                    {
                        symbol_ptr = symtab->SymbolAtIndex(*pos);
                        if (symbol_ptr != global_symbol && symbol_ptr->IsDebug() == false)
                        {
                            global_symbol->GetAddress() = symbol_ptr->GetAddress();
                            break;
                        }
                    }
                }
            }
        }
        
        uint32_t synthetic_sym_id = symtab_load_command.nsyms;

        if (function_starts_count > 0)
        {
            char synthetic_function_symbol[PATH_MAX];
            uint32_t num_synthetic_function_symbols = 0;
            for (i=0; i<function_starts_count; ++i)
            {
                if (function_starts.GetEntryRef (i).data == false)
                    ++num_synthetic_function_symbols;
            }
            
            if (num_synthetic_function_symbols > 0)
            {
                if (num_syms < sym_idx + num_synthetic_function_symbols)
                {
                    num_syms = sym_idx + num_synthetic_function_symbols;
                    sym = symtab->Resize (num_syms);
                }
                uint32_t synthetic_function_symbol_idx = 0;
                for (i=0; i<function_starts_count; ++i)
                {
                    const FunctionStarts::Entry *func_start_entry = function_starts.GetEntryAtIndex (i);
                    if (func_start_entry->data == false)
                    {
                        addr_t symbol_file_addr = func_start_entry->addr;
                        uint32_t symbol_flags = 0;
                        if (is_arm)
                        {
                            if (symbol_file_addr & 1)
                                symbol_flags = MACHO_NLIST_ARM_SYMBOL_IS_THUMB;
                            symbol_file_addr &= 0xfffffffffffffffeull;
                        }
                        Address symbol_addr;
                        if (module_sp->ResolveFileAddress (symbol_file_addr, symbol_addr))
                        {
                            SectionSP symbol_section (symbol_addr.GetSection());
                            uint32_t symbol_byte_size = 0;
                            if (symbol_section)
                            {
                                const addr_t section_file_addr = symbol_section->GetFileAddress();
                                const FunctionStarts::Entry *next_func_start_entry = function_starts.FindNextEntry (func_start_entry);
                                const addr_t section_end_file_addr = section_file_addr + symbol_section->GetByteSize();
                                if (next_func_start_entry)
                                {
                                    addr_t next_symbol_file_addr = next_func_start_entry->addr;
                                    if (is_arm)
                                        next_symbol_file_addr &= 0xfffffffffffffffeull;
                                    symbol_byte_size = std::min<lldb::addr_t>(next_symbol_file_addr - symbol_file_addr, section_end_file_addr - symbol_file_addr);
                                }
                                else
                                {
                                    symbol_byte_size = section_end_file_addr - symbol_file_addr;
                                }
                                snprintf (synthetic_function_symbol,
                                          sizeof(synthetic_function_symbol),
                                          "___lldb_unnamed_function%u$$%s",
                                          ++synthetic_function_symbol_idx,
                                          module_sp->GetFileSpec().GetFilename().GetCString());
                                sym[sym_idx].SetID (synthetic_sym_id++);
                                sym[sym_idx].GetMangled().SetDemangledName(ConstString(synthetic_function_symbol));
                                sym[sym_idx].SetType (eSymbolTypeCode);
                                sym[sym_idx].SetIsSynthetic (true);
                                sym[sym_idx].GetAddress() = symbol_addr;
                                if (symbol_flags)
                                    sym[sym_idx].SetFlags (symbol_flags);
                                if (symbol_byte_size)
                                    sym[sym_idx].SetByteSize (symbol_byte_size);
                                ++sym_idx;
                            }
                        }
                    }
                }
            }
        }

        // Trim our symbols down to just what we ended up with after
        // removing any symbols.
        if (sym_idx < num_syms)
        {
            num_syms = sym_idx;
            sym = symtab->Resize (num_syms);
        }

        // Now synthesize indirect symbols
        if (m_dysymtab.nindirectsyms != 0)
        {
            DataExtractor indirect_symbol_index_data (m_data, m_dysymtab.indirectsymoff, m_dysymtab.nindirectsyms * 4);

            if (indirect_symbol_index_data.GetByteSize())
            {
                NListIndexToSymbolIndexMap::const_iterator end_index_pos = m_nlist_idx_to_sym_idx.end();

                for (uint32_t sect_idx = 1; sect_idx < m_mach_sections.size(); ++sect_idx)
                {
                    if ((m_mach_sections[sect_idx].flags & SectionFlagMaskSectionType) == SectionTypeSymbolStubs)
                    {
                        uint32_t symbol_stub_byte_size = m_mach_sections[sect_idx].reserved2;
                        if (symbol_stub_byte_size == 0)
                            continue;

                        const uint32_t num_symbol_stubs = m_mach_sections[sect_idx].size / symbol_stub_byte_size;

                        if (num_symbol_stubs == 0)
                            continue;

                        const uint32_t symbol_stub_index_offset = m_mach_sections[sect_idx].reserved1;
                        for (uint32_t stub_idx = 0; stub_idx < num_symbol_stubs; ++stub_idx)
                        {
                            const uint32_t symbol_stub_index = symbol_stub_index_offset + stub_idx;
                            const lldb::addr_t symbol_stub_addr = m_mach_sections[sect_idx].addr + (stub_idx * symbol_stub_byte_size);
                            uint32_t symbol_stub_offset = symbol_stub_index * 4;
                            if (indirect_symbol_index_data.ValidOffsetForDataOfSize(symbol_stub_offset, 4))
                            {
                                const uint32_t stub_sym_id = indirect_symbol_index_data.GetU32 (&symbol_stub_offset);
                                if (stub_sym_id & (IndirectSymbolAbsolute | IndirectSymbolLocal))
                                    continue;

                                NListIndexToSymbolIndexMap::const_iterator index_pos = m_nlist_idx_to_sym_idx.find (stub_sym_id);
                                Symbol *stub_symbol = NULL;
                                if (index_pos != end_index_pos)
                                {
                                    // We have a remapping from the original nlist index to
                                    // a current symbol index, so just look this up by index
                                    stub_symbol = symtab->SymbolAtIndex (index_pos->second);
                                }
                                else 
                                {
                                    // We need to lookup a symbol using the original nlist
                                    // symbol index since this index is coming from the 
                                    // S_SYMBOL_STUBS
                                    stub_symbol = symtab->FindSymbolByID (stub_sym_id);
                                }

                                if (stub_symbol)
                                {
                                    Address so_addr(symbol_stub_addr, section_list);

                                    if (stub_symbol->GetType() == eSymbolTypeUndefined)
                                    {
                                        // Change the external symbol into a trampoline that makes sense
                                        // These symbols were N_UNDF N_EXT, and are useless to us, so we
                                        // can re-use them so we don't have to make up a synthetic symbol
                                        // for no good reason.
                                        stub_symbol->SetType (eSymbolTypeTrampoline);
                                        stub_symbol->SetExternal (false);
                                        stub_symbol->GetAddress() = so_addr;
                                        stub_symbol->SetByteSize (symbol_stub_byte_size);
                                    }
                                    else
                                    {
                                        // Make a synthetic symbol to describe the trampoline stub
                                        Mangled stub_symbol_mangled_name(stub_symbol->GetMangled());
                                        if (sym_idx >= num_syms)
                                        {
                                            sym = symtab->Resize (++num_syms);
                                            stub_symbol = NULL;  // this pointer no longer valid
                                        }
                                        sym[sym_idx].SetID (synthetic_sym_id++);
                                        sym[sym_idx].GetMangled() = stub_symbol_mangled_name;
                                        sym[sym_idx].SetType (eSymbolTypeTrampoline);
                                        sym[sym_idx].SetIsSynthetic (true);
                                        sym[sym_idx].GetAddress() = so_addr;
                                        sym[sym_idx].SetByteSize (symbol_stub_byte_size);
                                        ++sym_idx;
                                    }
                                }
                                else
                                {
                                    if (log)
                                        log->Warning ("symbol stub referencing symbol table symbol %u that isn't in our minimal symbol table, fix this!!!", stub_sym_id);
                                }
                            }
                        }
                    }
                }
            }
        }
        return symtab->GetNumSymbols();
    }
    return 0;
}


void
ObjectFileMachO::Dump (Stream *s)
{
    ModuleSP module_sp(GetModule());
    if (module_sp)
    {
        lldb_private::Mutex::Locker locker(module_sp->GetMutex());
        s->Printf("%p: ", this);
        s->Indent();
        if (m_header.magic == HeaderMagic64 || m_header.magic == HeaderMagic64Swapped)
            s->PutCString("ObjectFileMachO64");
        else
            s->PutCString("ObjectFileMachO32");

        ArchSpec header_arch(eArchTypeMachO, m_header.cputype, m_header.cpusubtype);

        *s << ", file = '" << m_file << "', arch = " << header_arch.GetArchitectureName() << "\n";

        if (m_sections_ap.get())
            m_sections_ap->Dump(s, NULL, true, UINT32_MAX);

        if (m_symtab_ap.get())
            m_symtab_ap->Dump(s, NULL, eSortOrderNone);
    }
}


bool
ObjectFileMachO::GetUUID (lldb_private::UUID* uuid)
{
    ModuleSP module_sp(GetModule());
    if (module_sp)
    {
        lldb_private::Mutex::Locker locker(module_sp->GetMutex());
        struct uuid_command load_cmd;
        uint32_t offset = MachHeaderSizeFromMagic(m_header.magic);
        uint32_t i;
        for (i=0; i<m_header.ncmds; ++i)
        {
            const uint32_t cmd_offset = offset;
            if (m_data.GetU32(&offset, &load_cmd, 2) == NULL)
                break;

            if (load_cmd.cmd == LoadCommandUUID)
            {
                const uint8_t *uuid_bytes = m_data.PeekData(offset, 16);
                
                if (uuid_bytes)
                {
                    // OpenCL on Mac OS X uses the same UUID for each of its object files.
                    // We pretend these object files have no UUID to prevent crashing.
                    
                    const uint8_t opencl_uuid[] = { 0x8c, 0x8e, 0xb3, 0x9b,
                                                    0x3b, 0xa8,
                                                    0x4b, 0x16,
                                                    0xb6, 0xa4,
                                                    0x27, 0x63, 0xbb, 0x14, 0xf0, 0x0d };
                    
                    if (!memcmp(uuid_bytes, opencl_uuid, 16))
                        return false;
                    
                    uuid->SetBytes (uuid_bytes);
                    return true;
                }
                return false;
            }
            offset = cmd_offset + load_cmd.cmdsize;
        }
    }
    return false;
}


uint32_t
ObjectFileMachO::GetDependentModules (FileSpecList& files)
{
    uint32_t count = 0;
    ModuleSP module_sp(GetModule());
    if (module_sp)
    {
        lldb_private::Mutex::Locker locker(module_sp->GetMutex());
        struct load_command load_cmd;
        uint32_t offset = MachHeaderSizeFromMagic(m_header.magic);
        const bool resolve_path = false; // Don't resolve the dependend file paths since they may not reside on this system
        uint32_t i;
        for (i=0; i<m_header.ncmds; ++i)
        {
            const uint32_t cmd_offset = offset;
            if (m_data.GetU32(&offset, &load_cmd, 2) == NULL)
                break;

            switch (load_cmd.cmd)
            {
            case LoadCommandDylibLoad:
            case LoadCommandDylibLoadWeak:
            case LoadCommandDylibReexport:
            case LoadCommandDynamicLinkerLoad:
            case LoadCommandFixedVMShlibLoad:
            case LoadCommandDylibLoadUpward:
                {
                    uint32_t name_offset = cmd_offset + m_data.GetU32(&offset);
                    const char *path = m_data.PeekCStr(name_offset);
                    // Skip any path that starts with '@' since these are usually:
                    // @executable_path/.../file
                    // @rpath/.../file
                    if (path && path[0] != '@')
                    {
                        FileSpec file_spec(path, resolve_path);
                        if (files.AppendIfUnique(file_spec))
                            count++;
                    }
                }
                break;

            default:
                break;
            }
            offset = cmd_offset + load_cmd.cmdsize;
        }
    }
    return count;
}

lldb_private::Address
ObjectFileMachO::GetEntryPointAddress () 
{
    // If the object file is not an executable it can't hold the entry point.  m_entry_point_address
    // is initialized to an invalid address, so we can just return that.
    // If m_entry_point_address is valid it means we've found it already, so return the cached value.
    
    if (!IsExecutable() || m_entry_point_address.IsValid())
        return m_entry_point_address;
    
    // Otherwise, look for the UnixThread or Thread command.  The data for the Thread command is given in 
    // /usr/include/mach-o.h, but it is basically:
    //
    //  uint32_t flavor  - this is the flavor argument you would pass to thread_get_state
    //  uint32_t count   - this is the count of longs in the thread state data
    //  struct XXX_thread_state state - this is the structure from <machine/thread_status.h> corresponding to the flavor.
    //  <repeat this trio>
    // 
    // So we just keep reading the various register flavors till we find the GPR one, then read the PC out of there.
    // FIXME: We will need to have a "RegisterContext data provider" class at some point that can get all the registers
    // out of data in this form & attach them to a given thread.  That should underlie the MacOS X User process plugin,
    // and we'll also need it for the MacOS X Core File process plugin.  When we have that we can also use it here.
    //
    // For now we hard-code the offsets and flavors we need:
    //
    //

    ModuleSP module_sp(GetModule());
    if (module_sp)
    {
        lldb_private::Mutex::Locker locker(module_sp->GetMutex());
        struct load_command load_cmd;
        uint32_t offset = MachHeaderSizeFromMagic(m_header.magic);
        uint32_t i;
        lldb::addr_t start_address = LLDB_INVALID_ADDRESS;
        bool done = false;
        
        for (i=0; i<m_header.ncmds; ++i)
        {
            const uint32_t cmd_offset = offset;
            if (m_data.GetU32(&offset, &load_cmd, 2) == NULL)
                break;

            switch (load_cmd.cmd)
            {
            case LoadCommandUnixThread:
            case LoadCommandThread:
                {
                    while (offset < cmd_offset + load_cmd.cmdsize)
                    {
                        uint32_t flavor = m_data.GetU32(&offset);
                        uint32_t count = m_data.GetU32(&offset);
                        if (count == 0)
                        {
                            // We've gotten off somehow, log and exit;
                            return m_entry_point_address;
                        }
                        
                        switch (m_header.cputype)
                        {
                        case llvm::MachO::CPUTypeARM:
                           if (flavor == 1) // ARM_THREAD_STATE from mach/arm/thread_status.h
                           {
                               offset += 60;  // This is the offset of pc in the GPR thread state data structure.
                               start_address = m_data.GetU32(&offset);
                               done = true;
                            }
                        break;
                        case llvm::MachO::CPUTypeI386:
                           if (flavor == 1) // x86_THREAD_STATE32 from mach/i386/thread_status.h
                           {
                               offset += 40;  // This is the offset of eip in the GPR thread state data structure.
                               start_address = m_data.GetU32(&offset);
                               done = true;
                            }
                        break;
                        case llvm::MachO::CPUTypeX86_64:
                           if (flavor == 4) // x86_THREAD_STATE64 from mach/i386/thread_status.h
                           {
                               offset += 16 * 8;  // This is the offset of rip in the GPR thread state data structure.
                               start_address = m_data.GetU64(&offset);
                               done = true;
                            }
                        break;
                        default:
                            return m_entry_point_address;
                        }
                        // Haven't found the GPR flavor yet, skip over the data for this flavor:
                        if (done)
                            break;
                        offset += count * 4;
                    }
                }
                break;
            case LoadCommandMain:
                {
                    ConstString text_segment_name ("__TEXT");
                    uint64_t entryoffset = m_data.GetU64(&offset);
                    SectionSP text_segment_sp = GetSectionList()->FindSectionByName(text_segment_name);
                    if (text_segment_sp)
                    {
                        done = true;
                        start_address = text_segment_sp->GetFileAddress() + entryoffset;
                    }
                }

            default:
                break;
            }
            if (done)
                break;

            // Go to the next load command:
            offset = cmd_offset + load_cmd.cmdsize;
        }
        
        if (start_address != LLDB_INVALID_ADDRESS)
        {
            // We got the start address from the load commands, so now resolve that address in the sections 
            // of this ObjectFile:
            if (!m_entry_point_address.ResolveAddressUsingFileSections (start_address, GetSectionList()))
            {
                m_entry_point_address.Clear();
            }
        }
        else
        {
            // We couldn't read the UnixThread load command - maybe it wasn't there.  As a fallback look for the
            // "start" symbol in the main executable.
            
            ModuleSP module_sp (GetModule());
            
            if (module_sp)
            {
                SymbolContextList contexts;
                SymbolContext context;
                if (module_sp->FindSymbolsWithNameAndType(ConstString ("start"), eSymbolTypeCode, contexts))
                {
                    if (contexts.GetContextAtIndex(0, context))
                        m_entry_point_address = context.symbol->GetAddress();
                }
            }
        }
    }
    
    return m_entry_point_address;

}

lldb_private::Address
ObjectFileMachO::GetHeaderAddress ()
{
    lldb_private::Address header_addr;
    SectionList *section_list = GetSectionList();
    if (section_list)
    {
        SectionSP text_segment_sp (section_list->FindSectionByName (GetSegmentNameTEXT()));
        if (text_segment_sp)
        {
            header_addr.SetSection (text_segment_sp);
            header_addr.SetOffset (0);
        }
    }
    return header_addr;
}

uint32_t
ObjectFileMachO::GetNumThreadContexts ()
{
    ModuleSP module_sp(GetModule());
    if (module_sp)
    {
        lldb_private::Mutex::Locker locker(module_sp->GetMutex());
        if (!m_thread_context_offsets_valid)
        {
            m_thread_context_offsets_valid = true;
            uint32_t offset = MachHeaderSizeFromMagic(m_header.magic);
            FileRangeArray::Entry file_range;
            thread_command thread_cmd;
            for (uint32_t i=0; i<m_header.ncmds; ++i)
            {
                const uint32_t cmd_offset = offset;
                if (m_data.GetU32(&offset, &thread_cmd, 2) == NULL)
                    break;
                
                if (thread_cmd.cmd == LoadCommandThread)
                {
                    file_range.SetRangeBase (offset);
                    file_range.SetByteSize (thread_cmd.cmdsize - 8);
                    m_thread_context_offsets.Append (file_range);
                }
                offset = cmd_offset + thread_cmd.cmdsize;
            }
        }
    }
    return m_thread_context_offsets.GetSize();
}

lldb::RegisterContextSP
ObjectFileMachO::GetThreadContextAtIndex (uint32_t idx, lldb_private::Thread &thread)
{
    lldb::RegisterContextSP reg_ctx_sp;

    ModuleSP module_sp(GetModule());
    if (module_sp)
    {
        lldb_private::Mutex::Locker locker(module_sp->GetMutex());
        if (!m_thread_context_offsets_valid)
            GetNumThreadContexts ();

        const FileRangeArray::Entry *thread_context_file_range = m_thread_context_offsets.GetEntryAtIndex (idx);
        if (thread_context_file_range)
        {
        
            DataExtractor data (m_data, 
                                thread_context_file_range->GetRangeBase(), 
                                thread_context_file_range->GetByteSize());

            switch (m_header.cputype)
            {
                case llvm::MachO::CPUTypeARM:
                    reg_ctx_sp.reset (new RegisterContextDarwin_arm_Mach (thread, data));
                    break;
                    
                case llvm::MachO::CPUTypeI386:
                    reg_ctx_sp.reset (new RegisterContextDarwin_i386_Mach (thread, data));
                    break;
                    
                case llvm::MachO::CPUTypeX86_64:
                    reg_ctx_sp.reset (new RegisterContextDarwin_x86_64_Mach (thread, data));
                    break;
            }
        }
    }
    return reg_ctx_sp;
}


ObjectFile::Type
ObjectFileMachO::CalculateType()
{
    switch (m_header.filetype)
    {
        case HeaderFileTypeObject:                                          // 0x1u MH_OBJECT
            if (GetAddressByteSize () == 4)
            {
                // 32 bit kexts are just object files, but they do have a valid
                // UUID load command.
                UUID uuid;
                if (GetUUID(&uuid))
                {
                    // this checking for the UUID load command is not enough
                    // we could eventually look for the symbol named 
                    // "OSKextGetCurrentIdentifier" as this is required of kexts
                    if (m_strata == eStrataInvalid)
                        m_strata = eStrataKernel;
                    return eTypeSharedLibrary;
                }
            }
            return eTypeObjectFile;

        case HeaderFileTypeExecutable:          return eTypeExecutable;     // 0x2u MH_EXECUTE
        case HeaderFileTypeFixedVMShlib:        return eTypeSharedLibrary;  // 0x3u MH_FVMLIB
        case HeaderFileTypeCore:                return eTypeCoreFile;       // 0x4u MH_CORE
        case HeaderFileTypePreloadedExecutable: return eTypeSharedLibrary;  // 0x5u MH_PRELOAD
        case HeaderFileTypeDynamicShlib:        return eTypeSharedLibrary;  // 0x6u MH_DYLIB
        case HeaderFileTypeDynamicLinkEditor:   return eTypeDynamicLinker;  // 0x7u MH_DYLINKER
        case HeaderFileTypeBundle:              return eTypeSharedLibrary;  // 0x8u MH_BUNDLE
        case HeaderFileTypeDynamicShlibStub:    return eTypeStubLibrary;    // 0x9u MH_DYLIB_STUB
        case HeaderFileTypeDSYM:                return eTypeDebugInfo;      // 0xAu MH_DSYM
        case HeaderFileTypeKextBundle:          return eTypeSharedLibrary;  // 0xBu MH_KEXT_BUNDLE
        default:
            break;
    }
    return eTypeUnknown;
}

ObjectFile::Strata
ObjectFileMachO::CalculateStrata()
{
    switch (m_header.filetype)
    {
        case HeaderFileTypeObject:      // 0x1u MH_OBJECT
            {
                // 32 bit kexts are just object files, but they do have a valid
                // UUID load command.
                UUID uuid;
                if (GetUUID(&uuid))
                {
                    // this checking for the UUID load command is not enough
                    // we could eventually look for the symbol named 
                    // "OSKextGetCurrentIdentifier" as this is required of kexts
                    if (m_type == eTypeInvalid)
                        m_type = eTypeSharedLibrary;

                    return eStrataKernel;
                }
            }
            return eStrataUnknown;

        case HeaderFileTypeExecutable:                                     // 0x2u MH_EXECUTE
            // Check for the MH_DYLDLINK bit in the flags
            if (m_header.flags & HeaderFlagBitIsDynamicLinkObject)
            {
                return eStrataUser;
            }
            else 
            {
                SectionList *section_list = GetSectionList();
                if (section_list)
                {
                    static ConstString g_kld_section_name ("__KLD");
                    if (section_list->FindSectionByName(g_kld_section_name))
                        return eStrataKernel;
                }
            }
            return eStrataRawImage;

        case HeaderFileTypeFixedVMShlib:        return eStrataUser;         // 0x3u MH_FVMLIB
        case HeaderFileTypeCore:                return eStrataUnknown;      // 0x4u MH_CORE
        case HeaderFileTypePreloadedExecutable: return eStrataRawImage;     // 0x5u MH_PRELOAD
        case HeaderFileTypeDynamicShlib:        return eStrataUser;         // 0x6u MH_DYLIB
        case HeaderFileTypeDynamicLinkEditor:   return eStrataUser;         // 0x7u MH_DYLINKER
        case HeaderFileTypeBundle:              return eStrataUser;         // 0x8u MH_BUNDLE
        case HeaderFileTypeDynamicShlibStub:    return eStrataUser;         // 0x9u MH_DYLIB_STUB
        case HeaderFileTypeDSYM:                return eStrataUnknown;      // 0xAu MH_DSYM
        case HeaderFileTypeKextBundle:          return eStrataKernel;       // 0xBu MH_KEXT_BUNDLE
        default:
            break;
    }
    return eStrataUnknown;
}


uint32_t
ObjectFileMachO::GetVersion (uint32_t *versions, uint32_t num_versions)
{
    ModuleSP module_sp(GetModule());
    if (module_sp)
    {
        lldb_private::Mutex::Locker locker(module_sp->GetMutex());
        struct dylib_command load_cmd;
        uint32_t offset = MachHeaderSizeFromMagic(m_header.magic);
        uint32_t version_cmd = 0;
        uint64_t version = 0;
        uint32_t i;
        for (i=0; i<m_header.ncmds; ++i)
        {
            const uint32_t cmd_offset = offset;
            if (m_data.GetU32(&offset, &load_cmd, 2) == NULL)
                break;
            
            if (load_cmd.cmd == LoadCommandDylibIdent)
            {
                if (version_cmd == 0)
                {
                    version_cmd = load_cmd.cmd;
                    if (m_data.GetU32(&offset, &load_cmd.dylib, 4) == NULL)
                        break;
                    version = load_cmd.dylib.current_version;
                }
                break; // Break for now unless there is another more complete version 
                       // number load command in the future.
            }
            offset = cmd_offset + load_cmd.cmdsize;
        }
        
        if (version_cmd == LoadCommandDylibIdent)
        {
            if (versions != NULL && num_versions > 0)
            {
                if (num_versions > 0)
                    versions[0] = (version & 0xFFFF0000ull) >> 16;
                if (num_versions > 1)
                    versions[1] = (version & 0x0000FF00ull) >> 8;
                if (num_versions > 2)
                    versions[2] = (version & 0x000000FFull);
                // Fill in an remaining version numbers with invalid values
                for (i=3; i<num_versions; ++i)
                    versions[i] = UINT32_MAX;
            }
            // The LC_ID_DYLIB load command has a version with 3 version numbers
            // in it, so always return 3
            return 3;
        }
    }
    return false;
}

bool
ObjectFileMachO::GetArchitecture (ArchSpec &arch)
{
    ModuleSP module_sp(GetModule());
    if (module_sp)
    {
        lldb_private::Mutex::Locker locker(module_sp->GetMutex());
        arch.SetArchitecture (eArchTypeMachO, m_header.cputype, m_header.cpusubtype);
    
        // Files with type MH_PRELOAD are currently used in cases where the image
        // debugs at the addresses in the file itself. Below we set the OS to 
        // unknown to make sure we use the DynamicLoaderStatic()...
        if (m_header.filetype == HeaderFileTypePreloadedExecutable)
        {
            arch.GetTriple().setOS (llvm::Triple::UnknownOS);
        }
        return true;
    }
    return false;
}


//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
ObjectFileMachO::GetPluginName()
{
    return "ObjectFileMachO";
}

const char *
ObjectFileMachO::GetShortPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
ObjectFileMachO::GetPluginVersion()
{
    return 1;
}

