//===-- ObjectFileELF.cpp ------------------------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ObjectFileELF.h"

#include <cassert>
#include <algorithm>
#include <unordered_map>

#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/DataBuffer.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/FileSpecList.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/Timer.h"
#include "lldb/Symbol/DWARFCallFrameInfo.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/SectionLoadList.h"
#include "lldb/Target/Target.h"

#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"

#define CASE_AND_STREAM(s, def, width)                  \
    case def: s->Printf("%-*s", width, #def); break;

using namespace lldb;
using namespace lldb_private;
using namespace elf;
using namespace llvm::ELF;

namespace {

// ELF note owner definitions
const char *const LLDB_NT_OWNER_FREEBSD = "FreeBSD";
const char *const LLDB_NT_OWNER_GNU     = "GNU";
const char *const LLDB_NT_OWNER_NETBSD  = "NetBSD";
const char *const LLDB_NT_OWNER_CSR     = "csr";
const char *const LLDB_NT_OWNER_ANDROID = "Android";
const char *const LLDB_NT_OWNER_CORE    = "CORE";
const char *const LLDB_NT_OWNER_LINUX   = "LINUX";

// ELF note type definitions
const elf_word LLDB_NT_FREEBSD_ABI_TAG  = 0x01;
const elf_word LLDB_NT_FREEBSD_ABI_SIZE = 4;

const elf_word LLDB_NT_GNU_ABI_TAG      = 0x01;
const elf_word LLDB_NT_GNU_ABI_SIZE     = 16;

const elf_word LLDB_NT_GNU_BUILD_ID_TAG = 0x03;

const elf_word LLDB_NT_NETBSD_ABI_TAG   = 0x01;
const elf_word LLDB_NT_NETBSD_ABI_SIZE  = 4;

// GNU ABI note OS constants
const elf_word LLDB_NT_GNU_ABI_OS_LINUX   = 0x00;
const elf_word LLDB_NT_GNU_ABI_OS_HURD    = 0x01;
const elf_word LLDB_NT_GNU_ABI_OS_SOLARIS = 0x02;

// LLDB_NT_OWNER_CORE and LLDB_NT_OWNER_LINUX note contants
#define NT_PRSTATUS             1
#define NT_PRFPREG              2
#define NT_PRPSINFO             3
#define NT_TASKSTRUCT           4
#define NT_AUXV                 6
#define NT_SIGINFO              0x53494749
#define NT_FILE                 0x46494c45
#define NT_PRXFPREG             0x46e62b7f
#define NT_PPC_VMX              0x100
#define NT_PPC_SPE              0x101
#define NT_PPC_VSX              0x102
#define NT_386_TLS              0x200
#define NT_386_IOPERM           0x201
#define NT_X86_XSTATE           0x202
#define NT_S390_HIGH_GPRS       0x300
#define NT_S390_TIMER           0x301
#define NT_S390_TODCMP          0x302
#define NT_S390_TODPREG         0x303
#define NT_S390_CTRS            0x304
#define NT_S390_PREFIX          0x305
#define NT_S390_LAST_BREAK      0x306
#define NT_S390_SYSTEM_CALL     0x307
#define NT_S390_TDB             0x308
#define NT_S390_VXRS_LOW        0x309
#define NT_S390_VXRS_HIGH       0x30a
#define NT_ARM_VFP              0x400
#define NT_ARM_TLS              0x401
#define NT_ARM_HW_BREAK         0x402
#define NT_ARM_HW_WATCH         0x403
#define NT_ARM_SYSTEM_CALL      0x404
#define NT_METAG_CBUF           0x500
#define NT_METAG_RPIPE          0x501
#define NT_METAG_TLS            0x502

//===----------------------------------------------------------------------===//
/// @class ELFRelocation
/// @brief Generic wrapper for ELFRel and ELFRela.
///
/// This helper class allows us to parse both ELFRel and ELFRela relocation
/// entries in a generic manner.
class ELFRelocation
{
public:

    /// Constructs an ELFRelocation entry with a personality as given by @p
    /// type.
    ///
    /// @param type Either DT_REL or DT_RELA.  Any other value is invalid.
    ELFRelocation(unsigned type);

    ~ELFRelocation();

    bool
    Parse(const lldb_private::DataExtractor &data, lldb::offset_t *offset);

    static unsigned
    RelocType32(const ELFRelocation &rel);

    static unsigned
    RelocType64(const ELFRelocation &rel);

    static unsigned
    RelocSymbol32(const ELFRelocation &rel);

    static unsigned
    RelocSymbol64(const ELFRelocation &rel);

    static unsigned
    RelocOffset32(const ELFRelocation &rel);

    static unsigned
    RelocOffset64(const ELFRelocation &rel);

    static unsigned
    RelocAddend32(const ELFRelocation &rel);

    static unsigned
    RelocAddend64(const ELFRelocation &rel);

private:
    typedef llvm::PointerUnion<ELFRel*, ELFRela*> RelocUnion;

    RelocUnion reloc;
};

ELFRelocation::ELFRelocation(unsigned type)
{
    if (type == DT_REL || type == SHT_REL)
        reloc = new ELFRel();
    else if (type == DT_RELA || type == SHT_RELA)
        reloc = new ELFRela();
    else {
        assert(false && "unexpected relocation type");
        reloc = static_cast<ELFRel*>(NULL);
    }
}

ELFRelocation::~ELFRelocation()
{
    if (reloc.is<ELFRel*>())
        delete reloc.get<ELFRel*>();
    else
        delete reloc.get<ELFRela*>();
}

bool
ELFRelocation::Parse(const lldb_private::DataExtractor &data, lldb::offset_t *offset)
{
    if (reloc.is<ELFRel*>())
        return reloc.get<ELFRel*>()->Parse(data, offset);
    else
        return reloc.get<ELFRela*>()->Parse(data, offset);
}

unsigned
ELFRelocation::RelocType32(const ELFRelocation &rel)
{
    if (rel.reloc.is<ELFRel*>())
        return ELFRel::RelocType32(*rel.reloc.get<ELFRel*>());
    else
        return ELFRela::RelocType32(*rel.reloc.get<ELFRela*>());
}

unsigned
ELFRelocation::RelocType64(const ELFRelocation &rel)
{
    if (rel.reloc.is<ELFRel*>())
        return ELFRel::RelocType64(*rel.reloc.get<ELFRel*>());
    else
        return ELFRela::RelocType64(*rel.reloc.get<ELFRela*>());
}

unsigned
ELFRelocation::RelocSymbol32(const ELFRelocation &rel)
{
    if (rel.reloc.is<ELFRel*>())
        return ELFRel::RelocSymbol32(*rel.reloc.get<ELFRel*>());
    else
        return ELFRela::RelocSymbol32(*rel.reloc.get<ELFRela*>());
}

unsigned
ELFRelocation::RelocSymbol64(const ELFRelocation &rel)
{
    if (rel.reloc.is<ELFRel*>())
        return ELFRel::RelocSymbol64(*rel.reloc.get<ELFRel*>());
    else
        return ELFRela::RelocSymbol64(*rel.reloc.get<ELFRela*>());
}

unsigned
ELFRelocation::RelocOffset32(const ELFRelocation &rel)
{
    if (rel.reloc.is<ELFRel*>())
        return rel.reloc.get<ELFRel*>()->r_offset;
    else
        return rel.reloc.get<ELFRela*>()->r_offset;
}

unsigned
ELFRelocation::RelocOffset64(const ELFRelocation &rel)
{
    if (rel.reloc.is<ELFRel*>())
        return rel.reloc.get<ELFRel*>()->r_offset;
    else
        return rel.reloc.get<ELFRela*>()->r_offset;
}

unsigned
ELFRelocation::RelocAddend32(const ELFRelocation &rel)
{
    if (rel.reloc.is<ELFRel*>())
        return 0;
    else
        return rel.reloc.get<ELFRela*>()->r_addend;
}

unsigned
ELFRelocation::RelocAddend64(const ELFRelocation &rel)
{
    if (rel.reloc.is<ELFRel*>())
        return 0;
    else
        return rel.reloc.get<ELFRela*>()->r_addend;
}

} // end anonymous namespace

bool
ELFNote::Parse(const DataExtractor &data, lldb::offset_t *offset)
{
    // Read all fields.
    if (data.GetU32(offset, &n_namesz, 3) == NULL)
        return false;

    // The name field is required to be nul-terminated, and n_namesz
    // includes the terminating nul in observed implementations (contrary
    // to the ELF-64 spec).  A special case is needed for cores generated
    // by some older Linux versions, which write a note named "CORE"
    // without a nul terminator and n_namesz = 4.
    if (n_namesz == 4)
    {
        char buf[4];
        if (data.ExtractBytes (*offset, 4, data.GetByteOrder(), buf) != 4)
            return false;
        if (strncmp (buf, "CORE", 4) == 0)
        {
            n_name = "CORE";
            *offset += 4;
            return true;
        }
    }

    const char *cstr = data.GetCStr(offset, llvm::RoundUpToAlignment (n_namesz, 4));
    if (cstr == NULL)
    {
        Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_SYMBOLS));
        if (log)
            log->Printf("Failed to parse note name lacking nul terminator");

        return false;
    }
    n_name = cstr;
    return true;
}

static uint32_t
kalimbaVariantFromElfFlags(const elf::elf_word e_flags)
{
    const uint32_t dsp_rev = e_flags & 0xFF;
    uint32_t kal_arch_variant = LLDB_INVALID_CPUTYPE;
    switch(dsp_rev)
    {
        // TODO(mg11) Support more variants
        case 10:
            kal_arch_variant = llvm::Triple::KalimbaSubArch_v3;
            break;
        case 14:
            kal_arch_variant = llvm::Triple::KalimbaSubArch_v4;
            break;
        case 17:
        case 20:
            kal_arch_variant = llvm::Triple::KalimbaSubArch_v5;
            break;
        default:
            break;
    }
    return kal_arch_variant;
}

static uint32_t
mipsVariantFromElfFlags(const elf::elf_word e_flags, uint32_t endian)
{
    const uint32_t mips_arch = e_flags & llvm::ELF::EF_MIPS_ARCH;
    uint32_t arch_variant = ArchSpec::eMIPSSubType_unknown;

    switch (mips_arch)
    {
        case llvm::ELF::EF_MIPS_ARCH_1:
        case llvm::ELF::EF_MIPS_ARCH_2:
        case llvm::ELF::EF_MIPS_ARCH_32:
            return (endian == ELFDATA2LSB) ? ArchSpec::eMIPSSubType_mips32el : ArchSpec::eMIPSSubType_mips32;
        case llvm::ELF::EF_MIPS_ARCH_32R2:
            return (endian == ELFDATA2LSB) ? ArchSpec::eMIPSSubType_mips32r2el : ArchSpec::eMIPSSubType_mips32r2;
        case llvm::ELF::EF_MIPS_ARCH_32R6:
            return (endian == ELFDATA2LSB) ? ArchSpec::eMIPSSubType_mips32r6el : ArchSpec::eMIPSSubType_mips32r6;
        case llvm::ELF::EF_MIPS_ARCH_3:
        case llvm::ELF::EF_MIPS_ARCH_4:
        case llvm::ELF::EF_MIPS_ARCH_5:
        case llvm::ELF::EF_MIPS_ARCH_64:
            return (endian == ELFDATA2LSB) ? ArchSpec::eMIPSSubType_mips64el : ArchSpec::eMIPSSubType_mips64;
        case llvm::ELF::EF_MIPS_ARCH_64R2:
            return (endian == ELFDATA2LSB) ? ArchSpec::eMIPSSubType_mips64r2el : ArchSpec::eMIPSSubType_mips64r2;
        case llvm::ELF::EF_MIPS_ARCH_64R6:
            return (endian == ELFDATA2LSB) ? ArchSpec::eMIPSSubType_mips64r6el : ArchSpec::eMIPSSubType_mips64r6;
        default:
            break;
    }

    return arch_variant;
}

static uint32_t
subTypeFromElfHeader(const elf::ELFHeader& header)
{
    if (header.e_machine == llvm::ELF::EM_MIPS)
        return mipsVariantFromElfFlags (header.e_flags,
            header.e_ident[EI_DATA]);

    return
        llvm::ELF::EM_CSR_KALIMBA == header.e_machine ?
        kalimbaVariantFromElfFlags(header.e_flags) :
        LLDB_INVALID_CPUTYPE;
}

//! The kalimba toolchain identifies a code section as being
//! one with the SHT_PROGBITS set in the section sh_type and the top
//! bit in the 32-bit address field set.
static lldb::SectionType
kalimbaSectionType(
    const elf::ELFHeader& header,
    const elf::ELFSectionHeader& sect_hdr)
{
    if (llvm::ELF::EM_CSR_KALIMBA != header.e_machine)
    {
        return eSectionTypeOther;
    }

    if (llvm::ELF::SHT_NOBITS == sect_hdr.sh_type)
    {
        return eSectionTypeZeroFill;
    }

    if (llvm::ELF::SHT_PROGBITS == sect_hdr.sh_type)
    {
        const lldb::addr_t KAL_CODE_BIT = 1 << 31;
        return KAL_CODE_BIT & sect_hdr.sh_addr ?
             eSectionTypeCode  : eSectionTypeData;
    }

    return eSectionTypeOther;
}

// Arbitrary constant used as UUID prefix for core files.
const uint32_t
ObjectFileELF::g_core_uuid_magic(0xE210C);

//------------------------------------------------------------------
// Static methods.
//------------------------------------------------------------------
void
ObjectFileELF::Initialize()
{
    PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                  GetPluginDescriptionStatic(),
                                  CreateInstance,
                                  CreateMemoryInstance,
                                  GetModuleSpecifications);
}

void
ObjectFileELF::Terminate()
{
    PluginManager::UnregisterPlugin(CreateInstance);
}

lldb_private::ConstString
ObjectFileELF::GetPluginNameStatic()
{
    static ConstString g_name("elf");
    return g_name;
}

const char *
ObjectFileELF::GetPluginDescriptionStatic()
{
    return "ELF object file reader.";
}

ObjectFile *
ObjectFileELF::CreateInstance (const lldb::ModuleSP &module_sp,
                               DataBufferSP &data_sp,
                               lldb::offset_t data_offset,
                               const lldb_private::FileSpec* file,
                               lldb::offset_t file_offset,
                               lldb::offset_t length)
{
    if (!data_sp)
    {
        data_sp = file->MemoryMapFileContentsIfLocal(file_offset, length);
        data_offset = 0;
    }

    if (data_sp && data_sp->GetByteSize() > (llvm::ELF::EI_NIDENT + data_offset))
    {
        const uint8_t *magic = data_sp->GetBytes() + data_offset;
        if (ELFHeader::MagicBytesMatch(magic))
        {
            // Update the data to contain the entire file if it doesn't already
            if (data_sp->GetByteSize() < length) {
                data_sp = file->MemoryMapFileContentsIfLocal(file_offset, length);
                data_offset = 0;
                magic = data_sp->GetBytes();
            }
            unsigned address_size = ELFHeader::AddressSizeInBytes(magic);
            if (address_size == 4 || address_size == 8)
            {
                std::unique_ptr<ObjectFileELF> objfile_ap(new ObjectFileELF(module_sp, data_sp, data_offset, file, file_offset, length));
                ArchSpec spec;
                if (objfile_ap->GetArchitecture(spec) &&
                    objfile_ap->SetModulesArchitecture(spec))
                    return objfile_ap.release();
            }
        }
    }
    return NULL;
}


ObjectFile*
ObjectFileELF::CreateMemoryInstance (const lldb::ModuleSP &module_sp,
                                     DataBufferSP& data_sp,
                                     const lldb::ProcessSP &process_sp,
                                     lldb::addr_t header_addr)
{
    if (data_sp && data_sp->GetByteSize() > (llvm::ELF::EI_NIDENT))
    {
        const uint8_t *magic = data_sp->GetBytes();
        if (ELFHeader::MagicBytesMatch(magic))
        {
            unsigned address_size = ELFHeader::AddressSizeInBytes(magic);
            if (address_size == 4 || address_size == 8)
            {
                std::auto_ptr<ObjectFileELF> objfile_ap(new ObjectFileELF(module_sp, data_sp, process_sp, header_addr));
                ArchSpec spec;
                if (objfile_ap->GetArchitecture(spec) &&
                    objfile_ap->SetModulesArchitecture(spec))
                    return objfile_ap.release();
            }
        }
    }
    return NULL;
}

bool
ObjectFileELF::MagicBytesMatch (DataBufferSP& data_sp,
                                  lldb::addr_t data_offset,
                                  lldb::addr_t data_length)
{
    if (data_sp && data_sp->GetByteSize() > (llvm::ELF::EI_NIDENT + data_offset))
    {
        const uint8_t *magic = data_sp->GetBytes() + data_offset;
        return ELFHeader::MagicBytesMatch(magic);
    }
    return false;
}

/*
 * crc function from http://svnweb.freebsd.org/base/head/sys/libkern/crc32.c
 *
 *   COPYRIGHT (C) 1986 Gary S. Brown. You may use this program, or
 *   code or tables extracted from it, as desired without restriction.
 */
static uint32_t
calc_crc32(uint32_t crc, const void *buf, size_t size)
{
    static const uint32_t g_crc32_tab[] =
    {
        0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f,
        0xe963a535, 0x9e6495a3, 0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
        0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91, 0x1db71064, 0x6ab020f2,
        0xf3b97148, 0x84be41de, 0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
        0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec, 0x14015c4f, 0x63066cd9,
        0xfa0f3d63, 0x8d080df5, 0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172,
        0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b, 0x35b5a8fa, 0x42b2986c,
        0xdbbbc9d6, 0xacbcf940, 0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
        0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423,
        0xcfba9599, 0xb8bda50f, 0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924,
        0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d, 0x76dc4190, 0x01db7106,
        0x98d220bc, 0xefd5102a, 0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
        0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818, 0x7f6a0dbb, 0x086d3d2d,
        0x91646c97, 0xe6635c01, 0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e,
        0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457, 0x65b0d9c6, 0x12b7e950,
        0x8bbeb8ea, 0xfcb9887c, 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
        0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2, 0x4adfa541, 0x3dd895d7,
        0xa4d1c46d, 0xd3d6f4fb, 0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0,
        0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9, 0x5005713c, 0x270241aa,
        0xbe0b1010, 0xc90c2086, 0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
        0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4, 0x59b33d17, 0x2eb40d81,
        0xb7bd5c3b, 0xc0ba6cad, 0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a,
        0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683, 0xe3630b12, 0x94643b84,
        0x0d6d6a3e, 0x7a6a5aa8, 0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
        0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe, 0xf762575d, 0x806567cb,
        0x196c3671, 0x6e6b06e7, 0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc,
        0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5, 0xd6d6a3e8, 0xa1d1937e,
        0x38d8c2c4, 0x4fdff252, 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
        0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60, 0xdf60efc3, 0xa867df55,
        0x316e8eef, 0x4669be79, 0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236,
        0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f, 0xc5ba3bbe, 0xb2bd0b28,
        0x2bb45a92, 0x5cb36a04, 0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
        0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a, 0x9c0906a9, 0xeb0e363f,
        0x72076785, 0x05005713, 0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38,
        0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21, 0x86d3d2d4, 0xf1d4e242,
        0x68ddb3f8, 0x1fda836e, 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
        0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c, 0x8f659eff, 0xf862ae69,
        0x616bffd3, 0x166ccf45, 0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2,
        0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db, 0xaed16a4a, 0xd9d65adc,
        0x40df0b66, 0x37d83bf0, 0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
        0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6, 0xbad03605, 0xcdd70693,
        0x54de5729, 0x23d967bf, 0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94,
        0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d
    };
    const uint8_t *p = (const uint8_t *)buf;

    crc = crc ^ ~0U;
    while (size--)
        crc = g_crc32_tab[(crc ^ *p++) & 0xFF] ^ (crc >> 8);
    return crc ^ ~0U;
}

static uint32_t
calc_gnu_debuglink_crc32(const void *buf, size_t size)
{
    return calc_crc32(0U, buf, size);
}

uint32_t
ObjectFileELF::CalculateELFNotesSegmentsCRC32 (const ProgramHeaderColl& program_headers,
                                               DataExtractor& object_data)
{
    typedef ProgramHeaderCollConstIter Iter;

    uint32_t core_notes_crc = 0;

    for (Iter I = program_headers.begin(); I != program_headers.end(); ++I)
    {
        if (I->p_type == llvm::ELF::PT_NOTE)
        {
            const elf_off ph_offset = I->p_offset;
            const size_t ph_size = I->p_filesz;

            DataExtractor segment_data;
            if (segment_data.SetData(object_data, ph_offset, ph_size) != ph_size)
            {
                // The ELF program header contained incorrect data,
                // probably corefile is incomplete or corrupted.
                break;
            }

            core_notes_crc = calc_crc32(core_notes_crc,
                                        segment_data.GetDataStart(),
                                        segment_data.GetByteSize());
        }
    }

    return core_notes_crc;
}

static const char*
OSABIAsCString (unsigned char osabi_byte)
{
#define _MAKE_OSABI_CASE(x) case x: return #x
    switch (osabi_byte)
    {
        _MAKE_OSABI_CASE(ELFOSABI_NONE);
        _MAKE_OSABI_CASE(ELFOSABI_HPUX);
        _MAKE_OSABI_CASE(ELFOSABI_NETBSD);
        _MAKE_OSABI_CASE(ELFOSABI_GNU);
        _MAKE_OSABI_CASE(ELFOSABI_HURD);
        _MAKE_OSABI_CASE(ELFOSABI_SOLARIS);
        _MAKE_OSABI_CASE(ELFOSABI_AIX);
        _MAKE_OSABI_CASE(ELFOSABI_IRIX);
        _MAKE_OSABI_CASE(ELFOSABI_FREEBSD);
        _MAKE_OSABI_CASE(ELFOSABI_TRU64);
        _MAKE_OSABI_CASE(ELFOSABI_MODESTO);
        _MAKE_OSABI_CASE(ELFOSABI_OPENBSD);
        _MAKE_OSABI_CASE(ELFOSABI_OPENVMS);
        _MAKE_OSABI_CASE(ELFOSABI_NSK);
        _MAKE_OSABI_CASE(ELFOSABI_AROS);
        _MAKE_OSABI_CASE(ELFOSABI_FENIXOS);
        _MAKE_OSABI_CASE(ELFOSABI_C6000_ELFABI);
        _MAKE_OSABI_CASE(ELFOSABI_C6000_LINUX);
        _MAKE_OSABI_CASE(ELFOSABI_ARM);
        _MAKE_OSABI_CASE(ELFOSABI_STANDALONE);
        default:
            return "<unknown-osabi>";
    }
#undef _MAKE_OSABI_CASE
}

//
// WARNING : This function is being deprecated
// It's functionality has moved to ArchSpec::SetArchitecture
// This function is only being kept to validate the move.
//
// TODO : Remove this function
static bool
GetOsFromOSABI (unsigned char osabi_byte, llvm::Triple::OSType &ostype)
{
    switch (osabi_byte)
    {
        case ELFOSABI_AIX:      ostype = llvm::Triple::OSType::AIX; break;
        case ELFOSABI_FREEBSD:  ostype = llvm::Triple::OSType::FreeBSD; break;
        case ELFOSABI_GNU:      ostype = llvm::Triple::OSType::Linux; break;
        case ELFOSABI_NETBSD:   ostype = llvm::Triple::OSType::NetBSD; break;
        case ELFOSABI_OPENBSD:  ostype = llvm::Triple::OSType::OpenBSD; break;
        case ELFOSABI_SOLARIS:  ostype = llvm::Triple::OSType::Solaris; break;
        default:
            ostype = llvm::Triple::OSType::UnknownOS;
    }
    return ostype != llvm::Triple::OSType::UnknownOS;
}

size_t
ObjectFileELF::GetModuleSpecifications (const lldb_private::FileSpec& file,
                                        lldb::DataBufferSP& data_sp,
                                        lldb::offset_t data_offset,
                                        lldb::offset_t file_offset,
                                        lldb::offset_t length,
                                        lldb_private::ModuleSpecList &specs)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_MODULES));

    const size_t initial_count = specs.GetSize();

    if (ObjectFileELF::MagicBytesMatch(data_sp, 0, data_sp->GetByteSize()))
    {
        DataExtractor data;
        data.SetData(data_sp);
        elf::ELFHeader header;
        if (header.Parse(data, &data_offset))
        {
            if (data_sp)
            {
                ModuleSpec spec (file);

                const uint32_t sub_type = subTypeFromElfHeader(header);
                spec.GetArchitecture().SetArchitecture(eArchTypeELF,
                                                       header.e_machine,
                                                       sub_type,
                                                       header.e_ident[EI_OSABI]);

                if (spec.GetArchitecture().IsValid())
                {
                    llvm::Triple::OSType ostype;
                    llvm::Triple::VendorType vendor;
                    llvm::Triple::OSType spec_ostype = spec.GetArchitecture ().GetTriple ().getOS ();

                    if (log)
                        log->Printf ("ObjectFileELF::%s file '%s' module OSABI: %s", __FUNCTION__, file.GetPath ().c_str (), OSABIAsCString (header.e_ident[EI_OSABI]));

                    // SetArchitecture should have set the vendor to unknown
                    vendor = spec.GetArchitecture ().GetTriple ().getVendor ();
                    assert(vendor == llvm::Triple::UnknownVendor);

                    //
                    // Validate it is ok to remove GetOsFromOSABI
                    GetOsFromOSABI (header.e_ident[EI_OSABI], ostype);
                    assert(spec_ostype == ostype);
                    if (spec_ostype != llvm::Triple::OSType::UnknownOS)
                    {
                        if (log)
                            log->Printf ("ObjectFileELF::%s file '%s' set ELF module OS type from ELF header OSABI.", __FUNCTION__, file.GetPath ().c_str ());
                    }

                    // Try to get the UUID from the section list. Usually that's at the end, so
                    // map the file in if we don't have it already.
                    size_t section_header_end = header.e_shoff + header.e_shnum * header.e_shentsize;
                    if (section_header_end > data_sp->GetByteSize())
                    {
                        data_sp = file.MemoryMapFileContentsIfLocal (file_offset, section_header_end);
                        data.SetData(data_sp);
                    }

                    uint32_t gnu_debuglink_crc = 0;
                    std::string gnu_debuglink_file;
                    SectionHeaderColl section_headers;
                    lldb_private::UUID &uuid = spec.GetUUID();

                    GetSectionHeaderInfo(section_headers, data, header, uuid, gnu_debuglink_file, gnu_debuglink_crc, spec.GetArchitecture ());

                    llvm::Triple &spec_triple = spec.GetArchitecture ().GetTriple ();

                    if (log)
                        log->Printf ("ObjectFileELF::%s file '%s' module set to triple: %s (architecture %s)", __FUNCTION__, file.GetPath ().c_str (), spec_triple.getTriple ().c_str (), spec.GetArchitecture ().GetArchitectureName ());

                    if (!uuid.IsValid())
                    {
                        uint32_t core_notes_crc = 0;

                        if (!gnu_debuglink_crc)
                        {
                            lldb_private::Timer scoped_timer (__PRETTY_FUNCTION__,
                                                              "Calculating module crc32 %s with size %" PRIu64 " KiB",
                                                              file.GetLastPathComponent().AsCString(),
                                                              (file.GetByteSize()-file_offset)/1024);

                            // For core files - which usually don't happen to have a gnu_debuglink,
                            // and are pretty bulky - calculating whole contents crc32 would be too much of luxury.
                            // Thus we will need to fallback to something simpler.
                            if (header.e_type == llvm::ELF::ET_CORE)
                            {
                                size_t program_headers_end = header.e_phoff + header.e_phnum * header.e_phentsize;
                                if (program_headers_end > data_sp->GetByteSize())
                                {
                                    data_sp = file.MemoryMapFileContentsIfLocal(file_offset, program_headers_end);
                                    data.SetData(data_sp);
                                }
                                ProgramHeaderColl program_headers;
                                GetProgramHeaderInfo(program_headers, data, header);

                                size_t segment_data_end = 0;
                                for (ProgramHeaderCollConstIter I = program_headers.begin();
                                     I != program_headers.end(); ++I)
                                {
                                     segment_data_end = std::max<unsigned long long> (I->p_offset + I->p_filesz, segment_data_end);
                                }

                                if (segment_data_end > data_sp->GetByteSize())
                                {
                                    data_sp = file.MemoryMapFileContentsIfLocal(file_offset, segment_data_end);
                                    data.SetData(data_sp);
                                }

                                core_notes_crc = CalculateELFNotesSegmentsCRC32 (program_headers, data);
                            }
                            else
                            {
                                // Need to map entire file into memory to calculate the crc.
                                data_sp = file.MemoryMapFileContentsIfLocal (file_offset, SIZE_MAX);
                                data.SetData(data_sp);
                                gnu_debuglink_crc = calc_gnu_debuglink_crc32 (data.GetDataStart(), data.GetByteSize());
                            }
                        }
                        if (gnu_debuglink_crc)
                        {
                            // Use 4 bytes of crc from the .gnu_debuglink section.
                            uint32_t uuidt[4] = { gnu_debuglink_crc, 0, 0, 0 };
                            uuid.SetBytes (uuidt, sizeof(uuidt));
                        }
                        else if (core_notes_crc)
                        {
                            // Use 8 bytes - first 4 bytes for *magic* prefix, mainly to make it look different form
                            // .gnu_debuglink crc followed by 4 bytes of note segments crc.
                            uint32_t uuidt[4] = { g_core_uuid_magic, core_notes_crc, 0, 0 };
                            uuid.SetBytes (uuidt, sizeof(uuidt));
                        }
                    }

                    specs.Append(spec);
                }
            }
        }
    }

    return specs.GetSize() - initial_count;
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
lldb_private::ConstString
ObjectFileELF::GetPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
ObjectFileELF::GetPluginVersion()
{
    return m_plugin_version;
}
//------------------------------------------------------------------
// ObjectFile protocol
//------------------------------------------------------------------

ObjectFileELF::ObjectFileELF (const lldb::ModuleSP &module_sp,
                              DataBufferSP& data_sp,
                              lldb::offset_t data_offset,
                              const FileSpec* file,
                              lldb::offset_t file_offset,
                              lldb::offset_t length) :
    ObjectFile(module_sp, file, file_offset, length, data_sp, data_offset),
    m_header(),
    m_uuid(),
    m_gnu_debuglink_file(),
    m_gnu_debuglink_crc(0),
    m_program_headers(),
    m_section_headers(),
    m_dynamic_symbols(),
    m_filespec_ap(),
    m_entry_point_address(),
    m_arch_spec()
{
    if (file)
        m_file = *file;
    ::memset(&m_header, 0, sizeof(m_header));
}

ObjectFileELF::ObjectFileELF (const lldb::ModuleSP &module_sp,
                              DataBufferSP& header_data_sp,
                              const lldb::ProcessSP &process_sp,
                              addr_t header_addr) :
    ObjectFile(module_sp, process_sp, header_addr, header_data_sp),
    m_header(),
    m_uuid(),
    m_gnu_debuglink_file(),
    m_gnu_debuglink_crc(0),
    m_program_headers(),
    m_section_headers(),
    m_dynamic_symbols(),
    m_filespec_ap(),
    m_entry_point_address(),
    m_arch_spec()
{
    ::memset(&m_header, 0, sizeof(m_header));
}

ObjectFileELF::~ObjectFileELF()
{
}

bool
ObjectFileELF::IsExecutable() const
{
    return ((m_header.e_type & ET_EXEC) != 0) || (m_header.e_entry != 0);
}

bool
ObjectFileELF::SetLoadAddress (Target &target,
                               lldb::addr_t value,
                               bool value_is_offset)
{
    ModuleSP module_sp = GetModule();
    if (module_sp)
    {
        size_t num_loaded_sections = 0;
        SectionList *section_list = GetSectionList ();
        if (section_list)
        {
            if (!value_is_offset)
            {
                bool found_offset = false;
                for (size_t i = 0, count = GetProgramHeaderCount(); i < count; ++i)
                {
                    const elf::ELFProgramHeader* header = GetProgramHeaderByIndex(i);
                    if (header == nullptr)
                        continue;

                    if (header->p_type != PT_LOAD || header->p_offset != 0)
                        continue;

                    value = value - header->p_vaddr;
                    found_offset = true;
                    break;
                }
                if (!found_offset)
                    return false;
            }

            const size_t num_sections = section_list->GetSize();
            size_t sect_idx = 0;

            for (sect_idx = 0; sect_idx < num_sections; ++sect_idx)
            {
                // Iterate through the object file sections to find all
                // of the sections that have SHF_ALLOC in their flag bits.
                SectionSP section_sp (section_list->GetSectionAtIndex (sect_idx));
                // if (section_sp && !section_sp->IsThreadSpecific())
                if (section_sp && section_sp->Test(SHF_ALLOC))
                {
                    lldb::addr_t load_addr = section_sp->GetFileAddress() + value;

                    // On 32-bit systems the load address have to fit into 4 bytes. The rest of
                    // the bytes are the overflow from the addition.
                    if (GetAddressByteSize() == 4)
                        load_addr &= 0xFFFFFFFF;

                    if (target.GetSectionLoadList().SetSectionLoadAddress (section_sp, load_addr))
                        ++num_loaded_sections;
                }
            }
            return num_loaded_sections > 0;
        }
    }
    return false;
}

ByteOrder
ObjectFileELF::GetByteOrder() const
{
    if (m_header.e_ident[EI_DATA] == ELFDATA2MSB)
        return eByteOrderBig;
    if (m_header.e_ident[EI_DATA] == ELFDATA2LSB)
        return eByteOrderLittle;
    return eByteOrderInvalid;
}

uint32_t
ObjectFileELF::GetAddressByteSize() const
{
    return m_data.GetAddressByteSize();
}

AddressClass
ObjectFileELF::GetAddressClass (addr_t file_addr)
{
    Symtab* symtab = GetSymtab();
    if (!symtab)
        return eAddressClassUnknown;

    // The address class is determined based on the symtab. Ask it from the object file what
    // contains the symtab information.
    ObjectFile* symtab_objfile = symtab->GetObjectFile();
    if (symtab_objfile != nullptr && symtab_objfile != this)
        return symtab_objfile->GetAddressClass(file_addr);

    auto res = ObjectFile::GetAddressClass (file_addr);
    if (res != eAddressClassCode)
        return res;

    auto ub = m_address_class_map.upper_bound(file_addr);
    if (ub == m_address_class_map.begin())
    {
        // No entry in the address class map before the address. Return
        // default address class for an address in a code section.
        return eAddressClassCode;
    }

    // Move iterator to the address class entry preceding address
    --ub;

    return ub->second;
}

size_t
ObjectFileELF::SectionIndex(const SectionHeaderCollIter &I)
{
    return std::distance(m_section_headers.begin(), I) + 1u;
}

size_t
ObjectFileELF::SectionIndex(const SectionHeaderCollConstIter &I) const
{
    return std::distance(m_section_headers.begin(), I) + 1u;
}

bool
ObjectFileELF::ParseHeader()
{
    lldb::offset_t offset = 0;
    if (!m_header.Parse(m_data, &offset))
        return false;

    if (!IsInMemory())
        return true;

    // For in memory object files m_data might not contain the full object file. Try to load it
    // until the end of the "Section header table" what is at the end of the ELF file.
    addr_t file_size = m_header.e_shoff + m_header.e_shnum * m_header.e_shentsize;
    if (m_data.GetByteSize() < file_size)
    {
        ProcessSP process_sp (m_process_wp.lock());
        if (!process_sp)
            return false;

        DataBufferSP data_sp = ReadMemory(process_sp, m_memory_addr, file_size);
        if (!data_sp)
            return false;
        m_data.SetData(data_sp, 0, file_size);
    }

    return true;
}

bool
ObjectFileELF::GetUUID(lldb_private::UUID* uuid)
{
    // Need to parse the section list to get the UUIDs, so make sure that's been done.
    if (!ParseSectionHeaders() && GetType() != ObjectFile::eTypeCoreFile)
        return false;

    if (m_uuid.IsValid())
    {
        // We have the full build id uuid.
        *uuid = m_uuid;
        return true;
    }
    else if (GetType() == ObjectFile::eTypeCoreFile)
    {
        uint32_t core_notes_crc = 0;

        if (!ParseProgramHeaders())
            return false;

        core_notes_crc = CalculateELFNotesSegmentsCRC32(m_program_headers, m_data);

        if (core_notes_crc)
        {
            // Use 8 bytes - first 4 bytes for *magic* prefix, mainly to make it
            // look different form .gnu_debuglink crc - followed by 4 bytes of note
            // segments crc.
            uint32_t uuidt[4] = { g_core_uuid_magic, core_notes_crc, 0, 0 };
            m_uuid.SetBytes (uuidt, sizeof(uuidt));
        }
    }
    else
    {
        if (!m_gnu_debuglink_crc)
            m_gnu_debuglink_crc = calc_gnu_debuglink_crc32 (m_data.GetDataStart(), m_data.GetByteSize());
        if (m_gnu_debuglink_crc)
        {
            // Use 4 bytes of crc from the .gnu_debuglink section.
            uint32_t uuidt[4] = { m_gnu_debuglink_crc, 0, 0, 0 };
            m_uuid.SetBytes (uuidt, sizeof(uuidt));
        }
    }

    if (m_uuid.IsValid())
    {
        *uuid = m_uuid;
        return true;
    }

    return false;
}

lldb_private::FileSpecList
ObjectFileELF::GetDebugSymbolFilePaths()
{
    FileSpecList file_spec_list;

    if (!m_gnu_debuglink_file.empty())
    {
        FileSpec file_spec (m_gnu_debuglink_file.c_str(), false);
        file_spec_list.Append (file_spec);
    }
    return file_spec_list;
}

uint32_t
ObjectFileELF::GetDependentModules(FileSpecList &files)
{
    size_t num_modules = ParseDependentModules();
    uint32_t num_specs = 0;

    for (unsigned i = 0; i < num_modules; ++i)
    {
        if (files.AppendIfUnique(m_filespec_ap->GetFileSpecAtIndex(i)))
            num_specs++;
    }

    return num_specs;
}

Address
ObjectFileELF::GetImageInfoAddress(Target *target)
{
    if (!ParseDynamicSymbols())
        return Address();

    SectionList *section_list = GetSectionList();
    if (!section_list)
        return Address();

    // Find the SHT_DYNAMIC (.dynamic) section.
    SectionSP dynsym_section_sp (section_list->FindSectionByType (eSectionTypeELFDynamicLinkInfo, true));
    if (!dynsym_section_sp)
        return Address();
    assert (dynsym_section_sp->GetObjectFile() == this);

    user_id_t dynsym_id = dynsym_section_sp->GetID();
    const ELFSectionHeaderInfo *dynsym_hdr = GetSectionHeaderByIndex(dynsym_id);
    if (!dynsym_hdr)
        return Address();

    for (size_t i = 0; i < m_dynamic_symbols.size(); ++i)
    {
        ELFDynamic &symbol = m_dynamic_symbols[i];

        if (symbol.d_tag == DT_DEBUG)
        {
            // Compute the offset as the number of previous entries plus the
            // size of d_tag.
            addr_t offset = i * dynsym_hdr->sh_entsize + GetAddressByteSize();
            return Address(dynsym_section_sp, offset);
        }
        // MIPS executables uses DT_MIPS_RLD_MAP_REL to support PIE. DT_MIPS_RLD_MAP exists in non-PIE.
        else if ((symbol.d_tag == DT_MIPS_RLD_MAP || symbol.d_tag == DT_MIPS_RLD_MAP_REL) && target)
        {
            addr_t offset = i * dynsym_hdr->sh_entsize + GetAddressByteSize();
            addr_t dyn_base = dynsym_section_sp->GetLoadBaseAddress(target);
            if (dyn_base == LLDB_INVALID_ADDRESS)
                return Address();

            Error error;
            if (symbol.d_tag == DT_MIPS_RLD_MAP)
            {
                // DT_MIPS_RLD_MAP tag stores an absolute address of the debug pointer.
                Address addr;
                if (target->ReadPointerFromMemory(dyn_base + offset, false, error, addr))
                    return addr;
            }
            if (symbol.d_tag == DT_MIPS_RLD_MAP_REL)
            {
                // DT_MIPS_RLD_MAP_REL tag stores the offset to the debug pointer, relative to the address of the tag.
                uint64_t rel_offset;
                rel_offset = target->ReadUnsignedIntegerFromMemory(dyn_base + offset, false, GetAddressByteSize(), UINT64_MAX, error);
                if (error.Success() && rel_offset != UINT64_MAX)
                {
                    Address addr;
                    addr_t debug_ptr_address = dyn_base + (offset - GetAddressByteSize()) + rel_offset;
                    addr.SetOffset (debug_ptr_address);
                    return addr;
                }
            }
        }
    }

    return Address();
}

lldb_private::Address
ObjectFileELF::GetEntryPointAddress ()
{
    if (m_entry_point_address.IsValid())
        return m_entry_point_address;

    if (!ParseHeader() || !IsExecutable())
        return m_entry_point_address;

    SectionList *section_list = GetSectionList();
    addr_t offset = m_header.e_entry;

    if (!section_list)
        m_entry_point_address.SetOffset(offset);
    else
        m_entry_point_address.ResolveAddressUsingFileSections(offset, section_list);
    return m_entry_point_address;
}

//----------------------------------------------------------------------
// ParseDependentModules
//----------------------------------------------------------------------
size_t
ObjectFileELF::ParseDependentModules()
{
    if (m_filespec_ap.get())
        return m_filespec_ap->GetSize();

    m_filespec_ap.reset(new FileSpecList());

    if (!ParseSectionHeaders())
        return 0;

    SectionList *section_list = GetSectionList();
    if (!section_list)
        return 0;

    // Find the SHT_DYNAMIC section.
    Section *dynsym = section_list->FindSectionByType (eSectionTypeELFDynamicLinkInfo, true).get();
    if (!dynsym)
        return 0;
    assert (dynsym->GetObjectFile() == this);

    const ELFSectionHeaderInfo *header = GetSectionHeaderByIndex (dynsym->GetID());
    if (!header)
        return 0;
    // sh_link: section header index of string table used by entries in the section.
    Section *dynstr = section_list->FindSectionByID (header->sh_link + 1).get();
    if (!dynstr)
        return 0;

    DataExtractor dynsym_data;
    DataExtractor dynstr_data;
    if (ReadSectionData(dynsym, dynsym_data) &&
        ReadSectionData(dynstr, dynstr_data))
    {
        ELFDynamic symbol;
        const lldb::offset_t section_size = dynsym_data.GetByteSize();
        lldb::offset_t offset = 0;

        // The only type of entries we are concerned with are tagged DT_NEEDED,
        // yielding the name of a required library.
        while (offset < section_size)
        {
            if (!symbol.Parse(dynsym_data, &offset))
                break;

            if (symbol.d_tag != DT_NEEDED)
                continue;

            uint32_t str_index = static_cast<uint32_t>(symbol.d_val);
            const char *lib_name = dynstr_data.PeekCStr(str_index);
            m_filespec_ap->Append(FileSpec(lib_name, true));
        }
    }

    return m_filespec_ap->GetSize();
}

//----------------------------------------------------------------------
// GetProgramHeaderInfo
//----------------------------------------------------------------------
size_t
ObjectFileELF::GetProgramHeaderInfo(ProgramHeaderColl &program_headers,
                                    DataExtractor &object_data,
                                    const ELFHeader &header)
{
    // We have already parsed the program headers
    if (!program_headers.empty())
        return program_headers.size();

    // If there are no program headers to read we are done.
    if (header.e_phnum == 0)
        return 0;

    program_headers.resize(header.e_phnum);
    if (program_headers.size() != header.e_phnum)
        return 0;

    const size_t ph_size = header.e_phnum * header.e_phentsize;
    const elf_off ph_offset = header.e_phoff;
    DataExtractor data;
    if (data.SetData(object_data, ph_offset, ph_size) != ph_size)
        return 0;

    uint32_t idx;
    lldb::offset_t offset;
    for (idx = 0, offset = 0; idx < header.e_phnum; ++idx)
    {
        if (program_headers[idx].Parse(data, &offset) == false)
            break;
    }

    if (idx < program_headers.size())
        program_headers.resize(idx);

    return program_headers.size();

}

//----------------------------------------------------------------------
// ParseProgramHeaders
//----------------------------------------------------------------------
size_t
ObjectFileELF::ParseProgramHeaders()
{
    return GetProgramHeaderInfo(m_program_headers, m_data, m_header);
}

lldb_private::Error
ObjectFileELF::RefineModuleDetailsFromNote (lldb_private::DataExtractor &data, lldb_private::ArchSpec &arch_spec, lldb_private::UUID &uuid)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_MODULES));
    Error error;

    lldb::offset_t offset = 0;

    while (true)
    {
        // Parse the note header.  If this fails, bail out.
        const lldb::offset_t note_offset = offset;
        ELFNote note = ELFNote();
        if (!note.Parse(data, &offset))
        {
            // We're done.
            return error;
        }

        if (log)
            log->Printf ("ObjectFileELF::%s parsing note name='%s', type=%" PRIu32, __FUNCTION__, note.n_name.c_str (), note.n_type);

        // Process FreeBSD ELF notes.
        if ((note.n_name == LLDB_NT_OWNER_FREEBSD) &&
            (note.n_type == LLDB_NT_FREEBSD_ABI_TAG) &&
            (note.n_descsz == LLDB_NT_FREEBSD_ABI_SIZE))
        {
            // Pull out the min version info.
            uint32_t version_info;
            if (data.GetU32 (&offset, &version_info, 1) == nullptr)
            {
                error.SetErrorString ("failed to read FreeBSD ABI note payload");
                return error;
            }

            // Convert the version info into a major/minor number.
            const uint32_t version_major = version_info / 100000;
            const uint32_t version_minor = (version_info / 1000) % 100;

            char os_name[32];
            snprintf (os_name, sizeof (os_name), "freebsd%" PRIu32 ".%" PRIu32, version_major, version_minor);

            // Set the elf OS version to FreeBSD.  Also clear the vendor.
            arch_spec.GetTriple ().setOSName (os_name);
            arch_spec.GetTriple ().setVendor (llvm::Triple::VendorType::UnknownVendor);

            if (log)
                log->Printf ("ObjectFileELF::%s detected FreeBSD %" PRIu32 ".%" PRIu32 ".%" PRIu32, __FUNCTION__, version_major, version_minor, static_cast<uint32_t> (version_info % 1000));
        }
        // Process GNU ELF notes.
        else if (note.n_name == LLDB_NT_OWNER_GNU)
        {
            switch (note.n_type)
            {
                case LLDB_NT_GNU_ABI_TAG:
                    if (note.n_descsz == LLDB_NT_GNU_ABI_SIZE)
                    {
                        // Pull out the min OS version supporting the ABI.
                        uint32_t version_info[4];
                        if (data.GetU32 (&offset, &version_info[0], note.n_descsz / 4) == nullptr)
                        {
                            error.SetErrorString ("failed to read GNU ABI note payload");
                            return error;
                        }

                        // Set the OS per the OS field.
                        switch (version_info[0])
                        {
                            case LLDB_NT_GNU_ABI_OS_LINUX:
                                arch_spec.GetTriple ().setOS (llvm::Triple::OSType::Linux);
                                arch_spec.GetTriple ().setVendor (llvm::Triple::VendorType::UnknownVendor);
                                if (log)
                                    log->Printf ("ObjectFileELF::%s detected Linux, min version %" PRIu32 ".%" PRIu32 ".%" PRIu32, __FUNCTION__, version_info[1], version_info[2], version_info[3]);
                                // FIXME we have the minimal version number, we could be propagating that.  version_info[1] = OS Major, version_info[2] = OS Minor, version_info[3] = Revision.
                                break;
                            case LLDB_NT_GNU_ABI_OS_HURD:
                                arch_spec.GetTriple ().setOS (llvm::Triple::OSType::UnknownOS);
                                arch_spec.GetTriple ().setVendor (llvm::Triple::VendorType::UnknownVendor);
                                if (log)
                                    log->Printf ("ObjectFileELF::%s detected Hurd (unsupported), min version %" PRIu32 ".%" PRIu32 ".%" PRIu32, __FUNCTION__, version_info[1], version_info[2], version_info[3]);
                                break;
                            case LLDB_NT_GNU_ABI_OS_SOLARIS:
                                arch_spec.GetTriple ().setOS (llvm::Triple::OSType::Solaris);
                                arch_spec.GetTriple ().setVendor (llvm::Triple::VendorType::UnknownVendor);
                                if (log)
                                    log->Printf ("ObjectFileELF::%s detected Solaris, min version %" PRIu32 ".%" PRIu32 ".%" PRIu32, __FUNCTION__, version_info[1], version_info[2], version_info[3]);
                                break;
                            default:
                                if (log)
                                    log->Printf ("ObjectFileELF::%s unrecognized OS in note, id %" PRIu32 ", min version %" PRIu32 ".%" PRIu32 ".%" PRIu32, __FUNCTION__, version_info[0], version_info[1], version_info[2], version_info[3]);
                                break;
                        }
                    }
                    break;

                case LLDB_NT_GNU_BUILD_ID_TAG:
                    // Only bother processing this if we don't already have the uuid set.
                    if (!uuid.IsValid())
                    {
                        // 16 bytes is UUID|MD5, 20 bytes is SHA1
                        if ((note.n_descsz == 16 || note.n_descsz == 20))
                        {
                            uint8_t uuidbuf[20];
                            if (data.GetU8 (&offset, &uuidbuf, note.n_descsz) == nullptr)
                            {
                                error.SetErrorString ("failed to read GNU_BUILD_ID note payload");
                                return error;
                            }

                            // Save the build id as the UUID for the module.
                            uuid.SetBytes (uuidbuf, note.n_descsz);
                        }
                    }
                    break;
            }
        }
        // Process NetBSD ELF notes.
        else if ((note.n_name == LLDB_NT_OWNER_NETBSD) &&
                 (note.n_type == LLDB_NT_NETBSD_ABI_TAG) &&
                 (note.n_descsz == LLDB_NT_NETBSD_ABI_SIZE))
        {
            // Pull out the min version info.
            uint32_t version_info;
            if (data.GetU32 (&offset, &version_info, 1) == nullptr)
            {
                error.SetErrorString ("failed to read NetBSD ABI note payload");
                return error;
            }

            // Set the elf OS version to NetBSD.  Also clear the vendor.
            arch_spec.GetTriple ().setOS (llvm::Triple::OSType::NetBSD);
            arch_spec.GetTriple ().setVendor (llvm::Triple::VendorType::UnknownVendor);

            if (log)
                log->Printf ("ObjectFileELF::%s detected NetBSD, min version constant %" PRIu32, __FUNCTION__, version_info);
        }
        // Process CSR kalimba notes
        else if ((note.n_type == LLDB_NT_GNU_ABI_TAG) &&
                (note.n_name == LLDB_NT_OWNER_CSR))
        {
            arch_spec.GetTriple().setOS(llvm::Triple::OSType::UnknownOS);
            arch_spec.GetTriple().setVendor(llvm::Triple::VendorType::CSR);

            // TODO At some point the description string could be processed.
            // It could provide a steer towards the kalimba variant which
            // this ELF targets.
            if(note.n_descsz)
            {
                const char *cstr = data.GetCStr(&offset, llvm::RoundUpToAlignment (note.n_descsz, 4));
                (void)cstr;
            }
        }
        else if (note.n_name == LLDB_NT_OWNER_ANDROID)
        {
            arch_spec.GetTriple().setOS(llvm::Triple::OSType::Linux);
            arch_spec.GetTriple().setEnvironment(llvm::Triple::EnvironmentType::Android);
        }
        else if (note.n_name == LLDB_NT_OWNER_LINUX)
        {
            // This is sometimes found in core files and usually contains extended register info
            arch_spec.GetTriple().setOS(llvm::Triple::OSType::Linux);
        }
        else if (note.n_name == LLDB_NT_OWNER_CORE)
        {
            // Parse the NT_FILE to look for stuff in paths to shared libraries
            // As the contents look like:
            // count     = 0x000000000000000a (10)
            // page_size = 0x0000000000001000 (4096)
            // Index start              end                file_ofs           path
            // ===== ------------------ ------------------ ------------------ -------------------------------------
            // [  0] 0x0000000000400000 0x0000000000401000 0x0000000000000000 /tmp/a.out
            // [  1] 0x0000000000600000 0x0000000000601000 0x0000000000000000 /tmp/a.out
            // [  2] 0x0000000000601000 0x0000000000602000 0x0000000000000001 /tmp/a.out
            // [  3] 0x00007fa79c9ed000 0x00007fa79cba8000 0x0000000000000000 /lib/x86_64-linux-gnu/libc-2.19.so
            // [  4] 0x00007fa79cba8000 0x00007fa79cda7000 0x00000000000001bb /lib/x86_64-linux-gnu/libc-2.19.so
            // [  5] 0x00007fa79cda7000 0x00007fa79cdab000 0x00000000000001ba /lib/x86_64-linux-gnu/libc-2.19.so
            // [  6] 0x00007fa79cdab000 0x00007fa79cdad000 0x00000000000001be /lib/x86_64-linux-gnu/libc-2.19.so
            // [  7] 0x00007fa79cdb2000 0x00007fa79cdd5000 0x0000000000000000 /lib/x86_64-linux-gnu/ld-2.19.so
            // [  8] 0x00007fa79cfd4000 0x00007fa79cfd5000 0x0000000000000022 /lib/x86_64-linux-gnu/ld-2.19.so
            // [  9] 0x00007fa79cfd5000 0x00007fa79cfd6000 0x0000000000000023 /lib/x86_64-linux-gnu/ld-2.19.so
            if (note.n_type == NT_FILE)
            {
                uint64_t count = data.GetU64(&offset);
                offset += 8 + 3*8*count; // Skip page size and all start/end/file_ofs
                for (size_t i=0; i<count; ++i)
                {
                    llvm::StringRef path(data.GetCStr(&offset));
                    if (path.startswith("/lib/x86_64-linux-gnu"))
                    {
                        arch_spec.GetTriple().setOS(llvm::Triple::OSType::Linux);
                        break;
                    }
                }
            }
        }

        // Calculate the offset of the next note just in case "offset" has been used
        // to poke at the contents of the note data
        offset = note_offset + note.GetByteSize();
    }

    return error;
}


//----------------------------------------------------------------------
// GetSectionHeaderInfo
//----------------------------------------------------------------------
size_t
ObjectFileELF::GetSectionHeaderInfo(SectionHeaderColl &section_headers,
                                    lldb_private::DataExtractor &object_data,
                                    const elf::ELFHeader &header,
                                    lldb_private::UUID &uuid,
                                    std::string &gnu_debuglink_file,
                                    uint32_t &gnu_debuglink_crc,
                                    ArchSpec &arch_spec)
{
    // Don't reparse the section headers if we already did that.
    if (!section_headers.empty())
        return section_headers.size();

    // Only initialize the arch_spec to okay defaults if they're not already set.
    // We'll refine this with note data as we parse the notes.
    if (arch_spec.GetTriple ().getOS () == llvm::Triple::OSType::UnknownOS)
    {
        llvm::Triple::OSType ostype;
        llvm::Triple::OSType spec_ostype;
        const uint32_t sub_type = subTypeFromElfHeader(header);
        arch_spec.SetArchitecture (eArchTypeELF, header.e_machine, sub_type, header.e_ident[EI_OSABI]);
        //
        // Validate if it is ok to remove GetOsFromOSABI
        GetOsFromOSABI (header.e_ident[EI_OSABI], ostype);
        spec_ostype = arch_spec.GetTriple ().getOS ();
        assert(spec_ostype == ostype);
    }

    if (arch_spec.GetMachine() == llvm::Triple::mips || arch_spec.GetMachine() == llvm::Triple::mipsel
        || arch_spec.GetMachine() == llvm::Triple::mips64 || arch_spec.GetMachine() == llvm::Triple::mips64el)
    {
        switch (header.e_flags & llvm::ELF::EF_MIPS_ARCH_ASE)
        {
            case llvm::ELF::EF_MIPS_MICROMIPS:
                arch_spec.SetFlags (ArchSpec::eMIPSAse_micromips);
                break;
            case llvm::ELF::EF_MIPS_ARCH_ASE_M16:
                arch_spec.SetFlags (ArchSpec::eMIPSAse_mips16);
                break;
            case llvm::ELF::EF_MIPS_ARCH_ASE_MDMX:
                arch_spec.SetFlags (ArchSpec::eMIPSAse_mdmx);
                break;
            default:
                break;
        }
    }

    // If there are no section headers we are done.
    if (header.e_shnum == 0)
        return 0;

    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_MODULES));

    section_headers.resize(header.e_shnum);
    if (section_headers.size() != header.e_shnum)
        return 0;

    const size_t sh_size = header.e_shnum * header.e_shentsize;
    const elf_off sh_offset = header.e_shoff;
    DataExtractor sh_data;
    if (sh_data.SetData (object_data, sh_offset, sh_size) != sh_size)
        return 0;

    uint32_t idx;
    lldb::offset_t offset;
    for (idx = 0, offset = 0; idx < header.e_shnum; ++idx)
    {
        if (section_headers[idx].Parse(sh_data, &offset) == false)
            break;
    }
    if (idx < section_headers.size())
        section_headers.resize(idx);

    const unsigned strtab_idx = header.e_shstrndx;
    if (strtab_idx && strtab_idx < section_headers.size())
    {
        const ELFSectionHeaderInfo &sheader = section_headers[strtab_idx];
        const size_t byte_size = sheader.sh_size;
        const Elf64_Off offset = sheader.sh_offset;
        lldb_private::DataExtractor shstr_data;

        if (shstr_data.SetData (object_data, offset, byte_size) == byte_size)
        {
            for (SectionHeaderCollIter I = section_headers.begin();
                 I != section_headers.end(); ++I)
            {
                static ConstString g_sect_name_gnu_debuglink (".gnu_debuglink");
                const ELFSectionHeaderInfo &sheader = *I;
                const uint64_t section_size = sheader.sh_type == SHT_NOBITS ? 0 : sheader.sh_size;
                ConstString name(shstr_data.PeekCStr(I->sh_name));

                I->section_name = name;

                if (arch_spec.GetMachine() == llvm::Triple::mips || arch_spec.GetMachine() == llvm::Triple::mipsel
                    || arch_spec.GetMachine() == llvm::Triple::mips64 || arch_spec.GetMachine() == llvm::Triple::mips64el)
                {
                    uint32_t arch_flags = arch_spec.GetFlags ();
                    DataExtractor data;
                    if (sheader.sh_type == SHT_MIPS_ABIFLAGS)
                    {

                        if (section_size && (data.SetData (object_data, sheader.sh_offset, section_size) == section_size))
                        {
                            lldb::offset_t ase_offset = 12; // MIPS ABI Flags Version: 0
                            arch_flags |= data.GetU32 (&ase_offset);
                        }
                    }
                    // Settings appropriate ArchSpec ABI Flags
                    if (header.e_flags & llvm::ELF::EF_MIPS_ABI2)
                    {
                        arch_flags |= lldb_private::ArchSpec::eMIPSABI_N32;
                    }
                    else if (header.e_flags & llvm::ELF::EF_MIPS_ABI_O32)
                    {
                         arch_flags |= lldb_private::ArchSpec::eMIPSABI_O32;
                    }
                    arch_spec.SetFlags (arch_flags);
                }

                if (name == g_sect_name_gnu_debuglink)
                {
                    DataExtractor data;
                    if (section_size && (data.SetData (object_data, sheader.sh_offset, section_size) == section_size))
                    {
                        lldb::offset_t gnu_debuglink_offset = 0;
                        gnu_debuglink_file = data.GetCStr (&gnu_debuglink_offset);
                        gnu_debuglink_offset = llvm::RoundUpToAlignment (gnu_debuglink_offset, 4);
                        data.GetU32 (&gnu_debuglink_offset, &gnu_debuglink_crc, 1);
                    }
                }

                // Process ELF note section entries.
                bool is_note_header = (sheader.sh_type == SHT_NOTE);

                // The section header ".note.android.ident" is stored as a
                // PROGBITS type header but it is actually a note header.
                static ConstString g_sect_name_android_ident (".note.android.ident");
                if (!is_note_header && name == g_sect_name_android_ident)
                    is_note_header = true;

                if (is_note_header)
                {
                    // Allow notes to refine module info.
                    DataExtractor data;
                    if (section_size && (data.SetData (object_data, sheader.sh_offset, section_size) == section_size))
                    {
                        Error error = RefineModuleDetailsFromNote (data, arch_spec, uuid);
                        if (error.Fail ())
                        {
                            if (log)
                                log->Printf ("ObjectFileELF::%s ELF note processing failed: %s", __FUNCTION__, error.AsCString ());
                        }
                    }
                }
            }

            // Make any unknown triple components to be unspecified unknowns.
            if (arch_spec.GetTriple().getVendor() == llvm::Triple::UnknownVendor)
                arch_spec.GetTriple().setVendorName (llvm::StringRef());
            if (arch_spec.GetTriple().getOS() == llvm::Triple::UnknownOS)
                arch_spec.GetTriple().setOSName (llvm::StringRef());

            return section_headers.size();
        }
    }

    section_headers.clear();
    return 0;
}

size_t
ObjectFileELF::GetProgramHeaderCount()
{
    return ParseProgramHeaders();
}

const elf::ELFProgramHeader *
ObjectFileELF::GetProgramHeaderByIndex(lldb::user_id_t id)
{
    if (!id || !ParseProgramHeaders())
        return NULL;

    if (--id < m_program_headers.size())
        return &m_program_headers[id];

    return NULL;
}

DataExtractor
ObjectFileELF::GetSegmentDataByIndex(lldb::user_id_t id)
{
    const elf::ELFProgramHeader *segment_header = GetProgramHeaderByIndex(id);
    if (segment_header == NULL)
        return DataExtractor();
    return DataExtractor(m_data, segment_header->p_offset, segment_header->p_filesz);
}

std::string
ObjectFileELF::StripLinkerSymbolAnnotations(llvm::StringRef symbol_name) const
{
    size_t pos = symbol_name.find('@');
    return symbol_name.substr(0, pos).str();
}

//----------------------------------------------------------------------
// ParseSectionHeaders
//----------------------------------------------------------------------
size_t
ObjectFileELF::ParseSectionHeaders()
{
    return GetSectionHeaderInfo(m_section_headers, m_data, m_header, m_uuid, m_gnu_debuglink_file, m_gnu_debuglink_crc, m_arch_spec);
}

const ObjectFileELF::ELFSectionHeaderInfo *
ObjectFileELF::GetSectionHeaderByIndex(lldb::user_id_t id)
{
    if (!id || !ParseSectionHeaders())
        return NULL;

    if (--id < m_section_headers.size())
        return &m_section_headers[id];

    return NULL;
}

lldb::user_id_t
ObjectFileELF::GetSectionIndexByName(const char* name)
{
    if (!name || !name[0] || !ParseSectionHeaders())
        return 0;
    for (size_t i = 1; i < m_section_headers.size(); ++i)
        if (m_section_headers[i].section_name == ConstString(name))
            return i;
    return 0;
}

void
ObjectFileELF::CreateSections(SectionList &unified_section_list)
{
    if (!m_sections_ap.get() && ParseSectionHeaders())
    {
        m_sections_ap.reset(new SectionList());

        for (SectionHeaderCollIter I = m_section_headers.begin();
             I != m_section_headers.end(); ++I)
        {
            const ELFSectionHeaderInfo &header = *I;

            ConstString& name = I->section_name;
            const uint64_t file_size = header.sh_type == SHT_NOBITS ? 0 : header.sh_size;
            const uint64_t vm_size = header.sh_flags & SHF_ALLOC ? header.sh_size : 0;

            static ConstString g_sect_name_text (".text");
            static ConstString g_sect_name_data (".data");
            static ConstString g_sect_name_bss (".bss");
            static ConstString g_sect_name_tdata (".tdata");
            static ConstString g_sect_name_tbss (".tbss");
            static ConstString g_sect_name_dwarf_debug_abbrev (".debug_abbrev");
            static ConstString g_sect_name_dwarf_debug_addr (".debug_addr");
            static ConstString g_sect_name_dwarf_debug_aranges (".debug_aranges");
            static ConstString g_sect_name_dwarf_debug_frame (".debug_frame");
            static ConstString g_sect_name_dwarf_debug_info (".debug_info");
            static ConstString g_sect_name_dwarf_debug_line (".debug_line");
            static ConstString g_sect_name_dwarf_debug_loc (".debug_loc");
            static ConstString g_sect_name_dwarf_debug_macinfo (".debug_macinfo");
            static ConstString g_sect_name_dwarf_debug_macro (".debug_macro");
            static ConstString g_sect_name_dwarf_debug_pubnames (".debug_pubnames");
            static ConstString g_sect_name_dwarf_debug_pubtypes (".debug_pubtypes");
            static ConstString g_sect_name_dwarf_debug_ranges (".debug_ranges");
            static ConstString g_sect_name_dwarf_debug_str (".debug_str");
            static ConstString g_sect_name_dwarf_debug_str_offsets (".debug_str_offsets");
            static ConstString g_sect_name_dwarf_debug_abbrev_dwo (".debug_abbrev.dwo");
            static ConstString g_sect_name_dwarf_debug_info_dwo (".debug_info.dwo");
            static ConstString g_sect_name_dwarf_debug_line_dwo (".debug_line.dwo");
            static ConstString g_sect_name_dwarf_debug_macro_dwo (".debug_macro.dwo");
            static ConstString g_sect_name_dwarf_debug_loc_dwo (".debug_loc.dwo");
            static ConstString g_sect_name_dwarf_debug_str_dwo (".debug_str.dwo");
            static ConstString g_sect_name_dwarf_debug_str_offsets_dwo (".debug_str_offsets.dwo");
            static ConstString g_sect_name_eh_frame (".eh_frame");
            static ConstString g_sect_name_arm_exidx (".ARM.exidx");
            static ConstString g_sect_name_arm_extab (".ARM.extab");
            static ConstString g_sect_name_go_symtab (".gosymtab");

            SectionType sect_type = eSectionTypeOther;

            bool is_thread_specific = false;

            if      (name == g_sect_name_text)                  sect_type = eSectionTypeCode;
            else if (name == g_sect_name_data)                  sect_type = eSectionTypeData;
            else if (name == g_sect_name_bss)                   sect_type = eSectionTypeZeroFill;
            else if (name == g_sect_name_tdata)
            {
                sect_type = eSectionTypeData;
                is_thread_specific = true;
            }
            else if (name == g_sect_name_tbss)
            {
                sect_type = eSectionTypeZeroFill;
                is_thread_specific = true;
            }
            // .debug_abbrev – Abbreviations used in the .debug_info section
            // .debug_aranges – Lookup table for mapping addresses to compilation units
            // .debug_frame – Call frame information
            // .debug_info – The core DWARF information section
            // .debug_line – Line number information
            // .debug_loc – Location lists used in DW_AT_location attributes
            // .debug_macinfo – Macro information
            // .debug_pubnames – Lookup table for mapping object and function names to compilation units
            // .debug_pubtypes – Lookup table for mapping type names to compilation units
            // .debug_ranges – Address ranges used in DW_AT_ranges attributes
            // .debug_str – String table used in .debug_info
            // MISSING? .gnu_debugdata - "mini debuginfo / MiniDebugInfo" section, http://sourceware.org/gdb/onlinedocs/gdb/MiniDebugInfo.html
            // MISSING? .debug-index - http://src.chromium.org/viewvc/chrome/trunk/src/build/gdb-add-index?pathrev=144644
            // MISSING? .debug_types - Type descriptions from DWARF 4? See http://gcc.gnu.org/wiki/DwarfSeparateTypeInfo
            else if (name == g_sect_name_dwarf_debug_abbrev)          sect_type = eSectionTypeDWARFDebugAbbrev;
            else if (name == g_sect_name_dwarf_debug_addr)            sect_type = eSectionTypeDWARFDebugAddr;
            else if (name == g_sect_name_dwarf_debug_aranges)         sect_type = eSectionTypeDWARFDebugAranges;
            else if (name == g_sect_name_dwarf_debug_frame)           sect_type = eSectionTypeDWARFDebugFrame;
            else if (name == g_sect_name_dwarf_debug_info)            sect_type = eSectionTypeDWARFDebugInfo;
            else if (name == g_sect_name_dwarf_debug_line)            sect_type = eSectionTypeDWARFDebugLine;
            else if (name == g_sect_name_dwarf_debug_loc)             sect_type = eSectionTypeDWARFDebugLoc;
            else if (name == g_sect_name_dwarf_debug_macinfo)         sect_type = eSectionTypeDWARFDebugMacInfo;
            else if (name == g_sect_name_dwarf_debug_macro)           sect_type = eSectionTypeDWARFDebugMacro;
            else if (name == g_sect_name_dwarf_debug_pubnames)        sect_type = eSectionTypeDWARFDebugPubNames;
            else if (name == g_sect_name_dwarf_debug_pubtypes)        sect_type = eSectionTypeDWARFDebugPubTypes;
            else if (name == g_sect_name_dwarf_debug_ranges)          sect_type = eSectionTypeDWARFDebugRanges;
            else if (name == g_sect_name_dwarf_debug_str)             sect_type = eSectionTypeDWARFDebugStr;
            else if (name == g_sect_name_dwarf_debug_str_offsets)     sect_type = eSectionTypeDWARFDebugStrOffsets;
            else if (name == g_sect_name_dwarf_debug_abbrev_dwo)      sect_type = eSectionTypeDWARFDebugAbbrev;
            else if (name == g_sect_name_dwarf_debug_info_dwo)        sect_type = eSectionTypeDWARFDebugInfo;
            else if (name == g_sect_name_dwarf_debug_line_dwo)        sect_type = eSectionTypeDWARFDebugLine;
            else if (name == g_sect_name_dwarf_debug_macro_dwo)       sect_type = eSectionTypeDWARFDebugMacro;
            else if (name == g_sect_name_dwarf_debug_loc_dwo)         sect_type = eSectionTypeDWARFDebugLoc;
            else if (name == g_sect_name_dwarf_debug_str_dwo)         sect_type = eSectionTypeDWARFDebugStr;
            else if (name == g_sect_name_dwarf_debug_str_offsets_dwo) sect_type = eSectionTypeDWARFDebugStrOffsets;
            else if (name == g_sect_name_eh_frame)                    sect_type = eSectionTypeEHFrame;
            else if (name == g_sect_name_arm_exidx)                   sect_type = eSectionTypeARMexidx;
            else if (name == g_sect_name_arm_extab)                   sect_type = eSectionTypeARMextab;
            else if (name == g_sect_name_go_symtab)                   sect_type = eSectionTypeGoSymtab;

            switch (header.sh_type)
            {
                case SHT_SYMTAB:
                    assert (sect_type == eSectionTypeOther);
                    sect_type = eSectionTypeELFSymbolTable;
                    break;
                case SHT_DYNSYM:
                    assert (sect_type == eSectionTypeOther);
                    sect_type = eSectionTypeELFDynamicSymbols;
                    break;
                case SHT_RELA:
                case SHT_REL:
                    assert (sect_type == eSectionTypeOther);
                    sect_type = eSectionTypeELFRelocationEntries;
                    break;
                case SHT_DYNAMIC:
                    assert (sect_type == eSectionTypeOther);
                    sect_type = eSectionTypeELFDynamicLinkInfo;
                    break;
            }

            if (eSectionTypeOther == sect_type)
            {
                // the kalimba toolchain assumes that ELF section names are free-form. It does
                // support linkscripts which (can) give rise to various arbitrarily named
                // sections being "Code" or "Data".
                sect_type = kalimbaSectionType(m_header, header);
            }

            const uint32_t target_bytes_size =
                (eSectionTypeData == sect_type || eSectionTypeZeroFill == sect_type) ?
                m_arch_spec.GetDataByteSize() :
                    eSectionTypeCode == sect_type ?
                    m_arch_spec.GetCodeByteSize() : 1;

            elf::elf_xword log2align = (header.sh_addralign==0)
                                        ? 0
                                        : llvm::Log2_64(header.sh_addralign);
            SectionSP section_sp (new Section(GetModule(),        // Module to which this section belongs.
                                              this,               // ObjectFile to which this section belongs and should read section data from.
                                              SectionIndex(I),    // Section ID.
                                              name,               // Section name.
                                              sect_type,          // Section type.
                                              header.sh_addr,     // VM address.
                                              vm_size,            // VM size in bytes of this section.
                                              header.sh_offset,   // Offset of this section in the file.
                                              file_size,          // Size of the section as found in the file.
                                              log2align,          // Alignment of the section
                                              header.sh_flags,    // Flags for this section.
                                              target_bytes_size));// Number of host bytes per target byte

            if (is_thread_specific)
                section_sp->SetIsThreadSpecific (is_thread_specific);
            m_sections_ap->AddSection(section_sp);
        }
    }

    if (m_sections_ap.get())
    {
        if (GetType() == eTypeDebugInfo)
        {
            static const SectionType g_sections[] =
            {
                eSectionTypeDWARFDebugAbbrev,
                eSectionTypeDWARFDebugAddr,
                eSectionTypeDWARFDebugAranges,
                eSectionTypeDWARFDebugFrame,
                eSectionTypeDWARFDebugInfo,
                eSectionTypeDWARFDebugLine,
                eSectionTypeDWARFDebugLoc,
                eSectionTypeDWARFDebugMacInfo,
                eSectionTypeDWARFDebugPubNames,
                eSectionTypeDWARFDebugPubTypes,
                eSectionTypeDWARFDebugRanges,
                eSectionTypeDWARFDebugStr,
                eSectionTypeDWARFDebugStrOffsets,
                eSectionTypeELFSymbolTable,
            };
            SectionList *elf_section_list = m_sections_ap.get();
            for (size_t idx = 0; idx < sizeof(g_sections) / sizeof(g_sections[0]); ++idx)
            {
                SectionType section_type = g_sections[idx];
                SectionSP section_sp (elf_section_list->FindSectionByType (section_type, true));
                if (section_sp)
                {
                    SectionSP module_section_sp (unified_section_list.FindSectionByType (section_type, true));
                    if (module_section_sp)
                        unified_section_list.ReplaceSection (module_section_sp->GetID(), section_sp);
                    else
                        unified_section_list.AddSection (section_sp);
                }
            }
        }
        else
        {
            unified_section_list = *m_sections_ap;
        }
    }
}

// Find the arm/aarch64 mapping symbol character in the given symbol name. Mapping symbols have the
// form of "$<char>[.<any>]*". Additionally we recognize cases when the mapping symbol prefixed by
// an arbitrary string because if a symbol prefix added to each symbol in the object file with
// objcopy then the mapping symbols are also prefixed.
static char
FindArmAarch64MappingSymbol(const char* symbol_name)
{
    if (!symbol_name)
        return '\0';

    const char* dollar_pos = ::strchr(symbol_name, '$');
    if (!dollar_pos || dollar_pos[1] == '\0')
        return '\0';

    if (dollar_pos[2] == '\0' || dollar_pos[2] == '.')
        return dollar_pos[1];
    return '\0';
}

#define STO_MIPS_ISA            (3 << 6)
#define STO_MICROMIPS           (2 << 6)
#define IS_MICROMIPS(ST_OTHER)  (((ST_OTHER) & STO_MIPS_ISA) == STO_MICROMIPS)

// private
unsigned
ObjectFileELF::ParseSymbols (Symtab *symtab,
                             user_id_t start_id,
                             SectionList *section_list,
                             const size_t num_symbols,
                             const DataExtractor &symtab_data,
                             const DataExtractor &strtab_data)
{
    ELFSymbol symbol;
    lldb::offset_t offset = 0;

    static ConstString text_section_name(".text");
    static ConstString init_section_name(".init");
    static ConstString fini_section_name(".fini");
    static ConstString ctors_section_name(".ctors");
    static ConstString dtors_section_name(".dtors");

    static ConstString data_section_name(".data");
    static ConstString rodata_section_name(".rodata");
    static ConstString rodata1_section_name(".rodata1");
    static ConstString data2_section_name(".data1");
    static ConstString bss_section_name(".bss");
    static ConstString opd_section_name(".opd");    // For ppc64

    // On Android the oatdata and the oatexec symbols in system@framework@boot.oat covers the full
    // .text section what causes issues with displaying unusable symbol name to the user and very
    // slow unwinding speed because the instruction emulation based unwind plans try to emulate all
    // instructions in these symbols. Don't add these symbols to the symbol list as they have no
    // use for the debugger and they are causing a lot of trouble.
    // Filtering can't be restricted to Android because this special object file don't contain the
    // note section specifying the environment to Android but the custom extension and file name
    // makes it highly unlikely that this will collide with anything else.
    bool skip_oatdata_oatexec = m_file.GetFilename() == ConstString("system@framework@boot.oat");

    ArchSpec arch;
    GetArchitecture(arch);

    // Local cache to avoid doing a FindSectionByName for each symbol. The "const char*" key must
    // came from a ConstString object so they can be compared by pointer
    std::unordered_map<const char*, lldb::SectionSP> section_name_to_section;

    unsigned i;
    for (i = 0; i < num_symbols; ++i)
    {
        if (symbol.Parse(symtab_data, &offset) == false)
            break;

        const char *symbol_name = strtab_data.PeekCStr(symbol.st_name);

        // No need to add non-section symbols that have no names
        if (symbol.getType() != STT_SECTION &&
            (symbol_name == NULL || symbol_name[0] == '\0'))
            continue;

        // Skipping oatdata and oatexec sections if it is requested. See details above the
        // definition of skip_oatdata_oatexec for the reasons.
        if (skip_oatdata_oatexec && (::strcmp(symbol_name, "oatdata") == 0 || ::strcmp(symbol_name, "oatexec") == 0))
            continue;

        SectionSP symbol_section_sp;
        SymbolType symbol_type = eSymbolTypeInvalid;
        Elf64_Half symbol_idx = symbol.st_shndx;

        switch (symbol_idx)
        {
        case SHN_ABS:
            symbol_type = eSymbolTypeAbsolute;
            break;
        case SHN_UNDEF:
            symbol_type = eSymbolTypeUndefined;
            break;
        default:
            symbol_section_sp = section_list->GetSectionAtIndex(symbol_idx);
            break;
        }

        // If a symbol is undefined do not process it further even if it has a STT type
        if (symbol_type != eSymbolTypeUndefined)
        {
            switch (symbol.getType())
            {
            default:
            case STT_NOTYPE:
                // The symbol's type is not specified.
                break;

            case STT_OBJECT:
                // The symbol is associated with a data object, such as a variable,
                // an array, etc.
                symbol_type = eSymbolTypeData;
                break;

            case STT_FUNC:
                // The symbol is associated with a function or other executable code.
                symbol_type = eSymbolTypeCode;
                break;

            case STT_SECTION:
                // The symbol is associated with a section. Symbol table entries of
                // this type exist primarily for relocation and normally have
                // STB_LOCAL binding.
                break;

            case STT_FILE:
                // Conventionally, the symbol's name gives the name of the source
                // file associated with the object file. A file symbol has STB_LOCAL
                // binding, its section index is SHN_ABS, and it precedes the other
                // STB_LOCAL symbols for the file, if it is present.
                symbol_type = eSymbolTypeSourceFile;
                break;

            case STT_GNU_IFUNC:
                // The symbol is associated with an indirect function. The actual
                // function will be resolved if it is referenced.
                symbol_type = eSymbolTypeResolver;
                break;
            }
        }

        if (symbol_type == eSymbolTypeInvalid)
        {
            if (symbol_section_sp)
            {
                const ConstString &sect_name = symbol_section_sp->GetName();
                if (sect_name == text_section_name ||
                    sect_name == init_section_name ||
                    sect_name == fini_section_name ||
                    sect_name == ctors_section_name ||
                    sect_name == dtors_section_name)
                {
                    symbol_type = eSymbolTypeCode;
                }
                else if (sect_name == data_section_name ||
                         sect_name == data2_section_name ||
                         sect_name == rodata_section_name ||
                         sect_name == rodata1_section_name ||
                         sect_name == bss_section_name)
                {
                    symbol_type = eSymbolTypeData;
                }
            }
        }

        int64_t symbol_value_offset = 0;
        uint32_t additional_flags = 0;

        if (arch.IsValid())
        {
            if (arch.GetMachine() == llvm::Triple::arm)
            {
                if (symbol.getBinding() == STB_LOCAL)
                {
                    char mapping_symbol = FindArmAarch64MappingSymbol(symbol_name);
                    if (symbol_type == eSymbolTypeCode)
                    {
                        switch (mapping_symbol)
                        {
                            case 'a':
                                // $a[.<any>]* - marks an ARM instruction sequence
                                m_address_class_map[symbol.st_value] = eAddressClassCode;
                                break;
                            case 'b':
                            case 't':
                                // $b[.<any>]* - marks a THUMB BL instruction sequence
                                // $t[.<any>]* - marks a THUMB instruction sequence
                                m_address_class_map[symbol.st_value] = eAddressClassCodeAlternateISA;
                                break;
                            case 'd':
                                // $d[.<any>]* - marks a data item sequence (e.g. lit pool)
                                m_address_class_map[symbol.st_value] = eAddressClassData;
                                break;
                        }
                    }
                    if (mapping_symbol)
                        continue;
                }
            }
            else if (arch.GetMachine() == llvm::Triple::aarch64)
            {
                if (symbol.getBinding() == STB_LOCAL)
                {
                    char mapping_symbol = FindArmAarch64MappingSymbol(symbol_name);
                    if (symbol_type == eSymbolTypeCode)
                    {
                        switch (mapping_symbol)
                        {
                            case 'x':
                                // $x[.<any>]* - marks an A64 instruction sequence
                                m_address_class_map[symbol.st_value] = eAddressClassCode;
                                break;
                            case 'd':
                                // $d[.<any>]* - marks a data item sequence (e.g. lit pool)
                                m_address_class_map[symbol.st_value] = eAddressClassData;
                                break;
                        }
                    }
                    if (mapping_symbol)
                        continue;
                }
            }

            if (arch.GetMachine() == llvm::Triple::arm)
            {
                if (symbol_type == eSymbolTypeCode)
                {
                    if (symbol.st_value & 1)
                    {
                        // Subtracting 1 from the address effectively unsets
                        // the low order bit, which results in the address
                        // actually pointing to the beginning of the symbol.
                        // This delta will be used below in conjunction with
                        // symbol.st_value to produce the final symbol_value
                        // that we store in the symtab.
                        symbol_value_offset = -1;
                        m_address_class_map[symbol.st_value^1] = eAddressClassCodeAlternateISA;
                    }
                    else
                    {
                        // This address is ARM
                        m_address_class_map[symbol.st_value] = eAddressClassCode;
                    }
                }
            }

            /*
             * MIPS:
             * The bit #0 of an address is used for ISA mode (1 for microMIPS, 0 for MIPS).
             * This allows processer to switch between microMIPS and MIPS without any need
             * for special mode-control register. However, apart from .debug_line, none of
             * the ELF/DWARF sections set the ISA bit (for symbol or section). Use st_other
             * flag to check whether the symbol is microMIPS and then set the address class
             * accordingly.
            */
            const llvm::Triple::ArchType llvm_arch = arch.GetMachine();
            if (llvm_arch == llvm::Triple::mips || llvm_arch == llvm::Triple::mipsel
                || llvm_arch == llvm::Triple::mips64 || llvm_arch == llvm::Triple::mips64el)
            {
                if (IS_MICROMIPS(symbol.st_other))
                    m_address_class_map[symbol.st_value] = eAddressClassCodeAlternateISA;
                else if ((symbol.st_value & 1) && (symbol_type == eSymbolTypeCode))
                {
                    symbol.st_value = symbol.st_value & (~1ull);
                    m_address_class_map[symbol.st_value] = eAddressClassCodeAlternateISA;
                }
                else
                {
                    if (symbol_type == eSymbolTypeCode)
                        m_address_class_map[symbol.st_value] = eAddressClassCode;
                    else if (symbol_type == eSymbolTypeData)
                        m_address_class_map[symbol.st_value] = eAddressClassData;
                    else
                        m_address_class_map[symbol.st_value] = eAddressClassUnknown;
                }
            }
        }

        // symbol_value_offset may contain 0 for ARM symbols or -1 for
        // THUMB symbols. See above for more details.
        uint64_t symbol_value = symbol.st_value + symbol_value_offset;
        if (symbol_section_sp && CalculateType() != ObjectFile::Type::eTypeObjectFile)
            symbol_value -= symbol_section_sp->GetFileAddress();

        if (symbol_section_sp)
        {
            ModuleSP module_sp(GetModule());
            if (module_sp)
            {
                SectionList *module_section_list = module_sp->GetSectionList();
                if (module_section_list && module_section_list != section_list)
                {
                    const ConstString &sect_name = symbol_section_sp->GetName();
                    auto section_it = section_name_to_section.find(sect_name.GetCString());
                    if (section_it == section_name_to_section.end())
                        section_it = section_name_to_section.emplace(
                            sect_name.GetCString(),
                            module_section_list->FindSectionByName (sect_name)).first;
                    if (section_it->second && section_it->second->GetFileSize())
                        symbol_section_sp = section_it->second;
                }
            }
        }

        bool is_global = symbol.getBinding() == STB_GLOBAL;
        uint32_t flags = symbol.st_other << 8 | symbol.st_info | additional_flags;
        bool is_mangled = symbol_name ? (symbol_name[0] == '_' && symbol_name[1] == 'Z') : false;

        llvm::StringRef symbol_ref(symbol_name);

        // Symbol names may contain @VERSION suffixes. Find those and strip them temporarily.
        size_t version_pos = symbol_ref.find('@');
        bool has_suffix = version_pos != llvm::StringRef::npos;
        llvm::StringRef symbol_bare = symbol_ref.substr(0, version_pos);
        Mangled mangled(ConstString(symbol_bare), is_mangled);

        // Now append the suffix back to mangled and unmangled names. Only do it if the
        // demangling was successful (string is not empty).
        if (has_suffix)
        {
            llvm::StringRef suffix = symbol_ref.substr(version_pos);

            llvm::StringRef mangled_name = mangled.GetMangledName().GetStringRef();
            if (! mangled_name.empty())
                mangled.SetMangledName( ConstString((mangled_name + suffix).str()) );

            ConstString demangled = mangled.GetDemangledName(lldb::eLanguageTypeUnknown);
            llvm::StringRef demangled_name = demangled.GetStringRef();
            if (!demangled_name.empty())
                mangled.SetDemangledName( ConstString((demangled_name + suffix).str()) );
        }

        Symbol dc_symbol(
            i + start_id,       // ID is the original symbol table index.
            mangled,
            symbol_type,        // Type of this symbol
            is_global,          // Is this globally visible?
            false,              // Is this symbol debug info?
            false,              // Is this symbol a trampoline?
            false,              // Is this symbol artificial?
            AddressRange(
                symbol_section_sp,  // Section in which this symbol is defined or null.
                symbol_value,       // Offset in section or symbol value.
                symbol.st_size),    // Size in bytes of this symbol.
            true,                   // Symbol size is valid
            has_suffix,             // Contains linker annotations?
            flags);                 // Symbol flags.
        symtab->AddSymbol(dc_symbol);
    }
    return i;
}

unsigned
ObjectFileELF::ParseSymbolTable(Symtab *symbol_table,
                                user_id_t start_id,
                                lldb_private::Section *symtab)
{
    if (symtab->GetObjectFile() != this)
    {
        // If the symbol table section is owned by a different object file, have it do the
        // parsing.
        ObjectFileELF *obj_file_elf = static_cast<ObjectFileELF *>(symtab->GetObjectFile());
        return obj_file_elf->ParseSymbolTable (symbol_table, start_id, symtab);
    }

    // Get section list for this object file.
    SectionList *section_list = m_sections_ap.get();
    if (!section_list)
        return 0;

    user_id_t symtab_id = symtab->GetID();
    const ELFSectionHeaderInfo *symtab_hdr = GetSectionHeaderByIndex(symtab_id);
    assert(symtab_hdr->sh_type == SHT_SYMTAB ||
           symtab_hdr->sh_type == SHT_DYNSYM);

    // sh_link: section header index of associated string table.
    // Section ID's are ones based.
    user_id_t strtab_id = symtab_hdr->sh_link + 1;
    Section *strtab = section_list->FindSectionByID(strtab_id).get();

    if (symtab && strtab)
    {
        assert (symtab->GetObjectFile() == this);
        assert (strtab->GetObjectFile() == this);

        DataExtractor symtab_data;
        DataExtractor strtab_data;
        if (ReadSectionData(symtab, symtab_data) &&
            ReadSectionData(strtab, strtab_data))
        {
            size_t num_symbols = symtab_data.GetByteSize() / symtab_hdr->sh_entsize;

            return ParseSymbols(symbol_table, start_id, section_list,
                                num_symbols, symtab_data, strtab_data);
        }
    }

    return 0;
}

size_t
ObjectFileELF::ParseDynamicSymbols()
{
    if (m_dynamic_symbols.size())
        return m_dynamic_symbols.size();

    SectionList *section_list = GetSectionList();
    if (!section_list)
        return 0;

    // Find the SHT_DYNAMIC section.
    Section *dynsym = section_list->FindSectionByType (eSectionTypeELFDynamicLinkInfo, true).get();
    if (!dynsym)
        return 0;
    assert (dynsym->GetObjectFile() == this);

    ELFDynamic symbol;
    DataExtractor dynsym_data;
    if (ReadSectionData(dynsym, dynsym_data))
    {
        const lldb::offset_t section_size = dynsym_data.GetByteSize();
        lldb::offset_t cursor = 0;

        while (cursor < section_size)
        {
            if (!symbol.Parse(dynsym_data, &cursor))
                break;

            m_dynamic_symbols.push_back(symbol);
        }
    }

    return m_dynamic_symbols.size();
}

const ELFDynamic *
ObjectFileELF::FindDynamicSymbol(unsigned tag)
{
    if (!ParseDynamicSymbols())
        return NULL;

    DynamicSymbolCollIter I = m_dynamic_symbols.begin();
    DynamicSymbolCollIter E = m_dynamic_symbols.end();
    for ( ; I != E; ++I)
    {
        ELFDynamic *symbol = &*I;

        if (symbol->d_tag == tag)
            return symbol;
    }

    return NULL;
}

unsigned
ObjectFileELF::PLTRelocationType()
{
    // DT_PLTREL
    //  This member specifies the type of relocation entry to which the
    //  procedure linkage table refers. The d_val member holds DT_REL or
    //  DT_RELA, as appropriate. All relocations in a procedure linkage table
    //  must use the same relocation.
    const ELFDynamic *symbol = FindDynamicSymbol(DT_PLTREL);

    if (symbol)
        return symbol->d_val;

    return 0;
}

// Returns the size of the normal plt entries and the offset of the first normal plt entry. The
// 0th entry in the plt table is usually a resolution entry which have different size in some
// architectures then the rest of the plt entries.
static std::pair<uint64_t, uint64_t>
GetPltEntrySizeAndOffset(const ELFSectionHeader* rel_hdr, const ELFSectionHeader* plt_hdr)
{
    const elf_xword num_relocations = rel_hdr->sh_size / rel_hdr->sh_entsize;

    // Clang 3.3 sets entsize to 4 for 32-bit binaries, but the plt entries are 16 bytes.
    // So round the entsize up by the alignment if addralign is set.
    elf_xword plt_entsize = plt_hdr->sh_addralign ?
        llvm::RoundUpToAlignment (plt_hdr->sh_entsize, plt_hdr->sh_addralign) : plt_hdr->sh_entsize;

    if (plt_entsize == 0)
    {
        // The linker haven't set the plt_hdr->sh_entsize field. Try to guess the size of the plt
        // entries based on the number of entries and the size of the plt section with the
        // assumption that the size of the 0th entry is at least as big as the size of the normal
        // entries and it isn't much bigger then that.
        if (plt_hdr->sh_addralign)
            plt_entsize = plt_hdr->sh_size / plt_hdr->sh_addralign / (num_relocations + 1) * plt_hdr->sh_addralign;
        else
            plt_entsize = plt_hdr->sh_size / (num_relocations + 1);
    }

    elf_xword plt_offset = plt_hdr->sh_size - num_relocations * plt_entsize;

    return std::make_pair(plt_entsize, plt_offset);
}

static unsigned
ParsePLTRelocations(Symtab *symbol_table,
                    user_id_t start_id,
                    unsigned rel_type,
                    const ELFHeader *hdr,
                    const ELFSectionHeader *rel_hdr,
                    const ELFSectionHeader *plt_hdr,
                    const ELFSectionHeader *sym_hdr,
                    const lldb::SectionSP &plt_section_sp,
                    DataExtractor &rel_data,
                    DataExtractor &symtab_data,
                    DataExtractor &strtab_data)
{
    ELFRelocation rel(rel_type);
    ELFSymbol symbol;
    lldb::offset_t offset = 0;

    uint64_t plt_offset, plt_entsize;
    std::tie(plt_entsize, plt_offset) = GetPltEntrySizeAndOffset(rel_hdr, plt_hdr);
    const elf_xword num_relocations = rel_hdr->sh_size / rel_hdr->sh_entsize;

    typedef unsigned (*reloc_info_fn)(const ELFRelocation &rel);
    reloc_info_fn reloc_type;
    reloc_info_fn reloc_symbol;

    if (hdr->Is32Bit())
    {
        reloc_type = ELFRelocation::RelocType32;
        reloc_symbol = ELFRelocation::RelocSymbol32;
    }
    else
    {
        reloc_type = ELFRelocation::RelocType64;
        reloc_symbol = ELFRelocation::RelocSymbol64;
    }

    unsigned slot_type = hdr->GetRelocationJumpSlotType();
    unsigned i;
    for (i = 0; i < num_relocations; ++i)
    {
        if (rel.Parse(rel_data, &offset) == false)
            break;

        if (reloc_type(rel) != slot_type)
            continue;

        lldb::offset_t symbol_offset = reloc_symbol(rel) * sym_hdr->sh_entsize;
        if (!symbol.Parse(symtab_data, &symbol_offset))
            break;

        const char *symbol_name = strtab_data.PeekCStr(symbol.st_name);
        bool is_mangled = symbol_name ? (symbol_name[0] == '_' && symbol_name[1] == 'Z') : false;
        uint64_t plt_index = plt_offset + i * plt_entsize;

        Symbol jump_symbol(
            i + start_id,    // Symbol table index
            symbol_name,     // symbol name.
            is_mangled,      // is the symbol name mangled?
            eSymbolTypeTrampoline, // Type of this symbol
            false,           // Is this globally visible?
            false,           // Is this symbol debug info?
            true,            // Is this symbol a trampoline?
            true,            // Is this symbol artificial?
            plt_section_sp,  // Section in which this symbol is defined or null.
            plt_index,       // Offset in section or symbol value.
            plt_entsize,     // Size in bytes of this symbol.
            true,            // Size is valid
            false,           // Contains linker annotations?
            0);              // Symbol flags.

        symbol_table->AddSymbol(jump_symbol);
    }

    return i;
}

unsigned
ObjectFileELF::ParseTrampolineSymbols(Symtab *symbol_table,
                                      user_id_t start_id,
                                      const ELFSectionHeaderInfo *rel_hdr,
                                      user_id_t rel_id)
{
    assert(rel_hdr->sh_type == SHT_RELA || rel_hdr->sh_type == SHT_REL);

    // The link field points to the associated symbol table. The info field
    // points to the section holding the plt.
    user_id_t symtab_id = rel_hdr->sh_link;
    user_id_t plt_id = rel_hdr->sh_info;

    // If the link field doesn't point to the appropriate symbol name table then
    // try to find it by name as some compiler don't fill in the link fields.
    if (!symtab_id)
        symtab_id = GetSectionIndexByName(".dynsym");
    if (!plt_id)
        plt_id = GetSectionIndexByName(".plt");

    if (!symtab_id || !plt_id)
        return 0;

    // Section ID's are ones based;
    symtab_id++;
    plt_id++;

    const ELFSectionHeaderInfo *plt_hdr = GetSectionHeaderByIndex(plt_id);
    if (!plt_hdr)
        return 0;

    const ELFSectionHeaderInfo *sym_hdr = GetSectionHeaderByIndex(symtab_id);
    if (!sym_hdr)
        return 0;

    SectionList *section_list = m_sections_ap.get();
    if (!section_list)
        return 0;

    Section *rel_section = section_list->FindSectionByID(rel_id).get();
    if (!rel_section)
        return 0;

    SectionSP plt_section_sp (section_list->FindSectionByID(plt_id));
    if (!plt_section_sp)
        return 0;

    Section *symtab = section_list->FindSectionByID(symtab_id).get();
    if (!symtab)
        return 0;

    // sh_link points to associated string table.
    Section *strtab = section_list->FindSectionByID(sym_hdr->sh_link + 1).get();
    if (!strtab)
        return 0;

    DataExtractor rel_data;
    if (!ReadSectionData(rel_section, rel_data))
        return 0;

    DataExtractor symtab_data;
    if (!ReadSectionData(symtab, symtab_data))
        return 0;

    DataExtractor strtab_data;
    if (!ReadSectionData(strtab, strtab_data))
        return 0;

    unsigned rel_type = PLTRelocationType();
    if (!rel_type)
        return 0;

    return ParsePLTRelocations (symbol_table,
                                start_id,
                                rel_type,
                                &m_header,
                                rel_hdr,
                                plt_hdr,
                                sym_hdr,
                                plt_section_sp,
                                rel_data,
                                symtab_data,
                                strtab_data);
}

unsigned
ObjectFileELF::RelocateSection(Symtab* symtab, const ELFHeader *hdr, const ELFSectionHeader *rel_hdr,
                const ELFSectionHeader *symtab_hdr, const ELFSectionHeader *debug_hdr,
                DataExtractor &rel_data, DataExtractor &symtab_data,
                DataExtractor &debug_data, Section* rel_section)
{
    ELFRelocation rel(rel_hdr->sh_type);
    lldb::addr_t offset = 0;
    const unsigned num_relocations = rel_hdr->sh_size / rel_hdr->sh_entsize;
    typedef unsigned (*reloc_info_fn)(const ELFRelocation &rel);
    reloc_info_fn reloc_type;
    reloc_info_fn reloc_symbol;

    if (hdr->Is32Bit())
    {
        reloc_type = ELFRelocation::RelocType32;
        reloc_symbol = ELFRelocation::RelocSymbol32;
    }
    else
    {
        reloc_type = ELFRelocation::RelocType64;
        reloc_symbol = ELFRelocation::RelocSymbol64;
    }

    for (unsigned i = 0; i < num_relocations; ++i)
    {
        if (rel.Parse(rel_data, &offset) == false)
            break;

        Symbol* symbol = NULL;

        if (hdr->Is32Bit())
        {
            switch (reloc_type(rel)) {
            case R_386_32:
            case R_386_PC32:
            default:
                assert(false && "unexpected relocation type");
            }
        } else {
            switch (reloc_type(rel)) {
            case R_X86_64_64:
            {
                symbol = symtab->FindSymbolByID(reloc_symbol(rel));
                if (symbol)
                {
                    addr_t value = symbol->GetAddressRef().GetFileAddress();
                    DataBufferSP& data_buffer_sp = debug_data.GetSharedDataBuffer();
                    uint64_t* dst = reinterpret_cast<uint64_t*>(data_buffer_sp->GetBytes() + rel_section->GetFileOffset() + ELFRelocation::RelocOffset64(rel));
                    *dst = value + ELFRelocation::RelocAddend64(rel);
                }
                break;
            }
            case R_X86_64_32:
            case R_X86_64_32S:
            {
                symbol = symtab->FindSymbolByID(reloc_symbol(rel));
                if (symbol)
                {
                    addr_t value = symbol->GetAddressRef().GetFileAddress();
                    value += ELFRelocation::RelocAddend32(rel);
                    assert((reloc_type(rel) == R_X86_64_32 && (value <= UINT32_MAX)) ||
                           (reloc_type(rel) == R_X86_64_32S &&
                            ((int64_t)value <= INT32_MAX && (int64_t)value >= INT32_MIN)));
                    uint32_t truncated_addr = (value & 0xFFFFFFFF);
                    DataBufferSP& data_buffer_sp = debug_data.GetSharedDataBuffer();
                    uint32_t* dst = reinterpret_cast<uint32_t*>(data_buffer_sp->GetBytes() + rel_section->GetFileOffset() + ELFRelocation::RelocOffset32(rel));
                    *dst = truncated_addr;
                }
                break;
            }
            case R_X86_64_PC32:
            default:
                assert(false && "unexpected relocation type");
            }
        }
    }

    return 0;
}

unsigned
ObjectFileELF::RelocateDebugSections(const ELFSectionHeader *rel_hdr, user_id_t rel_id)
{
    assert(rel_hdr->sh_type == SHT_RELA || rel_hdr->sh_type == SHT_REL);

    // Parse in the section list if needed.
    SectionList *section_list = GetSectionList();
    if (!section_list)
        return 0;

    // Section ID's are ones based.
    user_id_t symtab_id = rel_hdr->sh_link + 1;
    user_id_t debug_id = rel_hdr->sh_info + 1;

    const ELFSectionHeader *symtab_hdr = GetSectionHeaderByIndex(symtab_id);
    if (!symtab_hdr)
        return 0;

    const ELFSectionHeader *debug_hdr = GetSectionHeaderByIndex(debug_id);
    if (!debug_hdr)
        return 0;

    Section *rel = section_list->FindSectionByID(rel_id).get();
    if (!rel)
        return 0;

    Section *symtab = section_list->FindSectionByID(symtab_id).get();
    if (!symtab)
        return 0;

    Section *debug = section_list->FindSectionByID(debug_id).get();
    if (!debug)
        return 0;

    DataExtractor rel_data;
    DataExtractor symtab_data;
    DataExtractor debug_data;

    if (ReadSectionData(rel, rel_data) &&
        ReadSectionData(symtab, symtab_data) &&
        ReadSectionData(debug, debug_data))
    {
        RelocateSection(m_symtab_ap.get(), &m_header, rel_hdr, symtab_hdr, debug_hdr,
                        rel_data, symtab_data, debug_data, debug);
    }

    return 0;
}

Symtab *
ObjectFileELF::GetSymtab()
{
    ModuleSP module_sp(GetModule());
    if (!module_sp)
        return NULL;

    // We always want to use the main object file so we (hopefully) only have one cached copy
    // of our symtab, dynamic sections, etc.
    ObjectFile *module_obj_file = module_sp->GetObjectFile();
    if (module_obj_file && module_obj_file != this)
        return module_obj_file->GetSymtab();

    if (m_symtab_ap.get() == NULL)
    {
        SectionList *section_list = module_sp->GetSectionList();
        if (!section_list)
            return NULL;

        uint64_t symbol_id = 0;
        lldb_private::Mutex::Locker locker(module_sp->GetMutex());

        // Sharable objects and dynamic executables usually have 2 distinct symbol
        // tables, one named ".symtab", and the other ".dynsym". The dynsym is a smaller
        // version of the symtab that only contains global symbols. The information found
        // in the dynsym is therefore also found in the symtab, while the reverse is not
        // necessarily true.
        Section *symtab = section_list->FindSectionByType (eSectionTypeELFSymbolTable, true).get();
        if (!symtab)
        {
            // The symtab section is non-allocable and can be stripped, so if it doesn't exist
            // then use the dynsym section which should always be there.
            symtab = section_list->FindSectionByType (eSectionTypeELFDynamicSymbols, true).get();
        }
        if (symtab)
        {
            m_symtab_ap.reset(new Symtab(symtab->GetObjectFile()));
            symbol_id += ParseSymbolTable (m_symtab_ap.get(), symbol_id, symtab);
        }

        // DT_JMPREL
        //      If present, this entry's d_ptr member holds the address of relocation
        //      entries associated solely with the procedure linkage table. Separating
        //      these relocation entries lets the dynamic linker ignore them during
        //      process initialization, if lazy binding is enabled. If this entry is
        //      present, the related entries of types DT_PLTRELSZ and DT_PLTREL must
        //      also be present.
        const ELFDynamic *symbol = FindDynamicSymbol(DT_JMPREL);
        if (symbol)
        {
            // Synthesize trampoline symbols to help navigate the PLT.
            addr_t addr = symbol->d_ptr;
            Section *reloc_section = section_list->FindSectionContainingFileAddress(addr).get();
            if (reloc_section)
            {
                user_id_t reloc_id = reloc_section->GetID();
                const ELFSectionHeaderInfo *reloc_header = GetSectionHeaderByIndex(reloc_id);
                assert(reloc_header);

                if (m_symtab_ap == nullptr)
                    m_symtab_ap.reset(new Symtab(reloc_section->GetObjectFile()));

                ParseTrampolineSymbols (m_symtab_ap.get(), symbol_id, reloc_header, reloc_id);
            }
        }

        // If we still don't have any symtab then create an empty instance to avoid do the section
        // lookup next time.
        if (m_symtab_ap == nullptr)
            m_symtab_ap.reset(new Symtab(this));

        m_symtab_ap->CalculateSymbolSizes();
    }

    for (SectionHeaderCollIter I = m_section_headers.begin();
         I != m_section_headers.end(); ++I)
    {
        if (I->sh_type == SHT_RELA || I->sh_type == SHT_REL)
        {
            if (CalculateType() == eTypeObjectFile)
            {
                const char *section_name = I->section_name.AsCString("");
                if (strstr(section_name, ".rela.debug") ||
                    strstr(section_name, ".rel.debug"))
                {
                    const ELFSectionHeader &reloc_header = *I;
                    user_id_t reloc_id = SectionIndex(I);
                    RelocateDebugSections(&reloc_header, reloc_id);
                }
            }
        }
    }
    return m_symtab_ap.get();
}

Symbol *
ObjectFileELF::ResolveSymbolForAddress(const Address& so_addr, bool verify_unique)
{
    if (!m_symtab_ap.get())
        return nullptr; // GetSymtab() should be called first.

    const SectionList *section_list = GetSectionList();
    if (!section_list)
        return nullptr;

    if (DWARFCallFrameInfo *eh_frame = GetUnwindTable().GetEHFrameInfo())
    {
        AddressRange range;
        if (eh_frame->GetAddressRange (so_addr, range))
        {
            const addr_t file_addr = range.GetBaseAddress().GetFileAddress();
            Symbol * symbol = verify_unique ? m_symtab_ap->FindSymbolContainingFileAddress(file_addr) : nullptr;
            if (symbol)
                return symbol;

            // Note that a (stripped) symbol won't be found by GetSymtab()...
            lldb::SectionSP eh_sym_section_sp = section_list->FindSectionContainingFileAddress(file_addr);
            if (eh_sym_section_sp.get())
            {
                addr_t section_base = eh_sym_section_sp->GetFileAddress();
                addr_t offset = file_addr - section_base;
                uint64_t symbol_id = m_symtab_ap->GetNumSymbols();

                Symbol eh_symbol(
                        symbol_id,            // Symbol table index.
                        "???",                // Symbol name.
                        false,                // Is the symbol name mangled?
                        eSymbolTypeCode,      // Type of this symbol.
                        true,                 // Is this globally visible?
                        false,                // Is this symbol debug info?
                        false,                // Is this symbol a trampoline?
                        true,                 // Is this symbol artificial?
                        eh_sym_section_sp,    // Section in which this symbol is defined or null.
                        offset,               // Offset in section or symbol value.
                        range.GetByteSize(),  // Size in bytes of this symbol.
                        true,                 // Size is valid.
                        false,                // Contains linker annotations?
                        0);                   // Symbol flags.
                if (symbol_id == m_symtab_ap->AddSymbol(eh_symbol))
                    return m_symtab_ap->SymbolAtIndex(symbol_id);
            }
        }
    }
    return nullptr;
}


bool
ObjectFileELF::IsStripped ()
{
    // TODO: determine this for ELF
    return false;
}

//===----------------------------------------------------------------------===//
// Dump
//
// Dump the specifics of the runtime file container (such as any headers
// segments, sections, etc).
//----------------------------------------------------------------------
void
ObjectFileELF::Dump(Stream *s)
{
    DumpELFHeader(s, m_header);
    s->EOL();
    DumpELFProgramHeaders(s);
    s->EOL();
    DumpELFSectionHeaders(s);
    s->EOL();
    SectionList *section_list = GetSectionList();
    if (section_list)
        section_list->Dump(s, NULL, true, UINT32_MAX);
    Symtab *symtab = GetSymtab();
    if (symtab)
        symtab->Dump(s, NULL, eSortOrderNone);
    s->EOL();
    DumpDependentModules(s);
    s->EOL();
}

//----------------------------------------------------------------------
// DumpELFHeader
//
// Dump the ELF header to the specified output stream
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFHeader(Stream *s, const ELFHeader &header)
{
    s->PutCString("ELF Header\n");
    s->Printf("e_ident[EI_MAG0   ] = 0x%2.2x\n", header.e_ident[EI_MAG0]);
    s->Printf("e_ident[EI_MAG1   ] = 0x%2.2x '%c'\n",
              header.e_ident[EI_MAG1], header.e_ident[EI_MAG1]);
    s->Printf("e_ident[EI_MAG2   ] = 0x%2.2x '%c'\n",
              header.e_ident[EI_MAG2], header.e_ident[EI_MAG2]);
    s->Printf("e_ident[EI_MAG3   ] = 0x%2.2x '%c'\n",
              header.e_ident[EI_MAG3], header.e_ident[EI_MAG3]);

    s->Printf("e_ident[EI_CLASS  ] = 0x%2.2x\n", header.e_ident[EI_CLASS]);
    s->Printf("e_ident[EI_DATA   ] = 0x%2.2x ", header.e_ident[EI_DATA]);
    DumpELFHeader_e_ident_EI_DATA(s, header.e_ident[EI_DATA]);
    s->Printf ("\ne_ident[EI_VERSION] = 0x%2.2x\n", header.e_ident[EI_VERSION]);
    s->Printf ("e_ident[EI_PAD    ] = 0x%2.2x\n", header.e_ident[EI_PAD]);

    s->Printf("e_type      = 0x%4.4x ", header.e_type);
    DumpELFHeader_e_type(s, header.e_type);
    s->Printf("\ne_machine   = 0x%4.4x\n", header.e_machine);
    s->Printf("e_version   = 0x%8.8x\n", header.e_version);
    s->Printf("e_entry     = 0x%8.8" PRIx64 "\n", header.e_entry);
    s->Printf("e_phoff     = 0x%8.8" PRIx64 "\n", header.e_phoff);
    s->Printf("e_shoff     = 0x%8.8" PRIx64 "\n", header.e_shoff);
    s->Printf("e_flags     = 0x%8.8x\n", header.e_flags);
    s->Printf("e_ehsize    = 0x%4.4x\n", header.e_ehsize);
    s->Printf("e_phentsize = 0x%4.4x\n", header.e_phentsize);
    s->Printf("e_phnum     = 0x%4.4x\n", header.e_phnum);
    s->Printf("e_shentsize = 0x%4.4x\n", header.e_shentsize);
    s->Printf("e_shnum     = 0x%4.4x\n", header.e_shnum);
    s->Printf("e_shstrndx  = 0x%4.4x\n", header.e_shstrndx);
}

//----------------------------------------------------------------------
// DumpELFHeader_e_type
//
// Dump an token value for the ELF header member e_type
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFHeader_e_type(Stream *s, elf_half e_type)
{
    switch (e_type)
    {
    case ET_NONE:   *s << "ET_NONE"; break;
    case ET_REL:    *s << "ET_REL"; break;
    case ET_EXEC:   *s << "ET_EXEC"; break;
    case ET_DYN:    *s << "ET_DYN"; break;
    case ET_CORE:   *s << "ET_CORE"; break;
    default:
        break;
    }
}

//----------------------------------------------------------------------
// DumpELFHeader_e_ident_EI_DATA
//
// Dump an token value for the ELF header member e_ident[EI_DATA]
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFHeader_e_ident_EI_DATA(Stream *s, unsigned char ei_data)
{
    switch (ei_data)
    {
    case ELFDATANONE:   *s << "ELFDATANONE"; break;
    case ELFDATA2LSB:   *s << "ELFDATA2LSB - Little Endian"; break;
    case ELFDATA2MSB:   *s << "ELFDATA2MSB - Big Endian"; break;
    default:
        break;
    }
}


//----------------------------------------------------------------------
// DumpELFProgramHeader
//
// Dump a single ELF program header to the specified output stream
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFProgramHeader(Stream *s, const ELFProgramHeader &ph)
{
    DumpELFProgramHeader_p_type(s, ph.p_type);
    s->Printf(" %8.8" PRIx64 " %8.8" PRIx64 " %8.8" PRIx64, ph.p_offset, ph.p_vaddr, ph.p_paddr);
    s->Printf(" %8.8" PRIx64 " %8.8" PRIx64 " %8.8x (", ph.p_filesz, ph.p_memsz, ph.p_flags);

    DumpELFProgramHeader_p_flags(s, ph.p_flags);
    s->Printf(") %8.8" PRIx64, ph.p_align);
}

//----------------------------------------------------------------------
// DumpELFProgramHeader_p_type
//
// Dump an token value for the ELF program header member p_type which
// describes the type of the program header
// ----------------------------------------------------------------------
void
ObjectFileELF::DumpELFProgramHeader_p_type(Stream *s, elf_word p_type)
{
    const int kStrWidth = 15;
    switch (p_type)
    {
    CASE_AND_STREAM(s, PT_NULL        , kStrWidth);
    CASE_AND_STREAM(s, PT_LOAD        , kStrWidth);
    CASE_AND_STREAM(s, PT_DYNAMIC     , kStrWidth);
    CASE_AND_STREAM(s, PT_INTERP      , kStrWidth);
    CASE_AND_STREAM(s, PT_NOTE        , kStrWidth);
    CASE_AND_STREAM(s, PT_SHLIB       , kStrWidth);
    CASE_AND_STREAM(s, PT_PHDR        , kStrWidth);
    CASE_AND_STREAM(s, PT_TLS         , kStrWidth);
    CASE_AND_STREAM(s, PT_GNU_EH_FRAME, kStrWidth);
    default:
        s->Printf("0x%8.8x%*s", p_type, kStrWidth - 10, "");
        break;
    }
}


//----------------------------------------------------------------------
// DumpELFProgramHeader_p_flags
//
// Dump an token value for the ELF program header member p_flags
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFProgramHeader_p_flags(Stream *s, elf_word p_flags)
{
    *s  << ((p_flags & PF_X) ? "PF_X" : "    ")
        << (((p_flags & PF_X) && (p_flags & PF_W)) ? '+' : ' ')
        << ((p_flags & PF_W) ? "PF_W" : "    ")
        << (((p_flags & PF_W) && (p_flags & PF_R)) ? '+' : ' ')
        << ((p_flags & PF_R) ? "PF_R" : "    ");
}

//----------------------------------------------------------------------
// DumpELFProgramHeaders
//
// Dump all of the ELF program header to the specified output stream
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFProgramHeaders(Stream *s)
{
    if (!ParseProgramHeaders())
        return;

    s->PutCString("Program Headers\n");
    s->PutCString("IDX  p_type          p_offset p_vaddr  p_paddr  "
                  "p_filesz p_memsz  p_flags                   p_align\n");
    s->PutCString("==== --------------- -------- -------- -------- "
                  "-------- -------- ------------------------- --------\n");

    uint32_t idx = 0;
    for (ProgramHeaderCollConstIter I = m_program_headers.begin();
         I != m_program_headers.end(); ++I, ++idx)
    {
        s->Printf("[%2u] ", idx);
        ObjectFileELF::DumpELFProgramHeader(s, *I);
        s->EOL();
    }
}

//----------------------------------------------------------------------
// DumpELFSectionHeader
//
// Dump a single ELF section header to the specified output stream
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFSectionHeader(Stream *s, const ELFSectionHeaderInfo &sh)
{
    s->Printf("%8.8x ", sh.sh_name);
    DumpELFSectionHeader_sh_type(s, sh.sh_type);
    s->Printf(" %8.8" PRIx64 " (", sh.sh_flags);
    DumpELFSectionHeader_sh_flags(s, sh.sh_flags);
    s->Printf(") %8.8" PRIx64 " %8.8" PRIx64 " %8.8" PRIx64, sh.sh_addr, sh.sh_offset, sh.sh_size);
    s->Printf(" %8.8x %8.8x", sh.sh_link, sh.sh_info);
    s->Printf(" %8.8" PRIx64 " %8.8" PRIx64, sh.sh_addralign, sh.sh_entsize);
}

//----------------------------------------------------------------------
// DumpELFSectionHeader_sh_type
//
// Dump an token value for the ELF section header member sh_type which
// describes the type of the section
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFSectionHeader_sh_type(Stream *s, elf_word sh_type)
{
    const int kStrWidth = 12;
    switch (sh_type)
    {
    CASE_AND_STREAM(s, SHT_NULL     , kStrWidth);
    CASE_AND_STREAM(s, SHT_PROGBITS , kStrWidth);
    CASE_AND_STREAM(s, SHT_SYMTAB   , kStrWidth);
    CASE_AND_STREAM(s, SHT_STRTAB   , kStrWidth);
    CASE_AND_STREAM(s, SHT_RELA     , kStrWidth);
    CASE_AND_STREAM(s, SHT_HASH     , kStrWidth);
    CASE_AND_STREAM(s, SHT_DYNAMIC  , kStrWidth);
    CASE_AND_STREAM(s, SHT_NOTE     , kStrWidth);
    CASE_AND_STREAM(s, SHT_NOBITS   , kStrWidth);
    CASE_AND_STREAM(s, SHT_REL      , kStrWidth);
    CASE_AND_STREAM(s, SHT_SHLIB    , kStrWidth);
    CASE_AND_STREAM(s, SHT_DYNSYM   , kStrWidth);
    CASE_AND_STREAM(s, SHT_LOPROC   , kStrWidth);
    CASE_AND_STREAM(s, SHT_HIPROC   , kStrWidth);
    CASE_AND_STREAM(s, SHT_LOUSER   , kStrWidth);
    CASE_AND_STREAM(s, SHT_HIUSER   , kStrWidth);
    default:
        s->Printf("0x%8.8x%*s", sh_type, kStrWidth - 10, "");
        break;
    }
}

//----------------------------------------------------------------------
// DumpELFSectionHeader_sh_flags
//
// Dump an token value for the ELF section header member sh_flags
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFSectionHeader_sh_flags(Stream *s, elf_xword sh_flags)
{
    *s  << ((sh_flags & SHF_WRITE) ? "WRITE" : "     ")
        << (((sh_flags & SHF_WRITE) && (sh_flags & SHF_ALLOC)) ? '+' : ' ')
        << ((sh_flags & SHF_ALLOC) ? "ALLOC" : "     ")
        << (((sh_flags & SHF_ALLOC) && (sh_flags & SHF_EXECINSTR)) ? '+' : ' ')
        << ((sh_flags & SHF_EXECINSTR) ? "EXECINSTR" : "         ");
}

//----------------------------------------------------------------------
// DumpELFSectionHeaders
//
// Dump all of the ELF section header to the specified output stream
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFSectionHeaders(Stream *s)
{
    if (!ParseSectionHeaders())
        return;

    s->PutCString("Section Headers\n");
    s->PutCString("IDX  name     type         flags                            "
                  "addr     offset   size     link     info     addralgn "
                  "entsize  Name\n");
    s->PutCString("==== -------- ------------ -------------------------------- "
                  "-------- -------- -------- -------- -------- -------- "
                  "-------- ====================\n");

    uint32_t idx = 0;
    for (SectionHeaderCollConstIter I = m_section_headers.begin();
         I != m_section_headers.end(); ++I, ++idx)
    {
        s->Printf("[%2u] ", idx);
        ObjectFileELF::DumpELFSectionHeader(s, *I);
        const char* section_name = I->section_name.AsCString("");
        if (section_name)
            *s << ' ' << section_name << "\n";
    }
}

void
ObjectFileELF::DumpDependentModules(lldb_private::Stream *s)
{
    size_t num_modules = ParseDependentModules();

    if (num_modules > 0)
    {
        s->PutCString("Dependent Modules:\n");
        for (unsigned i = 0; i < num_modules; ++i)
        {
            const FileSpec &spec = m_filespec_ap->GetFileSpecAtIndex(i);
            s->Printf("   %s\n", spec.GetFilename().GetCString());
        }
    }
}

bool
ObjectFileELF::GetArchitecture (ArchSpec &arch)
{
    if (!ParseHeader())
        return false;

    if (m_section_headers.empty())
    {
        // Allow elf notes to be parsed which may affect the detected architecture.
        ParseSectionHeaders();
    }

    if (CalculateType() == eTypeCoreFile && m_arch_spec.TripleOSIsUnspecifiedUnknown())
    {
        // Core files don't have section headers yet they have PT_NOTE program headers
        // that might shed more light on the architecture
        if (ParseProgramHeaders())
        {
            for (size_t i = 0, count = GetProgramHeaderCount(); i < count; ++i)
            {
                const elf::ELFProgramHeader* header = GetProgramHeaderByIndex(i);
                if (header && header->p_type == PT_NOTE && header->p_offset != 0 && header->p_filesz > 0)
                {
                    DataExtractor data;
                    if (data.SetData (m_data, header->p_offset, header->p_filesz) == header->p_filesz)
                    {
                        lldb_private::UUID uuid;
                        RefineModuleDetailsFromNote (data, m_arch_spec, uuid);
                    }
                }
            }
        }
    }
    arch = m_arch_spec;
    return true;
}

ObjectFile::Type
ObjectFileELF::CalculateType()
{
    switch (m_header.e_type)
    {
        case llvm::ELF::ET_NONE:
            // 0 - No file type
            return eTypeUnknown;

        case llvm::ELF::ET_REL:
            // 1 - Relocatable file
            return eTypeObjectFile;

        case llvm::ELF::ET_EXEC:
            // 2 - Executable file
            return eTypeExecutable;

        case llvm::ELF::ET_DYN:
            // 3 - Shared object file
            return eTypeSharedLibrary;

        case ET_CORE:
            // 4 - Core file
            return eTypeCoreFile;

        default:
            break;
    }
    return eTypeUnknown;
}

ObjectFile::Strata
ObjectFileELF::CalculateStrata()
{
    switch (m_header.e_type)
    {
        case llvm::ELF::ET_NONE:
            // 0 - No file type
            return eStrataUnknown;

        case llvm::ELF::ET_REL:
            // 1 - Relocatable file
            return eStrataUnknown;

        case llvm::ELF::ET_EXEC:
            // 2 - Executable file
            // TODO: is there any way to detect that an executable is a kernel
            // related executable by inspecting the program headers, section
            // headers, symbols, or any other flag bits???
            return eStrataUser;

        case llvm::ELF::ET_DYN:
            // 3 - Shared object file
            // TODO: is there any way to detect that an shared library is a kernel
            // related executable by inspecting the program headers, section
            // headers, symbols, or any other flag bits???
            return eStrataUnknown;

        case ET_CORE:
            // 4 - Core file
            // TODO: is there any way to detect that an core file is a kernel
            // related executable by inspecting the program headers, section
            // headers, symbols, or any other flag bits???
            return eStrataUnknown;

        default:
            break;
    }
    return eStrataUnknown;
}

