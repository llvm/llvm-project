//===-- ELFHeader.cpp ----------------------------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cstring>

#include "lldb/Core/DataExtractor.h"

#include "ELFHeader.h"

using namespace elf;
using namespace lldb;
using namespace llvm::ELF;

//------------------------------------------------------------------------------
// Static utility functions.
//
// GetMaxU64 and GetMaxS64 wrap the similarly named methods from DataExtractor
// with error handling code and provide for parsing a sequence of values.
static bool
GetMaxU64(const lldb_private::DataExtractor &data, 
          uint32_t *offset, uint64_t *value, uint32_t byte_size) 
{
    const uint32_t saved_offset = *offset;
    *value = data.GetMaxU64(offset, byte_size);
    return *offset != saved_offset;
}

static bool
GetMaxU64(const lldb_private::DataExtractor &data, 
          uint32_t *offset, uint64_t *value, uint32_t byte_size, 
          uint32_t count) 
{
    uint32_t saved_offset = *offset;

    for (uint32_t i = 0; i < count; ++i, ++value)
    {
        if (GetMaxU64(data, offset, value, byte_size) == false) 
        {
            *offset = saved_offset;
            return false;
        }
    }
    return true;
}

static bool
GetMaxS64(const lldb_private::DataExtractor &data, 
          uint32_t *offset, int64_t *value, uint32_t byte_size) 
{
    const uint32_t saved_offset = *offset;
    *value = data.GetMaxS64(offset, byte_size);
    return *offset != saved_offset;
}

static bool
GetMaxS64(const lldb_private::DataExtractor &data, 
          uint32_t *offset, int64_t *value, uint32_t byte_size, 
          uint32_t count) 
{
    uint32_t saved_offset = *offset;

    for (uint32_t i = 0; i < count; ++i, ++value)
    {
        if (GetMaxS64(data, offset, value, byte_size) == false) 
        {
            *offset = saved_offset;
            return false;
        }
    }
    return true;
}

//------------------------------------------------------------------------------
// ELFHeader

ELFHeader::ELFHeader()
{
    memset(this, 0, sizeof(ELFHeader)); 
}

ByteOrder
ELFHeader::GetByteOrder() const 
{
    if (e_ident[EI_DATA] == ELFDATA2MSB)
        return eByteOrderBig;
    if (e_ident[EI_DATA] == ELFDATA2LSB)
        return eByteOrderLittle;
    return eByteOrderInvalid;
}

bool
ELFHeader::Parse(lldb_private::DataExtractor &data, uint32_t *offset) 
{
    // Read e_ident.  This provides byte order and address size info.
    if (data.GetU8(offset, &e_ident, EI_NIDENT) == NULL)
        return false;

    const unsigned byte_size = Is32Bit() ? 4 : 8;
    data.SetByteOrder(GetByteOrder());
    data.SetAddressByteSize(byte_size);

    // Read e_type and e_machine.
    if (data.GetU16(offset, &e_type, 2) == NULL)
        return false;

    // Read e_version.
    if (data.GetU32(offset, &e_version, 1) == NULL)
        return false;

    // Read e_entry, e_phoff and e_shoff.
    if (GetMaxU64(data, offset, &e_entry, byte_size, 3) == false)
        return false;

    // Read e_flags.
    if (data.GetU32(offset, &e_flags, 1) == NULL)
        return false;

    // Read e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum and
    // e_shstrndx.
    if (data.GetU16(offset, &e_ehsize, 6) == NULL)
        return false;

    return true;
}

bool
ELFHeader::MagicBytesMatch(const uint8_t *magic)
{
    return memcmp(magic, ElfMagic, strlen(ElfMagic)) == 0;
}

unsigned
ELFHeader::AddressSizeInBytes(const uint8_t *magic)
{
    unsigned address_size = 0;

    switch (magic[EI_CLASS]) 
    {
    case ELFCLASS32:
        address_size = 4;
        break;
            
    case ELFCLASS64:
        address_size = 8;
        break;
    }
    return address_size;
}

unsigned
ELFHeader::GetRelocationJumpSlotType() const
{
    unsigned slot = 0;

    switch (e_machine)
    {
    default:
        assert(false && "architecture not supported");
        break;
    case EM_386:
    case EM_486:
        slot = R_386_JUMP_SLOT;
        break;
    case EM_X86_64:
        slot = R_X86_64_JUMP_SLOT;
        break;
    case EM_ARM:
        slot = R_ARM_JUMP_SLOT;
        break;
    case EM_MBLAZE:
        slot = R_MICROBLAZE_JUMP_SLOT;
    }

    return slot;
}

//------------------------------------------------------------------------------
// ELFSectionHeader

ELFSectionHeader::ELFSectionHeader() 
{
    memset(this, 0, sizeof(ELFSectionHeader));
}

bool
ELFSectionHeader::Parse(const lldb_private::DataExtractor &data,
                        uint32_t *offset) 
{
    const unsigned byte_size = data.GetAddressByteSize();

    // Read sh_name and sh_type.
    if (data.GetU32(offset, &sh_name, 2) == NULL)
        return false;

    // Read sh_flags.
    if (GetMaxU64(data, offset, &sh_flags, byte_size) == false)
        return false;

    // Read sh_addr, sh_off and sh_size.
    if (GetMaxU64(data, offset, &sh_addr, byte_size, 3) == false)
        return false;

    // Read sh_link and sh_info.
    if (data.GetU32(offset, &sh_link, 2) == NULL)
        return false;

    // Read sh_addralign and sh_entsize.
    if (GetMaxU64(data, offset, &sh_addralign, byte_size, 2) == false)
        return false;

    return true;
}

//------------------------------------------------------------------------------
// ELFSymbol

ELFSymbol::ELFSymbol() 
{
    memset(this, 0, sizeof(ELFSymbol));
}

bool
ELFSymbol::Parse(const lldb_private::DataExtractor &data, uint32_t *offset) 
{
    const unsigned byte_size = data.GetAddressByteSize();
    const bool parsing_32 = byte_size == 4;

    // Read st_name.
    if (data.GetU32(offset, &st_name, 1) == NULL)
        return false;

    if (parsing_32) 
    {
        // Read st_value and st_size.
        if (GetMaxU64(data, offset, &st_value, byte_size, 2) == false)
            return false;

        // Read st_info and st_other.
        if (data.GetU8(offset, &st_info, 2) == NULL)
            return false;
            
        // Read st_shndx.
        if (data.GetU16(offset, &st_shndx, 1) == NULL)
            return false;
    }
    else 
    {
        // Read st_info and st_other.
        if (data.GetU8(offset, &st_info, 2) == NULL)
            return false;
            
        // Read st_shndx.
        if (data.GetU16(offset, &st_shndx, 1) == NULL)
            return false;

        // Read st_value and st_size.
        if (data.GetU64(offset, &st_value, 2) == NULL)
            return false;
    }
    return true;
}

//------------------------------------------------------------------------------
// ELFProgramHeader

ELFProgramHeader::ELFProgramHeader() 
{
    memset(this, 0, sizeof(ELFProgramHeader));
}

bool
ELFProgramHeader::Parse(const lldb_private::DataExtractor &data, 
                        uint32_t *offset) 
{
    const uint32_t byte_size = data.GetAddressByteSize();
    const bool parsing_32 = byte_size == 4;

    // Read p_type;
    if (data.GetU32(offset, &p_type, 1) == NULL)
        return false;

    if (parsing_32) {
        // Read p_offset, p_vaddr, p_paddr, p_filesz and p_memsz.
        if (GetMaxU64(data, offset, &p_offset, byte_size, 5) == false)
            return false;

        // Read p_flags.
        if (data.GetU32(offset, &p_flags, 1) == NULL)
            return false;

        // Read p_align.
        if (GetMaxU64(data, offset, &p_align, byte_size) == false)
            return false;
    }
    else {
        // Read p_flags.
        if (data.GetU32(offset, &p_flags, 1) == NULL)
            return false;

        // Read p_offset, p_vaddr, p_paddr, p_filesz, p_memsz and p_align.
        if (GetMaxU64(data, offset, &p_offset, byte_size, 6) == false)
            return false;
    }

    return true;
}

//------------------------------------------------------------------------------
// ELFDynamic

ELFDynamic::ELFDynamic() 
{ 
    memset(this, 0, sizeof(ELFDynamic)); 
}

bool
ELFDynamic::Parse(const lldb_private::DataExtractor &data, uint32_t *offset)
{
    const unsigned byte_size = data.GetAddressByteSize();
    return GetMaxS64(data, offset, &d_tag, byte_size, 2);
}

//------------------------------------------------------------------------------
// ELFRel

ELFRel::ELFRel()
{
    memset(this, 0, sizeof(ELFRel));
}

bool
ELFRel::Parse(const lldb_private::DataExtractor &data, uint32_t *offset)
{
    const unsigned byte_size = data.GetAddressByteSize();

    // Read r_offset and r_info.
    if (GetMaxU64(data, offset, &r_offset, byte_size, 2) == false)
        return false;

    return true;
}

//------------------------------------------------------------------------------
// ELFRela

ELFRela::ELFRela()
{
    memset(this, 0, sizeof(ELFRela));
}

bool
ELFRela::Parse(const lldb_private::DataExtractor &data, uint32_t *offset)
{
    const unsigned byte_size = data.GetAddressByteSize();

    // Read r_offset and r_info.
    if (GetMaxU64(data, offset, &r_offset, byte_size, 2) == false)
        return false;

    // Read r_addend;
    if (GetMaxS64(data, offset, &r_addend, byte_size) == false)
        return false;

    return true;
}


