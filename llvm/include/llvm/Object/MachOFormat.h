//===- MachOFormat.h - Mach-O Format Structures And Constants ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares various structures and constants which are platform
// independent and can be shared by any client which wishes to interact with
// Mach object files.
//
// The definitions here are purposely chosen to match the LLVM style as opposed
// to following the platform specific definition of the format.
//
// On a Mach system, see the <mach-o/...> includes for more information, in
// particular <mach-o/loader.h>.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_MACHOFORMAT_H
#define LLVM_OBJECT_MACHOFORMAT_H

#include "llvm/Support/DataTypes.h"

namespace llvm {
namespace object {

/// General Mach platform information.
namespace mach {
  /// @name CPU Type and Subtype Information
  /// {

  /// \brief Capability bits used in CPU type encoding.
  enum CPUTypeFlagsMask {
    CTFM_ArchMask =  0xFF000000,
    CTFM_ArchABI64 = 0x01000000
  };

  /// \brief Machine type IDs used in CPU type encoding.
  enum CPUTypeMachine {
    CTM_i386      = 7,
    CTM_x86_64    = CTM_i386 | CTFM_ArchABI64,
    CTM_ARM       = 12,
    CTM_SPARC     = 14,
    CTM_PowerPC   = 18,
    CTM_PowerPC64 = CTM_PowerPC | CTFM_ArchABI64
  };

  /// \brief Capability bits used in CPU subtype encoding.
  enum CPUSubtypeFlagsMask {
    CSFM_SubtypeMask =  0xFF000000,
    CSFM_SubtypeLib64 = 0x80000000
  };

  /// \brief ARM Machine Subtypes.
  enum CPUSubtypeARM {
    CSARM_ALL    = 0,
    CSARM_V4T    = 5,
    CSARM_V6     = 6,
    CSARM_V5TEJ  = 7,
    CSARM_XSCALE = 8,
    CSARM_V7     = 9,
    CSARM_V7F    = 10,
    CSARM_V7S    = 11,
    CSARM_V7K    = 12
  };

  /// \brief PowerPC Machine Subtypes.
  enum CPUSubtypePowerPC {
    CSPPC_ALL = 0
  };

  /// \brief SPARC Machine Subtypes.
  enum CPUSubtypeSPARC {
    CSSPARC_ALL = 0
  };

  /// \brief x86 Machine Subtypes.
  enum CPUSubtypeX86 {
    CSX86_ALL = 3
  };

  /// @}

} // end namespace mach

/// Format information for Mach object files.
namespace macho {
  /// \brief Constants for structure sizes.
  enum StructureSizes {
    Header32Size = 28,
    Header64Size = 32,
    SegmentLoadCommand32Size = 56,
    SegmentLoadCommand64Size = 72,
    Section32Size = 68,
    Section64Size = 80,
    SymtabLoadCommandSize = 24,
    DysymtabLoadCommandSize = 80,
    Nlist32Size = 12,
    Nlist64Size = 16,
    RelocationInfoSize = 8,
    LinkeditLoadCommandSize = 16
  };

  /// \brief Constants for header magic field.
  enum HeaderMagic {
    HM_Object32 = 0xFEEDFACE,  ///< 32-bit mach object file
    HM_Object64 = 0xFEEDFACF,  ///< 64-bit mach object file
    HM_Universal = 0xCAFEBABE  ///< Universal object file
  };

  /// \brief Header common to all Mach object files.
  struct Header {
    uint32_t Magic;
    uint32_t CPUType;
    uint32_t CPUSubtype;
    uint32_t FileType;
    uint32_t NumLoadCommands;
    uint32_t SizeOfLoadCommands;
    uint32_t Flags;
  };

  /// \brief Extended header for 64-bit object files.
  struct Header64Ext {
    uint32_t Reserved;
  };

  // See <mach-o/loader.h>.
  enum HeaderFileType {
    HFT_Object = 0x1
  };

  enum HeaderFlags {
    HF_SubsectionsViaSymbols = 0x2000
  };

  enum LoadCommandType {
    LCT_Segment = 0x1,
    LCT_Symtab = 0x2,
    LCT_Dysymtab = 0xb,
    LCT_Segment64 = 0x19,
    LCT_UUID = 0x1b,
    LCT_CodeSignature = 0x1d,
    LCT_SegmentSplitInfo = 0x1e,
    LCT_FunctionStarts = 0x26,
    LCT_DataInCode = 0x29
  };

  /// \brief Load command structure.
  struct LoadCommand {
    uint32_t Type;
    uint32_t Size;
  };

  /// @name Load Command Structures
  /// @{

  struct SegmentLoadCommand {
    uint32_t Type;
    uint32_t Size;
    char Name[16];
    uint32_t VMAddress;
    uint32_t VMSize;
    uint32_t FileOffset;
    uint32_t FileSize;
    uint32_t MaxVMProtection;
    uint32_t InitialVMProtection;
    uint32_t NumSections;
    uint32_t Flags;
  };

  struct Segment64LoadCommand {
    uint32_t Type;
    uint32_t Size;
    char Name[16];
    uint64_t VMAddress;
    uint64_t VMSize;
    uint64_t FileOffset;
    uint64_t FileSize;
    uint32_t MaxVMProtection;
    uint32_t InitialVMProtection;
    uint32_t NumSections;
    uint32_t Flags;
  };

  struct SymtabLoadCommand {
    uint32_t Type;
    uint32_t Size;
    uint32_t SymbolTableOffset;
    uint32_t NumSymbolTableEntries;
    uint32_t StringTableOffset;
    uint32_t StringTableSize;
  };

  struct DysymtabLoadCommand {
    uint32_t Type;
    uint32_t Size;

    uint32_t LocalSymbolsIndex;
    uint32_t NumLocalSymbols;

    uint32_t ExternalSymbolsIndex;
    uint32_t NumExternalSymbols;

    uint32_t UndefinedSymbolsIndex;
    uint32_t NumUndefinedSymbols;

    uint32_t TOCOffset;
    uint32_t NumTOCEntries;

    uint32_t ModuleTableOffset;
    uint32_t NumModuleTableEntries;

    uint32_t ReferenceSymbolTableOffset;
    uint32_t NumReferencedSymbolTableEntries;

    uint32_t IndirectSymbolTableOffset;
    uint32_t NumIndirectSymbolTableEntries;

    uint32_t ExternalRelocationTableOffset;
    uint32_t NumExternalRelocationTableEntries;

    uint32_t LocalRelocationTableOffset;
    uint32_t NumLocalRelocationTableEntries;
  };

  struct LinkeditDataLoadCommand {
    uint32_t Type;
    uint32_t Size;
    uint32_t DataOffset;
    uint32_t DataSize;
  };

  /// @}
  /// @name Section Data
  /// @{

  struct Section {
    char Name[16];
    char SegmentName[16];
    uint32_t Address;
    uint32_t Size;
    uint32_t Offset;
    uint32_t Align;
    uint32_t RelocationTableOffset;
    uint32_t NumRelocationTableEntries;
    uint32_t Flags;
    uint32_t Reserved1;
    uint32_t Reserved2;
  };
  struct Section64 {
    char Name[16];
    char SegmentName[16];
    uint64_t Address;
    uint64_t Size;
    uint32_t Offset;
    uint32_t Align;
    uint32_t RelocationTableOffset;
    uint32_t NumRelocationTableEntries;
    uint32_t Flags;
    uint32_t Reserved1;
    uint32_t Reserved2;
    uint32_t Reserved3;
  };

  /// @}
  /// @name Symbol Table Entries
  /// @{

  struct SymbolTableEntry {
    uint32_t StringIndex;
    uint8_t Type;
    uint8_t SectionIndex;
    uint16_t Flags;
    uint32_t Value;
  };
  // Despite containing a uint64_t, this structure is only 4-byte aligned within
  // a MachO file.
#pragma pack(push)
#pragma pack(4)
  struct Symbol64TableEntry {
    uint32_t StringIndex;
    uint8_t Type;
    uint8_t SectionIndex;
    uint16_t Flags;
    uint64_t Value;
  };
#pragma pack(pop)

  /// @}
  /// @name Data-in-code Table Entry
  /// @{

  // See <mach-o/loader.h>.
  enum DataRegionType { Data = 1, JumpTable8, JumpTable16, JumpTable32 };
  struct DataInCodeTableEntry {
    uint32_t Offset;  /* from mach_header to start of data region */
    uint16_t Length;  /* number of bytes in data region */
    uint16_t Kind;    /* a DataRegionType value  */
  };

  /// @}
  /// @name Indirect Symbol Table
  /// @{

  struct IndirectSymbolTableEntry {
    uint32_t Index;
  };

  /// @}
  /// @name Relocation Data
  /// @{

  struct RelocationEntry {
    uint32_t Word0;
    uint32_t Word1;
  };

  /// @}

  // See <mach-o/nlist.h>.
  enum SymbolTypeType {
    STT_Undefined = 0x00,
    STT_Absolute  = 0x02,
    STT_Section   = 0x0e
  };

  enum SymbolTypeFlags {
    // If any of these bits are set, then the entry is a stab entry number (see
    // <mach-o/stab.h>. Otherwise the other masks apply.
    STF_StabsEntryMask = 0xe0,

    STF_TypeMask       = 0x0e,
    STF_External       = 0x01,
    STF_PrivateExtern  = 0x10
  };

  /// IndirectSymbolFlags - Flags for encoding special values in the indirect
  /// symbol entry.
  enum IndirectSymbolFlags {
    ISF_Local    = 0x80000000,
    ISF_Absolute = 0x40000000
  };

  /// RelocationFlags - Special flags for addresses.
  enum RelocationFlags {
    RF_Scattered = 0x80000000
  };

  /// Common relocation info types.
  enum RelocationInfoType {
    RIT_Vanilla             = 0,
    RIT_Pair                = 1,
    RIT_Difference          = 2
  };

  /// Generic relocation info types, which are shared by some (but not all)
  /// platforms.
  enum RelocationInfoType_Generic {
    RIT_Generic_PreboundLazyPointer = 3,
    RIT_Generic_LocalDifference     = 4,
    RIT_Generic_TLV                 = 5
  };

  /// X86_64 uses its own relocation types.
  enum RelocationInfoTypeX86_64 {
    // Note that x86_64 doesn't even share the common relocation types.
    RIT_X86_64_Unsigned   = 0,
    RIT_X86_64_Signed     = 1,
    RIT_X86_64_Branch     = 2,
    RIT_X86_64_GOTLoad    = 3,
    RIT_X86_64_GOT        = 4,
    RIT_X86_64_Subtractor = 5,
    RIT_X86_64_Signed1    = 6,
    RIT_X86_64_Signed2    = 7,
    RIT_X86_64_Signed4    = 8,
    RIT_X86_64_TLV        = 9
  };

  /// ARM uses its own relocation types.
  enum RelocationInfoTypeARM {
    RIT_ARM_LocalDifference = 3,
    RIT_ARM_PreboundLazyPointer = 4,
    RIT_ARM_Branch24Bit = 5,
    RIT_ARM_ThumbBranch22Bit = 6,
    RIT_ARM_ThumbBranch32Bit = 7,
    RIT_ARM_Half = 8,
    RIT_ARM_HalfDifference = 9

  };

} // end namespace macho

} // end namespace object
} // end namespace llvm

#endif
