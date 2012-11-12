//===-- llvm/Support/MachO.h - The MachO file format ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines manifest constants for the MachO object file format.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_MACHO_H
#define LLVM_SUPPORT_MACHO_H

#include "llvm/Support/DataTypes.h"

// NOTE: The enums in this file are intentially named to be different than those
// in the headers in /usr/include/mach (on darwin systems) to avoid conflicts
// with those macros.
namespace llvm {
  namespace MachO {
    // Enums from <mach-o/loader.h>
    enum {
      // Constants for the "magic" field in llvm::MachO::mach_header and
      // llvm::MachO::mach_header_64
      HeaderMagic32         = 0xFEEDFACEu, // MH_MAGIC
      HeaderMagic32Swapped  = 0xCEFAEDFEu, // MH_CIGAM
      HeaderMagic64         = 0xFEEDFACFu, // MH_MAGIC_64
      HeaderMagic64Swapped  = 0xCFFAEDFEu, // MH_CIGAM_64
      UniversalMagic        = 0xCAFEBABEu, // FAT_MAGIC
      UniversalMagicSwapped = 0xBEBAFECAu, // FAT_CIGAM

      // Constants for the "filetype" field in llvm::MachO::mach_header and
      // llvm::MachO::mach_header_64
      HeaderFileTypeObject              = 0x1u, // MH_OBJECT
      HeaderFileTypeExecutable          = 0x2u, // MH_EXECUTE
      HeaderFileTypeFixedVMShlib        = 0x3u, // MH_FVMLIB
      HeaderFileTypeCore                = 0x4u, // MH_CORE
      HeaderFileTypePreloadedExecutable = 0x5u, // MH_PRELOAD
      HeaderFileTypeDynamicShlib        = 0x6u, // MH_DYLIB
      HeaderFileTypeDynamicLinkEditor   = 0x7u, // MH_DYLINKER
      HeaderFileTypeBundle              = 0x8u, // MH_BUNDLE
      HeaderFileTypeDynamicShlibStub    = 0x9u, // MH_DYLIB_STUB
      HeaderFileTypeDSYM                = 0xAu, // MH_DSYM
      HeaderFileTypeKextBundle          = 0xBu, // MH_KEXT_BUNDLE

      // Constant bits for the "flags" field in llvm::MachO::mach_header and
      // llvm::MachO::mach_header_64
      HeaderFlagBitNoUndefinedSymbols     = 0x00000001u, // MH_NOUNDEFS
      HeaderFlagBitIsIncrementalLinkObject= 0x00000002u, // MH_INCRLINK
      HeaderFlagBitIsDynamicLinkObject    = 0x00000004u, // MH_DYLDLINK
      HeaderFlagBitBindAtLoad             = 0x00000008u, // MH_BINDATLOAD
      HeaderFlagBitPrebound               = 0x00000010u, // MH_PREBOUND
      HeaderFlagBitSplitSegments          = 0x00000020u, // MH_SPLIT_SEGS
      HeaderFlagBitLazyInit               = 0x00000040u, // MH_LAZY_INIT
      HeaderFlagBitTwoLevelNamespace      = 0x00000080u, // MH_TWOLEVEL
      HeaderFlagBitForceFlatNamespace     = 0x00000100u, // MH_FORCE_FLAT
      HeaderFlagBitNoMultipleDefintions   = 0x00000200u, // MH_NOMULTIDEFS
      HeaderFlagBitNoFixPrebinding        = 0x00000400u, // MH_NOFIXPREBINDING
      HeaderFlagBitPrebindable            = 0x00000800u, // MH_PREBINDABLE
      HeaderFlagBitAllModulesBound        = 0x00001000u, // MH_ALLMODSBOUND
      HeaderFlagBitSubsectionsViaSymbols  = 0x00002000u, // MH_SUBSECTIONS_VIA_SYMBOLS
      HeaderFlagBitCanonical              = 0x00004000u, // MH_CANONICAL
      HeaderFlagBitWeakDefines            = 0x00008000u, // MH_WEAK_DEFINES
      HeaderFlagBitBindsToWeak            = 0x00010000u, // MH_BINDS_TO_WEAK
      HeaderFlagBitAllowStackExecution    = 0x00020000u, // MH_ALLOW_STACK_EXECUTION
      HeaderFlagBitRootSafe               = 0x00040000u, // MH_ROOT_SAFE
      HeaderFlagBitSetUIDSafe             = 0x00080000u, // MH_SETUID_SAFE
      HeaderFlagBitNoReexportedDylibs     = 0x00100000u, // MH_NO_REEXPORTED_DYLIBS
      HeaderFlagBitPIE                    = 0x00200000u, // MH_PIE
      HeaderFlagBitDeadStrippableDylib    = 0x00400000u, // MH_DEAD_STRIPPABLE_DYLIB

      // Constants for the "cmd" field in llvm::MachO::load_command
      LoadCommandDynamicLinkerRequired    = 0x80000000u, // LC_REQ_DYLD
      LoadCommandSegment32                = 0x00000001u, // LC_SEGMENT
      LoadCommandSymtab                   = 0x00000002u, // LC_SYMTAB
      LoadCommandSymSeg                   = 0x00000003u, // LC_SYMSEG
      LoadCommandThread                   = 0x00000004u, // LC_THREAD
      LoadCommandUnixThread               = 0x00000005u, // LC_UNIXTHREAD
      LoadCommandFixedVMShlibLoad         = 0x00000006u, // LC_LOADFVMLIB
      LoadCommandFixedVMShlibIdent        = 0x00000007u, // LC_IDFVMLIB
      LoadCommandIdent                    = 0x00000008u, // LC_IDENT
      LoadCommandFixedVMFileInclusion     = 0x00000009u, // LC_FVMFILE
      LoadCommandPrePage                  = 0x0000000Au, // LC_PREPAGE
      LoadCommandDynamicSymtabInfo        = 0x0000000Bu, // LC_DYSYMTAB
      LoadCommandDylibLoad                = 0x0000000Cu, // LC_LOAD_DYLIB
      LoadCommandDylibIdent               = 0x0000000Du, // LC_ID_DYLIB
      LoadCommandDynamicLinkerLoad        = 0x0000000Eu, // LC_LOAD_DYLINKER
      LoadCommandDynamicLinkerIdent       = 0x0000000Fu, // LC_ID_DYLINKER
      LoadCommandDylibPrebound            = 0x00000010u, // LC_PREBOUND_DYLIB
      LoadCommandRoutines32               = 0x00000011u, // LC_ROUTINES
      LoadCommandSubFramework             = 0x00000012u, // LC_SUB_FRAMEWORK
      LoadCommandSubUmbrella              = 0x00000013u, // LC_SUB_UMBRELLA
      LoadCommandSubClient                = 0x00000014u, // LC_SUB_CLIENT
      LoadCommandSubLibrary               = 0x00000015u, // LC_SUB_LIBRARY
      LoadCommandTwoLevelHints            = 0x00000016u, // LC_TWOLEVEL_HINTS
      LoadCommandPreBindChecksum          = 0x00000017u, // LC_PREBIND_CKSUM
      LoadCommandDylibLoadWeak            = 0x80000018u, // LC_LOAD_WEAK_DYLIB
      LoadCommandSegment64                = 0x00000019u, // LC_SEGMENT_64
      LoadCommandRoutines64               = 0x0000001Au, // LC_ROUTINES_64
      LoadCommandUUID                     = 0x0000001Bu, // LC_UUID
      LoadCommandRunpath                  = 0x8000001Cu, // LC_RPATH
      LoadCommandCodeSignature            = 0x0000001Du, // LC_CODE_SIGNATURE
      LoadCommandSegmentSplitInfo         = 0x0000001Eu, // LC_SEGMENT_SPLIT_INFO
      LoadCommandDylibReexport            = 0x8000001Fu, // LC_REEXPORT_DYLIB
      LoadCommandDylibLazyLoad            = 0x00000020u, // LC_LAZY_LOAD_DYLIB
      LoadCommandEncryptionInfo           = 0x00000021u, // LC_ENCRYPTION_INFO
      LoadCommandDynamicLinkerInfo        = 0x00000022u, // LC_DYLD_INFO
      LoadCommandDynamicLinkerInfoOnly    = 0x80000022u, // LC_DYLD_INFO_ONLY
      LoadCommandDylibLoadUpward          = 0x80000023u, // LC_LOAD_UPWARD_DYLIB
      LoadCommandVersionMinMacOSX         = 0x00000024u, // LC_VERSION_MIN_MACOSX
      LoadCommandVersionMinIPhoneOS       = 0x00000025u, // LC_VERSION_MIN_IPHONEOS
      LoadCommandFunctionStarts           = 0x00000026u, // LC_FUNCTION_STARTS
      LoadCommandDyldEnvironment          = 0x00000027u, // LC_DYLD_ENVIRONMENT
      LoadCommandMain                     = 0x80000028u, // LC_MAIN
      LoadCommandDataInCode               = 0x00000029u, // LC_DATA_IN_CODE
      LoadCommandSourceVersion            = 0x0000002Au, // LC_SOURCE_VERSION
      LoadCommandCodeSignDRs              = 0x0000002Bu, // LC_DYLIB_CODE_SIGN_DRS

      // Constant bits for the "flags" field in llvm::MachO::segment_command
      SegmentCommandFlagBitHighVM             = 0x1u, // SG_HIGHVM
      SegmentCommandFlagBitFixedVMLibrary     = 0x2u, // SG_FVMLIB
      SegmentCommandFlagBitNoRelocations      = 0x4u, // SG_NORELOC
      SegmentCommandFlagBitProtectedVersion1  = 0x8u, // SG_PROTECTED_VERSION_1


      // Constant masks for the "flags" field in llvm::MachO::section and
      // llvm::MachO::section_64
      SectionFlagMaskSectionType      = 0x000000ffu, // SECTION_TYPE
      SectionFlagMaskAllAttributes    = 0xffffff00u, // SECTION_ATTRIBUTES
      SectionFlagMaskUserAttributes   = 0xff000000u, // SECTION_ATTRIBUTES_USR
      SectionFlagMaskSystemAttributes = 0x00ffff00u, // SECTION_ATTRIBUTES_SYS

      // Constant masks for the "flags[7:0]" field in llvm::MachO::section and
      // llvm::MachO::section_64 (mask "flags" with SECTION_TYPE)
      SectionTypeRegular                    = 0x00u, // S_REGULAR
      SectionTypeZeroFill                   = 0x01u, // S_ZEROFILL
      SectionTypeCStringLiterals            = 0x02u, // S_CSTRING_LITERALS
      SectionType4ByteLiterals              = 0x03u, // S_4BYTE_LITERALS
      SectionType8ByteLiterals              = 0x04u, // S_8BYTE_LITERALS
      SectionTypeLiteralPointers            = 0x05u, // S_LITERAL_POINTERS
      SectionTypeNonLazySymbolPointers      = 0x06u, // S_NON_LAZY_SYMBOL_POINTERS
      SectionTypeLazySymbolPointers         = 0x07u, // S_LAZY_SYMBOL_POINTERS
      SectionTypeSymbolStubs                = 0x08u, // S_SYMBOL_STUBS
      SectionTypeModuleInitFunctionPointers = 0x09u, // S_MOD_INIT_FUNC_POINTERS
      SectionTypeModuleTermFunctionPointers = 0x0au, // S_MOD_TERM_FUNC_POINTERS
      SectionTypeCoalesced                  = 0x0bu, // S_COALESCED
      SectionTypeZeroFillLarge              = 0x0cu, // S_GB_ZEROFILL
      SectionTypeInterposing                = 0x0du, // S_INTERPOSING
      SectionType16ByteLiterals             = 0x0eu, // S_16BYTE_LITERALS
      SectionTypeDTraceObjectFormat         = 0x0fu, // S_DTRACE_DOF
      SectionTypeLazyDylibSymbolPointers    = 0x10u, // S_LAZY_DYLIB_SYMBOL_POINTERS

      // Constant masks for the "flags[31:24]" field in llvm::MachO::section and
      // llvm::MachO::section_64 (mask "flags" with SECTION_ATTRIBUTES_USR)
      SectionAttrUserPureInstructions       = 0x80000000u, // S_ATTR_PURE_INSTRUCTIONS
      SectionAttrUserNoTableOfContents      = 0x40000000u, // S_ATTR_NO_TOC
      SectionAttrUserCanStripStaticSymbols  = 0x20000000u, // S_ATTR_STRIP_STATIC_SYMS
      SectionAttrUserNoDeadStrip            = 0x10000000u, // S_ATTR_NO_DEAD_STRIP
      SectionAttrUserLiveSupport            = 0x08000000u, // S_ATTR_LIVE_SUPPORT
      SectionAttrUserSelfModifyingCode      = 0x04000000u, // S_ATTR_SELF_MODIFYING_CODE
      SectionAttrUserDebug                  = 0x02000000u, // S_ATTR_DEBUG

      // Constant masks for the "flags[23:8]" field in llvm::MachO::section and
      // llvm::MachO::section_64 (mask "flags" with SECTION_ATTRIBUTES_SYS)
      SectionAttrSytemSomeInstructions      = 0x00000400u, // S_ATTR_SOME_INSTRUCTIONS
      SectionAttrSytemHasExternalRelocations= 0x00000200u, // S_ATTR_EXT_RELOC
      SectionAttrSytemHasLocalRelocations   = 0x00000100u, // S_ATTR_LOC_RELOC

      IndirectSymbolLocal                   = 0x80000000u, // INDIRECT_SYMBOL_LOCAL
      IndirectSymbolAbsolute                = 0x40000000u, // INDIRECT_SYMBOL_ABS

      RebaseTypePointer                     = 1u, // REBASE_TYPE_POINTER
      RebaseTypeTextAbsolute32              = 2u, // REBASE_TYPE_TEXT_ABSOLUTE32
      RebaseTypeTextPCRelative32            = 3u, // REBASE_TYPE_TEXT_PCREL32

      RebaseOpcodeMask                          = 0xF0u, // REBASE_OPCODE_MASK
      RebaseImmediateMask                       = 0x0Fu, // REBASE_IMMEDIATE_MASK
      RebaseOpcodeDone                          = 0x00u, // REBASE_OPCODE_DONE
      RebaseOpcodeSetTypeImmediate              = 0x10u, // REBASE_OPCODE_SET_TYPE_IMM
      RebaseOpcodeSetSegmentAndOffsetULEB       = 0x20u, // REBASE_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB
      RebaseOpcodeAddAddressULEB                = 0x30u, // REBASE_OPCODE_ADD_ADDR_ULEB
      RebaseOpcodeAddAddressImmediateScaled     = 0x40u, // REBASE_OPCODE_ADD_ADDR_IMM_SCALED
      RebaseOpcodeDoRebaseImmediateTimes        = 0x50u, // REBASE_OPCODE_DO_REBASE_IMM_TIMES
      RebaseOpcodeDoRebaseULEBTimes             = 0x60u, // REBASE_OPCODE_DO_REBASE_ULEB_TIMES
      RebaseOpcodeDoRebaseAddAddressULEB        = 0x70u, // REBASE_OPCODE_DO_REBASE_ADD_ADDR_ULEB
      RebaseOpcodeDoRebaseULEBTimesSkippingULEB = 0x80u, // REBASE_OPCODE_DO_REBASE_ULEB_TIMES_SKIPPING_ULEB


      BindTypePointer           = 1u, // BIND_TYPE_POINTER
      BindTypeTextAbsolute32    = 2u, // BIND_TYPE_TEXT_ABSOLUTE32
      BindTypeTextPCRelative32  = 3u, // BIND_TYPE_TEXT_PCREL32

      BindSpecialDylibSelf            =  0u, // BIND_SPECIAL_DYLIB_SELF
      BindSpecialDylibMainExecutable  = -1u, // BIND_SPECIAL_DYLIB_MAIN_EXECUTABLE
      BindSpecialDylibFlatLookup      = -2u, // BIND_SPECIAL_DYLIB_FLAT_LOOKUP

      BindSymbolFlagsWeakImport         = 0x1u, // BIND_SYMBOL_FLAGS_WEAK_IMPORT
      BindSymbolFlagsNonWeakDefinition  = 0x8u, // BIND_SYMBOL_FLAGS_NON_WEAK_DEFINITION

      BindOpcodeMask                            = 0xF0u, // BIND_OPCODE_MASK
      BindImmediateMask                         = 0x0Fu, // BIND_IMMEDIATE_MASK
      BindOpcodeDone                            = 0x00u, // BIND_OPCODE_DONE
      BindOpcodeSetDylibOrdinalImmediate        = 0x10u, // BIND_OPCODE_SET_DYLIB_ORDINAL_IMM
      BindOpcodeSetDylibOrdinalULEB             = 0x20u, // BIND_OPCODE_SET_DYLIB_ORDINAL_ULEB
      BindOpcodeSetDylibSpecialImmediate        = 0x30u, // BIND_OPCODE_SET_DYLIB_SPECIAL_IMM
      BindOpcodeSetSymbolTrailingFlagsImmediate = 0x40u, // BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM
      BindOpcodeSetTypeImmediate                = 0x50u, // BIND_OPCODE_SET_TYPE_IMM
      BindOpcodeSetAppendSLEB                   = 0x60u, // BIND_OPCODE_SET_ADDEND_SLEB
      BindOpcodeSetSegmentAndOffsetULEB         = 0x70u, // BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB
      BindOpcodeAddAddressULEB                  = 0x80u, // BIND_OPCODE_ADD_ADDR_ULEB
      BindOpcodeDoBind                          = 0x90u, // BIND_OPCODE_DO_BIND
      BindOpcodeDoBindAddAddressULEB            = 0xA0u, // BIND_OPCODE_DO_BIND_ADD_ADDR_ULEB
      BindOpcodeDoBindAddAddressImmediateScaled = 0xB0u, // BIND_OPCODE_DO_BIND_ADD_ADDR_IMM_SCALED
      BindOpcodeDoBindULEBTimesSkippingULEB     = 0xC0u, // BIND_OPCODE_DO_BIND_ULEB_TIMES_SKIPPING_ULEB

      ExportSymbolFlagsKindMask           = 0x03u, // EXPORT_SYMBOL_FLAGS_KIND_MASK
      ExportSymbolFlagsKindRegular        = 0x00u, // EXPORT_SYMBOL_FLAGS_KIND_REGULAR
      ExportSymbolFlagsKindThreadLocal    = 0x01u, // EXPORT_SYMBOL_FLAGS_KIND_THREAD_LOCAL
      ExportSymbolFlagsWeakDefinition     = 0x04u, // EXPORT_SYMBOL_FLAGS_WEAK_DEFINITION
      ExportSymbolFlagsIndirectDefinition = 0x08u, // EXPORT_SYMBOL_FLAGS_INDIRECT_DEFINITION
      ExportSymbolFlagsHasSpecializations = 0x10u, // EXPORT_SYMBOL_FLAGS_HAS_SPECIALIZATIONS


      // Constant masks for the "n_type" field in llvm::MachO::nlist and
      // llvm::MachO::nlist_64
      NlistMaskStab             = 0xe0, // N_STAB
      NlistMaskPrivateExternal  = 0x10, // N_PEXT
      NlistMaskType             = 0x0e, // N_TYPE
      NlistMaskExternal         = 0x01, // N_EXT

      // Constants for the "n_type & N_TYPE" llvm::MachO::nlist and
      // llvm::MachO::nlist_64
      NListTypeUndefined          = 0x0u, // N_UNDF
      NListTypeAbsolute           = 0x2u, // N_ABS
      NListTypeSection            = 0xeu, // N_SECT
      NListTypePreboundUndefined  = 0xcu, // N_PBUD
      NListTypeIndirect           = 0xau, // N_INDR

      // Constant masks for the "n_sect" field in llvm::MachO::nlist and
      // llvm::MachO::nlist_64
      NListSectionNoSection     = 0u, // NO_SECT
      NListSectionMaxSection    = 0xffu, // MAX_SECT

      NListDescWeakRef          = 0x40u,
      NListDescWeakDef          = 0x80u,

      // Constant values for the "n_type" field in llvm::MachO::nlist and
      // llvm::MachO::nlist_64 when "(n_type & NlistMaskStab) != 0"
      StabGlobalSymbol          = 0x20u,  // N_GSYM
      StabFunctionName          = 0x22u,  // N_FNAME
      StabFunction              = 0x24u,  // N_FUN
      StabStaticSymbol          = 0x26u,  // N_STSYM
      StabLocalCommon           = 0x28u,  // N_LCSYM
      StabBeginSymbol           = 0x2Eu,  // N_BNSYM
      StabSourceFileOptions     = 0x3Cu,  // N_OPT
      StabRegisterSymbol        = 0x40u,  // N_RSYM
      StabSourceLine            = 0x44u,  // N_SLINE
      StabEndSymbol             = 0x4Eu,  // N_ENSYM
      StabStructureType         = 0x60u,  // N_SSYM
      StabSourceFileName        = 0x64u,  // N_SO
      StabObjectFileName        = 0x66u,  // N_OSO
      StabLocalSymbol           = 0x80u,  // N_LSYM
      StabBeginIncludeFileName  = 0x82u,  // N_BINCL
      StabIncludeFileName       = 0x84u,  // N_SOL
      StabCompilerParameters    = 0x86u,  // N_PARAMS
      StabCompilerVersion       = 0x88u,  // N_VERSION
      StabCompilerOptLevel      = 0x8Au,  // N_OLEVEL
      StabParameter             = 0xA0u,  // N_PSYM
      StabEndIncludeFile        = 0xA2u,  // N_EINCL
      StabAlternateEntry        = 0xA4u,  // N_ENTRY
      StabLeftBracket           = 0xC0u,  // N_LBRAC
      StabDeletedIncludeFile    = 0xC2u,  // N_EXCL
      StabRightBracket          = 0xE0u,  // N_RBRAC
      StabBeginCommon           = 0xE2u,  // N_BCOMM
      StabEndCommon             = 0xE4u,  // N_ECOMM
      StabEndCommonLocal        = 0xE8u,  // N_ECOML
      StabLength                = 0xFEu   // N_LENG

    };

    // Structs from <mach-o/loader.h>

    struct mach_header {
      uint32_t magic;
      uint32_t cputype;
      uint32_t cpusubtype;
      uint32_t filetype;
      uint32_t ncmds;
      uint32_t sizeofcmds;
      uint32_t flags;
    };

    struct mach_header_64 {
      uint32_t magic;
      uint32_t cputype;
      uint32_t cpusubtype;
      uint32_t filetype;
      uint32_t ncmds;
      uint32_t sizeofcmds;
      uint32_t flags;
      uint32_t reserved;
    };

    struct load_command {
      uint32_t cmd;
      uint32_t cmdsize;
    };

    struct segment_command {
      uint32_t cmd;
      uint32_t cmdsize;
      char segname[16];
      uint32_t vmaddr;
      uint32_t vmsize;
      uint32_t fileoff;
      uint32_t filesize;
      uint32_t maxprot;
      uint32_t initprot;
      uint32_t nsects;
      uint32_t flags;
    };

    struct segment_command_64 {
      uint32_t cmd;
      uint32_t cmdsize;
      char segname[16];
      uint64_t vmaddr;
      uint64_t vmsize;
      uint64_t fileoff;
      uint64_t filesize;
      uint32_t maxprot;
      uint32_t initprot;
      uint32_t nsects;
      uint32_t flags;
    };

    struct section {
      char sectname[16];
      char segname[16];
      uint32_t addr;
      uint32_t size;
      uint32_t offset;
      uint32_t align;
      uint32_t reloff;
      uint32_t nreloc;
      uint32_t flags;
      uint32_t reserved1;
      uint32_t reserved2;
    };

    struct section_64 {
      char sectname[16];
      char segname[16];
      uint64_t addr;
      uint64_t size;
      uint32_t offset;
      uint32_t align;
      uint32_t reloff;
      uint32_t nreloc;
      uint32_t flags;
      uint32_t reserved1;
      uint32_t reserved2;
      uint32_t reserved3;
    };

    struct fvmlib {
      uint32_t name;
      uint32_t minor_version;
      uint32_t header_addr;
    };

    struct fvmlib_command {
      uint32_t  cmd;
      uint32_t cmdsize;
      struct fvmlib fvmlib;
    };

    struct dylib {
      uint32_t name;
      uint32_t timestamp;
      uint32_t current_version;
      uint32_t compatibility_version;
    };

    struct dylib_command {
      uint32_t cmd;
      uint32_t cmdsize;
      struct dylib dylib;
    };

    struct sub_framework_command {
      uint32_t cmd;
      uint32_t cmdsize;
      uint32_t umbrella;
    };

    struct sub_client_command {
      uint32_t cmd;
      uint32_t cmdsize;
      uint32_t client;
    };

    struct sub_umbrella_command {
      uint32_t cmd;
      uint32_t cmdsize;
      uint32_t sub_umbrella;
    };

    struct sub_library_command {
      uint32_t cmd;
      uint32_t cmdsize;
      uint32_t sub_library;
    };

    struct prebound_dylib_command {
      uint32_t cmd;
      uint32_t cmdsize;
      uint32_t name;
      uint32_t nmodules;
      uint32_t linked_modules;
    };

    struct dylinker_command {
      uint32_t cmd;
      uint32_t cmdsize;
      uint32_t name;
    };

    struct thread_command {
      uint32_t cmd;
      uint32_t cmdsize;
    };

    struct routines_command {
      uint32_t cmd;
      uint32_t cmdsize;
      uint32_t init_address;
      uint32_t init_module;
      uint32_t reserved1;
      uint32_t reserved2;
      uint32_t reserved3;
      uint32_t reserved4;
      uint32_t reserved5;
      uint32_t reserved6;
    };

    struct routines_command_64 {
      uint32_t cmd;
      uint32_t cmdsize;
      uint64_t init_address;
      uint64_t init_module;
      uint64_t reserved1;
      uint64_t reserved2;
      uint64_t reserved3;
      uint64_t reserved4;
      uint64_t reserved5;
      uint64_t reserved6;
    };

    struct symtab_command {
      uint32_t cmd;
      uint32_t cmdsize;
      uint32_t symoff;
      uint32_t nsyms;
      uint32_t stroff;
      uint32_t strsize;
    };

    struct dysymtab_command {
      uint32_t cmd;
      uint32_t cmdsize;
      uint32_t ilocalsym;
      uint32_t nlocalsym;
      uint32_t iextdefsym;
      uint32_t nextdefsym;
      uint32_t iundefsym;
      uint32_t nundefsym;
      uint32_t tocoff;
      uint32_t ntoc;
      uint32_t modtaboff;
      uint32_t nmodtab;
      uint32_t extrefsymoff;
      uint32_t nextrefsyms;
      uint32_t indirectsymoff;
      uint32_t nindirectsyms;
      uint32_t extreloff;
      uint32_t nextrel;
      uint32_t locreloff;
      uint32_t nlocrel;
    };

    struct dylib_table_of_contents {
      uint32_t symbol_index;
      uint32_t module_index;
    };

    struct dylib_module {
      uint32_t module_name;
      uint32_t iextdefsym;
      uint32_t nextdefsym;
      uint32_t irefsym;
      uint32_t nrefsym;
      uint32_t ilocalsym;
      uint32_t nlocalsym;
      uint32_t iextrel;
      uint32_t nextrel;
      uint32_t iinit_iterm;
      uint32_t ninit_nterm;
      uint32_t objc_module_info_addr;
      uint32_t objc_module_info_size;
    };

    struct dylib_module_64 {
      uint32_t module_name;
      uint32_t iextdefsym;
      uint32_t nextdefsym;
      uint32_t irefsym;
      uint32_t nrefsym;
      uint32_t ilocalsym;
      uint32_t nlocalsym;
      uint32_t iextrel;
      uint32_t nextrel;
      uint32_t iinit_iterm;
      uint32_t ninit_nterm;
      uint32_t objc_module_info_size;
      uint64_t objc_module_info_addr;
    };

    struct dylib_reference {
      uint32_t isym:24,
               flags:8;
    };


    struct twolevel_hints_command {
      uint32_t cmd;
      uint32_t cmdsize;
      uint32_t offset;
      uint32_t nhints;
    };

    struct twolevel_hint {
      uint32_t isub_image:8,
               itoc:24;
    };

    struct prebind_cksum_command {
      uint32_t cmd;
      uint32_t cmdsize;
      uint32_t cksum;
    };

    struct uuid_command {
      uint32_t cmd;
      uint32_t cmdsize;
      uint8_t uuid[16];
    };

    struct rpath_command {
      uint32_t cmd;
      uint32_t cmdsize;
      uint32_t path;
    };

    struct linkedit_data_command {
      uint32_t cmd;
      uint32_t cmdsize;
      uint32_t dataoff;
      uint32_t datasize;
    };

    struct encryption_info_command {
      uint32_t cmd;
      uint32_t cmdsize;
      uint32_t cryptoff;
      uint32_t cryptsize;
      uint32_t cryptid;
    };

    struct version_min_command {
      uint32_t cmd;
      uint32_t cmdsize;
      uint32_t version;
      uint32_t reserved;
    };

    struct dyld_info_command {
      uint32_t cmd;
      uint32_t cmdsize;
      uint32_t rebase_off;
      uint32_t rebase_size;
      uint32_t bind_off;
      uint32_t bind_size;
      uint32_t weak_bind_off;
      uint32_t weak_bind_size;
      uint32_t lazy_bind_off;
      uint32_t lazy_bind_size;
      uint32_t export_off;
      uint32_t export_size;
    };

    struct symseg_command {
      uint32_t cmd;
      uint32_t cmdsize;
      uint32_t offset;
      uint32_t size;
    };

    struct ident_command {
      uint32_t cmd;
      uint32_t cmdsize;
    };

    struct fvmfile_command {
      uint32_t cmd;
      uint32_t cmdsize;
      uint32_t name;
      uint32_t header_addr;
    };


    // Structs from <mach-o/fat.h>
    struct fat_header {
      uint32_t magic;
      uint32_t nfat_arch;
    };

    struct fat_arch {
      uint32_t cputype;
      uint32_t cpusubtype;
      uint32_t offset;
      uint32_t size;
      uint32_t align;
    };

    // Structs from <mach-o/fat.h>
    struct nlist {
      uint32_t n_strx;
      uint8_t n_type;
      uint8_t n_sect;
      int16_t n_desc;
      uint32_t n_value;
    };

    struct nlist_64 {
      uint32_t n_strx;
      uint8_t n_type;
      uint8_t n_sect;
      uint16_t n_desc;
      uint64_t n_value;
    };

    // Get/Set functions from <mach-o/nlist.h>

    static inline uint16_t GET_LIBRARY_ORDINAL(uint16_t n_desc)
    {
      return (((n_desc) >> 8u) & 0xffu);
    }

    static inline void SET_LIBRARY_ORDINAL(uint16_t &n_desc, uint8_t ordinal)
    {
      n_desc = (((n_desc) & 0x00ff) | (((ordinal) & 0xff) << 8));
    }

    static inline uint8_t GET_COMM_ALIGN (uint16_t n_desc)
    {
      return (n_desc >> 8u) & 0x0fu;
    }

    static inline void SET_COMM_ALIGN (uint16_t &n_desc, uint8_t align)
    {
      n_desc = ((n_desc & 0xf0ffu) | ((align & 0x0fu) << 8u));
    }

    // Enums from <mach/machine.h>
    enum {
      // Capability bits used in the definition of cpu_type.
      CPUArchMask = 0xff000000,   // Mask for architecture bits
      CPUArchABI64 = 0x01000000,  // 64 bit ABI

      // Constants for the cputype field.
      CPUTypeI386      = 7,
      CPUTypeX86_64    = CPUTypeI386 | CPUArchABI64,
      CPUTypeARM       = 12,
      CPUTypeSPARC     = 14,
      CPUTypePowerPC   = 18,
      CPUTypePowerPC64 = CPUTypePowerPC | CPUArchABI64,


      // Constants for the cpusubtype field.

      // X86
      CPUSubType_I386_ALL    = 3,
      CPUSubType_X86_64_ALL  = 3,

      // ARM
      CPUSubType_ARM_ALL     = 0,
      CPUSubType_ARM_V4T     = 5,
      CPUSubType_ARM_V5      = 7,
      CPUSubType_ARM_V6      = 6,
      CPUSubType_ARM_V7      = 9,

      // PowerPC
      CPUSubType_POWERPC_ALL = 0,

      CPUSubType_SPARC_ALL   = 0
    };
  } // end namespace MachO
} // end namespace llvm

#endif
