//===-- llvm/BinaryFormat/XCOFF.h - The XCOFF file format -------*- C++/-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines manifest constants for the XCOFF object file format.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BINARYFORMAT_XCOFF_H
#define LLVM_BINARYFORMAT_XCOFF_H

#include "llvm/ADT/StringRef.h"
#include <cstdint>

namespace llvm {
namespace XCOFF {

// Constants used in the XCOFF definition.
enum { FileNamePadSize = 6, NameSize = 8, SymbolTableEntrySize = 18 };

enum ReservedSectionNum { N_DEBUG = -2, N_ABS = -1, N_UNDEF = 0 };

// x_smclas field of x_csect from system header: /usr/include/syms.h
/// Storage Mapping Class definitions.
enum StorageMappingClass : uint8_t {
  //     READ ONLY CLASSES
  XMC_PR = 0,      ///< Program Code
  XMC_RO = 1,      ///< Read Only Constant
  XMC_DB = 2,      ///< Debug Dictionary Table
  XMC_GL = 6,      ///< Global Linkage (Interfile Interface Code)
  XMC_XO = 7,      ///< Extended Operation (Pseudo Machine Instruction)
  XMC_SV = 8,      ///< Supervisor Call (32-bit process only)
  XMC_SV64 = 17,   ///< Supervisor Call for 64-bit process
  XMC_SV3264 = 18, ///< Supervisor Call for both 32- and 64-bit processes
  XMC_TI = 12,     ///< Traceback Index csect
  XMC_TB = 13,     ///< Traceback Table csect

  //       READ WRITE CLASSES
  XMC_RW = 5,   ///< Read Write Data
  XMC_TC0 = 15, ///< TOC Anchor for TOC Addressability
  XMC_TC = 3,   ///< General TOC item
  XMC_TD = 16,  ///< Scalar data item in the TOC
  XMC_DS = 10,  ///< Descriptor csect
  XMC_UA = 4,   ///< Unclassified - Treated as Read Write
  XMC_BS = 9,   ///< BSS class (uninitialized static internal)
  XMC_UC = 11,  ///< Un-named Fortran Common

  XMC_TL = 20, ///< Initialized thread-local variable
  XMC_UL = 21, ///< Uninitialized thread-local variable
  XMC_TE = 22  ///< Symbol mapped at the end of TOC
};

// Flags for defining the section type. Used for the s_flags field of
// the section header structure. Defined in the system header `scnhdr.h`.
enum SectionTypeFlags {
  STYP_PAD = 0x0008,
  STYP_DWARF = 0x0010,
  STYP_TEXT = 0x0020,
  STYP_DATA = 0x0040,
  STYP_BSS = 0x0080,
  STYP_EXCEPT = 0x0100,
  STYP_INFO = 0x0200,
  STYP_TDATA = 0x0400,
  STYP_TBSS = 0x0800,
  STYP_LOADER = 0x1000,
  STYP_DEBUG = 0x2000,
  STYP_TYPCHK = 0x4000,
  STYP_OVRFLO = 0x8000
};

// STORAGE CLASSES, n_sclass field of syment.
// The values come from `storclass.h` and `dbxstclass.h`.
enum StorageClass : uint8_t {
  // Storage classes used for symbolic debugging symbols.
  C_FILE = 103,  // File name
  C_BINCL = 108, // Beginning of include file
  C_EINCL = 109, // Ending of include file
  C_GSYM = 128,  // Global variable
  C_STSYM = 133, // Statically allocated symbol
  C_BCOMM = 135, // Beginning of common block
  C_ECOMM = 137, // End of common block
  C_ENTRY = 141, // Alternate entry
  C_BSTAT = 143, // Beginning of static block
  C_ESTAT = 144, // End of static block
  C_GTLS = 145,  // Global thread-local variable
  C_STTLS = 146, // Static thread-local variable

  // Storage classes used for DWARF symbols.
  C_DWARF = 112, // DWARF section symbol

  // Storage classes used for absolute symbols.
  C_LSYM = 129,  // Automatic variable allocated on stack
  C_PSYM = 130,  // Argument to subroutine allocated on stack
  C_RSYM = 131,  // Register variable
  C_RPSYM = 132, // Argument to function or procedure stored in register
  C_ECOML = 136, // Local member of common block
  C_FUN = 142,   // Function or procedure

  // Storage classes used for undefined external symbols or
  // symbols of general sections.
  C_EXT = 2,       // External symbol
  C_WEAKEXT = 111, // Weak external symbol

  // Storage classes used for symbols of general sections.
  C_NULL = 0,
  C_STAT = 3,     // Static
  C_BLOCK = 100,  // ".bb" or ".eb"
  C_FCN = 101,    // ".bf" or ".ef"
  C_HIDEXT = 107, // Un-named external symbol
  C_INFO = 110,   // Comment string in .info section
  C_DECL = 140,   // Declaration of object (type)

  // Storage classes - Obsolete/Undocumented.
  C_AUTO = 1,     // Automatic variable
  C_REG = 4,      // Register variable
  C_EXTDEF = 5,   // External definition
  C_LABEL = 6,    // Label
  C_ULABEL = 7,   // Undefined label
  C_MOS = 8,      // Member of structure
  C_ARG = 9,      // Function argument
  C_STRTAG = 10,  // Structure tag
  C_MOU = 11,     // Member of union
  C_UNTAG = 12,   // Union tag
  C_TPDEF = 13,   // Type definition
  C_USTATIC = 14, // Undefined static
  C_ENTAG = 15,   // Enumeration tag
  C_MOE = 16,     // Member of enumeration
  C_REGPARM = 17, // Register parameter
  C_FIELD = 18,   // Bit field
  C_EOS = 102,    // End of structure
  C_LINE = 104,
  C_ALIAS = 105,  // Duplicate tag
  C_HIDDEN = 106, // Special storage class for external
  C_EFCN = 255,   // Physical end of function

  // Storage classes - reserved
  C_TCSYM = 134 // Reserved
};

enum SymbolType {
  XTY_ER = 0, ///< External reference.
  XTY_SD = 1, ///< Csect definition for initialized storage.
  XTY_LD = 2, ///< Label definition.
              ///< Defines an entry point to an initialized csect.
  XTY_CM = 3  ///< Common csect definition. For uninitialized storage.
};

// Relocation types, defined in `/usr/include/reloc.h`.
enum RelocationType : uint8_t {
  R_POS = 0x00, ///< Positive relocation. Provides the address of the referenced
                ///< symbol.
  R_RL = 0x0c,  ///< Positive indirect load relocation. Modifiable instruction.
  R_RLA = 0x0d, ///< Positive load address relocation. Modifiable instruction.

  R_NEG = 0x01, ///< Negative relocation. Provides the negative of the address
                ///< of the referenced symbol.
  R_REL = 0x02, ///< Relative to self relocation. Provides a displacement value
                ///< between the address of the referenced symbol and the
                ///< address being relocated.

  R_TOC = 0x03, ///< Relative to the TOC relocation. Provides a displacement
                ///< that is the difference between the address of the
                ///< referenced symbol and the TOC anchor csect.
  R_TRL = 0x12, ///< TOC relative indirect load relocation. Similar to R_TOC,
                ///< but not modifiable instruction.

  R_TRLA =
      0x13, ///< Relative to the TOC or to the thread-local storage base
            ///< relocation. Compilers are not permitted to generate this
            ///< relocation type. It is the result of a reversible
            ///< transformation by the linker of an R_TOC relation that turned a
            ///< load instruction into an add-immediate instruction.

  R_GL = 0x05, ///< Global linkage-external TOC address relocation. Provides the
               ///< address of the external TOC associated with a defined
               ///< external symbol.
  R_TCL = 0x06, ///< Local object TOC address relocation. Provides the address
                ///< of the local TOC entry of a defined external symbol.

  R_REF = 0x0f, ///< A non-relocating relocation. Used to prevent the binder
                ///< from garbage collecting a csect (such as code used for
                ///< dynamic initialization of non-local statics) for which
                ///< another csect has an implicit dependency.

  R_BA = 0x08, ///< Branch absolute relocation. Provides the address of the
               ///< referenced symbol. References a non-modifiable instruction.
  R_BR = 0x0a, ///< Branch relative to self relocation. Provides the
               ///< displacement that is the difference between the address of
               ///< the referenced symbol and the address of the referenced
               ///< branch instruction. References a non-modifiable instruction.
  R_RBA = 0x18, ///< Branch absolute relocation. Similar to R_BA but
                ///< references a modifiable instruction.
  R_RBR = 0x1a, ///< Branch relative to self relocation. Similar to the R_BR
                ///< relocation type, but references a modifiable instruction.

  R_TLS = 0x20,    ///< General-dynamic reference to TLS symbol.
  R_TLS_IE = 0x21, ///< Initial-exec reference to TLS symbol.
  R_TLS_LD = 0x22, ///< Local-dynamic reference to TLS symbol.
  R_TLS_LE = 0x23, ///< Local-exec reference to TLS symbol.
  R_TLSM = 0x24,  ///< Module reference to TLS. Provides a handle for the module
                  ///< containing the referenced symbol.
  R_TLSML = 0x25, ///< Module reference to the local TLS storage.

  R_TOCU = 0x30, ///< Relative to TOC upper. Specifies the high-order 16 bits of
                 ///< a large code model TOC-relative relocation.
  R_TOCL = 0x31 ///< Relative to TOC lower. Specifies the low-order 16 bits of a
                ///< large code model TOC-relative relocation.
};

struct FileHeader32 {
  uint16_t Magic;
  uint16_t NumberOfSections;
  int32_t TimeStamp;
  uint32_t SymbolTableFileOffset;
  int32_t NumberOfSymbolTableEntries;
  uint16_t AuxiliaryHeaderSize;
  uint16_t Flags;
};

struct SectionHeader32 {
  char Name[XCOFF::NameSize];
  uint32_t PhysicalAddress;
  uint32_t VirtualAddress;
  uint32_t Size;
  uint32_t FileOffsetToData;
  uint32_t FileOffsetToRelocations;
  uint32_t FileOffsetToLineNumbers;
  uint16_t NumberOfRelocations;
  uint16_t NumberOfLineNumbers;
  int32_t Flags;
};

enum CFileStringType : uint8_t {
  XFT_FN = 0,  ///< Specifies the source-file name.
  XFT_CT = 1,  ///< Specifies the compiler time stamp.
  XFT_CV = 2,  ///< Specifies the compiler version number.
  XFT_CD = 128 ///< Specifies compiler-defined information.
};

enum CFileLangId : uint8_t {
  TB_C = 0,        ///< C language.
  TB_CPLUSPLUS = 9 ///< C++ language.
};

enum CFileCpuId : uint8_t {
  TCPU_PPC64 = 2, ///< PowerPC common architecture 64-bit mode.
  TCPU_COM = 3,   ///< POWER and PowerPC architecture common.
  TCPU_970 = 19   ///< PPC970 - PowerPC 64-bit architecture.
};

StringRef getMappingClassString(XCOFF::StorageMappingClass SMC);

} // end namespace XCOFF
} // end namespace llvm

#endif
