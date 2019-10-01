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

#include <cstdint>

namespace llvm {
namespace XCOFF {

// Constants used in the XCOFF definition.
enum { SectionNameSize = 8, SymbolNameSize = 8 };
enum ReservedSectionNum { N_DEBUG = -2, N_ABS = -1, N_UNDEF = 0 };

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

} // end namespace XCOFF
} // end namespace llvm

#endif
