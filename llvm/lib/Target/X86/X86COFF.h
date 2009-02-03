//===--- X86COFF.h - Some definitions from COFF documentations ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file just defines some symbols found in COFF documentation. They are
// used to emit function type information for COFF targets (Cygwin/Mingw32).
//
//===----------------------------------------------------------------------===//

#ifndef X86COFF_H
#define X86COFF_H

namespace COFF 
{
/// Storage class tells where and what the symbol represents
enum StorageClass {
  C_EFCN =   -1,  ///< Physical end of function
  C_NULL    = 0,  ///< No symbol
  C_AUTO    = 1,  ///< External definition
  C_EXT     = 2,  ///< External symbol
  C_STAT    = 3,  ///< Static
  C_REG     = 4,  ///< Register variable
  C_EXTDEF  = 5,  ///< External definition
  C_LABEL   = 6,  ///< Label
  C_ULABEL  = 7,  ///< Undefined label
  C_MOS     = 8,  ///< Member of structure
  C_ARG     = 9,  ///< Function argument
  C_STRTAG  = 10, ///< Structure tag
  C_MOU     = 11, ///< Member of union
  C_UNTAG   = 12, ///< Union tag
  C_TPDEF   = 13, ///< Type definition
  C_USTATIC = 14, ///< Undefined static
  C_ENTAG   = 15, ///< Enumeration tag
  C_MOE     = 16, ///< Member of enumeration
  C_REGPARM = 17, ///< Register parameter
  C_FIELD   = 18, ///< Bit field

  C_BLOCK  = 100, ///< ".bb" or ".eb" - beginning or end of block
  C_FCN    = 101, ///< ".bf" or ".ef" - beginning or end of function
  C_EOS    = 102, ///< End of structure
  C_FILE   = 103, ///< File name
  C_LINE   = 104, ///< Line number, reformatted as symbol
  C_ALIAS  = 105, ///< Duplicate tag
  C_HIDDEN = 106  ///< External symbol in dmert public lib
};

/// The type of the symbol. This is made up of a base type and a derived type.
/// For example, pointer to int is "pointer to T" and "int"
enum SymbolType {
  T_NULL   = 0,  ///< No type info
  T_ARG    = 1,  ///< Void function argument (only used by compiler)
  T_VOID   = 1,  ///< The same as above. Just named differently in some specs.
  T_CHAR   = 2,  ///< Character
  T_SHORT  = 3,  ///< Short integer
  T_INT    = 4,  ///< Integer
  T_LONG   = 5,  ///< Long integer
  T_FLOAT  = 6,  ///< Floating point
  T_DOUBLE = 7,  ///< Double word
  T_STRUCT = 8,  ///< Structure
  T_UNION  = 9,  ///< Union
  T_ENUM   = 10, ///< Enumeration
  T_MOE    = 11, ///< Member of enumeration
  T_UCHAR  = 12, ///< Unsigned character
  T_USHORT = 13, ///< Unsigned short
  T_UINT   = 14, ///< Unsigned integer
  T_ULONG  = 15  ///< Unsigned long
};

/// Derived type of symbol
enum SymbolDerivedType {
  DT_NON = 0, ///< No derived type
  DT_PTR = 1, ///< Pointer to T
  DT_FCN = 2, ///< Function returning T
  DT_ARY = 3  ///< Array of T
};

/// Masks for extracting parts of type
enum SymbolTypeMasks {
  N_BTMASK = 017, ///< Mask for base type
  N_TMASK  = 060  ///< Mask for derived type
};

/// Offsets of parts of type
enum Shifts {
  N_BTSHFT = 4 ///< Type is formed as (base + derived << N_BTSHIFT)
};

}

#endif // X86COFF_H
