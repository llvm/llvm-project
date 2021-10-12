//===-- BTF.h --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the layout of .BTF and .BTF.ext ELF sections.
///
/// The binary layout for .BTF section:
///   struct Header
///   Type and Str subsections
/// The Type subsection is a collection of types with type id starting with 1.
/// The Str subsection is simply a collection of strings.
///
/// The binary layout for .BTF.ext section:
///   struct ExtHeader
///   FuncInfo, LineInfo, FieldReloc and ExternReloc subsections
/// The FuncInfo subsection is defined as below:
///   BTFFuncInfo Size
///   struct SecFuncInfo for ELF section #1
///   A number of struct BPFFuncInfo for ELF section #1
///   struct SecFuncInfo for ELF section #2
///   A number of struct BPFFuncInfo for ELF section #2
///   ...
/// The LineInfo subsection is defined as below:
///   BPFLineInfo Size
///   struct SecLineInfo for ELF section #1
///   A number of struct BPFLineInfo for ELF section #1
///   struct SecLineInfo for ELF section #2
///   A number of struct BPFLineInfo for ELF section #2
///   ...
/// The FieldReloc subsection is defined as below:
///   BPFFieldReloc Size
///   struct SecFieldReloc for ELF section #1
///   A number of struct BPFFieldReloc for ELF section #1
///   struct SecFieldReloc for ELF section #2
///   A number of struct BPFFieldReloc for ELF section #2
///   ...
///
/// The section formats are also defined at
///    https://github.com/torvalds/linux/blob/master/include/uapi/linux/btf.h
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_BPF_BTF_H
#define LLVM_LIB_TARGET_BPF_BTF_H

namespace llvm {
namespace BTF {

enum : uint32_t { MAGIC = 0xeB9F, VERSION = 1 };

/// Sizes in bytes of various things in the BTF format.
enum {
  HeaderSize = 24,
  ExtHeaderSize = 32,
  CommonTypeSize = 12,
  BTFArraySize = 12,
  BTFEnumSize = 8,
  BTFMemberSize = 12,
  BTFParamSize = 8,
  BTFDataSecVarSize = 12,
  SecFuncInfoSize = 8,
  SecLineInfoSize = 8,
  SecFieldRelocSize = 8,
  BPFFuncInfoSize = 8,
  BPFLineInfoSize = 16,
  BPFFieldRelocSize = 16,
};

/// The .BTF section header definition.
struct Header {
  uint16_t Magic;  ///< Magic value
  uint8_t Version; ///< Version number
  uint8_t Flags;   ///< Extra flags
  uint32_t HdrLen; ///< Length of this header

  /// All offsets are in bytes relative to the end of this header.
  uint32_t TypeOff; ///< Offset of type section
  uint32_t TypeLen; ///< Length of type section
  uint32_t StrOff;  ///< Offset of string section
  uint32_t StrLen;  ///< Length of string section
};

enum : uint32_t {
  MAX_VLEN = 0xffff ///< Max # of struct/union/enum members or func args
};

enum TypeKinds : uint8_t {
#define HANDLE_BTF_KIND(ID, NAME) BTF_KIND_##NAME = ID,
#include "BTF.def"
};

/// The BTF common type definition. Different kinds may have
/// additional information after this structure data.
struct CommonType {
  /// Type name offset in the string table.
  uint32_t NameOff;

  /// "Info" bits arrangement:
  /// Bits  0-15: vlen (e.g. # of struct's members)
  /// Bits 16-23: unused
  /// Bits 24-27: kind (e.g. int, ptr, array...etc)
  /// Bits 28-30: unused
  /// Bit     31: kind_flag, currently used by
  ///             struct, union, fwd and decl_tag
  uint32_t Info;

  /// "Size" is used by INT, ENUM, STRUCT and UNION.
  /// "Size" tells the size of the type it is describing.
  ///
  /// "Type" is used by PTR, TYPEDEF, VOLATILE, CONST, RESTRICT,
  /// FUNC, FUNC_PROTO, VAR and DECL_TAG.
  /// "Type" is a type_id referring to another type.
  union {
    uint32_t Size;
    uint32_t Type;
  };
};

// For some specific BTF_KIND, "struct CommonType" is immediately
// followed by extra data.

// BTF_KIND_INT is followed by a u32 and the following
// is the 32 bits arrangement:
// BTF_INT_ENCODING(VAL) : (((VAL) & 0x0f000000) >> 24)
// BTF_INT_OFFSET(VAL) : (((VAL & 0x00ff0000)) >> 16)
// BTF_INT_BITS(VAL) : ((VAL) & 0x000000ff)

/// Attributes stored in the INT_ENCODING.
enum : uint8_t {
  INT_SIGNED = (1 << 0),
  INT_CHAR = (1 << 1),
  INT_BOOL = (1 << 2)
};

/// BTF_KIND_ENUM is followed by multiple "struct BTFEnum".
/// The exact number of btf_enum is stored in the vlen (of the
/// info in "struct CommonType").
struct BTFEnum {
  uint32_t NameOff; ///< Enum name offset in the string table
  int32_t Val;      ///< Enum member value
};

/// BTF_KIND_ARRAY is followed by one "struct BTFArray".
struct BTFArray {
  uint32_t ElemType;  ///< Element type
  uint32_t IndexType; ///< Index type
  uint32_t Nelems;    ///< Number of elements for this array
};

/// BTF_KIND_STRUCT and BTF_KIND_UNION are followed
/// by multiple "struct BTFMember".  The exact number
/// of BTFMember is stored in the vlen (of the info in
/// "struct CommonType").
///
/// If the struct/union contains any bitfield member,
/// the Offset below represents BitOffset (bits 0 - 23)
/// and BitFieldSize(bits 24 - 31) with BitFieldSize = 0
/// for non bitfield members. Otherwise, the Offset
/// represents the BitOffset.
struct BTFMember {
  uint32_t NameOff; ///< Member name offset in the string table
  uint32_t Type;    ///< Member type
  uint32_t Offset;  ///< BitOffset or BitFieldSize+BitOffset
};

/// BTF_KIND_FUNC_PROTO are followed by multiple "struct BTFParam".
/// The exist number of BTFParam is stored in the vlen (of the info
/// in "struct CommonType").
struct BTFParam {
  uint32_t NameOff;
  uint32_t Type;
};

/// BTF_KIND_FUNC can be global, static or extern.
enum : uint8_t {
  FUNC_STATIC = 0,
  FUNC_GLOBAL = 1,
  FUNC_EXTERN = 2,
};

/// Variable scoping information.
enum : uint8_t {
  VAR_STATIC = 0,           ///< Linkage: InternalLinkage
  VAR_GLOBAL_ALLOCATED = 1, ///< Linkage: ExternalLinkage
  VAR_GLOBAL_EXTERNAL = 2,  ///< Linkage: ExternalLinkage
};

/// BTF_KIND_DATASEC are followed by multiple "struct BTFDataSecVar".
/// The exist number of BTFDataSec is stored in the vlen (of the info
/// in "struct CommonType").
struct BTFDataSec {
  uint32_t Type;   ///< A BTF_KIND_VAR type
  uint32_t Offset; ///< In-section offset
  uint32_t Size;   ///< Occupied memory size
};

/// The .BTF.ext section header definition.
struct ExtHeader {
  uint16_t Magic;
  uint8_t Version;
  uint8_t Flags;
  uint32_t HdrLen;

  uint32_t FuncInfoOff;    ///< Offset of func info section
  uint32_t FuncInfoLen;    ///< Length of func info section
  uint32_t LineInfoOff;    ///< Offset of line info section
  uint32_t LineInfoLen;    ///< Length of line info section
  uint32_t FieldRelocOff; ///< Offset of offset reloc section
  uint32_t FieldRelocLen; ///< Length of offset reloc section
};

/// Specifying one function info.
struct BPFFuncInfo {
  uint32_t InsnOffset; ///< Byte offset in the section
  uint32_t TypeId;     ///< Type id referring to .BTF type section
};

/// Specifying function info's in one section.
struct SecFuncInfo {
  uint32_t SecNameOff;  ///< Section name index in the .BTF string table
  uint32_t NumFuncInfo; ///< Number of func info's in this section
};

/// Specifying one line info.
struct BPFLineInfo {
  uint32_t InsnOffset;  ///< Byte offset in this section
  uint32_t FileNameOff; ///< File name index in the .BTF string table
  uint32_t LineOff;     ///< Line index in the .BTF string table
  uint32_t LineCol;     ///< Line num: line_col >> 10,
                        ///  col num: line_col & 0x3ff
};

/// Specifying line info's in one section.
struct SecLineInfo {
  uint32_t SecNameOff;  ///< Section name index in the .BTF string table
  uint32_t NumLineInfo; ///< Number of line info's in this section
};

/// Specifying one offset relocation.
struct BPFFieldReloc {
  uint32_t InsnOffset;    ///< Byte offset in this section
  uint32_t TypeID;        ///< TypeID for the relocation
  uint32_t OffsetNameOff; ///< The string to traverse types
  uint32_t RelocKind;     ///< What to patch the instruction
};

/// Specifying offset relocation's in one section.
struct SecFieldReloc {
  uint32_t SecNameOff;     ///< Section name index in the .BTF string table
  uint32_t NumFieldReloc; ///< Number of offset reloc's in this section
};

} // End namespace BTF.
} // End namespace llvm.

#endif
