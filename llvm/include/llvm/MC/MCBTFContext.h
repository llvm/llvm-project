//===- MCBTFContext.h ---------------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// This header file contains two parts. The first part is the BTF ELF
// specification in C format, and the second part is the various
// C++ classes to manipulate the data structure in order to generate
// the BTF related ELF sections.
//===----------------------------------------------------------------------===//
#ifndef LLVM_MC_MCBTFCONTEXT_H
#define LLVM_MC_MCBTFCONTEXT_H

#include <linux/types.h>

#define BTF_MAGIC	0xeB9F
#define BTF_VERSION	1

struct btf_header {
	__u16	magic;
	__u8	version;
	__u8	flags;
	__u32	hdr_len;

	/* All offsets are in bytes relative to the end of this header */
	__u32	type_off;	/* offset of type section	*/
	__u32	type_len;	/* length of type section	*/
	__u32	str_off;	/* offset of string section	*/
	__u32	str_len;	/* length of string section	*/
};

/* Max # of type identifier */
#define BTF_MAX_TYPE	0x0000ffff
/* Max offset into the string section */
#define BTF_MAX_NAME_OFFSET	0x0000ffff
/* Max # of struct/union/enum members or func args */
#define BTF_MAX_VLEN	0xffff

struct btf_type {
	__u32 name_off;
	/* "info" bits arrangement
	 * bits  0-15: vlen (e.g. # of struct's members)
	 * bits 16-23: unused
	 * bits 24-27: kind (e.g. int, ptr, array...etc)
	 * bits 28-31: unused
	 */
	__u32 info;
	/* "size" is used by INT, ENUM, STRUCT and UNION.
	 * "size" tells the size of the type it is describing.
	 *
	 * "type" is used by PTR, TYPEDEF, VOLATILE, CONST, RESTRICT,
	 * FUNC and FUNC_PROTO.
	 * "type" is a type_id referring to another type.
	 */
	union {
		__u32 size;
		__u32 type;
	};
};

#define BTF_INFO_KIND(info)	(((info) >> 24) & 0x0f)
#define BTF_INFO_VLEN(info)	((info) & 0xffff)

#define BTF_KIND_UNKN		0	/* Unknown	*/
#define BTF_KIND_INT		1	/* Integer	*/
#define BTF_KIND_PTR		2	/* Pointer	*/
#define BTF_KIND_ARRAY		3	/* Array	*/
#define BTF_KIND_STRUCT		4	/* Struct	*/
#define BTF_KIND_UNION		5	/* Union	*/
#define BTF_KIND_ENUM		6	/* Enumeration	*/
#define BTF_KIND_FWD		7	/* Forward	*/
#define BTF_KIND_TYPEDEF	8	/* Typedef	*/
#define BTF_KIND_VOLATILE	9	/* Volatile	*/
#define BTF_KIND_CONST		10	/* Const	*/
#define BTF_KIND_RESTRICT	11	/* Restrict	*/
#define BTF_KIND_FUNC		12	/* Function	*/
#define BTF_KIND_FUNC_PROTO	13	/* Function Prototype	*/
#define BTF_KIND_MAX		13
#define NR_BTF_KINDS		14

/* For some specific BTF_KIND, "struct btf_type" is immediately
 * followed by extra data.
 */

/* BTF_KIND_INT is followed by a u32 and the following
 * is the 32 bits arrangement:
 */
#define BTF_INT_ENCODING(VAL)	(((VAL) & 0x0f000000) >> 24)
#define BTF_INT_OFFSET(VAL)	(((VAL  & 0x00ff0000)) >> 16)
#define BTF_INT_BITS(VAL)	((VAL)  & 0x000000ff)

/* Attributes stored in the BTF_INT_ENCODING */
#define BTF_INT_SIGNED	(1 << 0)
#define BTF_INT_CHAR	(1 << 1)
#define BTF_INT_BOOL	(1 << 2)

/* BTF_KIND_ENUM is followed by multiple "struct btf_enum".
 * The exact number of btf_enum is stored in the vlen (of the
 * info in "struct btf_type").
 */
struct btf_enum {
	__u32	name_off;
	__s32	val;
};

/* BTF_KIND_ARRAY is followed by one "struct btf_array" */
struct btf_array {
	__u32	type;
	__u32	index_type;
	__u32	nelems;
};

/* BTF_KIND_STRUCT and BTF_KIND_UNION are followed
 * by multiple "struct btf_member".  The exact number
 * of btf_member is stored in the vlen (of the info in
 * "struct btf_type").
 */
struct btf_member {
	__u32	name_off;
	__u32	type;
	__u32	offset;	/* offset in bits */
};

/* .BTF.ext section contains func_info and line_info.
 */
struct btf_ext_header {
	__u16	magic;
	__u8	version;
	__u8	flags;
	__u32	hdr_len;

	__u32	func_info_off;
	__u32	func_info_len;
	__u32	line_info_off;
	__u32	line_info_len;
};

struct bpf_func_info {
	__u32	insn_offset;
	__u32	type_id;
};

struct btf_sec_func_info {
	__u32	sec_name_off;
	__u32	num_func_info;
};

struct bpf_line_info {
	__u32	insn_offset;
	__u32	file_name_off;
	__u32	line_off;
	__u32	line_col; /* line num: line_col >> 10, col num: line_col & 0x3ff */
};

struct btf_sec_line_info {
	__u32	sec_name_off;
	__u32	num_line_info;
};

namespace llvm {

const char *const btf_kind_str[NR_BTF_KINDS] = {
	[BTF_KIND_UNKN]		= "UNKNOWN",
	[BTF_KIND_INT]		= "INT",
	[BTF_KIND_PTR]		= "PTR",
	[BTF_KIND_ARRAY]	= "ARRAY",
	[BTF_KIND_STRUCT]	= "STRUCT",
	[BTF_KIND_UNION]	= "UNION",
	[BTF_KIND_ENUM]		= "ENUM",
	[BTF_KIND_FWD]		= "FWD",
	[BTF_KIND_TYPEDEF]	= "TYPEDEF",
	[BTF_KIND_VOLATILE]	= "VOLATILE",
	[BTF_KIND_CONST]	= "CONST",
	[BTF_KIND_RESTRICT]	= "RESTRICT",
	[BTF_KIND_FUNC]		= "FUNC",
	[BTF_KIND_FUNC_PROTO]	= "FUNC_PROTO",
};

#include "llvm/ADT/SmallVector.h"
#include <map>

class MCBTFContext;
class MCObjectStreamer;

// This is base class of all BTF KIND. It is also used directly
// by the reference kinds:
//   BTF_KIND_CONST,  BTF_KIND_PTR,  BTF_KIND_VOLATILE,
//   BTF_KIND_TYPEDEF, BTF_KIND_RESTRICT, and BTF_KIND_FWD
class BTFTypeEntry {
protected:
  size_t Id;  /* type index in the BTF list, started from 1 */
  struct btf_type BTFType;

public:
  BTFTypeEntry(size_t id, struct btf_type &type) :
    Id(id), BTFType(type) {}
  unsigned char getKind() { return BTF_INFO_KIND(BTFType.info); }
  void setId(size_t Id) { this->Id = Id; }
  size_t getId() { return Id; }
  void setNameOff(unsigned NameOff) { BTFType.name_off = NameOff; }

  unsigned getTypeIndex() { return BTFType.type; }
  unsigned getNameOff() { return BTFType.name_off; }
  virtual size_t getSize() { return sizeof(struct btf_type); }
  virtual void print(raw_ostream &s, MCBTFContext& BTFContext);
  virtual void emitData(MCObjectStreamer *MCOS);
};

// BTF_KIND_INT
class BTFTypeEntryInt : public BTFTypeEntry {
  unsigned IntVal;  // encoding, offset, bits

public:
  BTFTypeEntryInt(size_t id, struct btf_type &type, unsigned intval) :
    BTFTypeEntry(id, type), IntVal(intval) {}
  size_t getSize() { return BTFTypeEntry::getSize() + sizeof(unsigned); }
  void print(raw_ostream &s, MCBTFContext& BTFContext);
  void emitData(MCObjectStreamer *MCOS);
};

// BTF_KIND_ENUM
class BTFTypeEntryEnum : public BTFTypeEntry {
  std::vector<struct btf_enum> EnumValues;

public:
  BTFTypeEntryEnum(size_t id, struct btf_type &type,
                   std::vector<struct btf_enum> &values) :
    BTFTypeEntry(id, type), EnumValues(values) {}
  size_t getSize() {
    return BTFTypeEntry::getSize() +
      BTF_INFO_VLEN(BTFType.info) * sizeof(struct btf_enum);
  }
  void print(raw_ostream &s, MCBTFContext& BTFContext);
  void emitData(MCObjectStreamer *MCOS);
};

// BTF_KIND_ARRAY
class BTFTypeEntryArray : public BTFTypeEntry {
  struct btf_array ArrayInfo;

public:
  BTFTypeEntryArray(size_t id, struct btf_type &type,
                    struct btf_array &arrayinfo) :
    BTFTypeEntry(id, type), ArrayInfo(arrayinfo) {}
  size_t getSize() {
    return BTFTypeEntry::getSize() +  sizeof(struct btf_array);
  }
  void print(raw_ostream &s, MCBTFContext& BTFContext);
  void emitData(MCObjectStreamer *MCOS);
};

// BTF_KIND_STRUCT and BTF_KIND_UNION
class BTFTypeEntryStruct : public BTFTypeEntry {
  std::vector<struct btf_member> Members;

public:
  BTFTypeEntryStruct(size_t id, struct btf_type &type,
                     std::vector<struct btf_member> &members) :
    BTFTypeEntry(id, type), Members(members) {}
  size_t getSize() {
    return BTFTypeEntry::getSize() +
      BTF_INFO_VLEN(BTFType.info) * sizeof(struct btf_member);
  }
  void print(raw_ostream &s, MCBTFContext& BTFContext);
  void emitData(MCObjectStreamer *MCOS);
};

// BTF_KIND_FUNC and BTF_KIND_FUNC_PROTO
class BTFTypeEntryFunc : public BTFTypeEntry {
  std::vector<unsigned> Parameters;

public:
  BTFTypeEntryFunc(size_t id, struct btf_type &type,
                   std::vector<unsigned> &params) :
    BTFTypeEntry(id, type), Parameters(params) {}
  size_t getSize() {
    return BTFTypeEntry::getSize() +
      BTF_INFO_VLEN(BTFType.info) * sizeof(unsigned);
  }
  void print(raw_ostream &s, MCBTFContext& BTFContext);
  void emitData(MCObjectStreamer *MCOS);
};

class BTFStringTable {
  size_t Size;  // total size in bytes
  std::map<size_t, unsigned> OffsetToIdMap;
  std::vector<std::string> Table;

 public:
  BTFStringTable() : Size(0) {}
  size_t getSize() { return Size; }
  std::vector<std::string> &getTable() { return Table; }
  size_t addString(std::string S) {
    // check whether the string already exists
    for (auto &OffsetM : OffsetToIdMap) {
      if (Table[OffsetM.second] == S)
        return OffsetM.first;
    }
    // not find, add to the string table
    size_t Offset = Size;
    OffsetToIdMap[Offset] = Table.size();
    Table.push_back(S);
    Size += S.size() + 1;
    return Offset;
  }
  std::string &getStringAtOffset(size_t Offset) {
    return Table[OffsetToIdMap[Offset]];
  }
  void showTable(raw_ostream &OS) {
    for (auto OffsetM : OffsetToIdMap)
      OS << OffsetM.first << " : " << Table[OffsetM.second]
         << "\n";
  }
};

struct BTFFuncInfo  {
    const MCSymbol *Label;
    unsigned int TypeId;
};

struct BTFLineInfo  {
    MCSymbol *Label;
    unsigned int FileNameOff;
    unsigned int LineOff;
    unsigned int LineNum;
    unsigned int ColumnNum;
};

class MCBTFContext {
  std::vector<std::unique_ptr<BTFTypeEntry>> TypeEntries;
  BTFStringTable StringTable;
  std::map<unsigned, std::vector<BTFFuncInfo>> FuncInfoTable;
  std::map<unsigned, std::vector<BTFLineInfo>> LineInfoTable;

  friend class BTFTypeEntry;
  friend class BTFTypeEntryInt;
  friend class BTFTypeEntryEnum;
  friend class BTFTypeEntryArray;
  friend class BTFTypeEntryStruct;
  friend class BTFTypeEntryFunc;

public:
  void dump(raw_ostream& OS);
  void emitAll(MCObjectStreamer *MCOS);
  void emitCommonHeader(MCObjectStreamer *MCOS);
  void emitBTFSection(MCObjectStreamer *MCOS);
  void emitBTFExtSection(MCObjectStreamer *MCOS);

  size_t addString(std::string S) {
    return StringTable.addString(S);
  }
  void addTypeEntry(std::unique_ptr<BTFTypeEntry> Entry);
  void addFuncInfo(unsigned SecNameOff, BTFFuncInfo Info) {
    FuncInfoTable[SecNameOff].push_back(Info);
  }
  void addLineInfo(unsigned SecNameOff, BTFLineInfo Info) {
    LineInfoTable[SecNameOff].push_back(Info);
  }
};

}
#endif
