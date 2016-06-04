//===- RawTypes.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_RAWTYPES_H
#define LLVM_DEBUGINFO_PDB_RAW_RAWTYPES_H

#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/Support/Endian.h"

namespace llvm {
namespace pdb {
// This struct is defined as "SO" in langapi/include/pdb.h.
struct SectionOffset {
  support::ulittle32_t Off;
  support::ulittle16_t Isect;
  char Padding[2];
};

// This is HRFile.
struct PSHashRecord {
  support::ulittle32_t Off; // Offset in the symbol record stream
  support::ulittle32_t CRef;
};

// This struct is defined as `SC` in include/dbicommon.h
struct SectionContrib {
  support::ulittle16_t ISect;
  char Padding[2];
  support::little32_t Off;
  support::little32_t Size;
  support::ulittle32_t Characteristics;
  support::ulittle16_t Imod;
  char Padding2[2];
  support::ulittle32_t DataCrc;
  support::ulittle32_t RelocCrc;
};

// This struct is defined as `SC2` in include/dbicommon.h
struct SectionContrib2 {
  // To guarantee SectionContrib2 is standard layout, we cannot use inheritance.
  SectionContrib Base;
  support::ulittle32_t ISectCoff;
};

// This corresponds to the `OMFSegMap` structure.
struct SecMapHeader {
  support::ulittle16_t SecCount;    // Number of segment descriptors in table
  support::ulittle16_t SecCountLog; // Number of logical segment descriptors
};

// This corresponds to the `OMFSegMapDesc` structure.  The definition is not
// present in the reference implementation, but the layout is derived from
// code that accesses the fields.
struct SecMapEntry {
  support::ulittle16_t Flags; // Descriptor flags.  See OMFSegDescFlags
  support::ulittle16_t Ovl;   // Logical overlay number.
  support::ulittle16_t Group; // Group index into descriptor array.
  support::ulittle16_t Frame;
  support::ulittle16_t SecName;       // Byte index of the segment or group name
                                      // in the sstSegName table, or 0xFFFF.
  support::ulittle16_t ClassName;     // Byte index of the class name in the
                                      // sstSegName table, or 0xFFFF.
  support::ulittle32_t Offset;        // Byte offset of the logical segment
                                      // within the specified physical segment.
                                      // If group is set in flags, offset is the
                                      // offset of the group.
  support::ulittle32_t SecByteLength; // Byte count of the segment or group.
};

// Used for serialized hash table in TPI stream.
// In the reference, it is an array of TI and cbOff pair.
struct TypeIndexOffset {
  codeview::TypeIndex Type;
  support::ulittle32_t Offset;
};

} // namespace pdb
} // namespace llvm

#endif
