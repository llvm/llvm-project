//===- StringTable.h - CodeView String Table Reader/Writer ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_STRINGTABLE_H
#define LLVM_DEBUGINFO_CODEVIEW_STRINGTABLE_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include "llvm/Support/BinaryStreamRef.h"
#include "llvm/Support/Error.h"

#include <stdint.h>

namespace llvm {

class BinaryStreamReader;
class BinaryStreamRef;
class BinaryStreamWriter;

namespace codeview {

/// Represents a read-only view of a CodeView string table.  This is a very
/// simple flat buffer consisting of null-terminated strings, where strings
/// are retrieved by their offset in the buffer.  StringTableRef does not own
/// the underlying storage for the buffer.
class StringTableRef {
public:
  StringTableRef();

  Error initialize(BinaryStreamReader &Stream);

  StringRef getString(uint32_t Offset) const;

private:
  BinaryStreamRef Stream;
};

/// Represents a read-write view of a CodeView string table.  StringTable owns
/// the underlying storage for the table, and is capable of serializing the
/// string table into a format understood by StringTableRef.
class StringTable {
public:
  // If string S does not exist in the string table, insert it.
  // Returns the ID for S.
  uint32_t insert(StringRef S);

  uint32_t calculateSerializedSize() const;
  Error commit(BinaryStreamWriter &Writer) const;

  uint32_t size() const;

  StringMap<uint32_t>::const_iterator begin() const { return Strings.begin(); }

  StringMap<uint32_t>::const_iterator end() const { return Strings.end(); }

private:
  StringMap<uint32_t> Strings;
  uint32_t StringSize = 1;
};
}
}

#endif
