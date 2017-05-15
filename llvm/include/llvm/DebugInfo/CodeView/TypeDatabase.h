//===- TypeDatabase.h - A collection of CodeView type records ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_TYPEDATABASE_H
#define LLVM_DEBUGINFO_CODEVIEW_TYPEDATABASE_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/StringSaver.h"

namespace llvm {
namespace codeview {
class TypeDatabase {
  friend class RandomAccessTypeVisitor;

public:
  explicit TypeDatabase(uint32_t Capacity);

  /// Records the name of a type, and reserves its type index.
  TypeIndex appendType(StringRef Name, const CVType &Data);

  /// Records the name of a type, and reserves its type index.
  void recordType(StringRef Name, TypeIndex Index, const CVType &Data);

  /// Saves the name in a StringSet and creates a stable StringRef.
  StringRef saveTypeName(StringRef TypeName);

  StringRef getTypeName(TypeIndex Index) const;

  const CVType &getTypeRecord(TypeIndex Index) const;
  CVType &getTypeRecord(TypeIndex Index);

  bool contains(TypeIndex Index) const;

  uint32_t size() const;
  uint32_t capacity() const;
  bool empty() const;

  TypeIndex getAppendIndex() const;

private:
  void grow();

  BumpPtrAllocator Allocator;

  uint32_t Count = 0;

  /// All user defined type records in .debug$T live in here. Type indices
  /// greater than 0x1000 are user defined. Subtract 0x1000 from the index to
  /// index into this vector.
  SmallVector<StringRef, 10> CVUDTNames;
  SmallVector<CVType, 10> TypeRecords;

  StringSaver TypeNameStorage;

  BitVector ValidRecords;
};
}
}

#endif