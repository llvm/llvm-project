//===- TypeTableCollection.h ---------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_TYPETABLECOLLECTION_H
#define LLVM_DEBUGINFO_CODEVIEW_TYPETABLECOLLECTION_H

#include "llvm/DebugInfo/CodeView/TypeCollection.h"
#include "llvm/DebugInfo/CodeView/TypeDatabase.h"

namespace llvm {
namespace codeview {

class TypeTableCollection : public TypeCollection {
public:
  explicit TypeTableCollection(ArrayRef<MutableArrayRef<uint8_t>> Records);

  Optional<TypeIndex> getFirst() override;
  Optional<TypeIndex> getNext(TypeIndex Prev) override;

  CVType getType(TypeIndex Index) override;
  StringRef getTypeName(TypeIndex Index) override;
  bool contains(TypeIndex Index) override;
  uint32_t size() override;
  uint32_t capacity() override;

private:
  bool hasCapacityFor(TypeIndex Index) const;
  void ensureTypeExists(TypeIndex Index);

  ArrayRef<MutableArrayRef<uint8_t>> Records;
  TypeDatabase Database;
};
}
}

#endif
