//===- TypeTableCollection.cpp -------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/TypeTableCollection.h"

#include "llvm/DebugInfo/CodeView/CVTypeVisitor.h"
#include "llvm/DebugInfo/CodeView/TypeDatabaseVisitor.h"
#include "llvm/DebugInfo/CodeView/TypeTableBuilder.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/BinaryStreamReader.h"

using namespace llvm;
using namespace llvm::codeview;

static void error(Error &&EC) {
  assert(!static_cast<bool>(EC));
  if (EC)
    consumeError(std::move(EC));
}

TypeTableCollection::TypeTableCollection(
  ArrayRef<ArrayRef<uint8_t>> Records)
    : Records(Records), Database(Records.size()) {}

Optional<TypeIndex> TypeTableCollection::getFirst() {
  if (empty())
    return None;
  return TypeIndex::fromArrayIndex(0);
}

Optional<TypeIndex> TypeTableCollection::getNext(TypeIndex Prev) {
  ++Prev;
  assert(Prev.toArrayIndex() <= size());
  if (Prev.toArrayIndex() == size())
    return None;
  return Prev;
}

void TypeTableCollection::ensureTypeExists(TypeIndex Index) {
  assert(hasCapacityFor(Index));

  if (Database.contains(Index))
    return;

  BinaryByteStream Bytes(Records[Index.toArrayIndex()], support::little);

  CVType Type;
  uint32_t Len;
  error(VarStreamArrayExtractor<CVType>::extract(Bytes, Len, Type));

  TypeDatabaseVisitor DBV(Database);
  error(codeview::visitTypeRecord(Type, Index, DBV));
  assert(Database.contains(Index));
}

CVType TypeTableCollection::getType(TypeIndex Index) {
  ensureTypeExists(Index);
  return Database.getTypeRecord(Index);
}

StringRef TypeTableCollection::getTypeName(TypeIndex Index) {
  if (!Index.isSimple())
    ensureTypeExists(Index);
  return Database.getTypeName(Index);
}

bool TypeTableCollection::contains(TypeIndex Index) {
  return Database.contains(Index);
}

uint32_t TypeTableCollection::size() { return Records.size(); }

uint32_t TypeTableCollection::capacity() { return Records.size(); }

bool TypeTableCollection::hasCapacityFor(TypeIndex Index) const {
  return Index.toArrayIndex() < Records.size();
}
