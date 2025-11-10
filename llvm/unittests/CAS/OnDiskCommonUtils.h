//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Helper functions to test OnDiskCASDatabases.
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/BuiltinObjectHasher.h"
#include "llvm/CAS/OnDiskGraphDB.h"
#include "llvm/CAS/OnDiskKeyValueDB.h"
#include "llvm/CAS/UnifiedOnDiskCache.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Testing/Support/Error.h"

namespace llvm::unittest::cas {

using namespace llvm::cas;
using namespace llvm::cas::ondisk;

using HasherT = BLAKE3;
using HashType = decltype(HasherT::hash(std::declval<ArrayRef<uint8_t> &>()));
using ValueType = std::array<char, 20>;

inline HashType digest(StringRef Data, ArrayRef<ArrayRef<uint8_t>> RefHashes) {
  return BuiltinObjectHasher<HasherT>::hashObject(
      RefHashes, arrayRefFromStringRef<char>(Data));
}

inline ObjectID digest(OnDiskGraphDB &DB, StringRef Data,
                       ArrayRef<ObjectID> Refs) {
  SmallVector<ArrayRef<uint8_t>, 8> RefHashes;
  for (ObjectID Ref : Refs)
    RefHashes.push_back(DB.getDigest(Ref));
  HashType Digest = digest(Data, RefHashes);
  std::optional<ObjectID> ID;
  EXPECT_THAT_ERROR(DB.getReference(Digest).moveInto(ID), Succeeded());
  return *ID;
}

inline HashType digest(StringRef Data) {
  return HasherT::hash(arrayRefFromStringRef(Data));
}

inline ValueType valueFromString(StringRef S) {
  ValueType Val = {};
  llvm::copy(S.substr(0, sizeof(Val)), Val.data());
  return Val;
}

inline Expected<ObjectID> store(OnDiskGraphDB &DB, StringRef Data,
                                ArrayRef<ObjectID> Refs) {
  ObjectID ID = digest(DB, Data, Refs);
  if (Error E = DB.store(ID, Refs, arrayRefFromStringRef<char>(Data)))
    return std::move(E);
  return ID;
}

inline Expected<ObjectID> cachePut(OnDiskKeyValueDB &DB, ArrayRef<uint8_t> Key,
                                   ObjectID ID) {
  auto Value = UnifiedOnDiskCache::getValueFromObjectID(ID);
  auto Result = DB.put(Key, Value);
  if (!Result)
    return Result.takeError();
  return UnifiedOnDiskCache::getObjectIDFromValue(*Result);
}

inline Expected<std::optional<ObjectID>> cacheGet(OnDiskKeyValueDB &DB,
                                                  ArrayRef<uint8_t> Key) {
  auto Result = DB.get(Key);
  if (!Result)
    return Result.takeError();
  if (!*Result)
    return std::nullopt;
  return UnifiedOnDiskCache::getObjectIDFromValue(**Result);
}

inline Error printTree(OnDiskGraphDB &DB, ObjectID ID, raw_ostream &OS,
                       unsigned Indent = 0) {
  std::optional<ondisk::ObjectHandle> Obj;
  if (Error E = DB.load(ID).moveInto(Obj))
    return E;
  if (!Obj)
    return Error::success();
  OS.indent(Indent) << toStringRef(DB.getObjectData(*Obj)) << '\n';
  for (ObjectID Ref : DB.getObjectRefs(*Obj)) {
    if (Error E = printTree(DB, Ref, OS, Indent + 2))
      return E;
  }
  return Error::success();
}

} // namespace llvm::unittest::cas
