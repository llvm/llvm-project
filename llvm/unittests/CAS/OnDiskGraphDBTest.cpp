//===- llvm/unittest/CAS/OnDiskGraphDBTest.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/OnDiskGraphDB.h"
#include "llvm/CAS/BuiltinObjectHasher.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

#if LLVM_ENABLE_ONDISK_CAS

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::cas::ondisk;

using HasherT = BLAKE3;
using HashType = decltype(HasherT::hash(std::declval<ArrayRef<uint8_t> &>()));

TEST(OnDiskGraphDBTest, Basic) {
  unittest::TempDir Temp("ondiskcas", /*Unique=*/true);
  std::unique_ptr<OnDiskGraphDB> DB;
  ASSERT_THAT_ERROR(
      OnDiskGraphDB::open(Temp.path(), "blake3", sizeof(HashType)).moveInto(DB),
      Succeeded());

  auto digest = [&DB](StringRef Data, ArrayRef<ObjectID> Refs) -> ObjectID {
    SmallVector<ArrayRef<uint8_t>, 8> RefHashes;
    for (ObjectID Ref : Refs)
      RefHashes.push_back(DB->getDigest(Ref));
    HashType Digest = BuiltinObjectHasher<HasherT>::hashObject(
        RefHashes, arrayRefFromStringRef<char>(Data));
    return DB->getReference(Digest);
  };

  auto store = [&](StringRef Data,
                   ArrayRef<ObjectID> Refs) -> Expected<ObjectID> {
    ObjectID ID = digest(Data, Refs);
    if (Error E = DB->store(ID, Refs, arrayRefFromStringRef<char>(Data)))
      return std::move(E);
    return ID;
  };

  std::optional<ObjectID> ID1;
  ASSERT_THAT_ERROR(store("hello", {}).moveInto(ID1), Succeeded());

  std::optional<ondisk::ObjectHandle> Obj1;
  ASSERT_THAT_ERROR(DB->load(*ID1).moveInto(Obj1), Succeeded());
  ASSERT_TRUE(Obj1.has_value());
  EXPECT_EQ(toStringRef(DB->getObjectData(*Obj1)), "hello");

  ArrayRef<uint8_t> Digest1 = DB->getDigest(*ID1);
  ObjectID ID2 = DB->getReference(Digest1);
  EXPECT_EQ(ID1, ID2);

  ObjectID ID3 = digest("world", {});
  EXPECT_FALSE(DB->containsObject(ID3));
  std::optional<ondisk::ObjectHandle> Obj2;
  ASSERT_THAT_ERROR(DB->load(ID3).moveInto(Obj2), Succeeded());
  EXPECT_FALSE(Obj2.has_value());

  ASSERT_THAT_ERROR(DB->store(ID3, {}, arrayRefFromStringRef<char>("world")),
                    Succeeded());
  EXPECT_TRUE(DB->containsObject(ID3));
  ASSERT_THAT_ERROR(DB->load(ID3).moveInto(Obj2), Succeeded());
  ASSERT_TRUE(Obj2.has_value());
  EXPECT_EQ(toStringRef(DB->getObjectData(*Obj2)), "world");
}

#endif // LLVM_ENABLE_ONDISK_CAS
