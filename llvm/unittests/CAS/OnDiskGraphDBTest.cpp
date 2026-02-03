//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CASTestConfig.h"
#include "OnDiskCommonUtils.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::cas::ondisk;
using namespace llvm::unittest::cas;

TEST_F(OnDiskCASTest, OnDiskGraphDBTest) {
  unittest::TempDir Temp("ondiskcas", /*Unique=*/true);
  std::unique_ptr<OnDiskGraphDB> DB;
  ASSERT_THAT_ERROR(
      OnDiskGraphDB::open(Temp.path(), "blake3", sizeof(HashType)).moveInto(DB),
      Succeeded());

  auto digest = [&DB](StringRef Data, ArrayRef<ObjectID> Refs) -> ObjectID {
    return ::digest(*DB, Data, Refs);
  };

  auto store = [&](StringRef Data,
                   ArrayRef<ObjectID> Refs) -> Expected<ObjectID> {
    return ::store(*DB, Data, Refs);
  };

  std::optional<ObjectID> ID1;
  ASSERT_THAT_ERROR(store("hello", {}).moveInto(ID1), Succeeded());

  std::optional<ondisk::ObjectHandle> Obj1;
  ASSERT_THAT_ERROR(DB->load(*ID1).moveInto(Obj1), Succeeded());
  ASSERT_TRUE(Obj1.has_value());
  EXPECT_EQ(toStringRef(DB->getObjectData(*Obj1)), "hello");

  ArrayRef<uint8_t> Digest1 = DB->getDigest(*ID1);
  std::optional<ObjectID> ID2;
  ASSERT_THAT_ERROR(DB->getReference(Digest1).moveInto(ID2), Succeeded());
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

  ASSERT_THAT_ERROR(DB->validateObjectID(*ID1), Succeeded());
  ASSERT_THAT_ERROR(DB->validateObjectID(ObjectID::fromOpaqueData(0)),
                    Failed());
  ASSERT_THAT_ERROR(DB->validateObjectID(ObjectID::fromOpaqueData(4)),
                    Failed());
  ASSERT_THAT_ERROR(DB->validateObjectID(ObjectID::fromOpaqueData(8)),
                    Failed());

  size_t LargeDataSize = 256LL * 1024LL; // 256K.
  // The precise size number is not important, we mainly check that the large
  // object will be properly accounted for.
  EXPECT_TRUE(DB->getStorageSize() > 10 &&
              DB->getStorageSize() < LargeDataSize);

  SmallString<16> Buffer;
  Buffer.resize(LargeDataSize);
  ASSERT_THAT_ERROR(store(Buffer, {}).moveInto(ID1), Succeeded());
  size_t StorageSize = DB->getStorageSize();
  EXPECT_TRUE(StorageSize > LargeDataSize);

  // Close & re-open the DB and check that it reports the same storage size.
  DB.reset();
  ASSERT_THAT_ERROR(
      OnDiskGraphDB::open(Temp.path(), "blake3", sizeof(HashType)).moveInto(DB),
      Succeeded());
  EXPECT_EQ(DB->getStorageSize(), StorageSize);
}

TEST_F(OnDiskCASTest, OnDiskGraphDBFaultInSingleNode) {
  unittest::TempDir TempUpstream("ondiskcas-upstream", /*Unique=*/true);
  std::unique_ptr<OnDiskGraphDB> UpstreamDB;
  ASSERT_THAT_ERROR(
      OnDiskGraphDB::open(TempUpstream.path(), "blake3", sizeof(HashType))
          .moveInto(UpstreamDB),
      Succeeded());
  {
    std::optional<ObjectID> ID1;
    ASSERT_THAT_ERROR(store(*UpstreamDB, "hello", {}).moveInto(ID1),
                      Succeeded());
    std::optional<ObjectID> ID2;
    ASSERT_THAT_ERROR(store(*UpstreamDB, "another", {}).moveInto(ID2),
                      Succeeded());
    std::optional<ObjectID> ID3;
    ASSERT_THAT_ERROR(store(*UpstreamDB, "world", {*ID1, *ID2}).moveInto(ID3),
                      Succeeded());
  }

  unittest::TempDir Temp("ondiskcas", /*Unique=*/true);
  std::unique_ptr<OnDiskGraphDB> DB;
  ASSERT_THAT_ERROR(
      OnDiskGraphDB::open(Temp.path(), "blake3", sizeof(HashType),
                          UpstreamDB.get(), /*Logger=*/nullptr,
                          OnDiskGraphDB::FaultInPolicy::SingleNode)
          .moveInto(DB),
      Succeeded());

  ObjectID ID1 = digest(*DB, "hello", {});
  ObjectID ID2 = digest(*DB, "another", {});
  ObjectID ID3 = digest(*DB, "world", {ID1, ID2});
  ObjectID ID4 = digest(*DB, "world", {});

  EXPECT_TRUE(DB->containsObject(ID1));
  EXPECT_TRUE(DB->containsObject(ID2));
  EXPECT_TRUE(DB->containsObject(ID3));
  EXPECT_FALSE(DB->containsObject(ID4));

  EXPECT_TRUE(DB->getExistingReference(digest("hello", {})).has_value());
  EXPECT_TRUE(DB->getExistingReference(DB->getDigest(ID3)).has_value());
  EXPECT_FALSE(DB->getExistingReference(digest("world", {})).has_value());

  {
    std::optional<ondisk::ObjectHandle> Obj;
    ASSERT_THAT_ERROR(DB->load(ID1).moveInto(Obj), Succeeded());
    ASSERT_TRUE(Obj.has_value());
    EXPECT_EQ(toStringRef(DB->getObjectData(*Obj)), "hello");
    auto Refs = DB->getObjectRefs(*Obj);
    EXPECT_TRUE(Refs.empty());
  }
  {
    std::optional<ondisk::ObjectHandle> Obj;
    ASSERT_THAT_ERROR(DB->load(ID3).moveInto(Obj), Succeeded());
    ASSERT_TRUE(Obj.has_value());
    EXPECT_EQ(toStringRef(DB->getObjectData(*Obj)), "world");
    auto Refs = DB->getObjectRefs(*Obj);
    ASSERT_EQ(std::distance(Refs.begin(), Refs.end()), 2);
    EXPECT_EQ(Refs.begin()[0], ID1);
    EXPECT_EQ(Refs.begin()[1], ID2);
  }
  {
    std::optional<ondisk::ObjectHandle> Obj;
    ASSERT_THAT_ERROR(DB->load(ID4).moveInto(Obj), Succeeded());
    EXPECT_FALSE(Obj.has_value());
  }

  // Re-open the primary without chaining, to verify the data were copied from
  // the upstream.
  ASSERT_THAT_ERROR(
      OnDiskGraphDB::open(Temp.path(), "blake3", sizeof(HashType),
                          /*UpstreamDB=*/nullptr, /*Logger=*/nullptr,
                          OnDiskGraphDB::FaultInPolicy::SingleNode)
          .moveInto(DB),
      Succeeded());
  ID1 = digest(*DB, "hello", {});
  ID2 = digest(*DB, "another", {});
  ID3 = digest(*DB, "world", {ID1, ID2});
  EXPECT_TRUE(DB->containsObject(ID1));
  EXPECT_FALSE(DB->containsObject(ID2));
  EXPECT_TRUE(DB->containsObject(ID3));
  {
    std::optional<ondisk::ObjectHandle> Obj;
    ASSERT_THAT_ERROR(DB->load(ID1).moveInto(Obj), Succeeded());
    ASSERT_TRUE(Obj.has_value());
    EXPECT_EQ(toStringRef(DB->getObjectData(*Obj)), "hello");
    auto Refs = DB->getObjectRefs(*Obj);
    EXPECT_TRUE(Refs.empty());
  }
}

TEST_F(OnDiskCASTest, OnDiskGraphDBFaultInFullTree) {
  unittest::TempDir TempUpstream("ondiskcas-upstream", /*Unique=*/true);
  std::unique_ptr<OnDiskGraphDB> UpstreamDB;
  ASSERT_THAT_ERROR(
      OnDiskGraphDB::open(TempUpstream.path(), "blake3", sizeof(HashType))
          .moveInto(UpstreamDB),
      Succeeded());
  HashType RootHash;
  {
    std::optional<ObjectID> ID11;
    ASSERT_THAT_ERROR(store(*UpstreamDB, "11", {}).moveInto(ID11), Succeeded());
    std::optional<ObjectID> ID121;
    ASSERT_THAT_ERROR(store(*UpstreamDB, "121", {}).moveInto(ID121),
                      Succeeded());
    std::optional<ObjectID> ID12;
    ASSERT_THAT_ERROR(store(*UpstreamDB, "12", {*ID121}).moveInto(ID12),
                      Succeeded());
    std::optional<ObjectID> ID1;
    ASSERT_THAT_ERROR(store(*UpstreamDB, "1", {*ID11, *ID12}).moveInto(ID1),
                      Succeeded());
    std::optional<ObjectID> ID21;
    ASSERT_THAT_ERROR(store(*UpstreamDB, "21", {}).moveInto(ID21), Succeeded());
    std::optional<ObjectID> ID22;
    ASSERT_THAT_ERROR(store(*UpstreamDB, "22", {}).moveInto(ID22), Succeeded());
    std::optional<ObjectID> ID2;
    ASSERT_THAT_ERROR(
        store(*UpstreamDB, "2", {*ID12, *ID21, *ID22}).moveInto(ID2),
        Succeeded());
    std::optional<ObjectID> IDRoot;
    ASSERT_THAT_ERROR(store(*UpstreamDB, "root", {*ID1, *ID2}).moveInto(IDRoot),
                      Succeeded());
    ArrayRef<uint8_t> Digest = UpstreamDB->getDigest(*IDRoot);
    ASSERT_EQ(Digest.size(), RootHash.size());
    llvm::copy(Digest, RootHash.data());
  }

  unittest::TempDir Temp("ondiskcas", /*Unique=*/true);
  std::unique_ptr<OnDiskGraphDB> DB;
  ASSERT_THAT_ERROR(OnDiskGraphDB::open(Temp.path(), "blake3", sizeof(HashType),
                                        UpstreamDB.get(),
                                        /*Logger=*/nullptr,
                                        OnDiskGraphDB::FaultInPolicy::FullTree)
                        .moveInto(DB),
                    Succeeded());

  {
    std::optional<ObjectID> IDRoot;
    ASSERT_THAT_ERROR(DB->getReference(RootHash).moveInto(IDRoot), Succeeded());
    std::optional<ondisk::ObjectHandle> Obj;
    ASSERT_THAT_ERROR(DB->load(*IDRoot).moveInto(Obj), Succeeded());
    ASSERT_TRUE(Obj.has_value());
    EXPECT_EQ(toStringRef(DB->getObjectData(*Obj)), "root");
    auto Refs = DB->getObjectRefs(*Obj);
    ASSERT_EQ(std::distance(Refs.begin(), Refs.end()), 2);
  }

  // Re-open the primary without chaining, to verify the data were copied from
  // the upstream.
  ASSERT_THAT_ERROR(OnDiskGraphDB::open(Temp.path(), "blake3", sizeof(HashType),
                                        /*UpstreamDB=*/nullptr,
                                        /*Logger=*/nullptr,
                                        OnDiskGraphDB::FaultInPolicy::FullTree)
                        .moveInto(DB),
                    Succeeded());

  std::optional<ObjectID> IDRoot;
  ASSERT_THAT_ERROR(DB->getReference(RootHash).moveInto(IDRoot), Succeeded());
  std::string PrintedTree;
  raw_string_ostream OS(PrintedTree);
  ASSERT_THAT_ERROR(printTree(*DB, *IDRoot, OS), Succeeded());
  StringRef Expected = R"(root
  1
    11
    12
      121
  2
    12
      121
    21
    22
)";
  EXPECT_EQ(PrintedTree, Expected);
}

TEST_F(OnDiskCASTest, OnDiskGraphDBFaultInPolicyConflict) {
  auto tryFaultInPolicyConflict = [](OnDiskGraphDB::FaultInPolicy Policy1,
                                     OnDiskGraphDB::FaultInPolicy Policy2) {
    unittest::TempDir TempUpstream("ondiskcas-upstream", /*Unique=*/true);
    std::unique_ptr<OnDiskGraphDB> UpstreamDB;
    ASSERT_THAT_ERROR(
        OnDiskGraphDB::open(TempUpstream.path(), "blake3", sizeof(HashType))
            .moveInto(UpstreamDB),
        Succeeded());

    unittest::TempDir Temp("ondiskcas", /*Unique=*/true);
    std::unique_ptr<OnDiskGraphDB> DB;
    ASSERT_THAT_ERROR(OnDiskGraphDB::open(Temp.path(), "blake3",
                                          sizeof(HashType), UpstreamDB.get(),
                                          /*Logger=*/nullptr, Policy1)
                          .moveInto(DB),
                      Succeeded());
    DB.reset();
    ASSERT_THAT_ERROR(OnDiskGraphDB::open(Temp.path(), "blake3",
                                          sizeof(HashType), UpstreamDB.get(),
                                          /*Logger=*/nullptr, Policy2)
                          .moveInto(DB),
                      Failed());
  };
  // Open as 'single', then as 'full'.
  tryFaultInPolicyConflict(OnDiskGraphDB::FaultInPolicy::SingleNode,
                           OnDiskGraphDB::FaultInPolicy::FullTree);
  // Open as 'full', then as 'single'.
  tryFaultInPolicyConflict(OnDiskGraphDB::FaultInPolicy::FullTree,
                           OnDiskGraphDB::FaultInPolicy::SingleNode);
}

#if defined(EXPENSIVE_CHECKS) && !defined(_WIN32)
TEST_F(OnDiskCASTest, OnDiskGraphDBSpaceLimit) {
  setMaxOnDiskCASMappingSize();
  unittest::TempDir Temp("ondiskcas", /*Unique=*/true);
  std::unique_ptr<OnDiskGraphDB> DB;
  ASSERT_THAT_ERROR(
      OnDiskGraphDB::open(Temp.path(), "blake3", sizeof(HashType)).moveInto(DB),
      Succeeded());

  std::optional<ObjectID> ID;
  std::string Data(500, '0');
  auto storeSmallObject = [&]() {
    SmallVector<ObjectID, 1> Refs;
    if (ID)
      Refs.push_back(*ID);
    ASSERT_THAT_ERROR(store(*DB, Data, Refs).moveInto(ID), Succeeded());
  };

  // Insert enough small elements to overflow the data pool.
  for (unsigned I = 0; I < 1024 * 256; ++I)
    storeSmallObject();

  EXPECT_GE(DB->getHardStorageLimitUtilization(), 99U);
}
#endif
