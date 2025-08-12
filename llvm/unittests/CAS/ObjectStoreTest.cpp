//===- ObjectStoreTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/RandomNumberGenerator.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

#include "CASTestConfig.h"

using namespace llvm;
using namespace llvm::cas;

TEST_P(CASTest, PrintIDs) {
  std::unique_ptr<ObjectStore> CAS = createObjectStore();

  std::optional<CASID> ID1, ID2;
  ASSERT_THAT_ERROR(CAS->createProxy({}, "1").moveInto(ID1), Succeeded());
  ASSERT_THAT_ERROR(CAS->createProxy({}, "2").moveInto(ID2), Succeeded());
  EXPECT_NE(ID1, ID2);
  std::string PrintedID1 = ID1->toString();
  std::string PrintedID2 = ID2->toString();
  EXPECT_NE(PrintedID1, PrintedID2);

  std::optional<CASID> ParsedID1, ParsedID2;
  ASSERT_THAT_ERROR(CAS->parseID(PrintedID1).moveInto(ParsedID1), Succeeded());
  ASSERT_THAT_ERROR(CAS->parseID(PrintedID2).moveInto(ParsedID2), Succeeded());
  EXPECT_EQ(ID1, ParsedID1);
  EXPECT_EQ(ID2, ParsedID2);
}

TEST_P(CASTest, Blobs) {
  std::unique_ptr<ObjectStore> CAS1 = createObjectStore();
  StringRef ContentStrings[] = {
      "word",
      "some longer text std::string's local memory",
      R"(multiline text multiline text multiline text multiline text
multiline text multiline text multiline text multiline text multiline text
multiline text multiline text multiline text multiline text multiline text
multiline text multiline text multiline text multiline text multiline text
multiline text multiline text multiline text multiline text multiline text
multiline text multiline text multiline text multiline text multiline text)",
  };

  SmallVector<CASID> IDs;
  for (StringRef Content : ContentStrings) {
    // Use StringRef::str() to create a temporary std::string. This could cause
    // problems if the CAS is storing references to the input string instead of
    // copying it.
    std::optional<ObjectProxy> Blob;
    ASSERT_THAT_ERROR(CAS1->createProxy({}, Content).moveInto(Blob),
                      Succeeded());
    IDs.push_back(Blob->getID());

    // Check basic printing of IDs.
    EXPECT_EQ(IDs.back().toString(), IDs.back().toString());
    if (IDs.size() > 2)
      EXPECT_NE(IDs.front().toString(), IDs.back().toString());
  }

  // Check that the blobs give the same IDs later.
  for (int I = 0, E = IDs.size(); I != E; ++I) {
    std::optional<ObjectProxy> Blob;
    ASSERT_THAT_ERROR(CAS1->createProxy({}, ContentStrings[I]).moveInto(Blob),
                      Succeeded());
    EXPECT_EQ(IDs[I], Blob->getID());
  }

  // Run validation on all CASIDs.
  for (int I = 0, E = IDs.size(); I != E; ++I)
    ASSERT_THAT_ERROR(CAS1->validate(IDs[I]), Succeeded());

  // Check that the blobs can be retrieved multiple times.
  for (int I = 0, E = IDs.size(); I != E; ++I) {
    for (int J = 0, JE = 3; J != JE; ++J) {
      std::optional<ObjectProxy> Buffer;
      ASSERT_THAT_ERROR(CAS1->getProxy(IDs[I]).moveInto(Buffer), Succeeded());
      EXPECT_EQ(ContentStrings[I], Buffer->getData());
    }
  }

  // Confirm these blobs don't exist in a fresh CAS instance.
  std::unique_ptr<ObjectStore> CAS2 = createObjectStore();
  for (int I = 0, E = IDs.size(); I != E; ++I) {
    std::optional<ObjectProxy> Proxy;
    EXPECT_THAT_ERROR(CAS2->getProxy(IDs[I]).moveInto(Proxy), Failed());
  }

  // Insert into the second CAS and confirm the IDs are stable. Getting them
  // should work now.
  for (int I = IDs.size(), E = 0; I != E; --I) {
    auto &ID = IDs[I - 1];
    auto &Content = ContentStrings[I - 1];
    std::optional<ObjectProxy> Blob;
    ASSERT_THAT_ERROR(CAS2->createProxy({}, Content).moveInto(Blob),
                      Succeeded());
    EXPECT_EQ(ID, Blob->getID());

    std::optional<ObjectProxy> Buffer;
    ASSERT_THAT_ERROR(CAS2->getProxy(ID).moveInto(Buffer), Succeeded());
    EXPECT_EQ(Content, Buffer->getData());
  }
}

TEST_P(CASTest, BlobsBig) {
  // A little bit of validation that bigger blobs are okay. Climb up to 1MB.
  std::unique_ptr<ObjectStore> CAS = createObjectStore();
  SmallString<256> String1 = StringRef("a few words");
  SmallString<256> String2 = StringRef("others");
  while (String1.size() < 1024U * 1024U) {
    std::optional<CASID> ID1;
    std::optional<CASID> ID2;
    ASSERT_THAT_ERROR(CAS->createProxy({}, String1).moveInto(ID1), Succeeded());
    ASSERT_THAT_ERROR(CAS->createProxy({}, String1).moveInto(ID2), Succeeded());
    ASSERT_THAT_ERROR(CAS->validate(*ID1), Succeeded());
    ASSERT_THAT_ERROR(CAS->validate(*ID2), Succeeded());
    ASSERT_EQ(ID1, ID2);

    String1.append(String2);
    ASSERT_THAT_ERROR(CAS->createProxy({}, String2).moveInto(ID1), Succeeded());
    ASSERT_THAT_ERROR(CAS->createProxy({}, String2).moveInto(ID2), Succeeded());
    ASSERT_THAT_ERROR(CAS->validate(*ID1), Succeeded());
    ASSERT_THAT_ERROR(CAS->validate(*ID2), Succeeded());
    ASSERT_EQ(ID1, ID2);
    String2.append(String1);
  }

  // Specifically check near 1MB for objects large enough they're likely to be
  // stored externally in an on-disk CAS and will be near a page boundary.
  SmallString<0> Storage;
  const size_t InterestingSize = 1024U * 1024ULL;
  const size_t SizeE = InterestingSize + 2;
  if (Storage.size() < SizeE)
    Storage.resize(SizeE, '\01');
  for (size_t Size = InterestingSize - 2; Size != SizeE; ++Size) {
    StringRef Data(Storage.data(), Size);
    std::optional<ObjectProxy> Blob;
    ASSERT_THAT_ERROR(CAS->createProxy({}, Data).moveInto(Blob), Succeeded());
    ASSERT_EQ(Data, Blob->getData());
    ASSERT_EQ(0, Blob->getData().end()[0]);
  }
}

TEST_P(CASTest, LeafNodes) {
  std::unique_ptr<ObjectStore> CAS1 = createObjectStore();
  StringRef ContentStrings[] = {
      "word",
      "some longer text std::string's local memory",
      R"(multiline text multiline text multiline text multiline text
multiline text multiline text multiline text multiline text multiline text
multiline text multiline text multiline text multiline text multiline text
multiline text multiline text multiline text multiline text multiline text
multiline text multiline text multiline text multiline text multiline text
multiline text multiline text multiline text multiline text multiline text)",
  };

  SmallVector<ObjectRef> Nodes;
  SmallVector<CASID> IDs;
  for (StringRef Content : ContentStrings) {
    // Use StringRef::str() to create a temporary std::string. This could cause
    // problems if the CAS is storing references to the input string instead of
    // copying it.
    std::optional<ObjectRef> Node;
    ASSERT_THAT_ERROR(
        CAS1->store({}, arrayRefFromStringRef<char>(Content)).moveInto(Node),
        Succeeded());
    Nodes.push_back(*Node);

    // Check basic printing of IDs.
    IDs.push_back(CAS1->getID(*Node));
    auto ID = CAS1->getID(Nodes.back());
    EXPECT_EQ(ID.toString(), IDs.back().toString());
    EXPECT_EQ(*Node, Nodes.back());
    EXPECT_EQ(ID, IDs.back());
    if (Nodes.size() <= 1)
      continue;
    EXPECT_NE(Nodes.front(), Nodes.back());
    EXPECT_NE(IDs.front(), IDs.back());
  }

  // Check that the blobs give the same IDs later.
  for (int I = 0, E = IDs.size(); I != E; ++I) {
    std::optional<ObjectRef> Node;
    ASSERT_THAT_ERROR(
        CAS1->store({}, arrayRefFromStringRef<char>(ContentStrings[I]))
            .moveInto(Node),
        Succeeded());
    EXPECT_EQ(IDs[I], CAS1->getID(*Node));
  }

  // Check that the blobs can be retrieved multiple times.
  for (int I = 0, E = IDs.size(); I != E; ++I) {
    for (int J = 0, JE = 3; J != JE; ++J) {
      std::optional<ObjectProxy> Object;
      ASSERT_THAT_ERROR(CAS1->getProxy(IDs[I]).moveInto(Object), Succeeded());
      ASSERT_TRUE(Object);
      EXPECT_EQ(ContentStrings[I], Object->getData());
    }
  }

  // Confirm these blobs don't exist in a fresh CAS instance.
  std::unique_ptr<ObjectStore> CAS2 = createObjectStore();
  for (int I = 0, E = IDs.size(); I != E; ++I) {
    std::optional<ObjectProxy> Object;
    EXPECT_THAT_ERROR(CAS2->getProxy(IDs[I]).moveInto(Object), Failed());
  }

  // Insert into the second CAS and confirm the IDs are stable. Getting them
  // should work now.
  for (int I = IDs.size(), E = 0; I != E; --I) {
    auto &ID = IDs[I - 1];
    auto &Content = ContentStrings[I - 1];
    std::optional<ObjectRef> Node;
    ASSERT_THAT_ERROR(
        CAS2->store({}, arrayRefFromStringRef<char>(Content)).moveInto(Node),
        Succeeded());
    EXPECT_EQ(ID, CAS2->getID(*Node));

    std::optional<ObjectProxy> Object;
    ASSERT_THAT_ERROR(CAS2->getProxy(ID).moveInto(Object), Succeeded());
    ASSERT_TRUE(Object);
    EXPECT_EQ(Content, Object->getData());
  }
}

TEST_P(CASTest, NodesBig) {
  std::unique_ptr<ObjectStore> CAS = createObjectStore();

  // Specifically check near 1MB for objects large enough they're likely to be
  // stored externally in an on-disk CAS, and such that one of them will be
  // near a page boundary.
  SmallString<0> Storage;
  constexpr size_t InterestingSize = 1024U * 1024ULL;
  constexpr size_t WordSize = sizeof(void *);

  // Start much smaller to account for headers.
  constexpr size_t SizeB = InterestingSize - 8 * WordSize;
  constexpr size_t SizeE = InterestingSize + 1;
  if (Storage.size() < SizeE)
    Storage.resize(SizeE, '\01');

  SmallVector<ObjectRef, 4> CreatedNodes;
  // Avoid checking every size because this is an expensive test. Just check
  // for data that is 8B-word-aligned, and one less. Also appending the created
  // nodes as the references in the next block to check references are created
  // correctly.
  for (size_t Size = SizeB; Size < SizeE; Size += WordSize) {
    for (bool IsAligned : {false, true}) {
      StringRef Data(Storage.data(), Size - (IsAligned ? 0 : 1));
      std::optional<ObjectProxy> Node;
      ASSERT_THAT_ERROR(CAS->createProxy(CreatedNodes, Data).moveInto(Node),
                        Succeeded());
      ASSERT_EQ(Data, Node->getData());
      ASSERT_EQ(0, Node->getData().end()[0]);
      ASSERT_EQ(Node->getNumReferences(), CreatedNodes.size());
      CreatedNodes.emplace_back(Node->getRef());
    }
  }

  for (auto ID : CreatedNodes)
    ASSERT_THAT_ERROR(CAS->validate(CAS->getID(ID)), Succeeded());
}

/// Common test functionality for creating blobs in parallel. You can vary which
/// cas instances are the same or different, and the size of the created blobs.
static void testBlobsParallel(ObjectStore &Read1, ObjectStore &Read2,
                              ObjectStore &Write1, ObjectStore &Write2,
                              uint64_t BlobSize) {
  SCOPED_TRACE(testBlobsParallel);
  unsigned BlobCount = 100;
  std::vector<std::string> Blobs;
  Blobs.reserve(BlobCount);
  for (unsigned I = 0; I < BlobCount; ++I) {
    std::string Blob;
    Blob.resize(BlobSize);
    getRandomBytes(Blob.data(), BlobSize);
    Blobs.push_back(std::move(Blob));
  }

  std::mutex NodesMtx;
  std::vector<std::optional<CASID>> CreatedNodes(BlobCount);

  auto Producer = [&](unsigned I, ObjectStore *CAS) {
    std::optional<ObjectProxy> Node;
    EXPECT_THAT_ERROR(CAS->createProxy({}, Blobs[I]).moveInto(Node),
                      Succeeded());
    {
      std::lock_guard<std::mutex> L(NodesMtx);
      CreatedNodes[I] = Node ? Node->getID() : CASID::getDenseMapTombstoneKey();
    }
  };

  auto Consumer = [&](unsigned I, ObjectStore *CAS) {
    std::optional<CASID> ID;
    while (!ID) {
      // Busy wait.
      std::lock_guard<std::mutex> L(NodesMtx);
      ID = CreatedNodes[I];
    }
    if (ID == CASID::getDenseMapTombstoneKey())
      // Producer failed; already reported.
      return;

    std::optional<ObjectProxy> Node;
    ASSERT_THAT_ERROR(CAS->getProxy(*ID).moveInto(Node), Succeeded());
    EXPECT_EQ(Node->getData(), Blobs[I]);
  };

  DefaultThreadPool Threads;
  for (unsigned I = 0; I < BlobCount; ++I) {
    Threads.async(Consumer, I, &Read1);
    Threads.async(Consumer, I, &Read2);
    Threads.async(Producer, I, &Write1);
    Threads.async(Producer, I, &Write2);
  }

  Threads.wait();
}

static void testBlobsParallel1(ObjectStore &CAS, uint64_t BlobSize) {
  SCOPED_TRACE(testBlobsParallel1);
  testBlobsParallel(CAS, CAS, CAS, CAS, BlobSize);
}

TEST_P(CASTest, BlobsParallel) {
  std::shared_ptr<ObjectStore> CAS = createObjectStore();
  uint64_t Size = 1ULL * 1024;
  ASSERT_NO_FATAL_FAILURE(testBlobsParallel1(*CAS, Size));
}

#ifdef EXPENSIVE_CHECKS
TEST_P(CASTest, BlobsBigParallel) {
  std::shared_ptr<ObjectStore> CAS = createObjectStore();
  // 100k is large enough to be standalone files in our on-disk cas.
  uint64_t Size = 100ULL * 1024;
  ASSERT_NO_FATAL_FAILURE(testBlobsParallel1(*CAS, Size));
}
#endif
