//===- ObjectStoreTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

#include "CASTestConfig.h"

using namespace llvm;
using namespace llvm::cas;

TEST_P(CASTest, PrintIDs) {
  std::unique_ptr<ObjectStore> CAS = createObjectStore();

  Optional<CASID> ID1, ID2;
  ASSERT_THAT_ERROR(CAS->createProxy(None, "1").moveInto(ID1), Succeeded());
  ASSERT_THAT_ERROR(CAS->createProxy(None, "2").moveInto(ID2), Succeeded());
  EXPECT_NE(ID1, ID2);
  std::string PrintedID1 = ID1->toString();
  std::string PrintedID2 = ID2->toString();
  EXPECT_NE(PrintedID1, PrintedID2);

  Optional<CASID> ParsedID1, ParsedID2;
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
    Optional<ObjectProxy> Blob;
    ASSERT_THAT_ERROR(CAS1->createProxy(None, Content).moveInto(Blob),
                      Succeeded());
    IDs.push_back(Blob->getID());

    // Check basic printing of IDs.
    EXPECT_EQ(IDs.back().toString(), IDs.back().toString());
    if (IDs.size() > 2)
      EXPECT_NE(IDs.front().toString(), IDs.back().toString());
  }

  // Check that the blobs give the same IDs later.
  for (int I = 0, E = IDs.size(); I != E; ++I) {
    Optional<ObjectProxy> Blob;
    ASSERT_THAT_ERROR(CAS1->createProxy(None, ContentStrings[I]).moveInto(Blob),
                      Succeeded());
    EXPECT_EQ(IDs[I], Blob->getID());
  }

  // Run validation on all CASIDs.
  for (int I = 0, E = IDs.size(); I != E; ++I)
    ASSERT_THAT_ERROR(CAS1->validate(IDs[I]), Succeeded());

  // Check that the blobs can be retrieved multiple times.
  for (int I = 0, E = IDs.size(); I != E; ++I) {
    for (int J = 0, JE = 3; J != JE; ++J) {
      Optional<ObjectProxy> Buffer;
      ASSERT_THAT_ERROR(CAS1->getProxy(IDs[I]).moveInto(Buffer), Succeeded());
      EXPECT_EQ(ContentStrings[I], Buffer->getData());
    }
  }

  // Confirm these blobs don't exist in a fresh CAS instance.
  std::unique_ptr<ObjectStore> CAS2 = createObjectStore();
  for (int I = 0, E = IDs.size(); I != E; ++I) {
    Optional<ObjectProxy> Proxy;
    EXPECT_THAT_ERROR(CAS2->getProxy(IDs[I]).moveInto(Proxy), Failed());
  }

  // Insert into the second CAS and confirm the IDs are stable. Getting them
  // should work now.
  for (int I = IDs.size(), E = 0; I != E; --I) {
    auto &ID = IDs[I - 1];
    auto &Content = ContentStrings[I - 1];
    Optional<ObjectProxy> Blob;
    ASSERT_THAT_ERROR(CAS2->createProxy(None, Content).moveInto(Blob),
                      Succeeded());
    EXPECT_EQ(ID, Blob->getID());

    Optional<ObjectProxy> Buffer;
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
    Optional<CASID> ID1;
    Optional<CASID> ID2;
    ASSERT_THAT_ERROR(CAS->createProxy(None, String1).moveInto(ID1),
                      Succeeded());
    ASSERT_THAT_ERROR(CAS->createProxy(None, String1).moveInto(ID2),
                      Succeeded());
    ASSERT_THAT_ERROR(CAS->validate(*ID1), Succeeded());
    ASSERT_THAT_ERROR(CAS->validate(*ID2), Succeeded());
    ASSERT_EQ(ID1, ID2);

    String1.append(String2);
    ASSERT_THAT_ERROR(CAS->createProxy(None, String2).moveInto(ID1),
                      Succeeded());
    ASSERT_THAT_ERROR(CAS->createProxy(None, String2).moveInto(ID2),
                      Succeeded());
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
    Optional<ObjectProxy> Blob;
    ASSERT_THAT_ERROR(CAS->createProxy(None, Data).moveInto(Blob), Succeeded());
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
    Optional<ObjectRef> Node;
    ASSERT_THAT_ERROR(
        CAS1->store(None, arrayRefFromStringRef<char>(Content)).moveInto(Node),
        Succeeded());
    Nodes.push_back(*Node);

    // Check basic printing of IDs.
    IDs.push_back(CAS1->getID(*Node));
    EXPECT_EQ(IDs.back().toString(), IDs.back().toString());
    EXPECT_EQ(Nodes.front(), Nodes.front());
    EXPECT_EQ(Nodes.back(), Nodes.back());
    EXPECT_EQ(IDs.front(), IDs.front());
    EXPECT_EQ(IDs.back(), IDs.back());
    if (Nodes.size() <= 1)
      continue;
    EXPECT_NE(Nodes.front(), Nodes.back());
    EXPECT_NE(IDs.front(), IDs.back());
  }

  // Check that the blobs give the same IDs later.
  for (int I = 0, E = IDs.size(); I != E; ++I) {
    Optional<ObjectRef> Node;
    ASSERT_THAT_ERROR(
        CAS1->store(None, arrayRefFromStringRef<char>(ContentStrings[I]))
            .moveInto(Node),
        Succeeded());
    EXPECT_EQ(IDs[I], CAS1->getID(*Node));
  }

  // Check that the blobs can be retrieved multiple times.
  for (int I = 0, E = IDs.size(); I != E; ++I) {
    for (int J = 0, JE = 3; J != JE; ++J) {
      Optional<ObjectProxy> Object;
      ASSERT_THAT_ERROR(CAS1->getProxy(IDs[I]).moveInto(Object), Succeeded());
      ASSERT_TRUE(Object);
      EXPECT_EQ(ContentStrings[I], Object->getData());
    }
  }

  // Confirm these blobs don't exist in a fresh CAS instance.
  std::unique_ptr<ObjectStore> CAS2 = createObjectStore();
  for (int I = 0, E = IDs.size(); I != E; ++I) {
    Optional<ObjectProxy> Object;
    EXPECT_THAT_ERROR(CAS2->getProxy(IDs[I]).moveInto(Object), Failed());
  }

  // Insert into the second CAS and confirm the IDs are stable. Getting them
  // should work now.
  for (int I = IDs.size(), E = 0; I != E; --I) {
    auto &ID = IDs[I - 1];
    auto &Content = ContentStrings[I - 1];
    Optional<ObjectRef> Node;
    ASSERT_THAT_ERROR(
        CAS2->store(None, arrayRefFromStringRef<char>(Content)).moveInto(Node),
        Succeeded());
    EXPECT_EQ(ID, CAS2->getID(*Node));

    Optional<ObjectProxy> Object;
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
      Optional<ObjectProxy> Node;
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
