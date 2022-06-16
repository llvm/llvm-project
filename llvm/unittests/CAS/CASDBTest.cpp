//===- CASDBTest.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/CASDB.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::cas;

struct TestingAndDir {
  std::unique_ptr<CASDB> DB;
  Optional<unittest::TempDir> Temp;
};

class CASDBTest
    : public testing::TestWithParam<std::function<TestingAndDir(int)>> {
protected:
  Optional<int> NextCASIndex;

  SmallVector<unittest::TempDir> Dirs;

  std::unique_ptr<CASDB> createCAS() {
    auto TD = GetParam()((*NextCASIndex)++);
    if (TD.Temp)
      Dirs.push_back(std::move(*TD.Temp));
    return std::move(TD.DB);
  }
  void SetUp() { NextCASIndex = 0; }
  void TearDown() {
    NextCASIndex = None;
    Dirs.clear();
  }
};

TEST_P(CASDBTest, PrintIDs) {
  std::unique_ptr<CASDB> CAS = createCAS();

  Optional<CASID> ID1, ID2;
  ASSERT_THAT_ERROR(CAS->createBlob("1").moveInto(ID1), Succeeded());
  ASSERT_THAT_ERROR(CAS->createBlob("2").moveInto(ID2), Succeeded());
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

TEST_P(CASDBTest, Blobs) {
  std::unique_ptr<CASDB> CAS1 = createCAS();
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
    Optional<BlobProxy> Blob;
    ASSERT_THAT_ERROR(CAS1->createBlob(Content).moveInto(Blob), Succeeded());
    IDs.push_back(Blob->getID());

    // Check basic printing of IDs.
    EXPECT_EQ(IDs.back().toString(), IDs.back().toString());
    if (IDs.size() > 2)
      EXPECT_NE(IDs.front().toString(), IDs.back().toString());
  }

  // Check that the blobs give the same IDs later.
  for (int I = 0, E = IDs.size(); I != E; ++I) {
    Optional<BlobProxy> Blob;
    ASSERT_THAT_ERROR(CAS1->createBlob(ContentStrings[I]).moveInto(Blob),
                      Succeeded());
    EXPECT_EQ(IDs[I], Blob->getID());
  }

  // Run validation on all CASIDs.
  for (int I = 0, E = IDs.size(); I != E; ++I)
    ASSERT_THAT_ERROR(CAS1->validateObject(IDs[I]), Succeeded());

  // Check that the blobs can be retrieved multiple times.
  for (int I = 0, E = IDs.size(); I != E; ++I) {
    for (int J = 0, JE = 3; J != JE; ++J) {
      Optional<BlobProxy> Buffer;
      ASSERT_THAT_ERROR(CAS1->getBlob(IDs[I]).moveInto(Buffer), Succeeded());
      EXPECT_EQ(ContentStrings[I], **Buffer);
    }
  }

  // Confirm these blobs don't exist in a fresh CAS instance.
  std::unique_ptr<CASDB> CAS2 = createCAS();
  for (int I = 0, E = IDs.size(); I != E; ++I)
    EXPECT_THAT_EXPECTED(CAS2->getBlob(IDs[I]), Failed());

  // Insert into the second CAS and confirm the IDs are stable. Getting them
  // should work now.
  for (int I = IDs.size(), E = 0; I != E; --I) {
    auto &ID = IDs[I - 1];
    auto &Content = ContentStrings[I - 1];
    Optional<BlobProxy> Blob;
    ASSERT_THAT_ERROR(CAS2->createBlob(Content).moveInto(Blob), Succeeded());
    EXPECT_EQ(ID, Blob->getID());

    Optional<BlobProxy> Buffer;
    ASSERT_THAT_ERROR(CAS2->getBlob(ID).moveInto(Buffer), Succeeded());
    EXPECT_EQ(Content, **Buffer);
  }
}

TEST_P(CASDBTest, BlobsBig) {
  // A little bit of validation that bigger blobs are okay. Climb up to 1MB.
  std::unique_ptr<CASDB> CAS = createCAS();
  SmallString<256> String1 = StringRef("a few words");
  SmallString<256> String2 = StringRef("others");
  while (String1.size() < 1024U * 1024U) {
    Optional<CASID> ID1;
    Optional<CASID> ID2;
    ASSERT_THAT_ERROR(CAS->createBlob(String1).moveInto(ID1), Succeeded());
    ASSERT_THAT_ERROR(CAS->createBlob(String1).moveInto(ID2), Succeeded());
    ASSERT_THAT_ERROR(CAS->validateObject(*ID1), Succeeded());
    ASSERT_THAT_ERROR(CAS->validateObject(*ID2), Succeeded());
    ASSERT_EQ(ID1, ID2);

    String1.append(String2);
    ASSERT_THAT_ERROR(CAS->createBlob(String2).moveInto(ID1), Succeeded());
    ASSERT_THAT_ERROR(CAS->createBlob(String2).moveInto(ID2), Succeeded());
    ASSERT_THAT_ERROR(CAS->validateObject(*ID1), Succeeded());
    ASSERT_THAT_ERROR(CAS->validateObject(*ID2), Succeeded());
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
    Optional<BlobProxy> Blob;
    ASSERT_THAT_ERROR(CAS->createBlob(Data).moveInto(Blob), Succeeded());
    ASSERT_EQ(Data, Blob->getData());
    ASSERT_EQ(0, Blob->getData().end()[0]);
  }
}

TEST_P(CASDBTest, Trees) {
  std::unique_ptr<CASDB> CAS1 = createCAS();
  std::unique_ptr<CASDB> CAS2 = createCAS();

  auto createBlobInBoth = [&](StringRef Content) {
    Optional<NodeHandle> H1, H2;
    EXPECT_THAT_ERROR(CAS1->storeNodeFromString(None, Content).moveInto(H1),
                      Succeeded());
    EXPECT_THAT_ERROR(CAS2->storeNodeFromString(None, Content).moveInto(H2),
                      Succeeded());
    EXPECT_EQ(CAS1->getObjectID(*H1), CAS2->getObjectID(*H2));
    return CAS1->getReference(*H1);
  };

  ObjectRef Blob1 = createBlobInBoth("blob1");
  ObjectRef Blob2 = createBlobInBoth("blob2");
  ObjectRef Blob3 = createBlobInBoth("blob3");

  SmallVector<SmallVector<NamedTreeEntry, 0>, 0> FlatTreeEntries = {
      {},
      {NamedTreeEntry(Blob1, TreeEntry::Regular, "regular")},
      {NamedTreeEntry(Blob2, TreeEntry::Executable, "executable")},
      {NamedTreeEntry(Blob3, TreeEntry::Symlink, "symlink")},
      {
          NamedTreeEntry(Blob1, TreeEntry::Regular, "various"),
          NamedTreeEntry(Blob1, TreeEntry::Regular, "names"),
          NamedTreeEntry(Blob1, TreeEntry::Regular, "that"),
          NamedTreeEntry(Blob1, TreeEntry::Regular, "do"),
          NamedTreeEntry(Blob1, TreeEntry::Regular, "not"),
          NamedTreeEntry(Blob1, TreeEntry::Regular, "conflict"),
          NamedTreeEntry(Blob1, TreeEntry::Regular, "but have spaces and..."),
          NamedTreeEntry(Blob1, TreeEntry::Regular,
                         "`~,!@#$%^&*()-+=[]{}\\<>'\""),
      },
  };

  SmallVector<ObjectRef> FlatRefs;
  SmallVector<CASID> FlatIDs;
  for (ArrayRef<NamedTreeEntry> Entries : FlatTreeEntries) {
    Optional<TreeHandle> H;
    ASSERT_THAT_ERROR(CAS1->storeTree(Entries).moveInto(H), Succeeded());
    FlatIDs.push_back(CAS1->getObjectID(*H));
    FlatRefs.push_back(CAS1->getReference(*H));
  }

  // Confirm we get the same IDs the second time and that the trees can be
  // visited (the entries themselves will be checked later).
  for (int I = 0, E = FlatIDs.size(); I != E; ++I) {
    Optional<TreeHandle> H;
    ASSERT_THAT_ERROR(CAS1->storeTree(FlatTreeEntries[I]).moveInto(H),
                      Succeeded());
    EXPECT_EQ(FlatRefs[I], CAS1->getReference(*H));
    TreeProxy Tree = TreeProxy::load(*CAS1, *H);
    EXPECT_EQ(FlatTreeEntries[I].size(), Tree.size());

    size_t NumCalls = 0;
    EXPECT_THAT_ERROR(Tree.forEachEntry([&NumCalls](const NamedTreeEntry &E) {
      ++NumCalls;
      return Error::success();
    }),
                      Succeeded());
    EXPECT_EQ(FlatTreeEntries[I].size(), NumCalls);
  }

  // Run validation.
  for (int I = 1, E = FlatIDs.size(); I != E; ++I)
    ASSERT_THAT_ERROR(CAS1->validateObject(FlatIDs[I]), Succeeded());

  // Confirm these trees don't exist in a fresh CAS instance. Skip the first
  // tree, which is empty and could be implicitly in some CAS.
  for (int I = 1, E = FlatIDs.size(); I != E; ++I)
    EXPECT_FALSE(CAS2->getReference(FlatIDs[I]));

  // Insert into the other CAS and confirm the IDs are stable.
  for (int I = FlatIDs.size(), E = 0; I != E; --I) {
    for (CASDB *CAS : {&*CAS1, &*CAS2}) {
      auto &ID = FlatIDs[I - 1];
      // Make a copy of the original entries and sort them.
      SmallVector<NamedTreeEntry> NewEntries;
      for (const NamedTreeEntry &Entry : FlatTreeEntries[I - 1]) {
        Optional<ObjectRef> NewRef =
            CAS->getReference(CAS1->getObjectID(Entry.getRef()));
        ASSERT_TRUE(NewRef);
        NewEntries.emplace_back(*NewRef, Entry.getKind(), Entry.getName());
      }
      llvm::sort(NewEntries);

      // Confirm we get the same tree out of CAS2.
      {
        Optional<TreeProxy> Tree;
        ASSERT_THAT_ERROR(CAS->createTree(NewEntries).moveInto(Tree),
                          Succeeded());
        EXPECT_EQ(ID, Tree->getID());
      }

      // Check that the correct entries come back.
      Optional<TreeProxy> Tree;
      ASSERT_THAT_ERROR(CAS->getTree(ID).moveInto(Tree), Succeeded());
      for (int I = 0, E = NewEntries.size(); I != E; ++I)
        EXPECT_EQ(NewEntries[I], Tree->get(I));
    }
  }

  // Create some nested trees.
  SmallVector<ObjectRef> NestedTrees = FlatRefs;
  for (int I = 0, E = FlatTreeEntries.size() * 3; I != E; ++I) {
    // Copy one of the flat entries and add some trees.
    auto OriginalEntries =
        makeArrayRef(FlatTreeEntries[I % FlatTreeEntries.size()]);
    SmallVector<NamedTreeEntry> Entries(OriginalEntries.begin(),
                                        OriginalEntries.end());
    std::string Name = ("tree" + Twine(I)).str();
    Entries.emplace_back(*CAS1->getReference(FlatIDs[(I + 4) % FlatIDs.size()]),
                         TreeEntry::Tree, Name);

    Optional<std::string> Name1, Name2;
    if (NestedTrees.size() >= 2) {
      int Nested1 = I % NestedTrees.size();
      int Nested2 = (I * 3 + 2) % NestedTrees.size();
      if (Nested2 == Nested1)
        Nested2 = (Nested1 + 1) % NestedTrees.size();
      ASSERT_NE(Nested1, Nested2);
      Name1.emplace(("tree" + Twine(I) + "-" + Twine(Nested1)).str());
      Name2.emplace(("tree" + Twine(I) + "-" + Twine(Nested2)).str());

      Entries.emplace_back(NestedTrees[I % NestedTrees.size()], TreeEntry::Tree,
                           *Name1);
      Entries.emplace_back(NestedTrees[(I * 3 + 2) % NestedTrees.size()],
                           TreeEntry::Tree, *Name2);
    }
    Optional<CASID> ID;
    {
      Optional<TreeProxy> Tree;
      ASSERT_THAT_ERROR(CAS1->createTree(Entries).moveInto(Tree), Succeeded());
      ID = Tree->getID();
    }

    llvm::sort(Entries);
    for (CASDB *CAS : {&*CAS1, &*CAS2}) {
      // Make a copy of the original entries and sort them.
      SmallVector<NamedTreeEntry> NewEntries;
      for (const NamedTreeEntry &Entry : Entries) {
        Optional<ObjectRef> NewRef =
            CAS->getReference(CAS1->getObjectID(Entry.getRef()));
        ASSERT_TRUE(NewRef);
        NewEntries.emplace_back(*NewRef, Entry.getKind(), Entry.getName());
      }
      llvm::sort(NewEntries);

      Optional<TreeProxy> Tree;
      ASSERT_THAT_ERROR(CAS->createTree(NewEntries).moveInto(Tree),
                        Succeeded());
      ASSERT_EQ(*ID, Tree->getID());
      ASSERT_THAT_ERROR(CAS->validateObject(*ID), Succeeded());
      Tree.reset();
      ASSERT_THAT_ERROR(CAS->getTree(*ID).moveInto(Tree), Succeeded());
      for (int I = 0, E = NewEntries.size(); I != E; ++I)
        EXPECT_EQ(NewEntries[I], Tree->get(I));
    }
  }
}

TEST_P(CASDBTest, LeafNodes) {
  std::unique_ptr<CASDB> CAS1 = createCAS();
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

  SmallVector<NodeHandle> Nodes;
  SmallVector<CASID> IDs;
  for (StringRef Content : ContentStrings) {
    // Use StringRef::str() to create a temporary std::string. This could cause
    // problems if the CAS is storing references to the input string instead of
    // copying it.
    Optional<NodeHandle> Node;
    ASSERT_THAT_ERROR(
        CAS1->storeNode(None, arrayRefFromStringRef<char>(Content))
            .moveInto(Node),
        Succeeded());
    Nodes.push_back(*Node);

    // Check basic printing of IDs.
    IDs.push_back(CAS1->getObjectID(*Node));
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
    Optional<NodeHandle> Node;
    ASSERT_THAT_ERROR(
        CAS1->storeNode(None, arrayRefFromStringRef<char>(ContentStrings[I]))
            .moveInto(Node),
        Succeeded());
    EXPECT_EQ(IDs[I], CAS1->getObjectID(*Node));
  }

  // Check that the blobs can be retrieved multiple times.
  for (int I = 0, E = IDs.size(); I != E; ++I) {
    for (int J = 0, JE = 3; J != JE; ++J) {
      Optional<AnyObjectHandle> Object;
      ASSERT_THAT_ERROR(CAS1->loadObject(IDs[I]).moveInto(Object), Succeeded());
      ASSERT_TRUE(Object);
      Optional<NodeHandle> Node = Object->dyn_cast<NodeHandle>();
      ASSERT_TRUE(Node);
      EXPECT_EQ(ContentStrings[I], CAS1->getDataString(*Node));
    }
  }

  // Confirm these blobs don't exist in a fresh CAS instance.
  std::unique_ptr<CASDB> CAS2 = createCAS();
  for (int I = 0, E = IDs.size(); I != E; ++I) {
    Optional<AnyObjectHandle> Object;
    EXPECT_THAT_EXPECTED(CAS2->loadObject(IDs[I]), Succeeded());
    EXPECT_FALSE(Object);
  }

  // Insert into the second CAS and confirm the IDs are stable. Getting them
  // should work now.
  for (int I = IDs.size(), E = 0; I != E; --I) {
    auto &ID = IDs[I - 1];
    auto &Content = ContentStrings[I - 1];
    Optional<NodeHandle> Node;
    ASSERT_THAT_ERROR(
        CAS2->storeNode(None, arrayRefFromStringRef<char>(Content))
            .moveInto(Node),
        Succeeded());
    EXPECT_EQ(ID, CAS2->getObjectID(*Node));

    Optional<AnyObjectHandle> Object;
    ASSERT_THAT_ERROR(CAS2->loadObject(ID).moveInto(Object), Succeeded());
    ASSERT_TRUE(Object);
    Node = Object->dyn_cast<NodeHandle>();
    ASSERT_TRUE(Node);
    EXPECT_EQ(Content, CAS2->getDataString(*Node));
  }
}

TEST_P(CASDBTest, NodesBig) {
  std::unique_ptr<CASDB> CAS = createCAS();

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

  SmallVector<CASID, 4> CreatedNodes;
  // Avoid checking every size because this is an expensive test. Just check
  // for data that is 8B-word-aligned, and one less. Also appending the created
  // nodes as the references in the next block to check references are created
  // correctly.
  for (size_t Size = SizeB; Size < SizeE; Size += WordSize) {
    for (bool IsAligned : {false, true}) {
      StringRef Data(Storage.data(), Size - (IsAligned ? 0 : 1));
      Optional<NodeProxy> Node;
      ASSERT_THAT_ERROR(CAS->createNode(CreatedNodes, Data).moveInto(Node),
                        Succeeded());
      ASSERT_EQ(Data, Node->getData());
      ASSERT_EQ(0, Node->getData().end()[0]);
      ASSERT_EQ(Node->getNumReferences(), CreatedNodes.size());
      CreatedNodes.emplace_back(Node->getID());
    }
  }

  for (auto ID: CreatedNodes)
    ASSERT_THAT_ERROR(CAS->validateObject(ID), Succeeded());
}

INSTANTIATE_TEST_SUITE_P(InMemoryCAS, CASDBTest, ::testing::Values([](int) {
                           return TestingAndDir{createInMemoryCAS(), None};
                         }));
static TestingAndDir createOnDisk(int I) {
  unittest::TempDir Temp("on-disk-cas", /*Unique=*/true);
  std::unique_ptr<CASDB> CAS;
  EXPECT_THAT_ERROR(createOnDiskCAS(Temp.path()).moveInto(CAS), Succeeded());
  return TestingAndDir{std::move(CAS), std::move(Temp)};
}
INSTANTIATE_TEST_SUITE_P(OnDiskCAS, CASDBTest, ::testing::Values(createOnDisk));
