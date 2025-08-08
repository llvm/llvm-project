//===- TreeSchemaTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/TreeSchema.h"
#include "llvm/CAS/HierarchicalTreeBuilder.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::cas;

TEST(TreeSchemaTest, Trees) {
  std::unique_ptr<ObjectStore> CAS1 = createInMemoryCAS();
  std::unique_ptr<ObjectStore> CAS2 = createInMemoryCAS();

  auto createBlobInBoth = [&](StringRef Content) {
    std::optional<ObjectRef> H1, H2;
    EXPECT_THAT_ERROR(CAS1->storeFromString({}, Content).moveInto(H1),
                      Succeeded());
    EXPECT_THAT_ERROR(CAS2->storeFromString({}, Content).moveInto(H2),
                      Succeeded());
    EXPECT_EQ(CAS1->getID(*H1), CAS2->getID(*H2));
    return *H1;
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
  TreeSchema Schema1(*CAS1);

  for (ArrayRef<NamedTreeEntry> Entries : FlatTreeEntries) {
    std::optional<TreeProxy> H;
    ASSERT_THAT_ERROR(Schema1.create(Entries).moveInto(H), Succeeded());
    FlatIDs.push_back(H->getID());
    FlatRefs.push_back(H->getRef());
  }

  // Confirm we get the same IDs the second time and that the trees can be
  // visited (the entries themselves will be checked later).
  for (int I = 0, E = FlatIDs.size(); I != E; ++I) {
    std::optional<TreeProxy> H;
    ASSERT_THAT_ERROR(Schema1.create(FlatTreeEntries[I]).moveInto(H),
                      Succeeded());
    EXPECT_EQ(FlatRefs[I], CAS1->getReference(*H));
    std::optional<TreeProxy> Tree;
    ASSERT_THAT_ERROR(TreeProxy::get(Schema1, *H).moveInto(Tree),
                      Succeeded());
    EXPECT_EQ(FlatTreeEntries[I].size(), Tree->size());

    size_t NumCalls = 0;
    EXPECT_THAT_ERROR(Tree->forEachEntry([&NumCalls](const NamedTreeEntry &E) {
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
    for (ObjectStore *CAS : {&*CAS1, &*CAS2}) {
      TreeSchema Schema(*CAS);
      auto &ID = FlatIDs[I - 1];
      // Make a copy of the original entries and sort them.
      SmallVector<NamedTreeEntry> NewEntries;
      for (const NamedTreeEntry &Entry : FlatTreeEntries[I - 1]) {
        std::optional<ObjectRef> NewRef =
            CAS->getReference(CAS1->getID(Entry.getRef()));
        ASSERT_TRUE(NewRef);
        NewEntries.emplace_back(*NewRef, Entry.getKind(), Entry.getName());
      }
      llvm::sort(NewEntries);

      // Confirm we get the same tree out of CAS2.
      {
        std::optional<TreeProxy> Tree;
        ASSERT_THAT_ERROR(Schema.create(NewEntries).moveInto(Tree),
                          Succeeded());
        EXPECT_EQ(ID, Tree->getID());
      }

      // Check that the correct entries come back.
      std::optional<ObjectRef> Ref = CAS->getReference(ID);
      ASSERT_TRUE(Ref);
      std::optional<TreeProxy> Tree;
      ASSERT_THAT_ERROR(Schema.load(*Ref).moveInto(Tree), Succeeded());
      for (int I = 0, E = NewEntries.size(); I != E; ++I)
        EXPECT_EQ(NewEntries[I], Tree->get(I));
    }
  }

  // Create some nested trees.
  SmallVector<ObjectRef> NestedTrees = FlatRefs;
  for (int I = 0, E = FlatTreeEntries.size() * 3; I != E; ++I) {
    // Copy one of the flat entries and add some trees.
    auto OriginalEntries =
        ArrayRef(FlatTreeEntries[I % FlatTreeEntries.size()]);
    SmallVector<NamedTreeEntry> Entries(OriginalEntries.begin(),
                                        OriginalEntries.end());
    std::string Name = ("tree" + Twine(I)).str();
    Entries.emplace_back(*CAS1->getReference(FlatIDs[(I + 4) % FlatIDs.size()]),
                         TreeEntry::Tree, Name);

    std::optional<std::string> Name1, Name2;
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
    std::optional<CASID> ID;
    {
      std::optional<TreeProxy> Tree;
      ASSERT_THAT_ERROR(Schema1.create(Entries).moveInto(Tree), Succeeded());
      ID = Tree->getID();
    }

    llvm::sort(Entries);
    for (ObjectStore *CAS : {&*CAS1, &*CAS2}) {
      // Make a copy of the original entries and sort them.
      SmallVector<NamedTreeEntry> NewEntries;
      for (const NamedTreeEntry &Entry : Entries) {
        std::optional<ObjectRef> NewRef =
            CAS->getReference(CAS1->getID(Entry.getRef()));
        ASSERT_TRUE(NewRef);
        NewEntries.emplace_back(*NewRef, Entry.getKind(), Entry.getName());
      }
      llvm::sort(NewEntries);

      TreeSchema Schema(*CAS);
      std::optional<TreeProxy> Tree;
      ASSERT_THAT_ERROR(Schema.create(NewEntries).moveInto(Tree),
                        Succeeded());
      ASSERT_EQ(*ID, Tree->getID());
      ASSERT_THAT_ERROR(CAS->validateObject(*ID), Succeeded());
      Tree.reset();
      std::optional<ObjectRef> Ref = CAS->getReference(*ID);
      ASSERT_TRUE(Ref);
      ASSERT_THAT_ERROR(Schema.load(*Ref).moveInto(Tree), Succeeded());
      for (int I = 0, E = NewEntries.size(); I != E; ++I)
        EXPECT_EQ(NewEntries[I], Tree->get(I));
    }
  }
}

TEST(TreeSchemaTest, Lookup) {
  std::unique_ptr<ObjectStore> CAS = createInMemoryCAS();
  std::optional<ObjectRef> Node;
  EXPECT_THAT_ERROR(CAS->storeFromString({}, "blob").moveInto(Node),
                    Succeeded());
  ObjectRef Blob = *Node;
  SmallVector<NamedTreeEntry> FlatTreeEntries = {
      NamedTreeEntry(Blob, TreeEntry::Regular, "e"),
      NamedTreeEntry(Blob, TreeEntry::Regular, "b"),
      NamedTreeEntry(Blob, TreeEntry::Regular, "f"),
      NamedTreeEntry(Blob, TreeEntry::Regular, "a"),
      NamedTreeEntry(Blob, TreeEntry::Regular, "c"),
      NamedTreeEntry(Blob, TreeEntry::Regular, "f"),
      NamedTreeEntry(Blob, TreeEntry::Regular, "d"),
  };
  std::optional<TreeProxy> Tree;
  TreeSchema Schema(*CAS);
  ASSERT_THAT_ERROR(Schema.create(FlatTreeEntries).moveInto(Tree),
                    Succeeded());
  ASSERT_EQ(Tree->size(), (size_t)6);
  auto CheckEntry = [&](StringRef Name) {
    auto MaybeEntry = Tree->lookup(Name);
    ASSERT_TRUE(MaybeEntry);
    ASSERT_EQ(MaybeEntry->getName(), Name);
  };
  CheckEntry("a");
  CheckEntry("b");
  CheckEntry("c");
  CheckEntry("d");
  CheckEntry("e");
  CheckEntry("f");
  ASSERT_FALSE(Tree->lookup("h"));
}

TEST(TreeSchemaTest, walkFileTreeRecursively) {
  std::unique_ptr<ObjectStore> CAS = createInMemoryCAS();

  auto make = [&](StringRef Content) {
    return cantFail(CAS->storeFromString({}, Content));
  };

  HierarchicalTreeBuilder Builder(sys::path::Style::posix);
  Builder.push(make("blob2"), TreeEntry::Regular, "/d2");
  Builder.push(make("blob1"), TreeEntry::Regular, "/t1/d1");
  Builder.push(make("blob3"), TreeEntry::Regular, "/t3/d3");
  Builder.push(make("blob1"), TreeEntry::Regular, "/t3/t1nested/d1");
  std::optional<ObjectProxy> Root;
  ASSERT_THAT_ERROR(Builder.create(*CAS).moveInto(Root), Succeeded());

  std::pair<std::string, bool> ExpectedEntries[] = {
      {"/", true},
      {"/d2", false},
      {"/t1", true},
      {"/t1/d1", false},
      {"/t3", true},
      {"/t3/d3", false},
      {"/t3/t1nested", true},
      {"/t3/t1nested/d1", false},
  };
  auto RemainingEntries = ArrayRef(ExpectedEntries);

  TreeSchema Schema(*CAS);
  Error E = Schema.walkFileTreeRecursively(
      *CAS, Root->getRef(),
      [&](const NamedTreeEntry &Entry, std::optional<TreeProxy> Tree) -> Error {
        if (RemainingEntries.empty())
          return createStringError(inconvertibleErrorCode(),
                                   "unexpected entry: '" + Entry.getName() +
                                       "'");
        auto ExpectedEntry = RemainingEntries.front();
        RemainingEntries = RemainingEntries.drop_front();
        EXPECT_EQ(ExpectedEntry.first, Entry.getName());
        EXPECT_EQ(ExpectedEntry.second, Tree.has_value());
        return Error::success();
      });
  EXPECT_THAT_ERROR(std::move(E), Succeeded());
}
