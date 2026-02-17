//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/NamedValuesSchema.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::cas;

TEST(NamedValuesSchemaTest, Basic) {
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

  SmallVector<SmallVector<NamedValuesEntry, 0>, 0> FlatEntries = {
      {},
      {NamedValuesEntry("regular", Blob1)},
      {NamedValuesEntry("executable", Blob2)},
      {NamedValuesEntry("symlink", Blob3)},
      {
          NamedValuesEntry("various", Blob1),
          NamedValuesEntry("names", Blob1),
          NamedValuesEntry("that", Blob1),
          NamedValuesEntry("do", Blob1),
          NamedValuesEntry("not", Blob1),
          NamedValuesEntry("conflict", Blob1),
          NamedValuesEntry("but have spaces and...", Blob1),
          NamedValuesEntry("`~,!@#$%^&*()-+=[]{}\\<>'\"", Blob1),
      },
  };

  SmallVector<ObjectRef> FlatRefs;
  SmallVector<CASID> FlatIDs;
  NamedValuesSchema Schema1 = cantFail(NamedValuesSchema::create(*CAS1));

  for (ArrayRef<NamedValuesEntry> Entries : FlatEntries) {
    std::optional<NamedValuesProxy> H;
    ASSERT_THAT_ERROR(Schema1.construct(Entries).moveInto(H), Succeeded());
    FlatIDs.push_back(H->getID());
    FlatRefs.push_back(H->getRef());
  }

  // Confirm we get the same IDs the second time and that the trees can be
  // visited (the entries themselves will be checked later).
  for (int I = 0, E = FlatIDs.size(); I != E; ++I) {
    std::optional<NamedValuesProxy> H;
    ASSERT_THAT_ERROR(Schema1.construct(FlatEntries[I]).moveInto(H),
                      Succeeded());
    EXPECT_EQ(FlatRefs[I], CAS1->getReference(*H));
    std::optional<NamedValuesProxy> Value;
    ASSERT_THAT_ERROR(Schema1.load(*H).moveInto(Value), Succeeded());
    EXPECT_EQ(FlatEntries[I].size(), Value->size());

    size_t NumCalls = 0;
    EXPECT_THAT_ERROR(
        Value->forEachEntry([&NumCalls](const NamedValuesEntry &E) {
          ++NumCalls;
          return Error::success();
        }),
        Succeeded());
    EXPECT_EQ(FlatEntries[I].size(), NumCalls);
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
      NamedValuesSchema Schema = cantFail(NamedValuesSchema::create(*CAS));
      auto &ID = FlatIDs[I - 1];
      // Make a copy of the original entries and sort them.
      SmallVector<NamedValuesEntry> NewEntries;
      for (const NamedValuesEntry &Entry : FlatEntries[I - 1]) {
        std::optional<ObjectRef> NewRef =
            CAS->getReference(CAS1->getID(Entry.Ref));
        ASSERT_TRUE(NewRef);
        NewEntries.emplace_back(Entry.Name, *NewRef);
      }
      llvm::sort(NewEntries);

      // Confirm we get the same tree out of CAS2.
      {
        std::optional<NamedValuesProxy> Value;
        ASSERT_THAT_ERROR(Schema.construct(NewEntries).moveInto(Value),
                          Succeeded());
        EXPECT_EQ(ID, Value->getID());
      }

      // Check that the correct entries come back.
      std::optional<ObjectRef> Ref = CAS->getReference(ID);
      ASSERT_TRUE(Ref);
      std::optional<NamedValuesProxy> Value;
      ASSERT_THAT_ERROR(Schema.load(*Ref).moveInto(Value), Succeeded());
      for (int I = 0, E = NewEntries.size(); I != E; ++I)
        EXPECT_EQ(NewEntries[I], Value->get(I));
    }
  }

  // Create some nested trees.
  SmallVector<ObjectRef> Nested = FlatRefs;
  for (int I = 0, E = FlatEntries.size() * 3; I != E; ++I) {
    // Copy one of the flat entries and add some trees.
    auto OriginalEntries = ArrayRef(FlatEntries[I % FlatEntries.size()]);
    SmallVector<NamedValuesEntry> Entries(OriginalEntries.begin(),
                                          OriginalEntries.end());
    std::string Name = ("tree" + Twine(I)).str();
    Entries.emplace_back(
        Name, *CAS1->getReference(FlatIDs[(I + 4) % FlatIDs.size()]));

    std::optional<std::string> Name1, Name2;
    if (Nested.size() >= 2) {
      int Nested1 = I % Nested.size();
      int Nested2 = (I * 3 + 2) % Nested.size();
      if (Nested2 == Nested1)
        Nested2 = (Nested1 + 1) % Nested.size();
      ASSERT_NE(Nested1, Nested2);
      Name1.emplace(("tree" + Twine(I) + "-" + Twine(Nested1)).str());
      Name2.emplace(("tree" + Twine(I) + "-" + Twine(Nested2)).str());

      Entries.emplace_back(*Name1, Nested[I % Nested.size()]);
      Entries.emplace_back(*Name2, Nested[(I * 3 + 2) % Nested.size()]);
    }
    std::optional<CASID> ID;
    {
      std::optional<NamedValuesProxy> Value;
      ASSERT_THAT_ERROR(Schema1.construct(Entries).moveInto(Value),
                        Succeeded());
      ID = Value->getID();
    }

    llvm::sort(Entries);
    for (ObjectStore *CAS : {&*CAS1, &*CAS2}) {
      NamedValuesSchema Schema = cantFail(NamedValuesSchema::create(*CAS));

      // Make a copy of the original entries and sort them.
      SmallVector<NamedValuesEntry> NewEntries;
      for (const NamedValuesEntry &Entry : Entries) {
        std::optional<ObjectRef> NewRef =
            CAS->getReference(CAS1->getID(Entry.Ref));
        ASSERT_TRUE(NewRef);
        NewEntries.emplace_back(Entry.Name, *NewRef);
      }
      llvm::sort(NewEntries);

      std::optional<NamedValuesProxy> Value;
      ASSERT_THAT_ERROR(Schema.construct(NewEntries).moveInto(Value),
                        Succeeded());
      ASSERT_EQ(*ID, Value->getID());
      ASSERT_THAT_ERROR(CAS->validateObject(*ID), Succeeded());
      Value.reset();
      std::optional<ObjectRef> Ref = CAS->getReference(*ID);
      ASSERT_TRUE(Ref);
      ASSERT_THAT_ERROR(Schema.load(*Ref).moveInto(Value), Succeeded());
      for (int I = 0, E = NewEntries.size(); I != E; ++I)
        EXPECT_EQ(NewEntries[I], Value->get(I));
    }
  }
}

TEST(NamedValuesSchemaTest, Lookup) {
  std::unique_ptr<ObjectStore> CAS = createInMemoryCAS();
  std::optional<ObjectRef> Node;
  EXPECT_THAT_ERROR(CAS->storeFromString({}, "blob").moveInto(Node),
                    Succeeded());
  ObjectRef Blob = *Node;
  SmallVector<NamedValuesEntry> FlatEntries = {
      NamedValuesEntry("e", Blob), NamedValuesEntry("b", Blob),
      NamedValuesEntry("f", Blob), NamedValuesEntry("a", Blob),
      NamedValuesEntry("c", Blob), NamedValuesEntry("d", Blob),
  };
  std::optional<NamedValuesProxy> Value;
  NamedValuesSchema Schema = cantFail(NamedValuesSchema::create(*CAS));

  ASSERT_THAT_ERROR(Schema.construct(FlatEntries).moveInto(Value), Succeeded());

  ASSERT_EQ(Value->size(), (size_t)6);
  auto CheckEntry = [&](StringRef Name) {
    auto MaybeEntry = Value->lookup(Name);
    ASSERT_TRUE(MaybeEntry);
    ASSERT_EQ(MaybeEntry->Name, Name);
  };
  CheckEntry("a");
  CheckEntry("b");
  CheckEntry("c");
  CheckEntry("d");
  CheckEntry("e");
  CheckEntry("f");
  ASSERT_FALSE(Value->lookup("h"));
}

TEST(NamedValuesSchemaTest, Builder) {
  std::unique_ptr<ObjectStore> CAS = createInMemoryCAS();
  ObjectRef Blob = cantFail(CAS->storeFromString({}, ""));

  NamedValuesSchema::Builder Builder(*CAS);
  Builder.add("a", Blob);
  Builder.add("a", Blob);
  ASSERT_THAT_EXPECTED(Builder.build(), Failed());

  NamedValuesSchema::Builder Builder2(*CAS);
  Builder2.add("a", Blob);
  Builder2.add("b", Blob);
  ASSERT_THAT_EXPECTED(Builder2.build(), Succeeded());
}

TEST(NamedValuesSchemaTest, forEachEntry) {
  std::unique_ptr<ObjectStore> CAS = createInMemoryCAS();

  auto make = [&](StringRef Content) {
    return cantFail(CAS->storeFromString({}, Content));
  };

  std::pair<std::string, std::string> ExpectedEntries[] = {
      {"i", "blob2"},
      {"l", "blob1"},
      {"s", "blob3"},
      {"t", "blob4"},
  };

  NamedValuesSchema::Builder Builder(*CAS);
  for (auto &E : ExpectedEntries)
    Builder.add(E.first, make(E.second));

  std::optional<ObjectProxy> Root;
  ASSERT_THAT_ERROR(Builder.build().moveInto(Root), Succeeded());

  auto RemainingEntries = ArrayRef(ExpectedEntries);

  NamedValuesSchema Schema = cantFail(NamedValuesSchema::create(*CAS));
  std::optional<NamedValuesProxy> Loaded;
  ASSERT_THAT_ERROR(Schema.load(*Root).moveInto(Loaded), Succeeded());
  Error E = Loaded->forEachEntry([&](const NamedValuesEntry &Entry) -> Error {
    if (RemainingEntries.empty())
      return createStringError(inconvertibleErrorCode(),
                               "unexpected entry: '" + Entry.Name + "'");
    auto ExpectedEntry = RemainingEntries.front();
    RemainingEntries = RemainingEntries.drop_front();
    EXPECT_EQ(ExpectedEntry.first, Entry.Name);
    EXPECT_EQ(make(ExpectedEntry.second), Entry.Ref);
    return Error::success();
  });
  EXPECT_THAT_ERROR(std::move(E), Succeeded());
}
