//===- HierarchicalTreeBuilderTest.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/HierarchicalTreeBuilder.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/Path.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <memory>

using namespace llvm;
using namespace llvm::cas;

static std::unique_ptr<MemoryBuffer> getBufferForName(ObjectStore &CAS,
                                                      TreeSchema &Tree,
                                                      ObjectRef Root,
                                                      StringRef Name) {
  std::unique_ptr<MemoryBuffer> Buffer = nullptr;
  StringRef Filename = sys::path::filename(Name, sys::path::Style::posix);
  StringRef Dirname = sys::path::parent_path(Name, sys::path::Style::posix);
  auto Err = Tree.walkFileTreeRecursively(
      CAS, Root,
      [&](const NamedTreeEntry &Entry,
          std::optional<TreeProxy> Proxy) -> Error {
        if (Proxy && Entry.getName() == Dirname) {
          if (auto File = Proxy->lookup(Filename)) {
            auto Ref = File->getRef();
            auto Loaded = CAS.getProxy(Ref);
            if (!Loaded)
              return Loaded.takeError();
            Buffer = Loaded->getMemoryBuffer();
          }
        }
        return Error::success();
      });
  EXPECT_THAT_ERROR(std::move(Err), Succeeded());
  return Buffer;
}

TEST(HierarchicalTreeBuilderTest, Flat) {
  std::unique_ptr<ObjectStore> CAS = createInMemoryCAS();

  auto make = [&](StringRef Content) {
    return *expectedToOptional(CAS->storeFromString(std::nullopt, Content));
  };

  HierarchicalTreeBuilder Builder;
  Builder.push(make("1"), TreeEntry::Regular, "/file1");
  Builder.push(make("1"), TreeEntry::Regular, "/1");
  Builder.push(make("2"), TreeEntry::Regular, "/2");
  std::optional<ObjectProxy> Root;
  ASSERT_THAT_ERROR(Builder.create(*CAS).moveInto(Root), Succeeded());

  TreeSchema Tree(*CAS);
  ASSERT_TRUE(Tree.isNode(*Root));

  std::unique_ptr<MemoryBuffer> F1 =
      getBufferForName(*CAS, Tree, Root->getRef(), "/1");
  std::unique_ptr<MemoryBuffer> F2 =
      getBufferForName(*CAS, Tree, Root->getRef(), "/2");
  std::unique_ptr<MemoryBuffer> Ffile1 =
      getBufferForName(*CAS, Tree, Root->getRef(), "/file1");
  ASSERT_TRUE(Ffile1);
  ASSERT_TRUE(F1);
  ASSERT_TRUE(F2);
  EXPECT_EQ("1", F1->getBuffer());
  EXPECT_EQ("2", F2->getBuffer());
  EXPECT_EQ("1", Ffile1->getBuffer());
}

TEST(HierarchicalTreeBuilderTest, Nested) {
  std::unique_ptr<ObjectStore> CAS = createInMemoryCAS();

  auto make = [&](StringRef Content) {
    return *expectedToOptional(CAS->storeFromString(std::nullopt, Content));
  };

  HierarchicalTreeBuilder Builder;
  Builder.push(make("blob2"), TreeEntry::Regular, "/d2");
  Builder.push(make("blob1"), TreeEntry::Regular, "/t1/d1");
  Builder.push(make("blob3"), TreeEntry::Regular, "/t3/d3");
  Builder.push(make("blob1"), TreeEntry::Regular, "/t3/t1nested/d1");
  Builder.push(make("blob1"), TreeEntry::Regular, "/t3/t2/d1also");
  Builder.push(make("blob2"), TreeEntry::Regular, "/t3/t2/d2");
  std::optional<ObjectProxy> Root;
  ASSERT_THAT_ERROR(Builder.create(*CAS).moveInto(Root), Succeeded());

  TreeSchema Tree(*CAS);
  ASSERT_TRUE(Tree.isNode(*Root));

  std::unique_ptr<MemoryBuffer> F1 =
      getBufferForName(*CAS, Tree, Root->getRef(), "/1");
  std::unique_ptr<MemoryBuffer> T1D1 =
      getBufferForName(*CAS, Tree, Root->getRef(), "/t1/d1");
  std::unique_ptr<MemoryBuffer> T1NestedD1 =
      getBufferForName(*CAS, Tree, Root->getRef(), "/t3/t1nested/d1");
  std::unique_ptr<MemoryBuffer> T3T2D1Also =
      getBufferForName(*CAS, Tree, Root->getRef(), "/t3/t2/d1also");
  std::unique_ptr<MemoryBuffer> T3TD3 =
      getBufferForName(*CAS, Tree, Root->getRef(), "/t3/d3");
  ASSERT_TRUE(T1D1);
  ASSERT_TRUE(T1NestedD1);
  ASSERT_TRUE(T3T2D1Also);
  ASSERT_TRUE(T3TD3);

  EXPECT_EQ("blob1", T1D1->getBuffer());
  EXPECT_EQ("blob1", T1NestedD1->getBuffer());
  EXPECT_EQ("blob1", T3T2D1Also->getBuffer());
  EXPECT_EQ("blob3", T3TD3->getBuffer());
}

TEST(HierarchicalTreeBuilderTest, MergeDirectories) {
  std::unique_ptr<ObjectStore> CAS = createInMemoryCAS();

  auto make = [&](StringRef Content) {
    return *expectedToOptional(CAS->storeFromString(std::nullopt, Content));
  };

  auto createRoot = [&](StringRef Blob, StringRef Path,
                        std::optional<ObjectRef> &Root) {
    HierarchicalTreeBuilder Builder;
    Builder.push(make(Blob), TreeEntry::Regular, Path);

    std::optional<ObjectProxy> H;
    ASSERT_THAT_ERROR(Builder.create(*CAS).moveInto(H), Succeeded());
    Root = CAS->getReference(*H);
  };

  std::optional<ObjectRef> Root1;
  createRoot("blob1", "/t1/d1", Root1);
  std::optional<ObjectRef> Root2;
  createRoot("blob2", "/t1/d2", Root2);
  std::optional<ObjectRef> Root3;
  createRoot("blob3", "/t1/nested/d1", Root3);

  HierarchicalTreeBuilder Builder;
  Builder.pushTreeContent(*Root1, "/");
  Builder.pushTreeContent(*Root2, "");
  Builder.pushTreeContent(*Root3, "/");
  Builder.pushTreeContent(*Root1, "");
  Builder.pushTreeContent(*Root1, "other1/nest");
  std::optional<ObjectProxy> Root;
  ASSERT_THAT_ERROR(Builder.create(*CAS).moveInto(Root), Succeeded());

  TreeSchema Tree(*CAS);
  ASSERT_TRUE(Tree.isNode(*Root));

  std::unique_ptr<MemoryBuffer> T1D1 =
      getBufferForName(*CAS, Tree, Root->getRef(), "/t1/d1");
  std::unique_ptr<MemoryBuffer> T1D2 =
      getBufferForName(*CAS, Tree, Root->getRef(), "/t1/d2");
  std::unique_ptr<MemoryBuffer> T1NestedD1 =
      getBufferForName(*CAS, Tree, Root->getRef(), "/t1/nested/d1");
  std::unique_ptr<MemoryBuffer> OtherT1D1 =
      getBufferForName(*CAS, Tree, Root->getRef(), "/other1/nest/t1/d1");
  ASSERT_TRUE(T1D1);
  ASSERT_TRUE(T1D2);
  ASSERT_TRUE(T1NestedD1);
  ASSERT_TRUE(OtherT1D1);

  EXPECT_EQ("blob1", T1D1->getBuffer());
  EXPECT_EQ("blob2", T1D2->getBuffer());
  EXPECT_EQ("blob3", T1NestedD1->getBuffer());
  EXPECT_EQ("blob1", OtherT1D1->getBuffer());
}

TEST(HierarchicalTreeBuilderTest, MergeDirectoriesConflict) {
  std::unique_ptr<ObjectStore> CAS = createInMemoryCAS();

  auto make = [&](StringRef Content) {
    return *expectedToOptional(CAS->storeFromString(std::nullopt, Content));
  };

  auto createRoot = [&](StringRef Blob, StringRef Path,
                        std::optional<ObjectProxy> &Root) {
    HierarchicalTreeBuilder Builder;
    Builder.push(make(Blob), TreeEntry::Regular, Path);
    ASSERT_THAT_ERROR(Builder.create(*CAS).moveInto(Root), Succeeded());
  };

  std::optional<ObjectProxy> Root1;
  createRoot("blob1", "/t1/d1", Root1);
  std::optional<ObjectProxy> Root2;
  createRoot("blob2", "/t1/d1", Root2);
  std::optional<ObjectProxy> Root3;
  createRoot("blob3", "/t1/d1/nested", Root3);

  {
    HierarchicalTreeBuilder Builder;
    Builder.pushTreeContent(Root1->getRef(), "");
    Builder.pushTreeContent(Root2->getRef(), "");
    std::optional<ObjectProxy> Root;
    EXPECT_THAT_ERROR(
        Builder.create(*CAS).moveInto(Root),
        FailedWithMessage("duplicate path '/t1/d1' with different ID"));
  }
  {
    HierarchicalTreeBuilder Builder;
    Builder.pushTreeContent(Root1->getRef(), "");
    Builder.pushTreeContent(Root3->getRef(), "");
    std::optional<ObjectProxy> Root;
    EXPECT_THAT_ERROR(Builder.create(*CAS).moveInto(Root),
                      FailedWithMessage("duplicate path '/t1/d1'"));
  }
}
