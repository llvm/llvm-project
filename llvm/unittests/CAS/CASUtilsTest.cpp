//===- CASUtilsTest.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/CASDB.h"
#include "llvm/CAS/HierarchicalTreeBuilder.h"
#include "llvm/CAS/Utils.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::cas;

TEST(CASUtilsTest, walkFileTreeRecursively) {
  std::unique_ptr<CASDB> CAS = createInMemoryCAS();

  auto make = [&](StringRef Content) {
    return CAS->getReference(cantFail(CAS->storeNodeFromString(None, Content)));
  };

  HierarchicalTreeBuilder Builder;
  Builder.push(make("blob2"), TreeEntry::Regular, "/d2");
  Builder.push(make("blob1"), TreeEntry::Regular, "/t1/d1");
  Builder.push(make("blob3"), TreeEntry::Regular, "/t3/d3");
  Builder.push(make("blob1"), TreeEntry::Regular, "/t3/t1nested/d1");
  Optional<TreeHandle> Root;
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
  auto RemainingEntries = makeArrayRef(ExpectedEntries);

  Error E = walkFileTreeRecursively(
      *CAS, *Root,
      [&](const NamedTreeEntry &Entry, Optional<TreeProxy> Tree) -> Error {
        if (RemainingEntries.empty())
          return createStringError(inconvertibleErrorCode(),
                                   "unexpected entry: '" + Entry.getName() +
                                       "'");
        auto ExpectedEntry = RemainingEntries.front();
        RemainingEntries = RemainingEntries.drop_front();
        EXPECT_EQ(ExpectedEntry.first, Entry.getName());
        EXPECT_EQ(ExpectedEntry.second, Tree.hasValue());
        return Error::success();
      });
  EXPECT_THAT_ERROR(std::move(E), Succeeded());
}
