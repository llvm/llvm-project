//===------------------ FindTypesInMangledNameTest.cpp --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cstdlib>
#include <vector>
#include "llvm/Demangle/Demangle.h"
#include "gtest/gtest.h"

TEST(FindTypesInMangledNameTest, Test) {
  std::vector<const char *> Types;
  const char *Mangled = "_Z1fiv";
  EXPECT_FALSE(llvm::itaniumFindTypesInMangledName(
      Mangled, static_cast<void *>(&Types), [](void *Ty, const char *P) {
        static_cast<std::vector<const char *> *>(Ty)->push_back(P);
      }));
  EXPECT_EQ(Types.size(), size_t(2));
  EXPECT_EQ(Mangled + 4, Types.front());
  EXPECT_EQ(Mangled + 5, Types.back());

  EXPECT_TRUE(llvm::itaniumFindTypesInMangledName(
      "Not a mangled name!", nullptr, [](void *, const char *) {}));

  int TC = 0;
  EXPECT_FALSE(llvm::itaniumFindTypesInMangledName(
      "_Z1fPRic", static_cast<void *>(&TC),
      [](void *Ctx, const char *) { ++*static_cast<int *>(Ctx); }));
  EXPECT_EQ(TC, 4); // pointer, reference, int, char.
}
