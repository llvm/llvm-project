//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypePool.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Parallel.h"
#include "gtest/gtest.h"
#include <limits>
#include <string>
#include <vector>

using namespace llvm;
using namespace llvm::dwarf_linker::parallel;

static void appendPart(TypeKey &Key, StringRef Text) { Key += Text; }

static void appendPart(TypeKey &Key, const TypeEntry *Ref) {
  Key.appendRef(Ref);
}

/// \returns the concatenation of text runs and references in \p Parts.
template <typename... PartTys> static std::string makeKey(PartTys... Parts) {
  TypeKey Key;
  (appendPart(Key, Parts), ...);
  return Key.str().str();
}

static int sign(int Value) { return (Value > 0) - (Value < 0); }

static std::vector<std::string> getChildKeys(TypeEntry *Entry) {
  std::vector<std::string> Keys;
  Entry->getValue().load()->Children.forEach(
      [&](TypeEntry *Child) { Keys.push_back(getExpandedTypeKey(*Child)); });
  return Keys;
}

namespace {

TEST(TypePoolTest, TakePart) {
  TypePool Pool;
  TypeEntry *Int = Pool.insert("{0}int");
  std::string Key = makeKey("{C}", "x", Int, Int, "y");
  StringRef Rest = Key;

  TypeKey::Part Text = TypeKey::takePart(Rest);
  EXPECT_EQ(Text.Text, "{C}x");
  EXPECT_EQ(Text.Ref, nullptr);
  for (int I = 0; I < 2; ++I) {
    TypeKey::Part Ref = TypeKey::takePart(Rest);
    EXPECT_TRUE(Ref.Text.empty());
    EXPECT_EQ(Ref.Ref, Int);
  }
  EXPECT_EQ(TypeKey::takePart(Rest).Text, "y");
  EXPECT_TRUE(Rest.empty());
}

TEST(TypePoolTest, GetExpandedTypeKey) {
  TypePool Pool;
  TypeEntry *Int = Pool.insert("{0}int");
  TypeEntry *IntPtr = Pool.insert(makeKey("{C}", Int));
  TypeEntry *NS = Pool.insert("{1}ns");
  TypeEntry *Foo = Pool.insert(makeKey(NS, ".{8}Foo<", Int, ", ", IntPtr, ">"));
  TypeEntry *Empty = Pool.insert("");

  EXPECT_EQ(getExpandedTypeKey(*Int), "{0}int");
  EXPECT_EQ(getExpandedTypeKey(*IntPtr), "{C}{0}int");
  EXPECT_EQ(getExpandedTypeKey(*Foo), "{1}ns.{8}Foo<{0}int, {C}{0}int>");
  EXPECT_EQ(getExpandedTypeKey(*Pool.insert(makeKey("a", Empty, "b"))), "ab");
  EXPECT_EQ(getExpandedTypeKey(*Pool.getRoot()), "");
}

TEST(TypePoolTest, CompareTypeKeysWithoutRefs) {
  std::pair<StringRef, StringRef> Pairs[] = {{"", ""},       {"", "a"},
                                             {"a", "b"},     {"ab", "a"},
                                             {"abc", "abd"}, {"a\xff", "ab"}};
  for (auto [LHS, RHS] : Pairs)
    EXPECT_EQ(sign(compareTypeKeys(LHS, RHS)), sign(LHS.compare(RHS)))
        << LHS << " vs " << RHS;
}

// Keys order as if every reference were expanded, and the order is total.
TEST(TypePoolTest, CompareTypeKeysAsIfExpanded) {
  TypePool Pool;
  TypeEntry *A = Pool.insert("a");
  TypeEntry *B = Pool.insert("b");
  TypeEntry *AB = Pool.insert("ab");
  TypeEntry *AC = Pool.insert("ac");
  TypeEntry *High = Pool.insert("\xff");
  TypeEntry *Empty = Pool.insert("");
  TypeEntry *Nested = Pool.insert(makeKey(A, B));
  TypeEntry *Deep = Pool.insert(makeKey("x", Nested, "y"));

  std::string Keys[] = {"",
                        "a",
                        "ab",
                        "abc",
                        "ac",
                        "xab",
                        "xaby",
                        "xac",
                        "\xff",
                        makeKey(A),
                        makeKey(Empty),
                        makeKey(A, B),
                        makeKey(A, "c"),
                        makeKey(AB, "c"),
                        makeKey(AB, "z"),
                        makeKey(AB, AC),
                        makeKey("x", AB),
                        makeKey("x", AC),
                        makeKey("x", AB, "y"),
                        makeKey("x", Nested),
                        makeKey("x", Empty, "a", B),
                        makeKey(Deep),
                        makeKey(Nested, AC),
                        makeKey(High),
                        makeKey(A, High)};
  SmallVector<TypeEntry *> Entries;
  for (StringRef Key : Keys)
    Entries.push_back(Pool.insert(Key));

  auto Compare = [](TypeEntry *LHS, TypeEntry *RHS) {
    return sign(compareTypeKeys(LHS->getKey(), RHS->getKey()));
  };
  for (TypeEntry *L : Entries) {
    for (TypeEntry *R : Entries) {
      std::string LExpanded = getExpandedTypeKey(*L);
      std::string RExpanded = getExpandedTypeKey(*R);
      SCOPED_TRACE(LExpanded + " vs " + RExpanded);
      EXPECT_EQ(Compare(L, R), -Compare(R, L));
      EXPECT_EQ(Compare(L, R) == 0, L == R);
      if (LExpanded != RExpanded)
        EXPECT_EQ(Compare(L, R), sign(StringRef(LExpanded).compare(RExpanded)));
    }
  }
  for (TypeEntry *L : Entries)
    for (TypeEntry *M : Entries)
      for (TypeEntry *R : Entries)
        if (Compare(L, M) < 0 && Compare(M, R) < 0)
          EXPECT_LT(Compare(L, R), 0);
}

// Keys that expand equally order by structure, never by entry address.
TEST(TypePoolTest, CompareTypeKeysTieBreak) {
  for (bool Reverse : {false, true}) {
    SCOPED_TRACE(Reverse ? "reverse" : "forward");
    TypePool Pool;
    auto InsertPair = [&](StringRef First, StringRef Second) {
      if (!Reverse) {
        TypeEntry *FirstEntry = Pool.insert(First);
        return std::make_pair(FirstEntry, Pool.insert(Second));
      }
      TypeEntry *SecondEntry = Pool.insert(Second);
      return std::make_pair(Pool.insert(First), SecondEntry);
    };
    auto [B, BC] = InsertPair("b", "bc");
    auto [P, Q] = InsertPair(makeKey("a", B), "ab");
    TypeEntry *C = Pool.insert("c");
    TypeEntry *Int = Pool.insert("{0}int");
    TypeEntry *Empty = Pool.insert("");

    auto ExpectOrdered = [&](StringRef LHS, StringRef RHS) {
      ASSERT_EQ(getExpandedTypeKey(*Pool.insert(LHS)),
                getExpandedTypeKey(*Pool.insert(RHS)));
      EXPECT_LT(compareTypeKeys(LHS, RHS), 0);
      EXPECT_GT(compareTypeKeys(RHS, LHS), 0);
    };
    // A reference sorts before text.
    ExpectOrdered(makeKey("{H}T", Int), "{H}T{0}int");
    ExpectOrdered(makeKey(P, "c"), makeKey("a", BC));
    // A shorter text run sorts first.
    ExpectOrdered(makeKey("a", BC), makeKey("ab", C));
    // Distinct references order by their own structure.
    ExpectOrdered(makeKey(P), makeKey(Q));
    ExpectOrdered(makeKey(Int, "x", B, "c"), makeKey(Int, "x", BC));
    // A key that ends first sorts first.
    ExpectOrdered("", makeKey(Empty));
  }
}

TEST(TypePoolTest, Root) {
  TypePool Pool;
  TypeEntry *Root = Pool.getRoot();
  EXPECT_EQ(Root->getKey(), "");
  ASSERT_NE(Root->getValue().load(), nullptr);
  EXPECT_TRUE(Root->getValue().load()->Children.empty());
}

TEST(TypePoolTest, Insert) {
  TypePool Pool;
  TypeEntry *Int = Pool.insert("{0}int");
  EXPECT_EQ(Int->getKey(), "{0}int");
  EXPECT_EQ(Int->getValue().load(), nullptr);
  EXPECT_EQ(Pool.insert("{0}int"), Int);
  TypeEntry *Long = Pool.insert("{0}long");
  EXPECT_NE(Long, Int);

  std::string IntPtrKey = makeKey("{C}", Int);
  TypeEntry *IntPtr = Pool.insert(IntPtrKey);
  EXPECT_EQ(IntPtr->getKey(), IntPtrKey);
  EXPECT_EQ(Pool.insert(IntPtrKey), IntPtr);
  // Truncating keys at the reference marker would conflate these.
  EXPECT_NE(Pool.insert(makeKey("{C}", Long)), IntPtr);
}

TEST(TypePoolTest, InsertParallel) {
  TypePool Pool;
  std::vector<TypeEntry *> Entries(1000);
  parallelFor(0, Entries.size(), [&](size_t I) {
    Entries[I] = Pool.insert(std::to_string(I % 10));
  });
  for (size_t I = 0; I < Entries.size(); ++I) {
    EXPECT_EQ(Entries[I], Entries[I % 10]);
    EXPECT_EQ(Entries[I]->getKey(), std::to_string(I % 10));
  }
}

TEST(TypePoolTest, GetOrCreateTypeEntryBody) {
  TypePool Pool;
  TypeEntry *Root = Pool.getRoot();
  TypeEntry *NS = Pool.insert("{1}ns");
  TypeEntry *A = Pool.insert(makeKey(NS, ".{8}A"));

  TypeEntryBody *NSBody = Pool.getOrCreateTypeEntryBody(NS, Root);
  ASSERT_NE(NSBody, nullptr);
  EXPECT_EQ(NS->getValue().load(), NSBody);
  EXPECT_EQ(Pool.getOrCreateTypeEntryBody(NS, Root), NSBody);
  EXPECT_EQ(getChildKeys(Root), std::vector<std::string>{"{1}ns"});

  TypeEntryBody *ABody = Pool.getOrCreateTypeEntryBody(A, NS);
  EXPECT_EQ(getChildKeys(NS), std::vector<std::string>{"{1}ns.{8}A"});
  EXPECT_EQ(getChildKeys(Root), std::vector<std::string>{"{1}ns"});

  EXPECT_EQ(ABody->Die.load(), nullptr);
  EXPECT_EQ(ABody->DeclarationDie.load(), nullptr);
  EXPECT_TRUE(ABody->DeclarationParentIsDeclaration);
  EXPECT_EQ(ABody->DiePriority.load(), std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(ABody->DeclarationDiePriority,
            std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(ABody->SortKey.load(), std::numeric_limits<uint32_t>::max());
  EXPECT_TRUE(ABody->Children.empty());
}

TEST(TypePoolTest, GetOrCreateTypeEntryBodyParallel) {
  TypePool Pool;
  std::vector<std::string> Keys;
  std::vector<TypeEntry *> Entries;
  for (unsigned I = 0; I < 10; ++I) {
    Keys.push_back(std::to_string(I));
    Entries.push_back(Pool.insert(Keys.back()));
  }

  std::vector<TypeEntryBody *> Bodies(1000);
  parallelFor(0, Bodies.size(), [&](size_t I) {
    Bodies[I] = Pool.getOrCreateTypeEntryBody(Entries[I % Entries.size()],
                                              Pool.getRoot());
  });
  for (size_t I = 0; I < Bodies.size(); ++I)
    EXPECT_EQ(Bodies[I], Entries[I % Entries.size()]->getValue().load());

  std::vector<std::string> ChildKeys = getChildKeys(Pool.getRoot());
  llvm::sort(ChildKeys);
  EXPECT_EQ(ChildKeys, Keys);
}

TEST(TypePoolTest, TypeEntryBodyFinalDie) {
  TypePool Pool;
  TypeEntryBody *Body =
      Pool.getOrCreateTypeEntryBody(Pool.insert("{8}A"), Pool.getRoot());
  BumpPtrAllocator Allocator;
  DIE *Declaration = DIE::get(Allocator, dwarf::DW_TAG_class_type);
  DIE *Definition = DIE::get(Allocator, dwarf::DW_TAG_class_type);

  Body->DeclarationDie = Declaration;
  EXPECT_TRUE(Body->hasOnlyDeclaration());
  EXPECT_EQ(&Body->getFinalDie(), Declaration);

  Body->Die = Definition;
  EXPECT_FALSE(Body->hasOnlyDeclaration());
  EXPECT_EQ(&Body->getFinalDie(), Definition);
}

TEST(TypePoolTest, SortTypes) {
  TypePool Pool;
  TypeEntry *Root = Pool.getRoot();
  // Allocated in reverse key order.
  TypeEntry *Z = Pool.insert("{0}z");
  TypeEntry *A = Pool.insert("{0}a");
  auto AddChild = [&](TypeEntry *Parent, StringRef Key,
                      uint32_t SortKey = std::numeric_limits<uint32_t>::max()) {
    TypeEntry *Entry = Pool.insert(Key);
    Pool.getOrCreateTypeEntryBody(Entry, Parent)->SortKey = SortKey;
    return Entry;
  };
  TypeEntry *S = AddChild(Root, "{F}S");
  AddChild(S, makeKey(S, ".{d}s"));
  AddChild(S, makeKey(S, ".{B}m1"), 2);
  AddChild(S, makeKey(S, ".{B}m0"), 2);
  AddChild(S, makeKey(S, ".{B}m2"), 1);
  AddChild(Root, makeKey("{C}", Z));
  AddChild(Root, makeKey("{C}", A));
  AddChild(Root, "{8}C");

  Pool.sortTypes();
  EXPECT_EQ(getChildKeys(Root),
            (std::vector<std::string>{"{8}C", "{C}{0}a", "{C}{0}z", "{F}S"}));
  EXPECT_EQ(getChildKeys(S),
            (std::vector<std::string>{"{F}S.{B}m2", "{F}S.{B}m0", "{F}S.{B}m1",
                                      "{F}S.{d}s"}));
}

TEST(TypePoolTest, SortTypesLargeList) {
  TypePool Pool;
  TypeEntry *Root = Pool.getRoot();
  SmallVector<TypeEntry *> Letters;
  for (char Letter = 'a'; Letter <= 'z'; ++Letter)
    Letters.push_back(Pool.insert(std::string("{0}") + Letter));

  // Enough children for the sort to run in parallel.
  constexpr size_t NumChildren = 4096;
  for (size_t I = 0; I < NumChildren; ++I) {
    // Multiplying by an odd constant permutes indices modulo a power of two.
    size_t J = (I * 2654435761u) % NumChildren;
    TypeEntry *Entry =
        Pool.insert(makeKey(Letters[J % Letters.size()], std::to_string(J)));
    Pool.getOrCreateTypeEntryBody(Entry, Root);
  }

  Pool.sortTypes();
  std::vector<std::string> Keys = getChildKeys(Root);
  EXPECT_EQ(Keys.size(), NumChildren);
  EXPECT_TRUE(llvm::is_sorted(Keys));
}

} // anonymous namespace
