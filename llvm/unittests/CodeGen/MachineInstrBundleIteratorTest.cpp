//===- MachineInstrBundleIteratorTest.cpp ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ilist_node.h"
#include "llvm/CodeGen/MachineInstrBundleIterator.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

struct MyBundledInstr
    : public ilist_node<MyBundledInstr, ilist_sentinel_tracking<true>> {
  bool isBundledWithPred() const { return true; }
  bool isBundledWithSucc() const { return true; }
};
typedef MachineInstrBundleIterator<MyBundledInstr> bundled_iterator;
typedef MachineInstrBundleIterator<const MyBundledInstr> const_bundled_iterator;
typedef MachineInstrBundleIterator<MyBundledInstr, true>
    reverse_bundled_iterator;
typedef MachineInstrBundleIterator<const MyBundledInstr, true>
    const_reverse_bundled_iterator;

#ifdef GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
TEST(MachineInstrBundleIteratorTest, CheckForBundles) {
  MyBundledInstr MBI;
  auto I = MBI.getIterator();
  auto RI = I.getReverse();

  // Confirm that MBI is always considered bundled.
  EXPECT_TRUE(MBI.isBundledWithPred());
  EXPECT_TRUE(MBI.isBundledWithSucc());

  // Confirm that iterators check in their constructor for bundled iterators.
  EXPECT_DEATH((void)static_cast<bundled_iterator>(I),
               "not legal to initialize");
  EXPECT_DEATH((void)static_cast<bundled_iterator>(MBI),
               "not legal to initialize");
  EXPECT_DEATH((void)static_cast<bundled_iterator>(&MBI),
               "not legal to initialize");
  EXPECT_DEATH((void)static_cast<const_bundled_iterator>(I),
               "not legal to initialize");
  EXPECT_DEATH((void)static_cast<const_bundled_iterator>(MBI),
               "not legal to initialize");
  EXPECT_DEATH((void)static_cast<const_bundled_iterator>(&MBI),
               "not legal to initialize");
  EXPECT_DEATH((void)static_cast<reverse_bundled_iterator>(RI),
               "not legal to initialize");
  EXPECT_DEATH((void)static_cast<reverse_bundled_iterator>(MBI),
               "not legal to initialize");
  EXPECT_DEATH((void)static_cast<reverse_bundled_iterator>(&MBI),
               "not legal to initialize");
  EXPECT_DEATH((void)static_cast<const_reverse_bundled_iterator>(RI),
               "not legal to initialize");
  EXPECT_DEATH((void)static_cast<const_reverse_bundled_iterator>(MBI),
               "not legal to initialize");
  EXPECT_DEATH((void)static_cast<const_reverse_bundled_iterator>(&MBI),
               "not legal to initialize");
}
#endif
#endif

TEST(MachineInstrBundleIteratorTest, CompareToBundledMI) {
  MyBundledInstr MBI;
  const MyBundledInstr &CMBI = MBI;
  bundled_iterator I;
  const_bundled_iterator CI;

  // Confirm that MBI is always considered bundled.
  EXPECT_TRUE(MBI.isBundledWithPred());
  EXPECT_TRUE(MBI.isBundledWithSucc());

  // These invocations will crash when !NDEBUG if a conversion is taking place.
  // These checks confirm that comparison operators don't use any conversion
  // operators.
  ASSERT_FALSE(MBI == I);
  ASSERT_FALSE(&MBI == I);
  ASSERT_FALSE(CMBI == I);
  ASSERT_FALSE(&CMBI == I);
  ASSERT_FALSE(I == MBI);
  ASSERT_FALSE(I == &MBI);
  ASSERT_FALSE(I == CMBI);
  ASSERT_FALSE(I == &CMBI);
  ASSERT_FALSE(MBI == CI);
  ASSERT_FALSE(&MBI == CI);
  ASSERT_FALSE(CMBI == CI);
  ASSERT_FALSE(&CMBI == CI);
  ASSERT_FALSE(CI == MBI);
  ASSERT_FALSE(CI == &MBI);
  ASSERT_FALSE(CI == CMBI);
  ASSERT_FALSE(CI == &CMBI);
  ASSERT_FALSE(MBI.getIterator() == I);
  ASSERT_FALSE(CMBI.getIterator() == I);
  ASSERT_FALSE(I == MBI.getIterator());
  ASSERT_FALSE(I == CMBI.getIterator());
  ASSERT_FALSE(MBI.getIterator() == CI);
  ASSERT_FALSE(CMBI.getIterator() == CI);
  ASSERT_FALSE(CI == MBI.getIterator());
  ASSERT_FALSE(CI == CMBI.getIterator());
  ASSERT_TRUE(MBI != I);
  ASSERT_TRUE(&MBI != I);
  ASSERT_TRUE(CMBI != I);
  ASSERT_TRUE(&CMBI != I);
  ASSERT_TRUE(I != MBI);
  ASSERT_TRUE(I != &MBI);
  ASSERT_TRUE(I != CMBI);
  ASSERT_TRUE(I != &CMBI);
  ASSERT_TRUE(MBI != CI);
  ASSERT_TRUE(&MBI != CI);
  ASSERT_TRUE(CMBI != CI);
  ASSERT_TRUE(&CMBI != CI);
  ASSERT_TRUE(CI != MBI);
  ASSERT_TRUE(CI != &MBI);
  ASSERT_TRUE(CI != CMBI);
  ASSERT_TRUE(CI != &CMBI);
  ASSERT_TRUE(MBI.getIterator() != I);
  ASSERT_TRUE(CMBI.getIterator() != I);
  ASSERT_TRUE(I != MBI.getIterator());
  ASSERT_TRUE(I != CMBI.getIterator());
  ASSERT_TRUE(MBI.getIterator() != CI);
  ASSERT_TRUE(CMBI.getIterator() != CI);
  ASSERT_TRUE(CI != MBI.getIterator());
  ASSERT_TRUE(CI != CMBI.getIterator());
}

struct MyUnbundledInstr
    : ilist_node<MyUnbundledInstr, ilist_sentinel_tracking<true>> {
  bool isBundledWithPred() const { return false; }
  bool isBundledWithSucc() const { return false; }
};
typedef MachineInstrBundleIterator<MyUnbundledInstr> unbundled_iterator;
typedef MachineInstrBundleIterator<const MyUnbundledInstr>
    const_unbundled_iterator;
typedef MachineInstrBundleIterator<MyUnbundledInstr, true>
    reverse_unbundled_iterator;
typedef MachineInstrBundleIterator<const MyUnbundledInstr, true>
    const_reverse_unbundled_iterator;

TEST(MachineInstrBundleIteratorTest, ReverseConstructor) {
  simple_ilist<MyUnbundledInstr, ilist_sentinel_tracking<true>> L;
  const auto &CL = L;
  MyUnbundledInstr A, B;
  L.insert(L.end(), A);
  L.insert(L.end(), B);

  // Save typing.
  typedef MachineInstrBundleIterator<MyUnbundledInstr> iterator;
  typedef MachineInstrBundleIterator<MyUnbundledInstr, true> reverse_iterator;
  typedef MachineInstrBundleIterator<const MyUnbundledInstr> const_iterator;
  typedef MachineInstrBundleIterator<const MyUnbundledInstr, true>
      const_reverse_iterator;

  // Convert to bundle iterators.
  auto begin = [&]() -> iterator { return L.begin(); };
  auto end = [&]() -> iterator { return L.end(); };
  auto rbegin = [&]() -> reverse_iterator { return L.rbegin(); };
  auto rend = [&]() -> reverse_iterator { return L.rend(); };
  auto cbegin = [&]() -> const_iterator { return CL.begin(); };
  auto cend = [&]() -> const_iterator { return CL.end(); };
  auto crbegin = [&]() -> const_reverse_iterator { return CL.rbegin(); };
  auto crend = [&]() -> const_reverse_iterator { return CL.rend(); };

  // Check conversion values.
  EXPECT_EQ(begin(), iterator(rend()));
  EXPECT_EQ(++begin(), iterator(++rbegin()));
  EXPECT_EQ(end(), iterator(rbegin()));
  EXPECT_EQ(rbegin(), reverse_iterator(end()));
  EXPECT_EQ(++rbegin(), reverse_iterator(++begin()));
  EXPECT_EQ(rend(), reverse_iterator(begin()));

  // Check const iterator constructors.
  EXPECT_EQ(cbegin(), const_iterator(rend()));
  EXPECT_EQ(cbegin(), const_iterator(crend()));
  EXPECT_EQ(crbegin(), const_reverse_iterator(end()));
  EXPECT_EQ(crbegin(), const_reverse_iterator(cend()));

  // Confirm lack of implicit conversions.
  static_assert(!std::is_convertible<iterator, reverse_iterator>::value,
                "unexpected implicit conversion");
  static_assert(!std::is_convertible<reverse_iterator, iterator>::value,
                "unexpected implicit conversion");
  static_assert(
      !std::is_convertible<const_iterator, const_reverse_iterator>::value,
      "unexpected implicit conversion");
  static_assert(
      !std::is_convertible<const_reverse_iterator, const_iterator>::value,
      "unexpected implicit conversion");
}

} // end namespace
