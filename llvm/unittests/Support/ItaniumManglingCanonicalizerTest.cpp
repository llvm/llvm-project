//===-------------- ItaniumManglingCanonicalizerTest.cpp ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ItaniumManglingCanonicalizer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"

#include <cstdlib>
#include <map>
#include <vector>

using namespace llvm;

namespace {

using EquivalenceError = llvm::ItaniumManglingCanonicalizer::EquivalenceError;
using FragmentKind = llvm::ItaniumManglingCanonicalizer::FragmentKind;

struct Equivalence {
  FragmentKind Kind;
  llvm::StringRef First;
  llvm::StringRef Second;
};

// A set of manglings that should all be considered equivalent.
using EquivalenceClass = std::vector<llvm::StringRef>;

struct Testcase {
  // A set of equivalences to register.
  std::vector<Equivalence> Equivalences;
  // A set of distinct equivalence classes created by registering the
  // equivalences.
  std::vector<EquivalenceClass> Classes;
};

// A function that returns a set of test cases.
static std::vector<Testcase> getTestcases() {
  return {
    // Three different manglings for std::string (old libstdc++, new libstdc++,
    // libc++).
    {
      {
        {FragmentKind::Type, "Ss",
         "NSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEE"},
        {FragmentKind::Type, "Ss",
         "NSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE"},
      },
      {
        {"_Z1fv"},
        {"_Z1fSs",
         "_Z1fNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEE",
         "_Z1fNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE"},
        {"_ZNKSs4sizeEv",
         "_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv",
         "_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE4sizeEv"},
      }
    },

    // Check that substitutions are properly handled.
    {
      {
        // ::X <-> ::N::X<int>
        {FragmentKind::Type, "1X", "N1N1XIiEE"},
        // ::T<T<int, int>, T<int, int>> <-> T<int>
        {FragmentKind::Type, "1TIS_IiiES0_E", "1TIiE"},
        // A::B::foo <-> AB::foo
        {FragmentKind::Name, "N1A1B3fooE", "N2AB3fooE"},
      },
      {
        {"_Z1f1XPS_RS_", "_Z1fN1N1XIiEEPS1_RS1_"},
        {"_ZN1A1B3fooE1TIS1_IiiES2_EPS3_RS3_", "_ZN2AB3fooE1TIiEPS1_RS1_"},
      }
    },

    // Check that nested equivalences are properly handled.
    {
      {
        // std::__1::char_traits == std::__cxx11::char_traits
        // (Note that this is unused and should make no difference,
        // but it should not cause us to fail to match up the cases
        // below.)
        {FragmentKind::Name,
         "NSt3__111char_traitsE",
         "NSt7__cxx1111char_traitsE"},
        // std::__1::allocator == std::allocator
        {FragmentKind::Name,
         "NSt3__19allocatorE",
         "Sa"}, // "Sa" is not strictly a <name> but we accept it as one.
        // std::__1::vector == std::vector
        {FragmentKind::Name,
         "St6vector",
         "NSt3__16vectorE"},
        // std::__1::basic_string<
        //   char
        //   std::__1::char_traits<char>,
        //   std::__1::allocator<char>> ==
        // std::__cxx11::basic_string<
        //   char,
        //   std::char_traits<char>,
        //   std::allocator<char>>
        {FragmentKind::Type,
         "NSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEE",
         "NSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE"},
        // X<A> <-> X<B>
        {FragmentKind::Type, "1XI1AE", "1XI1BE"},
        // X <-> Y
        {FragmentKind::Name, "1X", "1Y"},
      },
      {
        // f(std::string)
        {"_Z1fNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEE",
         "_Z1fNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE"},
        // f(std::vector<int>)
        {"_Z1fSt6vectorIiSaIiEE", "_Z1fNSt3__16vectorIiNS_9allocatorIiEEEE"},
        // f(X<A>), f(X<B>), f(Y<A>), f(Y<B>)
        {"_Z1f1XI1AE", "_Z1f1XI1BE", "_Z1f1YI1AE", "_Z1f1YI1BE"},
        // f(X<C>), f(Y<C>)
        {"_Z1f1XI1CE", "_Z1f1YI1CE"},
      }
    },

    // Check namespace equivalences.
    {
      {
        // std::__1 == std::__cxx11
        {FragmentKind::Name, "St3__1", "St7__cxx11"},
        // std::__1::allocator == std::allocator
        {FragmentKind::Name, "NSt3__19allocatorE", "Sa"},
        // std::vector == std::__1::vector
        {FragmentKind::Name, "St6vector", "NSt3__16vectorE"},
        // std::__cxx11::char_traits == std::char_traits
        // (This indirectly means that std::__1::char_traits == std::char_traits,
        // due to the std::__cxx11 == std::__1 equivalence, which is what we rely
        // on below.)
        {FragmentKind::Name, "NSt7__cxx1111char_traitsE", "St11char_traits"},
      },
      {
        // f(std::foo)
        {"_Z1fNSt7__cxx113fooE",
         "_Z1fNSt3__13fooE"},
        // f(std::string)
        {"_Z1fNSt7__cxx1111char_traitsIcEE",
         "_Z1fNSt3__111char_traitsIcEE",
         "_Z1fSt11char_traitsIcE"},
        // f(std::string)
        {"_Z1fNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEE",
         "_Z1fNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE"},
        // f(std::vector<int>)
        {"_Z1fSt6vectorIiSaIiEE", "_Z1fNSt3__16vectorIiNS_9allocatorIiEEEE"},
      }
    },

    // Check namespace equivalences for namespace 'std'. We support using 'St'
    // for this, despite it not technically being a <name>.
    {
      {
        // std::__1 == std
        {FragmentKind::Name, "St3__1", "St"},
        // std::__1 == std::__cxx11
        {FragmentKind::Name, "St3__1", "St7__cxx11"},
        // FIXME: Should a 'std' equivalence also cover the predefined
        // substitutions?
        // std::__1::allocator == std::allocator
        {FragmentKind::Name, "NSt3__19allocatorE", "Sa"},
      },
      {
        {"_Z1fSt3foo", "_Z1fNSt3__13fooE", "_Z1fNSt7__cxx113fooE"},
        {"_Z1fNSt3bar3bazE", "_Z1fNSt3__13bar3bazE"},
        // f(std::string)
        {"_Z1fNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEE",
         "_Z1fNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE"},
        // f(std::vector<int>)
        {"_Z1fSt6vectorIiSaIiEE", "_Z1fNSt3__16vectorIiNS_9allocatorIiEEEE"},
      }
    },

    // Check mutually-recursive equivalences.
    {
      {
        {FragmentKind::Type, "1A", "1B"},
        {FragmentKind::Type, "1A", "1C"},
        {FragmentKind::Type, "1D", "1B"},
        {FragmentKind::Type, "1C", "1E"},
      },
      {
        {"_Z1f1A", "_Z1f1B", "_Z1f1C", "_Z1f1D", "_Z1f1E"},
        {"_Z1f1F"},
      }
    },

    // Check <encoding>s.
    {
      {
        {FragmentKind::Encoding, "1fv", "1gv"},
      },
      {
        // f(void) -> g(void)
        {"_Z1fv", "_Z1gv"},
        // static local 'n' in f(void) -> static local 'n' in g(void)
        {"_ZZ1fvE1n", "_ZZ1gvE1n"},
      }
    },

    // Corner case: the substitution can appear within its own expansion.
    {
      {
        // X <-> Y<X>
        {FragmentKind::Type, "1X", "1YI1XE"},
        // A<B> <-> B
        {FragmentKind::Type, "1AI1BE", "1B"},
      },
      {
        // f(X) == f(Y<X>) == f(Y<Y<X>>) == f(Y<Y<Y<X>>>)
        {"_Z1f1X", "_Z1f1YI1XE", "_Z1f1YIS_I1XEE", "_Z1f1YIS_IS_I1XEEE"},
        // f(B) == f(A<B>) == f(A<A<B>>) == f(A<A<A<B>>>)
        {"_Z1f1B", "_Z1f1AI1BE", "_Z1f1AIS_I1BEE", "_Z1f1AIS_IS_I1BEEE"},
      }
    },

    // Redundant equivalences are accepted (and have no effect).
    {
      {
        {FragmentKind::Name, "3std", "St"},
        {FragmentKind::Name, "1X", "1Y"},
        {FragmentKind::Name, "N1X1ZE", "N1Y1ZE"},
      },
      {}
    },

    // Check that ctor and dtor variants are considered distinct.
    {
      {},
      {{"_ZN1XC1Ev"}, {"_ZN1XC2Ev"}, {"_ZN1XD1Ev"}, {"_ZN1XD2Ev"}}
    },

    // Ensure array types with and without bounds are handled properly.
    {
      {
        {FragmentKind::Type, "A_i", "A1_f"},
      },
      {
        {"_Z1fRA_i", "_Z1fRA_i", "_Z1fRA1_f"},
        {"_Z1fRA1_i"}, {"_Z1fRA_f"},
      }
    },

    // Unmangled names can be remapped as complete encodings.
    {
      {
        {FragmentKind::Encoding, "3foo", "3bar"},
      },
      {
        // foo == bar
        {"foo", "bar"},
        // void f<foo>() == void f<bar>()
        {"_Z1fIL_Z3fooEEvv", "_Z1fIL_Z3barEEvv"},
      }
    },
  };
}

// A function to get a set of test cases for forward template references.
static std::vector<Testcase> getForwardTemplateReferenceTestcases() {
  return {
    // ForwardTemplateReference does not support canonicalization.
    // FIXME: We should consider ways of fixing this, perhaps by eliminating
    // the ForwardTemplateReference node with a tree transformation.
    {
      {
        // X::operator T() <with T = A> == Y::operator T() <with T = A>
        {FragmentKind::Encoding, "N1XcvT_I1AEEv", "N1YcvT_I1AEEv"},
        // A == B
        {FragmentKind::Name, "1A", "1B"},
      },
      {
        // All combinations result in unique equivalence classes.
        {"_ZN1XcvT_I1AEEv"},
        {"_ZN1XcvT_I1BEEv"},
        {"_ZN1YcvT_I1AEEv"},
        {"_ZN1YcvT_I1BEEv"},
        // Even giving the same string twice gives a new class.
        {"_ZN1XcvT_I1AEEv"},
      }
    },
  };
}

template<bool CanonicalizeFirst>
static void testTestcases(ArrayRef<Testcase> Testcases) {
  for (const auto &Testcase : Testcases) {
    llvm::ItaniumManglingCanonicalizer Canonicalizer;
    for (const auto &Equiv : Testcase.Equivalences) {
      auto Result =
          Canonicalizer.addEquivalence(Equiv.Kind, Equiv.First, Equiv.Second);
      EXPECT_EQ(Result, EquivalenceError::Success)
          << "couldn't add equivalence between " << Equiv.First << " and "
          << Equiv.Second;
    }

    using CanonKey = llvm::ItaniumManglingCanonicalizer::Key;

    std::map<const EquivalenceClass*, CanonKey> Keys;
    if (CanonicalizeFirst)
      for (const auto &Class : Testcase.Classes)
        Keys.insert({&Class, Canonicalizer.canonicalize(*Class.begin())});

    std::map<CanonKey, llvm::StringRef> Found;
    for (const auto &Class : Testcase.Classes) {
      CanonKey ClassKey = Keys[&Class];
      for (llvm::StringRef Str : Class) {
        // Force a copy to be made when calling lookup to test that it doesn't
        // retain any part of the provided string.
        CanonKey ThisKey = CanonicalizeFirst
                               ? Canonicalizer.lookup(std::string(Str))
                               : Canonicalizer.canonicalize(Str);
        EXPECT_NE(ThisKey, CanonKey()) << "couldn't canonicalize " << Str;
        if (ClassKey) {
          EXPECT_EQ(ThisKey, ClassKey)
              << Str << " not in the same class as " << *Class.begin();
        } else {
          ClassKey = ThisKey;
        }
      }
      EXPECT_TRUE(Found.insert({ClassKey, *Class.begin()}).second)
          << *Class.begin() << " is in the same class as " << Found[ClassKey];
    }
  }
}

TEST(ItaniumManglingCanonicalizerTest, TestCanonicalize) {
  testTestcases<false>(getTestcases());
}

TEST(ItaniumManglingCanonicalizerTest, TestLookup) {
  testTestcases<true>(getTestcases());
}

TEST(ItaniumManglingCanonicalizerTest, TestForwardTemplateReference) {
  // lookup(...) after canonicalization (intentionally) returns different
  // values for this testcase.
  testTestcases<false>(getForwardTemplateReferenceTestcases());
}


TEST(ItaniumManglingCanonicalizerTest, TestInvalidManglings) {
  llvm::ItaniumManglingCanonicalizer Canonicalizer;
  EXPECT_EQ(Canonicalizer.addEquivalence(FragmentKind::Type, "", "1X"),
            EquivalenceError::InvalidFirstMangling);
  EXPECT_EQ(Canonicalizer.addEquivalence(FragmentKind::Type, "1X", "1ab"),
            EquivalenceError::InvalidSecondMangling);
  EXPECT_EQ(Canonicalizer.canonicalize("_Z3fooE"),
            llvm::ItaniumManglingCanonicalizer::Key());
  EXPECT_EQ(Canonicalizer.canonicalize("_Zfoo"),
            llvm::ItaniumManglingCanonicalizer::Key());

  // A reference to a template parameter ('T_' etc) cannot appear in a <name>,
  // because we don't have template arguments to bind to it. (The arguments in
  // an 'I ... E' construct in the <name> aren't registered as
  // backreferenceable arguments in this sense, because they're not part of
  // the template argument list of an <encoding>.
  EXPECT_EQ(Canonicalizer.addEquivalence(FragmentKind::Name, "N1XcvT_I1AEE",
                                         "1f"),
            EquivalenceError::InvalidFirstMangling);
}

TEST(ItaniumManglingCanonicalizerTest, TestBadEquivalenceOrder) {
  llvm::ItaniumManglingCanonicalizer Canonicalizer;
  EXPECT_EQ(Canonicalizer.addEquivalence(FragmentKind::Type, "N1P1XE", "N1Q1XE"),
            EquivalenceError::Success);
  EXPECT_EQ(Canonicalizer.addEquivalence(FragmentKind::Type, "1P", "1Q"),
            EquivalenceError::ManglingAlreadyUsed);

  EXPECT_EQ(Canonicalizer.addEquivalence(FragmentKind::Type, "N1C1XE", "N1A1YE"),
            EquivalenceError::Success);
  EXPECT_EQ(Canonicalizer.addEquivalence(FragmentKind::Type, "1A", "1B"),
            EquivalenceError::Success);
  EXPECT_EQ(Canonicalizer.addEquivalence(FragmentKind::Type, "1C", "1D"),
            EquivalenceError::Success);
  EXPECT_EQ(Canonicalizer.addEquivalence(FragmentKind::Type, "1B", "1D"),
            EquivalenceError::ManglingAlreadyUsed);
}

} // end anonymous namespace
