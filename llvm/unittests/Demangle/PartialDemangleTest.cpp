//===----------------------- PartialDemangleTest.cpp ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cstdlib>
#include "llvm/Demangle/Demangle.h"
#include "gtest/gtest.h"

struct ChoppedName {
  const char *Mangled;
  const char *ContextName, *BaseName, *ReturnType, *Params;
};

static ChoppedName NamesToTest[] = {
  {"_Z1fv", "", "f", "", "()"},
  {"_ZN1a1b1cIiiiEEvm", "a::b", "c", "void", "(unsigned long)"},
  {"_ZZ5OuterIiEivEN5Inner12inner_memberEv",
   "int Outer<int>()::Inner", "inner_member", "", "()"},
  {"_Z1fIiEPFvvEv", "", "f", "void (*)()", "()"},
  {"_ZN1S1fIiEEvv", "S", "f", "void", "()"},

  // Call operator for a lambda in f().
  {"_ZZ1fvENK3$_0clEi", "f()::$_0", "operator()", "", "(int)"},

  // A call operator for a lambda in a lambda in f().
  {"_ZZZ1fvENK3$_0clEvENKUlvE_clEv",
   "f()::$_0::operator()() const::'lambda'()", "operator()", "", "()"},

  {"_ZZN1S1fEiiEd0_NKUlvE_clEv",
   "S::f(int, int)::'lambda'()", "operator()", "", "()"},

  {"_ZN1Scv7MuncherIJDpPT_EEIJFivEA_iEEEv",
   "S", "operator Muncher<int (*)(), int (*) []>", "", "()"},

  // Attributes.
  {"_ZN5test4IdE1fEUa9enable_ifIXeqfL0p_Li1EEXeqfL0p0_Li2EEEi",
   "test4<double>", "f", "", "(int)"},
  {"_ZN1SC2B8ctor_tagEv", "S", "S", "", "()"},
  {"_ZN1S1fB4MERPIiEEvv", "S", "f", "void", "()"},

  {"_ZNSsC1EmcRKSaIcE",
   "std::basic_string<char, std::char_traits<char>, std::allocator<char> >",
   "basic_string", "", "(unsigned long, char, std::allocator<char> const&)"},
  {"_ZNSsixEm", "std::string", "operator[]", "", "(unsigned long)"},
  {"_ZSt17__throw_bad_allocv", "std", "__throw_bad_alloc", "", "()"},

  {"_ZN1AI1BEC2Ev", "A<B>", "A", "", "()"},
  {"_ZN1AI1BED2Ev", "A<B>", "~A", "", "()"},
  {"_ZN1AI1BECI24BaseEi", "A<B>", "A", "", "(int)"},
  {"_ZNKR1AI1BE1fIiEEiv", "A<B>", "f", "int", "()"},

  {"_ZN1SIJicfEE3mfnIJjcdEEEvicfDpT_", "S<int, char, float>",
   "mfn", "void", "(int, char, float, unsigned int, char, double)"},
};

TEST(PartialDemangleTest, TestNameChopping) {
  size_t Size = 1;
  char *Buf = static_cast<char *>(std::malloc(Size));

  llvm::ItaniumPartialDemangler D;

  for (ChoppedName &N : NamesToTest) {
    EXPECT_FALSE(D.partialDemangle(N.Mangled));
    EXPECT_TRUE(D.isFunction());
    EXPECT_FALSE(D.isData());
    EXPECT_FALSE(D.isSpecialName());

    Buf = D.getFunctionDeclContextName(Buf, &Size);
    EXPECT_STREQ(Buf, N.ContextName);

    Buf = D.getFunctionBaseName(Buf, &Size);
    EXPECT_STREQ(Buf, N.BaseName);

    Buf = D.getFunctionReturnType(Buf, &Size);
    EXPECT_STREQ(Buf, N.ReturnType);

    Buf = D.getFunctionParameters(Buf, &Size);
    EXPECT_STREQ(Buf, N.Params);
  }

  std::free(Buf);
}

TEST(PartialDemangleTest, TestNameMeta) {
  llvm::ItaniumPartialDemangler Demangler;

  EXPECT_FALSE(Demangler.partialDemangle("_ZNK1f1gEv"));
  EXPECT_TRUE(Demangler.isFunction());
  EXPECT_TRUE(Demangler.hasFunctionQualifiers());
  EXPECT_FALSE(Demangler.isSpecialName());
  EXPECT_FALSE(Demangler.isData());

  EXPECT_FALSE(Demangler.partialDemangle("_Z1fv"));
  EXPECT_FALSE(Demangler.hasFunctionQualifiers());

  EXPECT_FALSE(Demangler.partialDemangle("_ZTV1S"));
  EXPECT_TRUE(Demangler.isSpecialName());
  EXPECT_FALSE(Demangler.isData());
  EXPECT_FALSE(Demangler.isFunction());

  EXPECT_FALSE(Demangler.partialDemangle("_ZN1aDC1a1b1cEE"));
  EXPECT_FALSE(Demangler.isFunction());
  EXPECT_FALSE(Demangler.isSpecialName());
  EXPECT_TRUE(Demangler.isData());
}

TEST(PartialDemanglerTest, TestCtorOrDtor) {
  static const char *Pos[] = {
      "_ZN1AC1Ev",        // A::A()
      "_ZN1AC1IiEET_",    // A::A<int>(int)
      "_ZN1AD2Ev",        // A::~A()
      "_ZN1BIiEC1IcEET_", // B<int>::B<char>(char)
      "_ZN1AC1B1TEv",     // A::A[abi:T]()
      "_ZNSt1AD2Ev",      // std::A::~A()
      "_ZN2ns1AD1Ev",      // ns::A::~A()
  };
  static const char *Neg[] = {
      "_Z1fv",
      "_ZN1A1gIiEEvT_", // void A::g<int>(int)
  };

  llvm::ItaniumPartialDemangler D;
  for (const char *N : Pos) {
    EXPECT_FALSE(D.partialDemangle(N));
    EXPECT_TRUE(D.isCtorOrDtor());
  }
  for (const char *N : Neg) {
    EXPECT_FALSE(D.partialDemangle(N));
    EXPECT_FALSE(D.isCtorOrDtor());
  }
}

TEST(PartialDemanglerTest, TestMisc) {
  llvm::ItaniumPartialDemangler D1, D2;

  EXPECT_FALSE(D1.partialDemangle("_Z1fv"));
  EXPECT_FALSE(D2.partialDemangle("_Z1g"));
  std::swap(D1, D2);
  EXPECT_FALSE(D1.isFunction());
  EXPECT_TRUE(D2.isFunction());

  EXPECT_TRUE(D1.partialDemangle("Not a mangled name!"));

}
