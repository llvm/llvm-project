//===------------------ ItaniumDemangleTest.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Demangle/ItaniumDemangle.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Allocator.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <string_view>
#include <vector>

using namespace llvm;
using namespace llvm::itanium_demangle;

namespace {
class TestAllocator {
  BumpPtrAllocator Alloc;

public:
  void reset() { Alloc.Reset(); }

  template <typename T, typename... Args> T *makeNode(Args &&... args) {
    return new (Alloc.Allocate(sizeof(T), alignof(T)))
        T(std::forward<Args>(args)...);
  }

  void *allocateNodeArray(size_t sz) {
    return Alloc.Allocate(sizeof(Node *) * sz, alignof(Node *));
  }
};
} // namespace

namespace NodeMatcher {
// Make sure the node matchers provide constructor parameters. This is a
// compilation test.
template <typename NT> struct Ctor {
  template <typename... Args> void operator()(Args &&...args) {
    auto _ = NT(std::forward<Args>(args)...);
  }
};

template <typename NT> void Visit(const NT *Node) { Node->match(Ctor<NT>{}); }
#define NOMATCHER(X)                                                           \
  template <> void Visit<itanium_demangle::X>(const itanium_demangle::X *) {}
// Some nodes have no match member.
NOMATCHER(ForwardTemplateReference)
#undef NOMATCHER

void Visitor() {
#define NODE(X) Visit(static_cast<const itanium_demangle::X *>(nullptr));
#include "llvm/Demangle/ItaniumNodes.def"
}
} // namespace NodeMatcher

// Verify Operator table is ordered
TEST(ItaniumDemangle, OperatorOrdering) {
  struct TestParser : AbstractManglingParser<TestParser, TestAllocator> {};
  for (const auto *Op = &TestParser::Ops[0];
       Op != &TestParser::Ops[TestParser::NumOps - 1]; Op++)
    ASSERT_LT(Op[0], Op[1]);
}

TEST(ItaniumDemangle, MethodOverride) {
  struct TestParser : AbstractManglingParser<TestParser, TestAllocator> {
    std::vector<char> Types;

    TestParser(const char *Str)
        : AbstractManglingParser(Str, Str + strlen(Str)) {}

    Node *parseType() {
      Types.push_back(*First);
      return AbstractManglingParser<TestParser, TestAllocator>::parseType();
    }
  };

  TestParser Parser("_Z1fIiEjl");
  ASSERT_NE(nullptr, Parser.parse());
  EXPECT_THAT(Parser.Types, testing::ElementsAre('i', 'j', 'l'));
}

static std::string toString(OutputBuffer &OB) {
  std::string_view SV = OB;
  return {SV.begin(), SV.end()};
}

TEST(ItaniumDemangle, HalfType) {
  struct TestParser : AbstractManglingParser<TestParser, TestAllocator> {
    std::vector<std::string> Types;

    TestParser(const char *Str)
        : AbstractManglingParser(Str, Str + strlen(Str)) {}

    Node *parseType() {
      OutputBuffer OB;
      Node *N = AbstractManglingParser<TestParser, TestAllocator>::parseType();
      OB.printLeft(*N);
      std::string_view Name = N->getBaseName();
      if (!Name.empty())
        Types.push_back(std::string(Name.begin(), Name.end()));
      else
        Types.push_back(toString(OB));
      std::free(OB.getBuffer());
      return N;
    }
  };

  // void f(A<_Float16>, _Float16);
  TestParser Parser("_Z1f1AIDF16_EDF16_");
  ASSERT_NE(nullptr, Parser.parse());
  EXPECT_THAT(Parser.Types, testing::ElementsAre("_Float16", "A", "_Float16"));
}

struct DemangleTestCase {
  const char *mangled;
  const char *expected;
};

struct DemangleTestFixture : public ::testing::TestWithParam<DemangleTestCase> {
};

DemangleTestCase g_demangle_test_cases[] = {
#include "llvm/Testing/Demangle/DemangleTestCases.inc"
};

TEST_P(DemangleTestFixture, Demangle_Valid) {
  auto [mangled, expected] = GetParam();

  llvm::itanium_demangle::ManglingParser<TestAllocator> Parser(
      mangled, mangled + ::strlen(mangled));

  const auto *Root = Parser.parse();

  ASSERT_NE(nullptr, Root);

  OutputBuffer OB;
  Root->print(OB);
  auto demangled = std::string_view(OB);

  EXPECT_EQ(demangled, std::string_view(expected));
}

INSTANTIATE_TEST_SUITE_P(DemangleValidTests, DemangleTestFixture,
                         ::testing::ValuesIn(g_demangle_test_cases));

struct DemangleInvalidTestFixture
    : public ::testing::TestWithParam<const char *> {};

const char *g_demangle_invalid_test_cases[] = {
    // clang-format off
    "_ZIPPreEncode",
    "Agentt",
    "NSoERj5E=Y1[uM:ga",
    "Aon_PmKVPDk7?fg4XP5smMUL6;<WsI_mgbf23cCgsHbT<l8EE\0uVRkNOoXDrgdA4[8IU>Vl<>IL8ayHpiVDDDXTY;^o9;i",
    "_ZNSt16allocator_traitsISaIN4llvm3sys2fs18directory_iteratorEEE9constructIS3_IS3_EEEDTcl12_S_constructfp_fp0_spcl7forwardIT0_Efp1_EEERS4_PT_DpOS7_",
    "3FooILdaaaaaaaaaaAAAAaaEE",
    "3FooILdaaaaaaaaaaaaaaEE",
#if !LDBL_FP80
    "_ZN5test01hIfEEvRAcvjplstT_Le4001a000000000000000E_c",
#endif
    // The following test cases were found by libFuzzer+ASAN
    "\x44\x74\x70\x74\x71\x75\x34\x43\x41\x72\x4D\x6E\x65\x34\x9F\xC1\x43\x41\x72\x4D\x6E\x77\x38\x9A\x8E\x44\x6F\x64\x6C\x53\xF9\x5F\x70\x74\x70\x69\x45\x34\xD3\x73\x9E\x2A\x37",
    "\x4D\x41\x72\x63\x4E\x39\x44\x76\x72\x4D\x34\x44\x53\x4B\x6F\x44\x54\x6E\x61\x37\x47\x77\x78\x38\x43\x27\x41\x5F\x73\x70\x69\x45*",
    "\x41\x64\x6E\x32*",
    "\x43\x46\x41\x67\x73*",
    "\x72\x3A\x4E\x53\x64\x45\x39\x4F\x52\x4E\x1F\x43\x34\x64\x54\x5F\x49\x31\x41\x63\x6C\x37\x2A\x4D\x41\x67\x73\x76\x43\x54\x35\x5F\x49\x4B\x4C\x55\x6C\x73\x4C\x38\x64\x43\x41\x47\x4C\x5A\x28\x4F\x41\x6E\x77\x5F\x53\x6F\x70\x69\x45\x5F\x63\x47\x61\x4C\x31\x4F\x4C\x33\x3E\x41\x4C\x4B\x4C\x55\x6C\x73\x4C\x38\x64\x43\x66\x41\x47\x4C\x5A\x28\x4F\x41\x6E\x77\x5F\x53\x6F\x70\x69\x45\x5F\x37\x41*",
    "\x2D\x5F\x63\x47\x4F\x63\xD3",
    "\x44\x74\x70\x74\x71\x75\x32\x43\x41\x38\x65\x6E\x9B\x72\x4D\xC1\x43\x41\x72\x4D\x6E\x77\x38\x9A\x8E\x44\x6F\x64\xC3\x53\xF9\x5F\x70\x74\x70\x69\x45\x38\xD3\x73\x9E\x2A\x37",
    "\x4C\x5A\x4C\x55\x6C\x4D\x41\x5F\x41\x67\x74\x71\x75\x34\x4D\x41\x64\x73\x4C\x44\x76\x72\x4D\x34\x44\x4B\x44\x54\x6E\x61\x37\x47\x77\x78\x38\x43\x27\x41\x5F\x73\x70\x69\x45\x6D\x73\x72\x53\x41\x6F\x41\x7B",
    "\x44\x74\x70\x74\x71\x75\x32\x43\x41\x38\x65\x6E\x9B\x72\x4D\xC1\x43\x41\x72\x4D\x6E\x77\x38\x9A\x8E\x44\x6F\x64\x2C\x53\xF9\x5F\x70\x74\x70\x69\x45\xB4\xD3\x73\x9F\x2A\x37",
    "\x4C\x5A\x4C\x55\x6C\x69\x4D\x73\x72\x53\x6F\x7A\x41\x5F\x41\x67\x74\x71\x75\x32\x4D\x41\x64\x73\x39\x28\x76\x72\x4D\x34\x44\x4B\x45\x54\x6E\x61\x37\x47\x77\x78\x38\x43\x27\x41\x5F\x73\x70\x69\x45\x6F\x45\x49\x6D\x1A\x4C\x53\x38\x6A\x7A\x5A",
    "\x44\x74\x63*",
    "\x44\x74\x71\x75\x35\x2A\xDF\x74\x44\x61\x73\x63\x35\x2A\x3B\x41\x72\x4D\x6E\x65\x34\x9F\xC1\x63\x41\x72\x4D\x6E\x77\x38\x9A\x8E\x44\x6F\x64\x6C\x53\xF9\x5F\x70\x74\x70\x69\x45\x33\x44\x76\x35",
    "\x44\x74\x70\x74\x71\x75\x32\x43\x41\x38\x65\x6E\x9B\x72\x4D\xC1\x43\x41\x72\x4D\x6E\x77\x38\x9A\x8E\x44\x6F\x64\x6C\x53\xF9\x5F\x70\x74\x70\x69\x45\x38\xD3\x73\x9E\x2A\x37",
    "\x46\x44\x74\x70\x74\x71\x75\x32\x43\x41\x72\x4D\x6E\x65\x34\x9F\xC1\x43\x41\x72\x4D\x6E\x77\x38\x9A\x8E\x44\x6F\x64\x6C\x53\xF9\x5F\x70\x74\x70\x69\x45\x34\xD3\x73\x9E\x2A\x37\x72\x33\x8E\x3A\x29\x8E\x44\x35",
    "_ZcvCiIJEEDvT__FFFFT_vT_v",
    "Z1JIJ1_T_EE3o00EUlT_E0",
    "___Z2i_D1D1D1D1D1D1D1D1D1D1D1D1D1D1D1D1D1D1D1D1D1D1D1D1D1D1D1D1D1D",
    "ZcvSdIZcvSdIZcvSdIZcvSdIZcvSdIZcvSdIDv_ZcvSdIZcvSdIZcvSdIZcvSdIZcvSdIZcvSdIDv_ZcvSdIZcvSdIZcvSdIZcvSdIZcvSdIZcvSdIDv_Dv_Dv_Dv_Dv_dZcvSdIZcvSdIZcvSdIZcvSdIZcvSdIZcvSdIDv_ZcvSdIZcvSdIZcvSdIZcvSdIZcvSdIZcvSdIDv_ZcvSdIZcvSdIZcvSdIZcvSdIZcvSdIZcvSdIDv_Dv_Dv_Dv_Dv_d",
    "Z1 Z1 IJEEAcvZcvT_EcvT_T_",
    "T_IZaaIJEEAnaaaT_T__",
    "PT_IJPNT_IJEET_T_T_T_)J)JKE",
    "1 IJEVNT_T_T_EE",
    "AT__ZSiIJEEAnwscT_T__",
    "FSiIJEENT_IoE ",
    "ZTVSiIZTVSiIZTVSiIZTVSiINIJEET_T_T_T_T_ ",
    "Ana_T_E_T_IJEffffffffffffffersfffffrsrsffffffbgE",

    "_ZN3TPLS_E",
    "_ZN3CLSIiEIiEE",
    "_ZN3CLSDtLi0EEE",
    "_ZN3CLSIiEEvNS_T_Ev",

    "_ZN1fIiEEvNTUt_E",
    "_ZNDTUt_Ev",

    "_Z1fIXfLpt1x1yEEvv",
    "_Z1fIXfLdt1x1yEEvv",

    "_ZN1fIXawLi0EEEEvv",

    "_ZNWUt_3FOOEv",
    "_ZWDC3FOOEv",
    "_ZGI3Foo",
    "_ZGIW3Foov",
    "W1x",
    // clang-format on
};

TEST_P(DemangleInvalidTestFixture, Demangle_Invalid) {
  const char *mangled = GetParam();

  llvm::itanium_demangle::ManglingParser<TestAllocator> Parser(
      mangled, mangled + ::strlen(mangled));

  const auto *Root = Parser.parse();

  EXPECT_EQ(nullptr, Root);
}

INSTANTIATE_TEST_SUITE_P(DemangleInvalidTests, DemangleInvalidTestFixture,
                         ::testing::ValuesIn(g_demangle_invalid_test_cases));

// Is long double fp80?  (Only x87 extended double has 64-bit mantissa)
#define LDBL_FP80 (__LDBL_MANT_DIG__ == 64)
// Is long double fp128?
#define LDBL_FP128 (__LDBL_MANT_DIG__ == 113)

struct FPLiteralCase {
  const char *mangled;
  // There are four possible demanglings of a given float.
  std::string expecting[4];
};

struct DemangleFPLiteralTestFixture
    : public ::testing::TestWithParam<FPLiteralCase> {};

FPLiteralCase g_fp_literal_cases[] = {
    // clang-format off
    {"_ZN5test01gIfEEvRAszplcvT__ELf40a00000E_c",
     {
         "void test0::g<float>(char (&) [sizeof ((float)() + 0x1.4p+2f)])",
         "void test0::g<float>(char (&) [sizeof ((float)() + 0x2.8p+1f)])",
         "void test0::g<float>(char (&) [sizeof ((float)() + 0x5p+0f)])",
         "void test0::g<float>(char (&) [sizeof ((float)() + 0xap-1f)])",
     }},
    {"_ZN5test01hIfEEvRAszplcvT__ELd4014000000000000E_c",
     {
         "void test0::h<float>(char (&) [sizeof ((float)() + 0x1.4p+2)])",
         "void test0::h<float>(char (&) [sizeof ((float)() + 0x2.8p+1)])",
         "void test0::h<float>(char (&) [sizeof ((float)() + 0x5p+0)])",
         "void test0::h<float>(char (&) [sizeof ((float)() + 0xap-1)])",
     }},
#if LDBL_FP80
    {"_ZN5test01hIfEEvRAcvjplstT_Le4001a000000000000000E_c",
     {
         "void test0::h<float>(char (&) [(unsigned int)(sizeof (float) + 0x1.4p+2L)])",
         "void test0::h<float>(char (&) [(unsigned int)(sizeof (float) + 0x2.8p+1L)])",
         "void test0::h<float>(char (&) [(unsigned int)(sizeof (float) + 0x5p+0L)])",
         "void test0::h<float>(char (&) [(unsigned int)(sizeof (float) + 0xap-1L)])",
     }},
#endif
#if LDBL_FP128
    // A 32-character FP literal of long double type
    {"3FooILeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeEE",
     {"Foo<-0x1.eeeeeeeeeeeeeeeeeeeeeeeeeeeep+12015L>"}},
#endif
    // clang-format on
};

TEST_P(DemangleFPLiteralTestFixture, Demangle_FPLiteral) {
  auto [mangled, expected] = GetParam();

  llvm::itanium_demangle::ManglingParser<TestAllocator> Parser(
      mangled, mangled + ::strlen(mangled));

  const auto *Root = Parser.parse();

  ASSERT_NE(nullptr, Root);

  OutputBuffer OB;
  Root->print(OB);
  auto demangled = std::string_view(OB);

  EXPECT_TRUE(llvm::find(expected, demangled) != std::end(expected));
}

INSTANTIATE_TEST_SUITE_P(DemangleFPLiteralTests, DemangleFPLiteralTestFixture,
                         ::testing::ValuesIn(g_fp_literal_cases));
