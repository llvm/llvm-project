//===-- JSONTest.cpp - JSON unit tests --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace llvm {
namespace json {

namespace {

std::string s(const Value &E) { return llvm::formatv("{0}", E).str(); }
std::string sp(const Value &E) { return llvm::formatv("{0:2}", E).str(); }

TEST(JSONTest, Types) {
  EXPECT_EQ("true", s(true));
  EXPECT_EQ("null", s(nullptr));
  EXPECT_EQ("2.5", s(2.5));
  EXPECT_EQ(R"("foo")", s("foo"));
  EXPECT_EQ("[1,2,3]", s({1, 2, 3}));
  EXPECT_EQ(R"({"x":10,"y":20})", s(Object{{"x", 10}, {"y", 20}}));

#ifdef NDEBUG
  EXPECT_EQ(R"("��")", s("\xC0\x80"));
  EXPECT_EQ(R"({"��":0})", s(Object{{"\xC0\x80", 0}}));
#else
  EXPECT_DEATH(s("\xC0\x80"), "Invalid UTF-8");
  EXPECT_DEATH(s(Object{{"\xC0\x80", 0}}), "Invalid UTF-8");
#endif
}

TEST(JSONTest, Constructors) {
  // Lots of edge cases around empty and singleton init lists.
  EXPECT_EQ("[[[3]]]", s({{{3}}}));
  EXPECT_EQ("[[[]]]", s({{{}}}));
  EXPECT_EQ("[[{}]]", s({{Object{}}}));
  EXPECT_EQ(R"({"A":{"B":{}}})", s(Object{{"A", Object{{"B", Object{}}}}}));
  EXPECT_EQ(R"({"A":{"B":{"X":"Y"}}})",
            s(Object{{"A", Object{{"B", Object{{"X", "Y"}}}}}}));
  EXPECT_EQ("null", s(std::optional<double>()));
  EXPECT_EQ("2.5", s(std::optional<double>(2.5)));
  EXPECT_EQ("[[2.5,null]]", s(std::vector<std::vector<std::optional<double>>>{
                                {2.5, std::nullopt}}));
}

TEST(JSONTest, StringOwnership) {
  char X[] = "Hello";
  Value Alias = static_cast<const char *>(X);
  X[1] = 'a';
  EXPECT_EQ(R"("Hallo")", s(Alias));

  std::string Y = "Hello";
  Value Copy = Y;
  Y[1] = 'a';
  EXPECT_EQ(R"("Hello")", s(Copy));
}

TEST(JSONTest, CanonicalOutput) {
  // Objects are sorted (but arrays aren't)!
  EXPECT_EQ(R"({"a":1,"b":2,"c":3})", s(Object{{"a", 1}, {"c", 3}, {"b", 2}}));
  EXPECT_EQ(R"(["a","c","b"])", s({"a", "c", "b"}));
  EXPECT_EQ("3", s(3.0));
}

TEST(JSONTest, Escaping) {
  std::string Test = {
      0,                    // Strings may contain nulls.
      '\b',   '\f',         // Have mnemonics, but we escape numerically.
      '\r',   '\n',   '\t', // Escaped with mnemonics.
      'S',    '\"',   '\\', // Printable ASCII characters.
      '\x7f',               // Delete is not escaped.
      '\xce', '\x94',       // Non-ASCII UTF-8 is not escaped.
  };

  std::string TestString = R"("\u0000\u0008\u000c\r\n\tS\"\\)"
                           "\x7f\xCE\x94\"";

  EXPECT_EQ(TestString, s(Test));

  EXPECT_EQ(R"({"object keys are\nescaped":true})",
            s(Object{{"object keys are\nescaped", true}}));
}

TEST(JSONTest, PrettyPrinting) {
  const char Str[] = R"({
  "empty_array": [],
  "empty_object": {},
  "full_array": [
    1,
    null
  ],
  "full_object": {
    "nested_array": [
      {
        "property": "value"
      }
    ]
  }
})";

  EXPECT_EQ(Str, sp(Object{
                     {"empty_object", Object{}},
                     {"empty_array", {}},
                     {"full_array", {1, nullptr}},
                     {"full_object",
                      Object{
                          {"nested_array",
                           {Object{
                               {"property", "value"},
                           }}},
                      }},
                 }));
}

TEST(JSONTest, Array) {
  Array A{1, 2};
  A.emplace_back(3);
  A.emplace(++A.begin(), 0);
  A.push_back(4);
  A.insert(++++A.begin(), 99);

  EXPECT_EQ(A.size(), 6u);
  EXPECT_EQ(R"([1,0,99,2,3,4])", s(std::move(A)));
}

TEST(JSONTest, Object) {
  Object O{{"a", 1}, {"b", 2}, {"c", 3}};
  EXPECT_TRUE(O.try_emplace("d", 4).second);
  EXPECT_FALSE(O.try_emplace("a", 4).second);

  auto D = O.find("d");
  EXPECT_NE(D, O.end());
  auto E = O.find("e");
  EXPECT_EQ(E, O.end());

  O.erase("b");
  O.erase(D);
  EXPECT_EQ(O.size(), 2u);
  EXPECT_EQ(R"({"a":1,"c":3})", s(std::move(O)));
}

TEST(JSONTest, Parse) {
  auto Compare = [](llvm::StringRef S, Value Expected) {
    if (auto E = parse(S)) {
      // Compare both string forms and with operator==, in case we have bugs.
      EXPECT_EQ(*E, Expected);
      EXPECT_EQ(sp(*E), sp(Expected));
    } else {
      handleAllErrors(E.takeError(), [S](const llvm::ErrorInfoBase &E) {
        FAIL() << "Failed to parse JSON >>> " << S << " <<<: " << E.message();
      });
    }
  };

  Compare(R"(true)", true);
  Compare(R"(false)", false);
  Compare(R"(null)", nullptr);

  Compare(R"(42)", 42);
  Compare(R"(2.5)", 2.5);
  Compare(R"(2e50)", 2e50);
  Compare(R"(1.2e3456789)", std::numeric_limits<double>::infinity());

  Compare(R"("foo")", "foo");
  Compare(R"("\"\\\b\f\n\r\t")", "\"\\\b\f\n\r\t");
  Compare(R"("\u0000")", llvm::StringRef("\0", 1));
  Compare("\"\x7f\"", "\x7f");
  Compare(R"("\ud801\udc37")", // UTF-16 surrogate pair escape.
          /*U+10437*/ "\xf0\x90\x90\xb7");
  Compare("\"\xE2\x82\xAC\xF0\x9D\x84\x9E\"", // UTF-8
          /*U+20AC U+1D11E*/ "\xe2\x82\xac\xf0\x9d\x84\x9e");
  Compare(
      // Invalid unicode.
      R"("LoneLeading=\ud801, LoneTrailing=\udc01, LeadLeadTrail=\ud801\ud801\udc37")",
      "LoneLeading=" /*U+FFFD*/ "\xef\xbf\xbd, "
      "LoneTrailing=" /*U+FFFD*/ "\xef\xbf\xbd, "
      "LeadLeadTrail=" /*U+FFFD U+10437*/ "\xef\xbf\xbd\xf0\x90\x90\xb7");

  Compare(R"({"":0,"":0})", Object{{"", 0}});
  Compare(R"({"obj":{},"arr":[]})", Object{{"obj", Object{}}, {"arr", {}}});
  Compare(R"({"\n":{"\u0000":[[[[]]]]}})",
          Object{{"\n", Object{
                            {llvm::StringRef("\0", 1), {{{{}}}}},
                        }}});
  Compare("\r[\n\t] ", {});
}

TEST(JSONTest, ParseErrors) {
  auto ExpectErr = [](llvm::StringRef Msg, llvm::StringRef S) {
    if (auto E = parse(S)) {
      // Compare both string forms and with operator==, in case we have bugs.
      FAIL() << "Parsed JSON >>> " << S << " <<< but wanted error: " << Msg;
    } else {
      handleAllErrors(E.takeError(), [S, Msg](const llvm::ErrorInfoBase &E) {
        EXPECT_THAT(E.message(), testing::HasSubstr(std::string(Msg))) << S;
      });
    }
  };
  ExpectErr("Unexpected EOF", "");
  ExpectErr("Unexpected EOF", "[");
  ExpectErr("Text after end of document", "[][]");
  ExpectErr("Invalid JSON value (false?)", "fuzzy");
  ExpectErr("Expected , or ]", "[2?]");
  ExpectErr("Expected object key", "{a:2}");
  ExpectErr("Expected : after object key", R"({"a",2})");
  ExpectErr("Expected , or } after object property", R"({"a":2 "b":3})");
  ExpectErr("Invalid JSON value", R"([&%!])");
  ExpectErr("Invalid JSON value (number?)", "1e1.0");
  ExpectErr("Unterminated string", R"("abc\"def)");
  ExpectErr("Control character in string", "\"abc\ndef\"");
  ExpectErr("Invalid escape sequence", R"("\030")");
  ExpectErr("Invalid \\u escape sequence", R"("\usuck")");
  ExpectErr("[3:3, byte=19]", R"({
  "valid": 1,
  invalid: 2
})");
  ExpectErr("Invalid UTF-8 sequence", "\"\xC0\x80\""); // WTF-8 null
}

// Direct tests of isUTF8 and fixUTF8. Internal uses are also tested elsewhere.
TEST(JSONTest, UTF8) {
  for (const char *Valid : {
           "this is ASCII text",
           "thïs tëxt häs BMP chäräctërs",
           "𐌶𐌰L𐌾𐍈 C𐍈𐌼𐌴𐍃",
       }) {
    EXPECT_TRUE(isUTF8(Valid)) << Valid;
    EXPECT_EQ(fixUTF8(Valid), Valid);
  }
  for (auto Invalid : std::vector<std::pair<const char *, const char *>>{
           {"lone trailing \x81\x82 bytes", "lone trailing �� bytes"},
           {"missing trailing \xD0 bytes", "missing trailing � bytes"},
           {"truncated character \xD0", "truncated character �"},
           {"not \xC1\x80 the \xE0\x9f\xBF shortest \xF0\x83\x83\x83 encoding",
            "not �� the ��� shortest ���� encoding"},
           {"too \xF9\x80\x80\x80\x80 long", "too ����� long"},
           {"surrogate \xED\xA0\x80 invalid \xF4\x90\x80\x80",
            "surrogate ��� invalid ����"}}) {
    EXPECT_FALSE(isUTF8(Invalid.first)) << Invalid.first;
    EXPECT_EQ(fixUTF8(Invalid.first), Invalid.second);
  }
}

TEST(JSONTest, Inspection) {
  llvm::Expected<Value> Doc = parse(R"(
    {
      "null": null,
      "boolean": false,
      "number": 2.78,
      "string": "json",
      "array": [null, true, 3.14, "hello", [1,2,3], {"time": "arrow"}],
      "object": {"fruit": "banana"}
    }
  )");
  EXPECT_TRUE(!!Doc);

  Object *O = Doc->getAsObject();
  ASSERT_TRUE(O);

  EXPECT_FALSE(O->getNull("missing"));
  EXPECT_FALSE(O->getNull("boolean"));
  EXPECT_TRUE(O->getNull("null"));

  EXPECT_EQ(O->getNumber("number"), std::optional<double>(2.78));
  EXPECT_FALSE(O->getInteger("number"));
  EXPECT_EQ(O->getString("string"), std::optional<llvm::StringRef>("json"));
  ASSERT_FALSE(O->getObject("missing"));
  ASSERT_FALSE(O->getObject("array"));
  ASSERT_TRUE(O->getObject("object"));
  EXPECT_EQ(*O->getObject("object"), (Object{{"fruit", "banana"}}));

  Array *A = O->getArray("array");
  ASSERT_TRUE(A);
  EXPECT_EQ((*A)[1].getAsBoolean(), std::optional<bool>(true));
  ASSERT_TRUE((*A)[4].getAsArray());
  EXPECT_EQ(*(*A)[4].getAsArray(), (Array{1, 2, 3}));
  EXPECT_EQ((*(*A)[4].getAsArray())[1].getAsInteger(),
            std::optional<int64_t>(2));
  int I = 0;
  for (Value &E : *A) {
    if (I++ == 5) {
      ASSERT_TRUE(E.getAsObject());
      EXPECT_EQ(E.getAsObject()->getString("time"),
                std::optional<llvm::StringRef>("arrow"));
    } else
      EXPECT_FALSE(E.getAsObject());
  }
}

// Verify special integer handling - we try to preserve exact int64 values.
TEST(JSONTest, Integers) {
  struct {
    const char *Desc;
    Value Val;
    const char *Str;
    std::optional<int64_t> AsInt;
    std::optional<double> AsNumber;
  } TestCases[] = {
    {
        "Non-integer. Stored as double, not convertible.",
        double{1.5},
        "1.5",
        std::nullopt,
        1.5,
    },

    {
        "Integer, not exact double. Stored as int64, convertible.",
        int64_t{0x4000000000000001},
        "4611686018427387905",
        int64_t{0x4000000000000001},
        double{0x4000000000000000},
    },

    {
        "Negative integer, not exact double. Stored as int64, convertible.",
        int64_t{-0x4000000000000001},
        "-4611686018427387905",
        int64_t{-0x4000000000000001},
        double{-0x4000000000000000},
    },

      // PR46470,
      // https://developercommunity.visualstudio.com/content/problem/1093399/incorrect-result-when-printing-6917529027641081856.html
#if !defined(_MSC_VER) || _MSC_VER < 1926
    {
        "Dynamically exact integer. Stored as double, convertible.",
        double{0x6000000000000000},
        "6.9175290276410819e+18",
        int64_t{0x6000000000000000},
        double{0x6000000000000000},
    },
#endif

    {
        "Dynamically integer, >64 bits. Stored as double, not convertible.",
        1.5 * double{0x8000000000000000},
        "1.3835058055282164e+19",
        std::nullopt,
        1.5 * double{0x8000000000000000},
    },
  };
  for (const auto &T : TestCases) {
    EXPECT_EQ(T.Str, s(T.Val)) << T.Desc;
    llvm::Expected<Value> Doc = parse(T.Str);
    EXPECT_TRUE(!!Doc) << T.Desc;
    EXPECT_EQ(Doc->getAsInteger(), T.AsInt) << T.Desc;
    EXPECT_EQ(Doc->getAsNumber(), T.AsNumber) << T.Desc;
    EXPECT_EQ(T.Val, *Doc) << T.Desc;
    EXPECT_EQ(T.Str, s(*Doc)) << T.Desc;
  }
}

// Verify uint64_t type.
TEST(JSONTest, U64Integers) {
  Value Val = uint64_t{3100100100};
  uint64_t Var = 3100100100;
  EXPECT_EQ(Val, Var);

  Val = uint64_t{std::numeric_limits<uint64_t>::max()};
  Var = std::numeric_limits<uint64_t>::max();
  EXPECT_EQ(Val, Var);

  // Test the parse() part.
  {
    const char *Str = "4611686018427387905";
    llvm::Expected<Value> Doc = parse(Str);

    EXPECT_TRUE(!!Doc);
    EXPECT_EQ(Doc->getAsInteger(), int64_t{4611686018427387905});
    EXPECT_EQ(Doc->getAsUINT64(), uint64_t{4611686018427387905});
  }

  {
    const char *Str = "-78278238238328222";
    llvm::Expected<Value> Doc = parse(Str);

    EXPECT_TRUE(!!Doc);
    EXPECT_EQ(Doc->getAsInteger(), int64_t{-78278238238328222});
    EXPECT_EQ(Doc->getAsUINT64(), std::nullopt);
  }

  // Test with the largest 64 signed int.
  {
    const char *Str = "9223372036854775807";
    llvm::Expected<Value> Doc = parse(Str);

    EXPECT_TRUE(!!Doc);
    EXPECT_EQ(Doc->getAsInteger(), int64_t{9223372036854775807});
    EXPECT_EQ(Doc->getAsUINT64(), uint64_t{9223372036854775807});
  }

  // Test with the largest 64 unsigned int.
  {
    const char *Str = "18446744073709551615";
    llvm::Expected<Value> Doc = parse(Str);

    EXPECT_TRUE(!!Doc);
    EXPECT_EQ(Doc->getAsInteger(), std::nullopt);
    EXPECT_EQ(Doc->getAsUINT64(), uint64_t{18446744073709551615u});
  }

  // Test with a number that is too big for 64 bits.
  {
    const char *Str = "184467440737095516150";
    llvm::Expected<Value> Doc = parse(Str);

    EXPECT_TRUE(!!Doc);
    EXPECT_EQ(Doc->getAsInteger(), std::nullopt);
    EXPECT_EQ(Doc->getAsUINT64(), std::nullopt);
    // The number was parsed as a double.
    EXPECT_TRUE(!!Doc->getAsNumber());
  }

  // Test with a negative number that is too small for 64 bits.
  {
    const char *Str = "-18446744073709551615";
    llvm::Expected<Value> Doc = parse(Str);

    EXPECT_TRUE(!!Doc);
    EXPECT_EQ(Doc->getAsInteger(), std::nullopt);
    EXPECT_EQ(Doc->getAsUINT64(), std::nullopt);
    // The number was parsed as a double.
    EXPECT_TRUE(!!Doc->getAsNumber());
  }
  // Test with a large number that is malformed.
  {
    const char *Str = "184467440737095516150.12.12";
    llvm::Expected<Value> Doc = parse(Str);

    EXPECT_EQ("[1:27, byte=27]: Invalid JSON value (number?)",
              llvm::toString(Doc.takeError()));
  }
}

template <typename T> void checkCppIntegers() {
  Value Val = T{10};
  T Var = 10;
  EXPECT_EQ(Val, Var);

  Val = T{std::numeric_limits<T>::max()};
  Var = std::numeric_limits<T>::max();
  EXPECT_EQ(Val, Var);

  Val = T{std::numeric_limits<T>::min()};
  Var = std::numeric_limits<T>::min();
  EXPECT_EQ(Val, Var);
}

// Test that underlying C++ integer types behave as expected.
TEST(JSONTest, CppIntegers) {
  checkCppIntegers<char>();
  checkCppIntegers<signed char>();
  checkCppIntegers<unsigned char>();

  checkCppIntegers<short>();
  checkCppIntegers<unsigned short>();

  checkCppIntegers<int>();
  checkCppIntegers<unsigned int>();

  checkCppIntegers<long>();
  checkCppIntegers<unsigned long>();

  checkCppIntegers<long long>();
  checkCppIntegers<unsigned long long>();
}

// Sample struct with typical JSON-mapping rules.
struct CustomStruct {
  CustomStruct() : B(false) {}
  CustomStruct(std::string S, std::optional<int> I, bool B)
      : S(S), I(I), B(B) {}
  std::string S;
  std::optional<int> I;
  bool B;
};
inline bool operator==(const CustomStruct &L, const CustomStruct &R) {
  return L.S == R.S && L.I == R.I && L.B == R.B;
}
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const CustomStruct &S) {
  return OS << "(" << S.S << ", " << (S.I ? std::to_string(*S.I) : "None")
            << ", " << S.B << ")";
}
bool fromJSON(const Value &E, CustomStruct &R, Path P) {
  ObjectMapper O(E, P);
  return O && O.map("str", R.S) && O.map("int", R.I) &&
         O.mapOptional("bool", R.B);
}

static std::string errorContext(const Value &V, const Path::Root &R) {
  std::string Context;
  llvm::raw_string_ostream OS(Context);
  R.printErrorContext(V, OS);
  return Context;
}

TEST(JSONTest, Deserialize) {
  std::map<std::string, std::vector<CustomStruct>> R;
  CustomStruct ExpectedStruct = {"foo", 42, true};
  std::map<std::string, std::vector<CustomStruct>> Expected;
  Value J = Object{{"foo", Array{
                               Object{
                                   {"str", "foo"},
                                   {"int", 42},
                                   {"bool", true},
                                   {"unknown", "ignored"},
                               },
                               Object{{"str", "bar"}},
                           }}};
  Expected["foo"] = {
      CustomStruct("foo", 42, true),
      CustomStruct("bar", std::nullopt, false),
  };
  Path::Root Root("CustomStruct");
  ASSERT_TRUE(fromJSON(J, R, Root));
  EXPECT_EQ(R, Expected);

  (*J.getAsObject()->getArray("foo"))[0] = 123;
  ASSERT_FALSE(fromJSON(J, R, Root));
  EXPECT_EQ("expected object at CustomStruct.foo[0]",
            toString(Root.getError()));
  const char *ExpectedDump = R"({
  "foo": [
    /* error: expected object */
    123,
    { ... }
  ]
})";
  EXPECT_EQ(ExpectedDump, errorContext(J, Root));

  CustomStruct V;
  EXPECT_FALSE(fromJSON(nullptr, V, Root));
  EXPECT_EQ("expected object when parsing CustomStruct",
            toString(Root.getError()));

  EXPECT_FALSE(fromJSON(Object{}, V, Root));
  EXPECT_EQ("missing value at CustomStruct.str", toString(Root.getError()));

  EXPECT_FALSE(fromJSON(Object{{"str", 1}}, V, Root));
  EXPECT_EQ("expected string at CustomStruct.str", toString(Root.getError()));

  // std::optional<T> must parse as the correct type if present.
  EXPECT_FALSE(fromJSON(Object{{"str", "1"}, {"int", "string"}}, V, Root));
  EXPECT_EQ("expected integer at CustomStruct.int", toString(Root.getError()));

  // mapOptional must parse as the correct type if present.
  EXPECT_FALSE(fromJSON(Object{{"str", "1"}, {"bool", "string"}}, V, Root));
  EXPECT_EQ("expected boolean at CustomStruct.bool", toString(Root.getError()));
}

TEST(JSONTest, ParseDeserialize) {
  auto E = parse<std::vector<CustomStruct>>(R"json(
    [{"str": "foo", "int": 42}, {"int": 42}]
  )json");
  EXPECT_THAT_EXPECTED(E, FailedWithMessage("missing value at (root)[1].str"));

  E = parse<std::vector<CustomStruct>>(R"json(
    [{"str": "foo", "int": 42}, {"str": "bar"}
  )json");
  EXPECT_THAT_EXPECTED(
      E,
      FailedWithMessage("[3:2, byte=50]: Expected , or ] after array element"));

  E = parse<std::vector<CustomStruct>>(R"json(
    [{"str": "foo", "int": 42}]
  )json");
  EXPECT_THAT_EXPECTED(E, Succeeded());
  EXPECT_THAT(*E, testing::SizeIs(1));
}

TEST(JSONTest, Stream) {
  auto StreamStuff = [](unsigned Indent) {
    std::string S;
    llvm::raw_string_ostream OS(S);
    OStream J(OS, Indent);
    J.comment("top*/level");
    J.object([&] {
      J.attributeArray("foo", [&] {
        J.value(nullptr);
        J.comment("element");
        J.value(42.5);
        J.arrayBegin();
        J.value(43);
        J.arrayEnd();
        J.rawValue([](raw_ostream &OS) { OS << "'unverified\nraw value'"; });
      });
      J.comment("attribute");
      J.attributeBegin("bar");
      J.comment("attribute value");
      J.objectBegin();
      J.objectEnd();
      J.attributeEnd();
      J.attribute("baz", "xyz");
    });
    return S;
  };

  const char *Plain =
      R"(/*top* /level*/{"foo":[null,/*element*/42.5,[43],'unverified
raw value'],/*attribute*/"bar":/*attribute value*/{},"baz":"xyz"})";
  EXPECT_EQ(Plain, StreamStuff(0));
  const char *Pretty = R"(/* top* /level */
{
  "foo": [
    null,
    /* element */
    42.5,
    [
      43
    ],
    'unverified
raw value'
  ],
  /* attribute */
  "bar": /* attribute value */ {},
  "baz": "xyz"
})";
  EXPECT_EQ(Pretty, StreamStuff(2));
}

TEST(JSONTest, Path) {
  Path::Root R("foo");
  Path P = R, A = P.field("a"), B = P.field("b");
  P.report("oh no");
  EXPECT_THAT_ERROR(R.getError(), FailedWithMessage("oh no when parsing foo"));
  A.index(1).field("c").index(2).report("boom");
  EXPECT_THAT_ERROR(R.getError(), FailedWithMessage("boom at foo.a[1].c[2]"));
  B.field("d").field("e").report("bam");
  EXPECT_THAT_ERROR(R.getError(), FailedWithMessage("bam at foo.b.d.e"));

  Value V = Object{
      {"a", Array{42}},
      {"b",
       Object{{"d",
               Object{
                   {"e", Array{1, Object{{"x", "y"}}}},
                   {"f", "a moderately long string: 48 characters in total"},
               }}}},
  };
  const char *Expected = R"({
  "a": [ ... ],
  "b": {
    "d": {
      "e": /* error: bam */ [
        1,
        { ... }
      ],
      "f": "a moderately long string: 48 characte..."
    }
  }
})";
  EXPECT_EQ(Expected, errorContext(V, R));
}

} // namespace
} // namespace json
} // namespace llvm
