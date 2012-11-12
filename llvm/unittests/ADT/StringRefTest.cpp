//===- llvm/unittest/ADT/StringRefTest.cpp - StringRef unit tests ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace llvm {

std::ostream &operator<<(std::ostream &OS, const StringRef &S) {
  OS << S.str();
  return OS;
}

std::ostream &operator<<(std::ostream &OS,
                         const std::pair<StringRef, StringRef> &P) {
  OS << "(" << P.first << ", " << P.second << ")";
  return OS;
}

}

namespace {
TEST(StringRefTest, Construction) {
  EXPECT_EQ("", StringRef());
  EXPECT_EQ("hello", StringRef("hello"));
  EXPECT_EQ("hello", StringRef("hello world", 5));
  EXPECT_EQ("hello", StringRef(std::string("hello")));
}

TEST(StringRefTest, Iteration) {
  StringRef S("hello");
  const char *p = "hello";
  for (const char *it = S.begin(), *ie = S.end(); it != ie; ++it, ++p)
    EXPECT_EQ(*it, *p);
}

TEST(StringRefTest, StringOps) {
  const char *p = "hello";
  EXPECT_EQ(p, StringRef(p, 0).data());
  EXPECT_TRUE(StringRef().empty());
  EXPECT_EQ((size_t) 5, StringRef("hello").size());
  EXPECT_EQ(-1, StringRef("aab").compare("aad"));
  EXPECT_EQ( 0, StringRef("aab").compare("aab"));
  EXPECT_EQ( 1, StringRef("aab").compare("aaa"));
  EXPECT_EQ(-1, StringRef("aab").compare("aabb"));
  EXPECT_EQ( 1, StringRef("aab").compare("aa"));
  EXPECT_EQ( 1, StringRef("\xFF").compare("\1"));

  EXPECT_EQ(-1, StringRef("AaB").compare_lower("aAd"));
  EXPECT_EQ( 0, StringRef("AaB").compare_lower("aab"));
  EXPECT_EQ( 1, StringRef("AaB").compare_lower("AAA"));
  EXPECT_EQ(-1, StringRef("AaB").compare_lower("aaBb"));
  EXPECT_EQ( 1, StringRef("AaB").compare_lower("aA"));
  EXPECT_EQ( 1, StringRef("\xFF").compare_lower("\1"));

  EXPECT_EQ(-1, StringRef("aab").compare_numeric("aad"));
  EXPECT_EQ( 0, StringRef("aab").compare_numeric("aab"));
  EXPECT_EQ( 1, StringRef("aab").compare_numeric("aaa"));
  EXPECT_EQ(-1, StringRef("aab").compare_numeric("aabb"));
  EXPECT_EQ( 1, StringRef("aab").compare_numeric("aa"));
  EXPECT_EQ(-1, StringRef("1").compare_numeric("10"));
  EXPECT_EQ( 0, StringRef("10").compare_numeric("10"));
  EXPECT_EQ( 0, StringRef("10a").compare_numeric("10a"));
  EXPECT_EQ( 1, StringRef("2").compare_numeric("1"));
  EXPECT_EQ( 0, StringRef("llvm_v1i64_ty").compare_numeric("llvm_v1i64_ty"));
  EXPECT_EQ( 1, StringRef("\xFF").compare_numeric("\1"));
  EXPECT_EQ( 1, StringRef("V16").compare_numeric("V1_q0"));
  EXPECT_EQ(-1, StringRef("V1_q0").compare_numeric("V16"));
  EXPECT_EQ(-1, StringRef("V8_q0").compare_numeric("V16"));
  EXPECT_EQ( 1, StringRef("V16").compare_numeric("V8_q0"));
  EXPECT_EQ(-1, StringRef("V1_q0").compare_numeric("V8_q0"));
  EXPECT_EQ( 1, StringRef("V8_q0").compare_numeric("V1_q0"));
}

TEST(StringRefTest, Operators) {
  EXPECT_EQ("", StringRef());
  EXPECT_TRUE(StringRef("aab") < StringRef("aad"));
  EXPECT_FALSE(StringRef("aab") < StringRef("aab"));
  EXPECT_TRUE(StringRef("aab") <= StringRef("aab"));
  EXPECT_FALSE(StringRef("aab") <= StringRef("aaa"));
  EXPECT_TRUE(StringRef("aad") > StringRef("aab"));
  EXPECT_FALSE(StringRef("aab") > StringRef("aab"));
  EXPECT_TRUE(StringRef("aab") >= StringRef("aab"));
  EXPECT_FALSE(StringRef("aaa") >= StringRef("aab"));
  EXPECT_EQ(StringRef("aab"), StringRef("aab"));
  EXPECT_FALSE(StringRef("aab") == StringRef("aac"));
  EXPECT_FALSE(StringRef("aab") != StringRef("aab"));
  EXPECT_TRUE(StringRef("aab") != StringRef("aac"));
  EXPECT_EQ('a', StringRef("aab")[1]);
}

TEST(StringRefTest, Substr) {
  StringRef Str("hello");
  EXPECT_EQ("lo", Str.substr(3));
  EXPECT_EQ("", Str.substr(100));
  EXPECT_EQ("hello", Str.substr(0, 100));
  EXPECT_EQ("o", Str.substr(4, 10));
}

TEST(StringRefTest, Slice) {
  StringRef Str("hello");
  EXPECT_EQ("l", Str.slice(2, 3));
  EXPECT_EQ("ell", Str.slice(1, 4));
  EXPECT_EQ("llo", Str.slice(2, 100));
  EXPECT_EQ("", Str.slice(2, 1));
  EXPECT_EQ("", Str.slice(10, 20));
}

TEST(StringRefTest, Split) {
  StringRef Str("hello");
  EXPECT_EQ(std::make_pair(StringRef("hello"), StringRef("")),
            Str.split('X'));
  EXPECT_EQ(std::make_pair(StringRef("h"), StringRef("llo")),
            Str.split('e'));
  EXPECT_EQ(std::make_pair(StringRef(""), StringRef("ello")),
            Str.split('h'));
  EXPECT_EQ(std::make_pair(StringRef("he"), StringRef("lo")),
            Str.split('l'));
  EXPECT_EQ(std::make_pair(StringRef("hell"), StringRef("")),
            Str.split('o'));

  EXPECT_EQ(std::make_pair(StringRef("hello"), StringRef("")),
            Str.rsplit('X'));
  EXPECT_EQ(std::make_pair(StringRef("h"), StringRef("llo")),
            Str.rsplit('e'));
  EXPECT_EQ(std::make_pair(StringRef(""), StringRef("ello")),
            Str.rsplit('h'));
  EXPECT_EQ(std::make_pair(StringRef("hel"), StringRef("o")),
            Str.rsplit('l'));
  EXPECT_EQ(std::make_pair(StringRef("hell"), StringRef("")),
            Str.rsplit('o'));
}

TEST(StringRefTest, Split2) {
  SmallVector<StringRef, 5> parts;
  SmallVector<StringRef, 5> expected;

  expected.push_back("ab"); expected.push_back("c");
  StringRef(",ab,,c,").split(parts, ",", -1, false);
  EXPECT_TRUE(parts == expected);

  expected.clear(); parts.clear();
  expected.push_back(""); expected.push_back("ab"); expected.push_back("");
  expected.push_back("c"); expected.push_back("");
  StringRef(",ab,,c,").split(parts, ",", -1, true);
  EXPECT_TRUE(parts == expected);

  expected.clear(); parts.clear();
  expected.push_back("");
  StringRef("").split(parts, ",", -1, true);
  EXPECT_TRUE(parts == expected);

  expected.clear(); parts.clear();
  StringRef("").split(parts, ",", -1, false);
  EXPECT_TRUE(parts == expected);

  expected.clear(); parts.clear();
  StringRef(",").split(parts, ",", -1, false);
  EXPECT_TRUE(parts == expected);

  expected.clear(); parts.clear();
  expected.push_back(""); expected.push_back("");
  StringRef(",").split(parts, ",", -1, true);
  EXPECT_TRUE(parts == expected);

  expected.clear(); parts.clear();
  expected.push_back("a"); expected.push_back("b");
  StringRef("a,b").split(parts, ",", -1, true);
  EXPECT_TRUE(parts == expected);

  // Test MaxSplit
  expected.clear(); parts.clear();
  expected.push_back("a,,b,c");
  StringRef("a,,b,c").split(parts, ",", 0, true);
  EXPECT_TRUE(parts == expected);

  expected.clear(); parts.clear();
  expected.push_back("a,,b,c");
  StringRef("a,,b,c").split(parts, ",", 0, false);
  EXPECT_TRUE(parts == expected);

  expected.clear(); parts.clear();
  expected.push_back("a"); expected.push_back(",b,c");
  StringRef("a,,b,c").split(parts, ",", 1, true);
  EXPECT_TRUE(parts == expected);

  expected.clear(); parts.clear();
  expected.push_back("a"); expected.push_back(",b,c");
  StringRef("a,,b,c").split(parts, ",", 1, false);
  EXPECT_TRUE(parts == expected);

  expected.clear(); parts.clear();
  expected.push_back("a"); expected.push_back(""); expected.push_back("b,c");
  StringRef("a,,b,c").split(parts, ",", 2, true);
  EXPECT_TRUE(parts == expected);

  expected.clear(); parts.clear();
  expected.push_back("a"); expected.push_back("b,c");
  StringRef("a,,b,c").split(parts, ",", 2, false);
  EXPECT_TRUE(parts == expected);

  expected.clear(); parts.clear();
  expected.push_back("a"); expected.push_back(""); expected.push_back("b");
  expected.push_back("c");
  StringRef("a,,b,c").split(parts, ",", 3, true);
  EXPECT_TRUE(parts == expected);

  expected.clear(); parts.clear();
  expected.push_back("a"); expected.push_back("b"); expected.push_back("c");
  StringRef("a,,b,c").split(parts, ",", 3, false);
  EXPECT_TRUE(parts == expected);
}

TEST(StringRefTest, Trim) {
  StringRef Str0("hello");
  StringRef Str1(" hello ");
  StringRef Str2("  hello  ");

  EXPECT_EQ(StringRef("hello"), Str0.rtrim());
  EXPECT_EQ(StringRef(" hello"), Str1.rtrim());
  EXPECT_EQ(StringRef("  hello"), Str2.rtrim());
  EXPECT_EQ(StringRef("hello"), Str0.ltrim());
  EXPECT_EQ(StringRef("hello "), Str1.ltrim());
  EXPECT_EQ(StringRef("hello  "), Str2.ltrim());
  EXPECT_EQ(StringRef("hello"), Str0.trim());
  EXPECT_EQ(StringRef("hello"), Str1.trim());
  EXPECT_EQ(StringRef("hello"), Str2.trim());

  EXPECT_EQ(StringRef("ello"), Str0.trim("hhhhhhhhhhh"));

  EXPECT_EQ(StringRef(""), StringRef("").trim());
  EXPECT_EQ(StringRef(""), StringRef(" ").trim());
  EXPECT_EQ(StringRef("\0", 1), StringRef(" \0 ", 3).trim());
  EXPECT_EQ(StringRef("\0\0", 2), StringRef("\0\0", 2).trim());
  EXPECT_EQ(StringRef("x"), StringRef("\0\0x\0\0", 5).trim(StringRef("\0", 1)));
}

TEST(StringRefTest, StartsWith) {
  StringRef Str("hello");
  EXPECT_TRUE(Str.startswith("he"));
  EXPECT_FALSE(Str.startswith("helloworld"));
  EXPECT_FALSE(Str.startswith("hi"));
}

TEST(StringRefTest, EndsWith) {
  StringRef Str("hello");
  EXPECT_TRUE(Str.endswith("lo"));
  EXPECT_FALSE(Str.endswith("helloworld"));
  EXPECT_FALSE(Str.endswith("worldhello"));
  EXPECT_FALSE(Str.endswith("so"));
}

TEST(StringRefTest, Find) {
  StringRef Str("hello");
  EXPECT_EQ(2U, Str.find('l'));
  EXPECT_EQ(StringRef::npos, Str.find('z'));
  EXPECT_EQ(StringRef::npos, Str.find("helloworld"));
  EXPECT_EQ(0U, Str.find("hello"));
  EXPECT_EQ(1U, Str.find("ello"));
  EXPECT_EQ(StringRef::npos, Str.find("zz"));
  EXPECT_EQ(2U, Str.find("ll", 2));
  EXPECT_EQ(StringRef::npos, Str.find("ll", 3));
  EXPECT_EQ(0U, Str.find(""));
  StringRef LongStr("hellx xello hell ello world foo bar hello");
  EXPECT_EQ(36U, LongStr.find("hello"));
  EXPECT_EQ(28U, LongStr.find("foo"));
  EXPECT_EQ(12U, LongStr.find("hell", 2));
  EXPECT_EQ(0U, LongStr.find(""));

  EXPECT_EQ(3U, Str.rfind('l'));
  EXPECT_EQ(StringRef::npos, Str.rfind('z'));
  EXPECT_EQ(StringRef::npos, Str.rfind("helloworld"));
  EXPECT_EQ(0U, Str.rfind("hello"));
  EXPECT_EQ(1U, Str.rfind("ello"));
  EXPECT_EQ(StringRef::npos, Str.rfind("zz"));

  EXPECT_EQ(2U, Str.find_first_of('l'));
  EXPECT_EQ(1U, Str.find_first_of("el"));
  EXPECT_EQ(StringRef::npos, Str.find_first_of("xyz"));

  EXPECT_EQ(1U, Str.find_first_not_of('h'));
  EXPECT_EQ(4U, Str.find_first_not_of("hel"));
  EXPECT_EQ(StringRef::npos, Str.find_first_not_of("hello"));

  EXPECT_EQ(3U, Str.find_last_not_of('o'));
  EXPECT_EQ(1U, Str.find_last_not_of("lo"));
  EXPECT_EQ(StringRef::npos, Str.find_last_not_of("helo"));
}

TEST(StringRefTest, Count) {
  StringRef Str("hello");
  EXPECT_EQ(2U, Str.count('l'));
  EXPECT_EQ(1U, Str.count('o'));
  EXPECT_EQ(0U, Str.count('z'));
  EXPECT_EQ(0U, Str.count("helloworld"));
  EXPECT_EQ(1U, Str.count("hello"));
  EXPECT_EQ(1U, Str.count("ello"));
  EXPECT_EQ(0U, Str.count("zz"));
}

TEST(StringRefTest, EditDistance) {
  StringRef Str("hello");
  EXPECT_EQ(2U, Str.edit_distance("hill"));
}

TEST(StringRefTest, Misc) {
  std::string Storage;
  raw_string_ostream OS(Storage);
  OS << StringRef("hello");
  EXPECT_EQ("hello", OS.str());
}

TEST(StringRefTest, Hashing) {
  EXPECT_EQ(hash_value(std::string()), hash_value(StringRef()));
  EXPECT_EQ(hash_value(std::string()), hash_value(StringRef("")));
  std::string S = "hello world";
  hash_code H = hash_value(S);
  EXPECT_EQ(H, hash_value(StringRef("hello world")));
  EXPECT_EQ(H, hash_value(StringRef(S)));
  EXPECT_NE(H, hash_value(StringRef("hello worl")));
  EXPECT_EQ(hash_value(std::string("hello worl")),
            hash_value(StringRef("hello worl")));
  EXPECT_NE(H, hash_value(StringRef("hello world ")));
  EXPECT_EQ(hash_value(std::string("hello world ")),
            hash_value(StringRef("hello world ")));
  EXPECT_EQ(H, hash_value(StringRef("hello world\0")));
  EXPECT_NE(hash_value(std::string("ello worl")),
            hash_value(StringRef("hello world").slice(1, -1)));
}

struct UnsignedPair {
  const char *Str;
  uint64_t Expected;
} Unsigned[] =
  { {"0", 0}
  , {"255", 255}
  , {"256", 256}
  , {"65535", 65535}
  , {"65536", 65536}
  , {"4294967295", 4294967295ULL}
  , {"4294967296", 4294967296ULL}
  , {"18446744073709551615", 18446744073709551615ULL}
  , {"042", 34}
  , {"0x42", 66}
  , {"0b101010", 42}
  };

struct SignedPair {
  const char *Str;
  int64_t Expected;
} Signed[] =
  { {"0", 0}
  , {"-0", 0}
  , {"127", 127}
  , {"128", 128}
  , {"-128", -128}
  , {"-129", -129}
  , {"32767", 32767}
  , {"32768", 32768}
  , {"-32768", -32768}
  , {"-32769", -32769}
  , {"2147483647", 2147483647LL}
  , {"2147483648", 2147483648LL}
  , {"-2147483648", -2147483648LL}
  , {"-2147483649", -2147483649LL}
  , {"-9223372036854775808", -(9223372036854775807LL) - 1}
  , {"042", 34}
  , {"0x42", 66}
  , {"0b101010", 42}
  , {"-042", -34}
  , {"-0x42", -66}
  , {"-0b101010", -42}
  };

TEST(StringRefTest, getAsInteger) {
  uint8_t U8;
  uint16_t U16;
  uint32_t U32;
  uint64_t U64;

  for (size_t i = 0; i < array_lengthof(Unsigned); ++i) {
    bool U8Success = StringRef(Unsigned[i].Str).getAsInteger(0, U8);
    if (static_cast<uint8_t>(Unsigned[i].Expected) == Unsigned[i].Expected) {
      ASSERT_FALSE(U8Success);
      EXPECT_EQ(U8, Unsigned[i].Expected);
    } else {
      ASSERT_TRUE(U8Success);
    }
    bool U16Success = StringRef(Unsigned[i].Str).getAsInteger(0, U16);
    if (static_cast<uint16_t>(Unsigned[i].Expected) == Unsigned[i].Expected) {
      ASSERT_FALSE(U16Success);
      EXPECT_EQ(U16, Unsigned[i].Expected);
    } else {
      ASSERT_TRUE(U16Success);
    }
    bool U32Success = StringRef(Unsigned[i].Str).getAsInteger(0, U32);
    if (static_cast<uint32_t>(Unsigned[i].Expected) == Unsigned[i].Expected) {
      ASSERT_FALSE(U32Success);
      EXPECT_EQ(U32, Unsigned[i].Expected);
    } else {
      ASSERT_TRUE(U32Success);
    }
    bool U64Success = StringRef(Unsigned[i].Str).getAsInteger(0, U64);
    if (static_cast<uint64_t>(Unsigned[i].Expected) == Unsigned[i].Expected) {
      ASSERT_FALSE(U64Success);
      EXPECT_EQ(U64, Unsigned[i].Expected);
    } else {
      ASSERT_TRUE(U64Success);
    }
  }

  int8_t S8;
  int16_t S16;
  int32_t S32;
  int64_t S64;

  for (size_t i = 0; i < array_lengthof(Signed); ++i) {
    bool S8Success = StringRef(Signed[i].Str).getAsInteger(0, S8);
    if (static_cast<int8_t>(Signed[i].Expected) == Signed[i].Expected) {
      ASSERT_FALSE(S8Success);
      EXPECT_EQ(S8, Signed[i].Expected);
    } else {
      ASSERT_TRUE(S8Success);
    }
    bool S16Success = StringRef(Signed[i].Str).getAsInteger(0, S16);
    if (static_cast<int16_t>(Signed[i].Expected) == Signed[i].Expected) {
      ASSERT_FALSE(S16Success);
      EXPECT_EQ(S16, Signed[i].Expected);
    } else {
      ASSERT_TRUE(S16Success);
    }
    bool S32Success = StringRef(Signed[i].Str).getAsInteger(0, S32);
    if (static_cast<int32_t>(Signed[i].Expected) == Signed[i].Expected) {
      ASSERT_FALSE(S32Success);
      EXPECT_EQ(S32, Signed[i].Expected);
    } else {
      ASSERT_TRUE(S32Success);
    }
    bool S64Success = StringRef(Signed[i].Str).getAsInteger(0, S64);
    if (static_cast<int64_t>(Signed[i].Expected) == Signed[i].Expected) {
      ASSERT_FALSE(S64Success);
      EXPECT_EQ(S64, Signed[i].Expected);
    } else {
      ASSERT_TRUE(S64Success);
    }
  }
}


static const char* BadStrings[] = {
    "18446744073709551617"  // value just over max
  , "123456789012345678901" // value way too large
  , "4t23v"                 // illegal decimal characters
  , "0x123W56"              // illegal hex characters
  , "0b2"                   // illegal bin characters
  , "08"                    // illegal oct characters
  , "0o8"                   // illegal oct characters
  , "-123"                  // negative unsigned value
};


TEST(StringRefTest, getAsUnsignedIntegerBadStrings) {
  unsigned long long U64;
  for (size_t i = 0; i < array_lengthof(BadStrings); ++i) {
    bool IsBadNumber = StringRef(BadStrings[i]).getAsInteger(0, U64);
    ASSERT_TRUE(IsBadNumber);
  }
}



} // end anonymous namespace
