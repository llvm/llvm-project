//===- TwineTest.cpp - Twine unit tests -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {

std::string repr(const Twine &Value) {
  std::string res;
  llvm::raw_string_ostream OS(res);
  Value.printRepr(OS);
  return OS.str();
}

TEST(TwineTest, Construction) {
  EXPECT_EQ("", Twine().str());
  EXPECT_EQ("hi", Twine("hi").str());
  EXPECT_EQ("hi", Twine(std::string("hi")).str());
  EXPECT_EQ("hi", Twine(StringRef("hi")).str());
  EXPECT_EQ("hi", Twine(StringRef(std::string("hi"))).str());
  EXPECT_EQ("hi", Twine(StringRef("hithere", 2)).str());
}

TEST(TwineTest, Numbers) {
  EXPECT_EQ("123", Twine(123U).str());
  EXPECT_EQ("123", Twine(123).str());
  EXPECT_EQ("-123", Twine(-123).str());
  EXPECT_EQ("123", Twine(123).str());
  EXPECT_EQ("-123", Twine(-123).str());

  EXPECT_EQ("7b", Twine::utohexstr(123).str());
}

TEST(TwineTest, Characters) {
  EXPECT_EQ("x", Twine('x').str());
  EXPECT_EQ("x", Twine(static_cast<unsigned char>('x')).str());
  EXPECT_EQ("x", Twine(static_cast<signed char>('x')).str());
}

TEST(TwineTest, Concat) {
  // Check verse repr, since we care about the actual representation not just
  // the result.

  // Concat with null.
  EXPECT_EQ("(Twine null empty)", 
            repr(Twine("hi").concat(Twine::createNull())));
  EXPECT_EQ("(Twine null empty)", 
            repr(Twine::createNull().concat(Twine("hi"))));
  
  // Concat with empty.
  EXPECT_EQ("(Twine cstring:\"hi\" empty)", 
            repr(Twine("hi").concat(Twine())));
  EXPECT_EQ("(Twine cstring:\"hi\" empty)", 
            repr(Twine().concat(Twine("hi"))));

  // Concatenation of unary ropes.
  EXPECT_EQ("(Twine cstring:\"a\" cstring:\"b\")", 
            repr(Twine("a").concat(Twine("b"))));

  // Concatenation of other ropes.
  EXPECT_EQ("(Twine rope:(Twine cstring:\"a\" cstring:\"b\") cstring:\"c\")", 
            repr(Twine("a").concat(Twine("b")).concat(Twine("c"))));
  EXPECT_EQ("(Twine cstring:\"a\" rope:(Twine cstring:\"b\" cstring:\"c\"))",
            repr(Twine("a").concat(Twine("b").concat(Twine("c")))));
}

TEST(TwineTest, toNullTerminatedStringRef) {
  SmallString<8> storage;
  EXPECT_EQ(0, *Twine("hello").toNullTerminatedStringRef(storage).end());
  EXPECT_EQ(0,
           *Twine(StringRef("hello")).toNullTerminatedStringRef(storage).end());
}

  // I suppose linking in the entire code generator to add a unit test to check
  // the code size of the concat operation is overkill... :)

} // end anonymous namespace
