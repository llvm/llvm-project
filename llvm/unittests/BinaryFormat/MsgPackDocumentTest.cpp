//===- MsgPackDocumentTest.cpp --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace msgpack;

TEST(MsgPackDocument, TestReadInt) {
  Document Doc;
  bool Ok = Doc.readFromBlob(StringRef("\xd0\x00", 2), /*Multi=*/false);
  ASSERT_TRUE(Ok);
  ASSERT_EQ(Doc.getRoot().getKind(), Type::Int);
  ASSERT_EQ(Doc.getRoot().getInt(), 0);
}

TEST(MsgPackDocument, TestReadArray) {
  Document Doc;
  bool Ok = Doc.readFromBlob(StringRef("\x92\xd0\x01\xc0"), /*Multi=*/false);
  ASSERT_TRUE(Ok);
  ASSERT_EQ(Doc.getRoot().getKind(), Type::Array);
  auto A = Doc.getRoot().getArray();
  ASSERT_EQ(A.size(), 2u);
  auto SI = A[0];
  ASSERT_EQ(SI.getKind(), Type::Int);
  ASSERT_EQ(SI.getInt(), 1);
  auto SN = A[1];
  ASSERT_EQ(SN.getKind(), Type::Nil);
}

TEST(MsgPackDocument, TestReadMap) {
  Document Doc;
  bool Ok = Doc.readFromBlob(StringRef("\x82\xa3"
                                       "foo"
                                       "\xd0\x01\xa3"
                                       "bar"
                                       "\xd0\x02"),
                             /*Multi=*/false);
  ASSERT_TRUE(Ok);
  ASSERT_EQ(Doc.getRoot().getKind(), Type::Map);
  auto M = Doc.getRoot().getMap();
  ASSERT_EQ(M.size(), 2u);
  auto FooS = M["foo"];
  ASSERT_EQ(FooS.getKind(), Type::Int);
  ASSERT_EQ(FooS.getInt(), 1);
  auto BarS = M["bar"];
  ASSERT_EQ(BarS.getKind(), Type::Int);
  ASSERT_EQ(BarS.getInt(), 2);
}

TEST(MsgPackDocument, TestWriteInt) {
  Document Doc;
  Doc.getRoot() = Doc.getNode(int64_t(1));
  std::string Buffer;
  Doc.writeToBlob(Buffer);
  ASSERT_EQ(Buffer, "\x01");
}

TEST(MsgPackDocument, TestWriteArray) {
  Document Doc;
  auto A = Doc.getRoot().getArray(/*Convert=*/true);
  A.push_back(Doc.getNode(int64_t(1)));
  A.push_back(Doc.getNode());
  std::string Buffer;
  Doc.writeToBlob(Buffer);
  ASSERT_EQ(Buffer, "\x92\x01\xc0");
}

TEST(MsgPackDocument, TestWriteMap) {
  Document Doc;
  auto M = Doc.getRoot().getMap(/*Convert=*/true);
  M["foo"] = Doc.getNode(int64_t(1));
  M["bar"] = Doc.getNode(int64_t(2));
  std::string Buffer;
  Doc.writeToBlob(Buffer);
  ASSERT_EQ(Buffer, "\x82\xa3"
                    "bar"
                    "\x02\xa3"
                    "foo"
                    "\x01");
}

TEST(MsgPackDocument, TestOutputYAMLArray) {
  Document Doc;
  auto A = Doc.getRoot().getArray(/*Convert=*/true);
  A.push_back(Doc.getNode(int64_t(1)));
  A.push_back(Doc.getNode(int64_t(2)));
  std::string Buffer;
  raw_string_ostream OStream(Buffer);
  Doc.toYAML(OStream);
  ASSERT_EQ(OStream.str(), "---\n- 1\n- 2\n...\n");
}

TEST(MsgPackDocument, TestInputYAMLArray) {
  Document Doc;
  bool Ok = Doc.fromYAML("---\n- !int 0x1\n- !str 2\n...\n");
  ASSERT_TRUE(Ok);
  ASSERT_EQ(Doc.getRoot().getKind(), Type::Array);
  auto A = Doc.getRoot().getArray();
  ASSERT_EQ(A.size(), 2u);
  auto SI = A[0];
  ASSERT_EQ(SI.getKind(), Type::UInt);
  ASSERT_EQ(SI.getUInt(), 1u);
  auto SS = A[1];
  ASSERT_EQ(SS.getKind(), Type::String);
  ASSERT_EQ(SS.getString(), "2");
}

TEST(MsgPackDocument, TestOutputYAMLMap) {
  Document Doc;
  auto M = Doc.getRoot().getMap(/*Convert=*/true);
  M["foo"] = Doc.getNode(int64_t(1));
  M["bar"] = Doc.getNode(uint64_t(2));
  auto N = Doc.getMapNode();
  M["qux"] = N;
  N["baz"] = Doc.getNode(true);
  std::string Buffer;
  raw_string_ostream OStream(Buffer);
  Doc.toYAML(OStream);
  ASSERT_EQ(OStream.str(), "---\n"
                           "bar:             2\n"
                           "foo:             1\n"
                           "qux:             \n"
                           "  baz:             true\n"
                           "...\n");
}

TEST(MsgPackDocument, TestOutputYAMLMapHex) {
  Document Doc;
  Doc.setHexMode();
  auto M = Doc.getRoot().getMap(/*Convert=*/true);
  M["foo"] = Doc.getNode(int64_t(1));
  M["bar"] = Doc.getNode(uint64_t(2));
  auto N = Doc.getMapNode();
  M["qux"] = N;
  N["baz"] = Doc.getNode(true);
  std::string Buffer;
  raw_string_ostream OStream(Buffer);
  Doc.toYAML(OStream);
  ASSERT_EQ(OStream.str(), "---\n"
                           "bar:             0x2\n"
                           "foo:             1\n"
                           "qux:             \n"
                           "  baz:             true\n"
                           "...\n");
}

TEST(MsgPackDocument, TestInputYAMLMap) {
  Document Doc;
  bool Ok = Doc.fromYAML("---\nfoo: !int 0x1\nbaz: !str 2\n...\n");
  ASSERT_TRUE(Ok);
  ASSERT_EQ(Doc.getRoot().getKind(), Type::Map);
  auto M = Doc.getRoot().getMap();
  ASSERT_EQ(M.size(), 2u);
  auto SI = M["foo"];
  ASSERT_EQ(SI.getKind(), Type::UInt);
  ASSERT_EQ(SI.getUInt(), 1u);
  auto SS = M["baz"];
  ASSERT_EQ(SS.getKind(), Type::String);
  ASSERT_EQ(SS.getString(), "2");
}
