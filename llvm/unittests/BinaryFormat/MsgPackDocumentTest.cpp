//===- MsgPackDocumentTest.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace msgpack;

TEST(MsgPackDocument, DocNodeTest) {
  Document Doc;

  DocNode Int1 = Doc.getNode(1), Int2 = Doc.getNode(2);
  DocNode Str1 = Doc.getNode("ab"), Str2 = Doc.getNode("ab");

  ASSERT_TRUE(Int1 != Int2);
  ASSERT_TRUE(Str1 == Str2);
}

TEST(MsgPackDocument, TestReadInt) {
  Document Doc;
  bool Ok = Doc.readFromBlob(StringRef("\xd0\x00", 2), /*Multi=*/false);
  ASSERT_TRUE(Ok);
  ASSERT_EQ(Doc.getRoot().getKind(), Type::Int);
  ASSERT_EQ(Doc.getRoot().getInt(), 0);
}

TEST(MsgPackDocument, TestReadBinary) {
  Document Doc;
  uint8_t data[] = {1, 2, 3, 4};
  bool Ok =
      Doc.readFromBlob(StringRef("\xC4\x4\x1\x2\x3\x4", 6), /*Multi=*/false);
  ASSERT_TRUE(Ok);
  ASSERT_EQ(Doc.getRoot().getKind(), Type::Binary);
  ASSERT_EQ(Doc.getRoot().getBinary().getBuffer(),
            StringRef(reinterpret_cast<const char *>(data), 4));
}

TEST(MsgPackDocument, TestReadMergeArray) {
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

  Ok = Doc.readFromBlob(StringRef("\x91\xd0\x2a"), /*Multi=*/false,
                        [](DocNode *DestNode, DocNode SrcNode, DocNode MapKey) {
                          // Allow array, merging into existing elements, ORing
                          // ints.
                          if (DestNode->getKind() == Type::Int &&
                              SrcNode.getKind() == Type::Int) {
                            *DestNode = DestNode->getDocument()->getNode(
                                DestNode->getInt() | SrcNode.getInt());
                            return 0;
                          }
                          return DestNode->isArray() && SrcNode.isArray() ? 0
                                                                          : -1;
                        });
  ASSERT_TRUE(Ok);
  A = Doc.getRoot().getArray();
  ASSERT_EQ(A.size(), 2u);
  SI = A[0];
  ASSERT_EQ(SI.getKind(), Type::Int);
  ASSERT_EQ(SI.getInt(), 43);
  SN = A[1];
  ASSERT_EQ(SN.getKind(), Type::Nil);
}

TEST(MsgPackDocument, TestReadAppendArray) {
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

  Ok = Doc.readFromBlob(StringRef("\x91\xd0\x2a"), /*Multi=*/false,
                        [](DocNode *DestNode, DocNode SrcNode, DocNode MapKey) {
                          // Allow array, appending after existing elements
                          return DestNode->isArray() && SrcNode.isArray()
                                     ? DestNode->getArray().size()
                                     : -1;
                        });
  ASSERT_TRUE(Ok);
  A = Doc.getRoot().getArray();
  ASSERT_EQ(A.size(), 3u);
  SI = A[0];
  ASSERT_EQ(SI.getKind(), Type::Int);
  ASSERT_EQ(SI.getInt(), 1);
  SN = A[1];
  ASSERT_EQ(SN.getKind(), Type::Nil);
  SI = A[2];
  ASSERT_EQ(SI.getKind(), Type::Int);
  ASSERT_EQ(SI.getInt(), 42);
}

TEST(MsgPackDocument, TestReadMergeMap) {
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

  Ok = Doc.readFromBlob(StringRef("\x82\xa3"
                                  "foz"
                                  "\xd0\x03\xa3"
                                  "baz"
                                  "\xd0\x04"),
                        /*Multi=*/false,
                        [](DocNode *DestNode, DocNode SrcNode, DocNode MapKey) {
                          return DestNode->isMap() && SrcNode.isMap() ? 0 : -1;
                        });
  ASSERT_TRUE(Ok);
  ASSERT_EQ(M.size(), 4u);
  FooS = M["foo"];
  ASSERT_EQ(FooS.getKind(), Type::Int);
  ASSERT_EQ(FooS.getInt(), 1);
  BarS = M["bar"];
  ASSERT_EQ(BarS.getKind(), Type::Int);
  ASSERT_EQ(BarS.getInt(), 2);
  auto FozS = M["foz"];
  ASSERT_EQ(FozS.getKind(), Type::Int);
  ASSERT_EQ(FozS.getInt(), 3);
  auto BazS = M["baz"];
  ASSERT_EQ(BazS.getKind(), Type::Int);
  ASSERT_EQ(BazS.getInt(), 4);

  Ok = Doc.readFromBlob(
      StringRef("\x82\xa3"
                "foz"
                "\xd0\x06\xa3"
                "bay"
                "\xd0\x08"),
      /*Multi=*/false, [](DocNode *Dest, DocNode Src, DocNode MapKey) {
        // Merger function that merges two ints by ORing their values, as long
        // as the map key is "foz".
        if (Src.isMap())
          return Dest->isMap();
        if (Src.isArray())
          return Dest->isArray();
        if (MapKey.isString() && MapKey.getString() == "foz" &&
            Dest->getKind() == Type::Int && Src.getKind() == Type::Int) {
          *Dest = Src.getDocument()->getNode(Dest->getInt() | Src.getInt());
          return true;
        }
        return false;
      });
  ASSERT_TRUE(Ok);
  ASSERT_EQ(M.size(), 5u);
  FooS = M["foo"];
  ASSERT_EQ(FooS.getKind(), Type::Int);
  ASSERT_EQ(FooS.getInt(), 1);
  BarS = M["bar"];
  ASSERT_EQ(BarS.getKind(), Type::Int);
  ASSERT_EQ(BarS.getInt(), 2);
  FozS = M["foz"];
  ASSERT_EQ(FozS.getKind(), Type::Int);
  ASSERT_EQ(FozS.getInt(), 7);
  BazS = M["baz"];
  ASSERT_EQ(BazS.getKind(), Type::Int);
  ASSERT_EQ(BazS.getInt(), 4);
  auto BayS = M["bay"];
  ASSERT_EQ(BayS.getKind(), Type::Int);
  ASSERT_EQ(BayS.getInt(), 8);
}

TEST(MsgPackDocument, TestWriteInt) {
  Document Doc;
  Doc.getRoot() = 1;
  std::string Buffer;
  Doc.writeToBlob(Buffer);
  ASSERT_EQ(Buffer, "\x01");
}

TEST(MsgPackDocument, TestWriteBinary) {
  uint8_t data[] = {1, 2, 3, 4};
  Document Doc;
  Doc.getRoot() = MemoryBufferRef(
      StringRef(reinterpret_cast<const char *>(data), sizeof(data)), "");
  std::string Buffer;
  Doc.writeToBlob(Buffer);
  ASSERT_EQ(Buffer, "\xC4\x4\x1\x2\x3\x4");
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
  M["foo"] = 1;
  M["bar"] = 2;
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
  M["foo"] = 1;
  M["bar"] = 2U;
  auto N = Doc.getMapNode();
  M["qux"] = N;
  N["baz"] = true;
  std::string Buffer;
  raw_string_ostream OStream(Buffer);
  Doc.toYAML(OStream);
  ASSERT_EQ(OStream.str(), "---\n"
                           "bar:             2\n"
                           "foo:             1\n"
                           "qux:\n"
                           "  baz:             true\n"
                           "...\n");
}

TEST(MsgPackDocument, TestOutputYAMLMapWithErase) {
  Document Doc;
  auto M = Doc.getRoot().getMap(/*Convert=*/true);
  M["foo"] = 1;
  M["bar"] = 2U;
  auto N = Doc.getMapNode();
  M["qux"] = N;
  N["baz"] = true;
  M.erase(Doc.getNode("bar"));
  std::string Buffer;
  raw_string_ostream OStream(Buffer);
  Doc.toYAML(OStream);
  ASSERT_EQ(OStream.str(), "---\n"
                           "foo:             1\n"
                           "qux:\n"
                           "  baz:             true\n"
                           "...\n");
}

TEST(MsgPackDocument, TestOutputYAMLMapHex) {
  Document Doc;
  Doc.setHexMode();
  auto M = Doc.getRoot().getMap(/*Convert=*/true);
  M["foo"] = 1;
  M["bar"] = 2U;
  auto N = Doc.getMapNode();
  M["qux"] = N;
  N["baz"] = true;
  std::string Buffer;
  raw_string_ostream OStream(Buffer);
  Doc.toYAML(OStream);
  ASSERT_EQ(OStream.str(), "---\n"
                           "bar:             0x2\n"
                           "foo:             1\n"
                           "qux:\n"
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
