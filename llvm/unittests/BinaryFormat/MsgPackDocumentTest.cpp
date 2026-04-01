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

TEST(MsgPackDocument, DocNodeEmptyAndNilEquality) {
  Document Doc1, Doc2;

  // Two empty nodes are equal.
  EXPECT_FALSE(Doc1.getEmptyNode() != Doc1.getEmptyNode());
  // Cross-document empty nodes are equal.
  EXPECT_FALSE(Doc1.getEmptyNode() != Doc2.getEmptyNode());
  // Default-constructed (no document) empty nodes are equal.
  DocNode Default1, Default2;
  EXPECT_FALSE(Default1 != Default2);
  // Empty vs non-empty are not equal.
  EXPECT_FALSE(Doc1.getEmptyNode() == Doc1.getNode(0));

  // Two nil nodes are equal.
  EXPECT_FALSE(Doc1.getNode() != Doc1.getNode());
  // Cross-document nil nodes are equal.
  EXPECT_FALSE(Doc1.getNode() != Doc2.getNode());
}

TEST(MsgPackDocument, DocNodeDifferentTypesNotEqual) {
  Document Doc;
  // Int vs String.
  EXPECT_FALSE(Doc.getNode(1) == Doc.getNode("1"));
  // Int vs UInt.
  EXPECT_FALSE(Doc.getNode(int64_t(1)) == Doc.getNode(uint64_t(1)));
  // Boolean vs Int.
  EXPECT_FALSE(Doc.getNode(true) == Doc.getNode(1));
  // Scalar vs Array.
  DocNode Arr = Doc.getArrayNode();
  EXPECT_FALSE(Doc.getNode(1) == Arr);
  // Scalar vs Map.
  DocNode Map = Doc.getMapNode();
  EXPECT_FALSE(Doc.getNode(1) == Map);
  // Array vs Map.
  EXPECT_FALSE(Arr == Map);
}

TEST(MsgPackDocument, DocNodeCrossDocumentScalarEquality) {
  Document Doc1, Doc2;
  // Same values across documents should be equal.
  EXPECT_FALSE(Doc1.getNode(42) != Doc2.getNode(42));
  EXPECT_FALSE(Doc1.getNode(42U) != Doc2.getNode(42U));
  EXPECT_FALSE(Doc1.getNode(int64_t(-1)) != Doc2.getNode(int64_t(-1)));
  EXPECT_FALSE(Doc1.getNode(true) != Doc2.getNode(true));
  EXPECT_FALSE(Doc1.getNode(3.14) != Doc2.getNode(3.14));
  EXPECT_FALSE(Doc1.getNode("hello") != Doc2.getNode("hello"));

  // Different values across documents should not be equal.
  EXPECT_FALSE(Doc1.getNode(1) == Doc2.getNode(2));
  EXPECT_FALSE(Doc1.getNode("foo") == Doc2.getNode("bar"));
  EXPECT_FALSE(Doc1.getNode(true) == Doc2.getNode(false));
  EXPECT_FALSE(Doc1.getNode(1U) == Doc2.getNode(2U));
}

TEST(MsgPackDocument, DocNodeCrossDocumentScalarOrdering) {
  Document Doc1, Doc2;
  // operator< should compare by value across documents.
  EXPECT_TRUE(Doc1.getNode(1) < Doc2.getNode(2));
  EXPECT_FALSE(Doc1.getNode(2) < Doc2.getNode(1));
  EXPECT_FALSE(Doc1.getNode(1) < Doc2.getNode(1));

  EXPECT_TRUE(Doc1.getNode("abc") < Doc2.getNode("def"));
  EXPECT_FALSE(Doc1.getNode("def") < Doc2.getNode("abc"));

  EXPECT_TRUE(Doc1.getNode(1U) < Doc2.getNode(2U));
  EXPECT_FALSE(Doc1.getNode(2U) < Doc2.getNode(1U));
}

TEST(MsgPackDocument, DocNodeArrayEquality) {
  Document Doc;
  auto A1 = Doc.getArrayNode();
  A1.push_back(Doc.getNode(int64_t(1)));
  A1.push_back(Doc.getNode(int64_t(2)));

  auto A2 = Doc.getArrayNode();
  A2.push_back(Doc.getNode(int64_t(1)));
  A2.push_back(Doc.getNode(int64_t(2)));

  // Same contents should be equal.
  DocNode N1 = A1, N2 = A2;
  EXPECT_FALSE(N1 != N2);

  // Different contents should not be equal.
  auto A3 = Doc.getArrayNode();
  A3.push_back(Doc.getNode(int64_t(1)));
  A3.push_back(Doc.getNode(int64_t(99)));
  DocNode N3 = A3;
  EXPECT_FALSE(N1 == N3);

  // Different sizes should not be equal.
  auto A4 = Doc.getArrayNode();
  A4.push_back(Doc.getNode(int64_t(1)));
  DocNode N4 = A4;
  EXPECT_FALSE(N1 == N4);
}

TEST(MsgPackDocument, DocNodeMapEquality) {
  Document Doc;
  auto M1 = Doc.getMapNode();
  M1["x"] = 10;
  M1["y"] = 20;

  auto M2 = Doc.getMapNode();
  M2["x"] = 10;
  M2["y"] = 20;

  DocNode N1 = M1, N2 = M2;
  EXPECT_FALSE(N1 != N2);

  // Different value.
  auto M3 = Doc.getMapNode();
  M3["x"] = 10;
  M3["y"] = 99;
  DocNode N3 = M3;
  EXPECT_FALSE(N1 == N3);

  // Different key.
  auto M4 = Doc.getMapNode();
  M4["x"] = 10;
  M4["z"] = 20;
  DocNode N4 = M4;
  EXPECT_FALSE(N1 == N4);
}

TEST(MsgPackDocument, DocNodeCrossDocumentArrayEquality) {
  Document Doc1, Doc2;
  auto A1 = Doc1.getArrayNode();
  A1.push_back(Doc1.getNode("hello"));
  A1.push_back(Doc1.getNode(42));

  auto A2 = Doc2.getArrayNode();
  A2.push_back(Doc2.getNode("hello"));
  A2.push_back(Doc2.getNode(42));

  DocNode N1 = A1, N2 = A2;
  EXPECT_FALSE(N1 != N2);

  auto A3 = Doc2.getArrayNode();
  A3.push_back(Doc2.getNode("hello"));
  A3.push_back(Doc2.getNode(99));
  DocNode N3 = A3;
  EXPECT_FALSE(N1 == N3);
}

TEST(MsgPackDocument, DocNodeCrossDocumentMapEquality) {
  Document Doc1, Doc2;
  auto M1 = Doc1.getMapNode();
  M1["key"] = Doc1.getNode("value");

  auto M2 = Doc2.getMapNode();
  M2["key"] = Doc2.getNode("value");

  DocNode N1 = M1, N2 = M2;
  EXPECT_FALSE(N1 != N2);

  auto M3 = Doc2.getMapNode();
  M3["key"] = Doc2.getNode("other");
  DocNode N3 = M3;
  EXPECT_FALSE(N1 == N3);
}

TEST(MsgPackDocument, CopyNodeEmpty) {
  Document Src, Dst;
  auto Copied = Dst.copyNode(Src.getEmptyNode());
  EXPECT_TRUE(Copied.isEmpty());
}

TEST(MsgPackDocument, CopyNodeScalar) {
  Document Src, Dst;
  auto Node = Src.getNode("hello", /*Copy=*/true);
  auto Copied = Dst.copyNode(Node);
  EXPECT_FALSE(Node != Copied);
  EXPECT_EQ(Copied.getKind(), Type::String);
  EXPECT_EQ(Copied.getString(), "hello");
}

TEST(MsgPackDocument, CopyNodeArray) {
  Document Src, Dst;
  auto A = Src.getArrayNode();
  A.push_back(Src.getNode(int64_t(1)));
  A.push_back(Src.getNode("two", /*Copy=*/true));
  A.push_back(Src.getNode(3.0));

  DocNode SrcNode = A;
  auto Copied = Dst.copyNode(SrcNode);
  EXPECT_FALSE(SrcNode != Copied);
  EXPECT_EQ(Copied.getArray().size(), 3u);
  EXPECT_EQ(Copied.getArray()[0].getInt(), int64_t(1));
  EXPECT_EQ(Copied.getArray()[1].getString(), "two");
  EXPECT_EQ(Copied.getArray()[2].getFloat(), 3.0);
}

TEST(MsgPackDocument, CopyNodeMap) {
  Document Src, Dst;
  auto M = Src.getMapNode();
  M["name"] = Src.getNode("test", /*Copy=*/true);
  M["count"] = 42;

  DocNode SrcNode = M;
  auto Copied = Dst.copyNode(SrcNode);
  EXPECT_FALSE(SrcNode != Copied);

  // Verify by iterating the map directly.
  auto &CopiedMap = Copied.getMap();
  EXPECT_EQ(CopiedMap.size(), 2u);
  for (auto &Entry : CopiedMap) {
    EXPECT_TRUE(Entry.first.isString());
    if (Entry.first.getString() == "name")
      EXPECT_EQ(Entry.second.getString(), "test");
    else if (Entry.first.getString() == "count")
      EXPECT_EQ(Entry.second.getInt(), int64_t(42));
    else
      FAIL() << "unexpected key: " << Entry.first.toString();
  }
}

TEST(MsgPackDocument, CopyNodeNested) {
  Document Src, Dst;
  auto M = Src.getMapNode();
  auto Inner = Src.getArrayNode();
  Inner.push_back(Src.getNode(int64_t(1)));
  Inner.push_back(Src.getNode(int64_t(2)));
  M["arr"] = Inner;
  M["val"] = Src.getNode("x", /*Copy=*/true);

  DocNode SrcNode = M;
  auto Copied = Dst.copyNode(SrcNode);
  EXPECT_FALSE(SrcNode != Copied);
  auto &CopiedMap = Copied.getMap();
  EXPECT_TRUE(CopiedMap["arr"].isArray());
  EXPECT_EQ(CopiedMap["arr"].getArray().size(), 2u);
  EXPECT_EQ(CopiedMap["arr"].getArray()[0].getInt(), int64_t(1));
}

TEST(MsgPackDocument, CopyNodeIndependence) {
  Document Src, Dst;
  auto A = Src.getArrayNode();
  A.push_back(Src.getNode(int64_t(1)));
  A.push_back(Src.getNode(int64_t(2)));

  DocNode SrcNode = A;
  auto Copied = Dst.copyNode(SrcNode);
  EXPECT_FALSE(SrcNode != Copied);

  // Mutate the source — the copy should be unaffected.
  A.push_back(Src.getNode(int64_t(3)));
  EXPECT_EQ(A.size(), 3u);
  EXPECT_EQ(Copied.getArray().size(), 2u);

  // Mutate the copy — the source should be unaffected.
  Copied.getArray().push_back(Dst.getNode(int64_t(10)));
  Copied.getArray().push_back(Dst.getNode(int64_t(20)));
  EXPECT_EQ(Copied.getArray().size(), 4u);
  EXPECT_EQ(A.size(), 3u);
}

TEST(MsgPackDocument, TestReadBoolean) {
  Document Doc1;
  bool Ok = Doc1.readFromBlob(StringRef("\xC2", 1), /*Multi=*/false);
  ASSERT_TRUE(Ok);
  ASSERT_EQ(Doc1.getRoot().getKind(), Type::Boolean);
  ASSERT_EQ(Doc1.getRoot().getBool(), false);
  Document Doc2;
  Ok = Doc2.readFromBlob(StringRef("\xC3", 1), /*Multi=*/false);
  ASSERT_TRUE(Ok);
  ASSERT_EQ(Doc2.getRoot().getKind(), Type::Boolean);
  ASSERT_EQ(Doc2.getRoot().getBool(), true);
}

TEST(MsgPackDocument, TestReadInt) {
  Document Doc1;
  bool Ok = Doc1.readFromBlob(StringRef("\xD0\x00", 2), /*Multi=*/false);
  ASSERT_TRUE(Ok);
  ASSERT_EQ(Doc1.getRoot().getKind(), Type::Int);
  ASSERT_EQ(Doc1.getRoot().getInt(), 0);
  Document Doc2;
  Ok = Doc2.readFromBlob(StringRef("\xFF", 1), /*Multi=*/false);
  ASSERT_TRUE(Ok);
  ASSERT_EQ(Doc2.getRoot().getKind(), Type::Int);
  ASSERT_EQ(Doc2.getRoot().getInt(), -1);
}

TEST(MsgPackDocument, TestReadUInt) {
  Document Doc1;
  bool Ok = Doc1.readFromBlob(StringRef("\xCC\x00", 2), /*Multi=*/false);
  ASSERT_TRUE(Ok);
  ASSERT_EQ(Doc1.getRoot().getKind(), Type::UInt);
  ASSERT_EQ(Doc1.getRoot().getUInt(), 0U);
  Document Doc2;
  Ok = Doc2.readFromBlob(StringRef("\x01", 1), /*Multi=*/false);
  ASSERT_TRUE(Ok);
  ASSERT_EQ(Doc2.getRoot().getKind(), Type::UInt);
  ASSERT_EQ(Doc2.getRoot().getUInt(), 1U);
}

TEST(MsgPackDocument, TestReadFloat) {
  Document Doc1;
  bool Ok =
      Doc1.readFromBlob(StringRef("\xCA\x3F\x80\x00\x00", 5), /*Multi=*/false);
  ASSERT_TRUE(Ok);
  ASSERT_EQ(Doc1.getRoot().getKind(), Type::Float);
  ASSERT_EQ(Doc1.getRoot().getFloat(), 1.0);
  Document Doc2;
  Ok = Doc2.readFromBlob(StringRef("\xCB\x48\x3D\x63\x29\xF1\xC3\x5C\xA5", 9),
                         /*Multi=*/false);
  ASSERT_TRUE(Ok);
  ASSERT_EQ(Doc2.getRoot().getKind(), Type::Float);
  ASSERT_EQ(Doc2.getRoot().getFloat(), 1e40);
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

TEST(MsgPackDocument, TestWriteBoolean) {
  Document Doc;
  Doc.getRoot() = true;
  std::string Buffer;
  Doc.writeToBlob(Buffer);
  ASSERT_EQ(Buffer, "\xc3");
  Doc.getRoot() = false;
  Doc.writeToBlob(Buffer);
  ASSERT_EQ(Buffer, "\xc2");
}

TEST(MsgPackDocument, TestWriteInt) {
  Document Doc;
  Doc.getRoot() = 1;
  std::string Buffer;
  Doc.writeToBlob(Buffer);
  ASSERT_EQ(Buffer, "\x01");
  Doc.getRoot() = -1;
  Doc.writeToBlob(Buffer);
  ASSERT_EQ(Buffer, "\xFF");
  Doc.getRoot() = -4096;
  Doc.writeToBlob(Buffer);
  ASSERT_EQ(Buffer, StringRef("\xD1\xF0\x00", 3));
}

TEST(MsgPackDocument, TestWriteUInt) {
  Document Doc;
  Doc.getRoot() = 1U;
  std::string Buffer;
  Doc.writeToBlob(Buffer);
  ASSERT_EQ(Buffer, "\x01");
  Doc.getRoot() = 4096U;
  Doc.writeToBlob(Buffer);
  ASSERT_EQ(Buffer, StringRef("\xCD\x10\x00", 3));
}

TEST(MsgPackDocument, TestWriteFloat) {
  Document Doc;
  Doc.getRoot() = 1.0;
  std::string Buffer;
  Doc.writeToBlob(Buffer);
  ASSERT_EQ(Buffer, StringRef("\xCA\x3F\x80\x00\x00", 5));
  Doc.getRoot() = 1.0f;
  Doc.writeToBlob(Buffer);
  ASSERT_EQ(Buffer, StringRef("\xCA\x3F\x80\x00\x00", 5));
  Doc.getRoot() = 1e40;
  Doc.writeToBlob(Buffer);
  ASSERT_EQ(Buffer, "\xCB\x48\x3D\x63\x29\xF1\xC3\x5C\xA5");
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

TEST(MsgPackDocument, TestYAMLBoolean) {
  Document Doc;
  auto GetFirst = [](Document &Doc) { return Doc.getRoot().getArray()[0]; };
  auto ToYAML = [](Document &Doc) {
    std::string S;
    raw_string_ostream OS(S);
    Doc.toYAML(OS);
    return S;
  };

  bool Ok;

  Ok = Doc.fromYAML("- n\n");
  ASSERT_TRUE(Ok);
  ASSERT_EQ(GetFirst(Doc).getKind(), Type::String);
  ASSERT_EQ(GetFirst(Doc).getString(), "n");
  ASSERT_EQ(ToYAML(Doc), "---\n- n\n...\n");

  Ok = Doc.fromYAML("- y\n");
  ASSERT_TRUE(Ok);
  ASSERT_EQ(GetFirst(Doc).getKind(), Type::String);
  ASSERT_EQ(GetFirst(Doc).getString(), "y");
  ASSERT_EQ(ToYAML(Doc), "---\n- y\n...\n");

  Ok = Doc.fromYAML("- no\n");
  ASSERT_TRUE(Ok);
  ASSERT_EQ(GetFirst(Doc).getKind(), Type::String);
  ASSERT_EQ(GetFirst(Doc).getString(), "no");
  ASSERT_EQ(ToYAML(Doc), "---\n- no\n...\n");

  Ok = Doc.fromYAML("- yes\n");
  ASSERT_TRUE(Ok);
  ASSERT_EQ(GetFirst(Doc).getKind(), Type::String);
  ASSERT_EQ(GetFirst(Doc).getString(), "yes");
  ASSERT_EQ(ToYAML(Doc), "---\n- yes\n...\n");

  Ok = Doc.fromYAML("- false\n");
  ASSERT_TRUE(Ok);
  ASSERT_EQ(GetFirst(Doc).getKind(), Type::Boolean);
  ASSERT_EQ(GetFirst(Doc).getBool(), false);
  ASSERT_EQ(ToYAML(Doc), "---\n- false\n...\n");

  Ok = Doc.fromYAML("- true\n");
  ASSERT_TRUE(Ok);
  ASSERT_EQ(GetFirst(Doc).getKind(), Type::Boolean);
  ASSERT_EQ(GetFirst(Doc).getBool(), true);
  ASSERT_EQ(ToYAML(Doc), "---\n- true\n...\n");

  Ok = Doc.fromYAML("- !str false\n");
  ASSERT_TRUE(Ok);
  ASSERT_EQ(GetFirst(Doc).getKind(), Type::String);
  ASSERT_EQ(GetFirst(Doc).getString(), "false");
  ASSERT_EQ(ToYAML(Doc), "---\n- !str 'false'\n...\n");

  Ok = Doc.fromYAML("- !str true\n");
  ASSERT_TRUE(Ok);
  ASSERT_EQ(GetFirst(Doc).getKind(), Type::String);
  ASSERT_EQ(GetFirst(Doc).getString(), "true");
  ASSERT_EQ(ToYAML(Doc), "---\n- !str 'true'\n...\n");

  Ok = Doc.fromYAML("- !bool false\n");
  ASSERT_TRUE(Ok);
  ASSERT_EQ(GetFirst(Doc).getKind(), Type::Boolean);
  ASSERT_EQ(GetFirst(Doc).getBool(), false);
  ASSERT_EQ(ToYAML(Doc), "---\n- false\n...\n");

  Ok = Doc.fromYAML("- !bool true\n");
  ASSERT_TRUE(Ok);
  ASSERT_EQ(GetFirst(Doc).getKind(), Type::Boolean);
  ASSERT_EQ(GetFirst(Doc).getBool(), true);
  ASSERT_EQ(ToYAML(Doc), "---\n- true\n...\n");

  // FIXME: A fix for these requires changes in YAMLParser/YAMLTraits.
  Ok = Doc.fromYAML("- \"false\"\n");
  ASSERT_TRUE(Ok);
  ASSERT_EQ(GetFirst(Doc).getKind(), Type::Boolean);
  ASSERT_EQ(GetFirst(Doc).getBool(), false);
  ASSERT_EQ(ToYAML(Doc), "---\n- false\n...\n");

  Ok = Doc.fromYAML("- \"true\"\n");
  ASSERT_TRUE(Ok);
  ASSERT_EQ(GetFirst(Doc).getKind(), Type::Boolean);
  ASSERT_EQ(GetFirst(Doc).getBool(), true);
  ASSERT_EQ(ToYAML(Doc), "---\n- true\n...\n");
}
