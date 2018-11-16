//===- MsgPackTypesTest.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/MsgPackTypes.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace msgpack;

TEST(MsgPackTypes, TestReadInt) {
  Reader MPReader(StringRef("\xd0\x00", 2));
  auto OptNodeOrErr = Node::read(MPReader);
  ASSERT_TRUE(static_cast<bool>(OptNodeOrErr));
  ASSERT_TRUE(*OptNodeOrErr);
  auto *S = dyn_cast<ScalarNode>((*OptNodeOrErr)->get());
  ASSERT_TRUE(S);
  ASSERT_EQ(S->getScalarKind(), ScalarNode::SK_Int);
  ASSERT_EQ(S->getInt(), 0);
}

TEST(MsgPackTypes, TestReadArray) {
  Reader MPReader(StringRef("\x92\xd0\x01\xc0"));
  auto OptNodeOrErr = Node::read(MPReader);
  ASSERT_TRUE(static_cast<bool>(OptNodeOrErr));
  ASSERT_TRUE(*OptNodeOrErr);
  auto *A = dyn_cast<ArrayNode>((*OptNodeOrErr)->get());
  ASSERT_TRUE(A);
  ASSERT_EQ(A->size(), 2u);
  auto *SI = dyn_cast<ScalarNode>((*A)[0].get());
  ASSERT_TRUE(SI);
  ASSERT_EQ(SI->getScalarKind(), ScalarNode::SK_Int);
  ASSERT_EQ(SI->getInt(), 1);
  auto *SN = dyn_cast<ScalarNode>((*A)[1].get());
  ASSERT_TRUE(SN);
  ASSERT_EQ(SN->getScalarKind(), ScalarNode::SK_Nil);
}

TEST(MsgPackTypes, TestReadMap) {
  Reader MPReader(StringRef("\x82\xa3"
                            "foo"
                            "\xd0\x01\xa3"
                            "bar"
                            "\xd0\x02"));
  auto OptNodeOrErr = Node::read(MPReader);
  ASSERT_TRUE(static_cast<bool>(OptNodeOrErr));
  ASSERT_TRUE(*OptNodeOrErr);
  auto *A = dyn_cast<MapNode>((*OptNodeOrErr)->get());
  ASSERT_TRUE(A);
  ASSERT_EQ(A->size(), 2u);
  auto *FooS = dyn_cast<ScalarNode>((*A)["foo"].get());
  ASSERT_TRUE(FooS);
  ASSERT_EQ(FooS->getScalarKind(), ScalarNode::SK_Int);
  ASSERT_EQ(FooS->getInt(), 1);
  auto *BarS = dyn_cast<ScalarNode>((*A)["bar"].get());
  ASSERT_TRUE(BarS);
  ASSERT_EQ(BarS->getScalarKind(), ScalarNode::SK_Int);
  ASSERT_EQ(BarS->getInt(), 2);
}

TEST(MsgPackTypes, TestWriteInt) {
  std::string Buffer;
  raw_string_ostream OStream(Buffer);
  Writer MPWriter(OStream);
  ScalarNode I(int64_t(1));
  I.write(MPWriter);
  ASSERT_EQ(OStream.str(), "\x01");
}

TEST(MsgPackTypes, TestWriteArray) {
  std::string Buffer;
  raw_string_ostream OStream(Buffer);
  Writer MPWriter(OStream);
  ArrayNode A;
  A.push_back(std::make_shared<ScalarNode>(int64_t(1)));
  A.push_back(std::make_shared<ScalarNode>());
  A.write(MPWriter);
  ASSERT_EQ(OStream.str(), "\x92\x01\xc0");
}

TEST(MsgPackTypes, TestWriteMap) {
  std::string Buffer;
  raw_string_ostream OStream(Buffer);
  Writer MPWriter(OStream);
  MapNode M;
  M["foo"] = std::make_shared<ScalarNode>(int64_t(1));
  M["bar"] = std::make_shared<ScalarNode>(int64_t(2));
  M.write(MPWriter);
  ASSERT_EQ(OStream.str(), "\x82\xa3"
                           "foo"
                           "\x01\xa3"
                           "bar"
                           "\x02");
}

TEST(MsgPackTypes, TestOutputYAMLArray) {
  std::string Buffer;
  raw_string_ostream OStream(Buffer);
  yaml::Output yout(OStream);
  ArrayNode A;
  A.push_back(std::make_shared<ScalarNode>(int64_t(1)));
  A.push_back(std::make_shared<ScalarNode>(int64_t(2)));
  yout << A;
  ASSERT_EQ(OStream.str(), "---\n- !int 1\n- !int 2\n...\n");
}

TEST(MsgPackTypes, TestInputYAMLArray) {
  NodePtr RootNode;
  yaml::Input yin("---\n- !int 1\n- !str 2\n...\n");
  yin >> RootNode;
  auto *A = dyn_cast<ArrayNode>(RootNode.get());
  ASSERT_TRUE(A);
  ASSERT_EQ(A->size(), 2u);
  auto *SI = dyn_cast<ScalarNode>((*A)[0].get());
  ASSERT_TRUE(SI);
  ASSERT_EQ(SI->getScalarKind(), ScalarNode::SK_UInt);
  ASSERT_EQ(SI->getUInt(), 1u);
  auto *SS = dyn_cast<ScalarNode>((*A)[1].get());
  ASSERT_TRUE(SS);
  ASSERT_EQ(SS->getScalarKind(), ScalarNode::SK_String);
  ASSERT_EQ(SS->getString(), "2");
}

TEST(MsgPackTypes, TestOutputYAMLMap) {
  std::string Buffer;
  raw_string_ostream OStream(Buffer);
  yaml::Output yout(OStream);
  MapNode M;
  M["foo"] = std::make_shared<ScalarNode>(int64_t(1));
  M["bar"] = std::make_shared<ScalarNode>(uint64_t(2));
  auto N = std::make_shared<MapNode>();
  (*N)["baz"] = std::make_shared<ScalarNode>(true);
  M["qux"] = std::move(N);
  yout << M;
  ASSERT_EQ(OStream.str(), "---\nfoo:             !int 1\nbar:             "
                           "!int 2\nqux:             \n  baz:             "
                           "!bool true\n...\n");
}

TEST(MsgPackTypes, TestInputYAMLMap) {
  NodePtr RootNode;
  yaml::Input yin("---\nfoo: !int 1\nbaz: !str 2\n...\n");
  yin >> RootNode;
  auto *M = dyn_cast<MapNode>(RootNode.get());
  ASSERT_TRUE(M);
  ASSERT_EQ(M->size(), 2u);
  auto *SI = dyn_cast<ScalarNode>((*M)["foo"].get());
  ASSERT_TRUE(SI);
  ASSERT_EQ(SI->getScalarKind(), ScalarNode::SK_UInt);
  ASSERT_EQ(SI->getUInt(), 1u);
  auto *SS = dyn_cast<ScalarNode>((*M)["baz"].get());
  ASSERT_TRUE(SS);
  ASSERT_EQ(SS->getScalarKind(), ScalarNode::SK_String);
  ASSERT_EQ(SS->getString(), "2");
}

// Test that the document is parsed into a tree of shared_ptr where each node
// can have multiple owners.
TEST(MsgPackTypes, TestInputShared) {
  yaml::Input yin("---\nfoo:\n  bar: !int 1\n...\n");
  NodePtr InnerMap;
  NodePtr IntNode;
  {
    {
      {
        NodePtr RootNode;
        yin >> RootNode;
        auto *M = dyn_cast<MapNode>(RootNode.get());
        ASSERT_TRUE(M);
        ASSERT_EQ(M->size(), 1u);
        InnerMap = (*M)["foo"];
      }
      auto *N = dyn_cast<MapNode>(InnerMap.get());
      ASSERT_TRUE(N);
      ASSERT_EQ(N->size(), 1u);
      IntNode = (*N)["bar"];
    }
    auto *S = dyn_cast<ScalarNode>(IntNode.get());
    ASSERT_TRUE(S);
    ASSERT_EQ(S->getScalarKind(), ScalarNode::SK_UInt);
    ASSERT_EQ(S->getUInt(), 1u);
  }
}
