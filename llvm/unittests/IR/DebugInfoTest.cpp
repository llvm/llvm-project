//===- llvm/unittest/IR/DebugInfo.cpp - DebugInfo tests -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugInfoMetadata.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(DINodeTest, getFlag) {
  // Some valid flags.
  EXPECT_EQ(DINode::FlagPublic, DINode::getFlag("DIFlagPublic"));
  EXPECT_EQ(DINode::FlagProtected, DINode::getFlag("DIFlagProtected"));
  EXPECT_EQ(DINode::FlagPrivate, DINode::getFlag("DIFlagPrivate"));
  EXPECT_EQ(DINode::FlagVector, DINode::getFlag("DIFlagVector"));
  EXPECT_EQ(DINode::FlagRValueReference,
            DINode::getFlag("DIFlagRValueReference"));

  // FlagAccessibility shouldn't work.
  EXPECT_EQ(0u, DINode::getFlag("DIFlagAccessibility"));

  // Some other invalid strings.
  EXPECT_EQ(0u, DINode::getFlag("FlagVector"));
  EXPECT_EQ(0u, DINode::getFlag("Vector"));
  EXPECT_EQ(0u, DINode::getFlag("other things"));
  EXPECT_EQ(0u, DINode::getFlag("DIFlagOther"));
}

TEST(DINodeTest, getFlagString) {
  // Some valid flags.
  EXPECT_EQ(StringRef("DIFlagPublic"),
            DINode::getFlagString(DINode::FlagPublic));
  EXPECT_EQ(StringRef("DIFlagProtected"),
            DINode::getFlagString(DINode::FlagProtected));
  EXPECT_EQ(StringRef("DIFlagPrivate"),
            DINode::getFlagString(DINode::FlagPrivate));
  EXPECT_EQ(StringRef("DIFlagVector"),
            DINode::getFlagString(DINode::FlagVector));
  EXPECT_EQ(StringRef("DIFlagRValueReference"),
            DINode::getFlagString(DINode::FlagRValueReference));

  // FlagAccessibility actually equals FlagPublic.
  EXPECT_EQ(StringRef("DIFlagPublic"),
            DINode::getFlagString(DINode::FlagAccessibility));

  // Some other invalid flags.
  EXPECT_EQ(StringRef(),
            DINode::getFlagString(DINode::FlagPublic | DINode::FlagVector));
  EXPECT_EQ(StringRef(), DINode::getFlagString(DINode::FlagFwdDecl |
                                               DINode::FlagArtificial));
  EXPECT_EQ(StringRef(), DINode::getFlagString(0xffff));
}

TEST(DINodeTest, splitFlags) {
// Some valid flags.
#define CHECK_SPLIT(FLAGS, VECTOR, REMAINDER)                                  \
  {                                                                            \
    SmallVector<unsigned, 8> V;                                                \
    EXPECT_EQ(REMAINDER, DINode::splitFlags(FLAGS, V));                        \
    EXPECT_TRUE(makeArrayRef(V).equals(VECTOR));                               \
  }
  CHECK_SPLIT(DINode::FlagPublic, {DINode::FlagPublic}, 0u);
  CHECK_SPLIT(DINode::FlagProtected, {DINode::FlagProtected}, 0u);
  CHECK_SPLIT(DINode::FlagPrivate, {DINode::FlagPrivate}, 0u);
  CHECK_SPLIT(DINode::FlagVector, {DINode::FlagVector}, 0u);
  CHECK_SPLIT(DINode::FlagRValueReference, {DINode::FlagRValueReference}, 0u);
  unsigned Flags[] = {DINode::FlagFwdDecl, DINode::FlagVector};
  CHECK_SPLIT(DINode::FlagFwdDecl | DINode::FlagVector, Flags, 0u);
  CHECK_SPLIT(0x100000u, {}, 0x100000u);
  CHECK_SPLIT(0x100000u | DINode::FlagVector, {DINode::FlagVector}, 0x100000u);
#undef CHECK_SPLIT
}

} // end namespace
