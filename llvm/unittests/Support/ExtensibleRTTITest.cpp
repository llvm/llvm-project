//===------ unittests/ExtensibleRTTITest.cpp - Extensible RTTI Tests ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ExtensibleRTTI.h"
#include "llvm/Support/Casting.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

class MyBaseType : public RTTIExtends<MyBaseType, RTTIRoot> {
public:
  static char ID;
};

class MyDerivedType : public RTTIExtends<MyDerivedType, MyBaseType> {
public:
  static char ID;
};

class MyOtherDerivedType : public RTTIExtends<MyOtherDerivedType, MyBaseType> {
public:
  static char ID;
};

class MyDeeperDerivedType
    : public RTTIExtends<MyDeeperDerivedType, MyDerivedType> {
public:
  static char ID;
};

char MyBaseType::ID = 0;
char MyDerivedType::ID = 0;
char MyOtherDerivedType::ID = 0;
char MyDeeperDerivedType::ID = 0;

TEST(ExtensibleRTTI, isa) {
  MyBaseType B;
  MyDerivedType D;
  MyDeeperDerivedType DD;

  EXPECT_TRUE(isa<MyBaseType>(B));
  EXPECT_FALSE(isa<MyDerivedType>(B));
  EXPECT_FALSE(isa<MyOtherDerivedType>(B));
  EXPECT_FALSE(isa<MyDeeperDerivedType>(B));

  EXPECT_TRUE(isa<MyBaseType>(D));
  EXPECT_TRUE(isa<MyDerivedType>(D));
  EXPECT_FALSE(isa<MyOtherDerivedType>(D));
  EXPECT_FALSE(isa<MyDeeperDerivedType>(D));

  EXPECT_TRUE(isa<MyBaseType>(DD));
  EXPECT_TRUE(isa<MyDerivedType>(DD));
  EXPECT_FALSE(isa<MyOtherDerivedType>(DD));
  EXPECT_TRUE(isa<MyDeeperDerivedType>(DD));
}

TEST(ExtensibleRTTI, cast) {
  MyDerivedType D;
  MyBaseType &BD = D;

  (void)cast<MyBaseType>(D);
  (void)cast<MyBaseType>(BD);
  (void)cast<MyDerivedType>(BD);
}

TEST(ExtensibleRTTI, dyn_cast) {
  MyBaseType B;
  MyDerivedType D;
  MyBaseType &BD = D;

  EXPECT_EQ(dyn_cast<MyDerivedType>(&B), nullptr);
  EXPECT_EQ(dyn_cast<MyDerivedType>(&D), &D);
  EXPECT_EQ(dyn_cast<MyBaseType>(&BD), &BD);
  EXPECT_EQ(dyn_cast<MyDerivedType>(&BD), &D);
}

} // namespace
