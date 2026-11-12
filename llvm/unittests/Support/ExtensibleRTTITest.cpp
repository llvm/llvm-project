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

class MyMultipleInheritanceType
    : public RTTIExtends<MyMultipleInheritanceType, MyDerivedType,
                         MyOtherDerivedType> {
public:
  static char ID;
};

class MyTypeWithConstructor
    : public RTTIExtends<MyTypeWithConstructor, MyBaseType> {
public:
  static char ID;

  MyTypeWithConstructor(int) {}
};

class MyDerivedTypeWithConstructor
    : public RTTIExtends<MyDerivedTypeWithConstructor, MyTypeWithConstructor> {
public:
  static char ID;

  MyDerivedTypeWithConstructor(int x) : RTTIExtends(x) {}
};

char MyBaseType::ID = 0;
char MyDerivedType::ID = 0;
char MyOtherDerivedType::ID = 0;
char MyDeeperDerivedType::ID = 0;
char MyMultipleInheritanceType::ID = 0;
char MyTypeWithConstructor::ID = 0;
char MyDerivedTypeWithConstructor::ID = 0;

TEST(ExtensibleRTTI, isa) {
  MyBaseType B;
  MyDerivedType D;
  MyDeeperDerivedType DD;
  MyMultipleInheritanceType MI;

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

  EXPECT_TRUE(isa<MyBaseType>(MI));
  EXPECT_TRUE(isa<MyDerivedType>(MI));
  EXPECT_TRUE(isa<MyOtherDerivedType>(MI));
  EXPECT_FALSE(isa<MyDeeperDerivedType>(MI));
  EXPECT_TRUE(isa<MyMultipleInheritanceType>(MI));
}

TEST(ExtensibleRTTI, cast) {
  MyMultipleInheritanceType MI;
  MyDerivedType &D = MI;
  MyOtherDerivedType &OD = MI;
  MyBaseType &B = D;

  EXPECT_EQ(&cast<MyBaseType>(D), &B);
  EXPECT_EQ(&cast<MyDerivedType>(MI), &D);
  EXPECT_EQ(&cast<MyOtherDerivedType>(MI), &OD);
  EXPECT_EQ(&cast<MyMultipleInheritanceType>(MI), &MI);
}

TEST(ExtensibleRTTI, dyn_cast) {
  MyMultipleInheritanceType MI;
  MyDerivedType &D = MI;
  MyOtherDerivedType &OD = MI;
  MyBaseType &BD = D;
  MyBaseType &BOD = OD;

  EXPECT_EQ(dyn_cast<MyBaseType>(&BD), &BD);
  EXPECT_EQ(dyn_cast<MyDerivedType>(&BD), &D);

  EXPECT_EQ(dyn_cast<MyBaseType>(&BOD), &BOD);
  EXPECT_EQ(dyn_cast<MyOtherDerivedType>(&BOD), &OD);

  EXPECT_EQ(dyn_cast<MyBaseType>(&D), &BD);
  EXPECT_EQ(dyn_cast<MyDerivedType>(&D), &D);
  EXPECT_EQ(dyn_cast<MyMultipleInheritanceType>(&D), &MI);

  EXPECT_EQ(dyn_cast<MyBaseType>(&OD), &BOD);
  EXPECT_EQ(dyn_cast<MyOtherDerivedType>(&OD), &OD);
  EXPECT_EQ(dyn_cast<MyMultipleInheritanceType>(&OD), &MI);

  EXPECT_EQ(dyn_cast<MyDerivedType>(&MI), &D);
  EXPECT_EQ(dyn_cast<MyMultipleInheritanceType>(&MI), &MI);

  EXPECT_EQ(dyn_cast<MyDerivedType>(&MI), &D);
  EXPECT_EQ(dyn_cast<MyOtherDerivedType>(&MI), &OD);
  EXPECT_EQ(dyn_cast<MyMultipleInheritanceType>(&MI), &MI);
}

TEST(ExtensibleRTTI, multiple_inheritance_constructor) {
  MyDerivedTypeWithConstructor V(42);
}

} // namespace
