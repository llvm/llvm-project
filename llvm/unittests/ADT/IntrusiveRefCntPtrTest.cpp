//===- unittest/ADT/IntrusiveRefCntPtrTest.cpp ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "gtest/gtest.h"

namespace llvm {

namespace {
struct SimpleRefCounted : public RefCountedBase<SimpleRefCounted> {
  SimpleRefCounted() { ++NumInstances; }
  SimpleRefCounted(const SimpleRefCounted &) : RefCountedBase() {
    ++NumInstances;
  }
  ~SimpleRefCounted() { --NumInstances; }

  static int NumInstances;
};
int SimpleRefCounted::NumInstances = 0;
} // anonymous namespace

TEST(IntrusiveRefCntPtr, RefCountedBaseCopyDoesNotLeak) {
  EXPECT_EQ(0, SimpleRefCounted::NumInstances);
  {
    SimpleRefCounted *S1 = new SimpleRefCounted;
    IntrusiveRefCntPtr<SimpleRefCounted> R1 = S1;
    SimpleRefCounted *S2 = new SimpleRefCounted(*S1);
    IntrusiveRefCntPtr<SimpleRefCounted> R2 = S2;
    EXPECT_EQ(2, SimpleRefCounted::NumInstances);
  }
  EXPECT_EQ(0, SimpleRefCounted::NumInstances);
}

struct InterceptRefCounted : public RefCountedBase<InterceptRefCounted> {
  InterceptRefCounted(bool *Released, bool *Retained)
    : Released(Released), Retained(Retained) {}
  bool * const Released;
  bool * const Retained;
};
template <> struct IntrusiveRefCntPtrInfo<InterceptRefCounted> {
  static void retain(InterceptRefCounted *I) {
    *I->Retained = true;
    I->Retain();
  }
  static void release(InterceptRefCounted *I) {
    *I->Released = true;
    I->Release();
  }
};
TEST(IntrusiveRefCntPtr, UsesTraitsToRetainAndRelease) {
  bool Released = false;
  bool Retained = false;
  {
    InterceptRefCounted *I = new InterceptRefCounted(&Released, &Retained);
    IntrusiveRefCntPtr<InterceptRefCounted> R = I;
  }
  EXPECT_TRUE(Released);
  EXPECT_TRUE(Retained);
}

} // end namespace llvm
