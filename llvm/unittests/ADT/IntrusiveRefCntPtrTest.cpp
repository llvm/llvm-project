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

namespace {
struct VirtualRefCounted : public llvm::RefCountedBaseVPTR {
  virtual void f() {}
};
}

namespace llvm {

// Run this test with valgrind to detect memory leaks.
TEST(IntrusiveRefCntPtr, RefCountedBaseVPTRCopyDoesNotLeak) {
  VirtualRefCounted *V1 = new VirtualRefCounted;
  IntrusiveRefCntPtr<VirtualRefCounted> R1 = V1;
  VirtualRefCounted *V2 = new VirtualRefCounted(*V1);
  IntrusiveRefCntPtr<VirtualRefCounted> R2 = V2;
}

struct SimpleRefCounted : public RefCountedBase<SimpleRefCounted> {};

// Run this test with valgrind to detect memory leaks.
TEST(IntrusiveRefCntPtr, RefCountedBaseCopyDoesNotLeak) {
  SimpleRefCounted *S1 = new SimpleRefCounted;
  IntrusiveRefCntPtr<SimpleRefCounted> R1 = S1;
  SimpleRefCounted *S2 = new SimpleRefCounted(*S1);
  IntrusiveRefCntPtr<SimpleRefCounted> R2 = S2;
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
