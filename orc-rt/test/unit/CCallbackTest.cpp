//===- CCallbackTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for orc-rt's CCallback.h APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/CCallback.h"

#include "gtest/gtest.h"

using namespace orc_rt;

namespace {

// Increments an int through a pointer member. Because the pointer is const in
// a const method but its pointee is not, even the const methods can mutate
// observable state -- this lets the const-callback tests confirm the wrapped
// method actually ran, and also exercises RetT / ArgTs plumbing.
class Incrementer {
public:
  Incrementer(int *P) : P(P) {}

  void inc() { ++*P; }
  void incNoexcept() noexcept { ++*P; }
  void incConst() const { ++*P; }
  void incConstNoexcept() const noexcept { ++*P; }

  // Non-void return and by-value argument.
  int advance(int Y) {
    *P += Y;
    return *P;
  }

private:
  int *P;
};

// A non-empty base placed *before* Incrementer so that the Incrementer
// subobject lands at a non-zero offset inside DerivedIncrementer. This is what
// makes context-pointer adjustment observable: handing the callback a raw
// pointer to the derived object (rather than the adjusted subobject pointer)
// would compute the wrong `this`.
class NonEmptyBase {
  [[maybe_unused]] void *Pad = nullptr;
};

class DerivedIncrementer : public NonEmptyBase, public Incrementer {
public:
  DerivedIncrementer(int *P) : Incrementer(P) {}
};

} // namespace

// C-style trampolines that only know the callback as a plain function pointer,
// mirroring how a C API would store and invoke it.
static void invokeVoidVoid(void (*Callback)(void *Ctx), void *Ctx) {
  Callback(Ctx);
}

static void invokeVoidVoidConst(void (*Callback)(const void *Ctx),
                                const void *Ctx) {
  Callback(Ctx);
}

TEST(CCallbackTest, NonConstVoidMethod) {
  int X = 0;
  Incrementer I(&X);
  invokeVoidVoid(asCCallback<&Incrementer::inc>,
                 asCCallbackContext<&Incrementer::inc>(I));
  EXPECT_EQ(X, 1);
}

TEST(CCallbackTest, NoexceptVoidMethod) {
  int X = 0;
  Incrementer I(&X);
  // A noexcept source method must match a specialization and yield a noexcept
  // function pointer; the explicit type here asserts both.
  void (*Callback)(void *) noexcept = asCCallback<&Incrementer::incNoexcept>;
  invokeVoidVoid(Callback, asCCallbackContext<&Incrementer::incNoexcept>(I));
  EXPECT_EQ(X, 1);
}

TEST(CCallbackTest, ConstVoidMethod) {
  int X = 0;
  const Incrementer I(&X);
  invokeVoidVoidConst(asCCallback<&Incrementer::incConst>,
                      asCCallbackContext<&Incrementer::incConst>(I));
  EXPECT_EQ(X, 1);
}

TEST(CCallbackTest, ConstNoexceptVoidMethod) {
  int X = 0;
  const Incrementer I(&X);
  void (*Callback)(const void *) noexcept =
      asCCallback<&Incrementer::incConstNoexcept>;
  invokeVoidVoidConst(Callback,
                      asCCallbackContext<&Incrementer::incConstNoexcept>(I));
  EXPECT_EQ(X, 1);
}

TEST(CCallbackTest, ForwardsArgumentAndReturnValue) {
  int X = 10;
  Incrementer I(&X);
  int (*Callback)(void *, int) = asCCallback<&Incrementer::advance>;
  EXPECT_EQ(Callback(asCCallbackContext<&Incrementer::advance>(I), 5), 15);
  EXPECT_EQ(X, 15);
}

TEST(CCallbackTest, NonConstContextWithBaseOffset) {
  int X = 0;
  DerivedIncrementer DI(&X);
  invokeVoidVoid(asCCallback<&Incrementer::inc>,
                 asCCallbackContext<&Incrementer::inc>(DI));
  EXPECT_EQ(X, 1);
}

TEST(CCallbackTest, ConstContextWithBaseOffset) {
  int X = 0;
  const DerivedIncrementer DI(&X);
  invokeVoidVoidConst(asCCallback<&Incrementer::incConst>,
                      asCCallbackContext<&Incrementer::incConst>(DI));
  EXPECT_EQ(X, 1);
}
