//===---------- llvm/unittest/Support/Casting.cpp - Casting tests ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"
#include <cstdlib>

namespace llvm {

// set up two example classes
// with conversion facility
//
struct bar {
  bar() {}
  struct foo *baz();
  struct foo *caz();
  struct foo *daz();
  struct foo *naz();
private:
  bar(const bar &);
};
struct foo {
  void ext() const;
  /*  static bool classof(const bar *X) {
    cerr << "Classof: " << X << "\n";
    return true;
    }*/
};

template <> struct isa_impl<foo, bar> {
  static inline bool doit(const bar &Val) {
    dbgs() << "Classof: " << &Val << "\n";
    return true;
  }
};

foo *bar::baz() {
    return cast<foo>(this);
}

foo *bar::caz() {
    return cast_or_null<foo>(this);
}

foo *bar::daz() {
    return dyn_cast<foo>(this);
}

foo *bar::naz() {
    return dyn_cast_or_null<foo>(this);
}


bar *fub();
} // End llvm namespace

using namespace llvm;

namespace {

const foo *null_foo = NULL;

bar B;
extern bar &B1;
bar &B1 = B;
extern const bar *B2;
// test various configurations of const
const bar &B3 = B1;
const bar *const B4 = B2;

TEST(CastingTest, isa) {
  EXPECT_TRUE(isa<foo>(B1));
  EXPECT_TRUE(isa<foo>(B2));
  EXPECT_TRUE(isa<foo>(B3));
  EXPECT_TRUE(isa<foo>(B4));
}

TEST(CastingTest, cast) {
  foo &F1 = cast<foo>(B1);
  EXPECT_NE(&F1, null_foo);
  const foo *F3 = cast<foo>(B2);
  EXPECT_NE(F3, null_foo);
  const foo *F4 = cast<foo>(B2);
  EXPECT_NE(F4, null_foo);
  const foo &F5 = cast<foo>(B3);
  EXPECT_NE(&F5, null_foo);
  const foo *F6 = cast<foo>(B4);
  EXPECT_NE(F6, null_foo);
  // Can't pass null pointer to cast<>.
  // foo *F7 = cast<foo>(fub());
  // EXPECT_EQ(F7, null_foo);
  foo *F8 = B1.baz();
  EXPECT_NE(F8, null_foo);
}

TEST(CastingTest, cast_or_null) {
  const foo *F11 = cast_or_null<foo>(B2);
  EXPECT_NE(F11, null_foo);
  const foo *F12 = cast_or_null<foo>(B2);
  EXPECT_NE(F12, null_foo);
  const foo *F13 = cast_or_null<foo>(B4);
  EXPECT_NE(F13, null_foo);
  const foo *F14 = cast_or_null<foo>(fub());  // Shouldn't print.
  EXPECT_EQ(F14, null_foo);
  foo *F15 = B1.caz();
  EXPECT_NE(F15, null_foo);
}

TEST(CastingTest, dyn_cast) {
  const foo *F1 = dyn_cast<foo>(B2);
  EXPECT_NE(F1, null_foo);
  const foo *F2 = dyn_cast<foo>(B2);
  EXPECT_NE(F2, null_foo);
  const foo *F3 = dyn_cast<foo>(B4);
  EXPECT_NE(F3, null_foo);
  // Can't pass null pointer to dyn_cast<>.
  // foo *F4 = dyn_cast<foo>(fub());
  // EXPECT_EQ(F4, null_foo);
  foo *F5 = B1.daz();
  EXPECT_NE(F5, null_foo);
}

TEST(CastingTest, dyn_cast_or_null) {
  const foo *F1 = dyn_cast_or_null<foo>(B2);
  EXPECT_NE(F1, null_foo);
  const foo *F2 = dyn_cast_or_null<foo>(B2);
  EXPECT_NE(F2, null_foo);
  const foo *F3 = dyn_cast_or_null<foo>(B4);
  EXPECT_NE(F3, null_foo);
  foo *F4 = dyn_cast_or_null<foo>(fub());
  EXPECT_EQ(F4, null_foo);
  foo *F5 = B1.naz();
  EXPECT_NE(F5, null_foo);
}

// These lines are errors...
//foo *F20 = cast<foo>(B2);  // Yields const foo*
//foo &F21 = cast<foo>(B3);  // Yields const foo&
//foo *F22 = cast<foo>(B4);  // Yields const foo*
//foo &F23 = cast_or_null<foo>(B1);
//const foo &F24 = cast_or_null<foo>(B3);

const bar *B2 = &B;
}  // anonymous namespace

bar *llvm::fub() { return 0; }

namespace {
namespace inferred_upcasting {
// This test case verifies correct behavior of inferred upcasts when the
// types are statically known to be OK to upcast. This is the case when,
// for example, Derived inherits from Base, and we do `isa<Base>(Derived)`.

// Note: This test will actually fail to compile without inferred
// upcasting.

class Base {
public:
  // No classof. We are testing that the upcast is inferred.
  Base() {}
};

class Derived : public Base {
public:
  Derived() {}
};

// Even with no explicit classof() in Base, we should still be able to cast
// Derived to its base class.
TEST(CastingTest, UpcastIsInferred) {
  Derived D;
  EXPECT_TRUE(isa<Base>(D));
  Base *BP = dyn_cast<Base>(&D);
  EXPECT_TRUE(BP != NULL);
}


// This test verifies that the inferred upcast takes precedence over an
// explicitly written one. This is important because it verifies that the
// dynamic check gets optimized away.
class UseInferredUpcast {
public:
  int Dummy;
  static bool classof(const UseInferredUpcast *) {
    return false;
  }
};

TEST(CastingTest, InferredUpcastTakesPrecedence) {
  UseInferredUpcast UIU;
  // Since the explicit classof() returns false, this will fail if the
  // explicit one is used.
  EXPECT_TRUE(isa<UseInferredUpcast>(&UIU));
}

} // end namespace inferred_upcasting
} // end anonymous namespace
