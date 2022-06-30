//===---------- llvm/unittest/Support/Casting.cpp - Casting tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Casting.h"
#include "llvm/IR/User.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <cstdlib>

namespace llvm {
// Used to test illegal cast. If a cast doesn't match any of the "real" ones,
// it will match this one.
struct IllegalCast;
template <typename T> IllegalCast *cast(...) { return nullptr; }

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
  foo(const bar &) {}
  void ext() const;
};

struct base {
  virtual ~base() {}
};

struct derived : public base {
  static bool classof(const base *B) { return true; }
};

template <> struct isa_impl<foo, bar> {
  static inline bool doit(const bar &Val) {
    dbgs() << "Classof: " << &Val << "\n";
    return true;
  }
};

// Note for the future - please don't do this. isa_impl is an internal template
// for the implementation of `isa` and should not be exposed this way.
// Completely unrelated types *should* result in compiler errors if you try to
// cast between them.
template <typename T> struct isa_impl<foo, T> {
  static inline bool doit(const T &Val) { return false; }
};

foo *bar::baz() { return cast<foo>(this); }

foo *bar::caz() { return cast_or_null<foo>(this); }

foo *bar::daz() { return dyn_cast<foo>(this); }

foo *bar::naz() { return dyn_cast_or_null<foo>(this); }

bar *fub();

template <> struct simplify_type<foo> {
  typedef int SimpleType;
  static SimpleType getSimplifiedValue(foo &Val) { return 0; }
};

struct T1 {};

struct T2 {
  T2(const T1 &x) {}
  static bool classof(const T1 *x) { return true; }
};

template <> struct CastInfo<T2, T1> : public OptionalValueCast<T2, T1> {};

struct T3 {
  T3(const T1 *x) : hasValue(x != nullptr) {}

  static bool classof(const T1 *x) { return true; }
  bool hasValue = false;
};

// T3 is convertible from a pointer to T1.
template <> struct CastInfo<T3, T1 *> : public ValueFromPointerCast<T3, T1> {};

struct T4 {
  T4() : hasValue(false) {}
  T4(const T3 &x) : hasValue(true) {}

  static bool classof(const T3 *x) { return true; }
  bool hasValue = false;
};

template <> struct ValueIsPresent<T3> {
  using UnwrappedType = T3;
  static inline bool isPresent(const T3 &t) { return t.hasValue; }
  static inline const T3 &unwrapValue(const T3 &t) { return t; }
};

template <> struct CastInfo<T4, T3> {
  using CastResultType = T4;
  static inline CastResultType doCast(const T3 &t) { return T4(t); }
  static inline CastResultType castFailed() { return CastResultType(); }
  static inline CastResultType doCastIfPossible(const T3 &f) {
    return doCast(f);
  }
};

} // namespace llvm

using namespace llvm;

// Test the peculiar behavior of Use in simplify_type.
static_assert(std::is_same<simplify_type<Use>::SimpleType, Value *>::value,
              "Use doesn't simplify correctly!");
static_assert(std::is_same<simplify_type<Use *>::SimpleType, Value *>::value,
              "Use doesn't simplify correctly!");

// Test that a regular class behaves as expected.
static_assert(std::is_same<simplify_type<foo>::SimpleType, int>::value,
              "Unexpected simplify_type result!");
static_assert(std::is_same<simplify_type<foo *>::SimpleType, foo *>::value,
              "Unexpected simplify_type result!");

namespace {

const foo *null_foo = nullptr;

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

TEST(CastingTest, isa_and_nonnull) {
  EXPECT_TRUE(isa_and_nonnull<foo>(B2));
  EXPECT_TRUE(isa_and_nonnull<foo>(B4));
  EXPECT_FALSE(isa_and_nonnull<foo>(fub()));
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

  std::unique_ptr<const bar> BP(B2);
  auto FP = cast<foo>(std::move(BP));
  static_assert(std::is_same<std::unique_ptr<const foo>, decltype(FP)>::value,
                "Incorrect deduced return type!");
  EXPECT_NE(FP.get(), null_foo);
  FP.release();
}

TEST(CastingTest, cast_or_null) {
  const foo *F11 = cast_or_null<foo>(B2);
  EXPECT_NE(F11, null_foo);
  const foo *F12 = cast_or_null<foo>(B2);
  EXPECT_NE(F12, null_foo);
  const foo *F13 = cast_or_null<foo>(B4);
  EXPECT_NE(F13, null_foo);
  const foo *F14 = cast_or_null<foo>(fub()); // Shouldn't print.
  EXPECT_EQ(F14, null_foo);
  foo *F15 = B1.caz();
  EXPECT_NE(F15, null_foo);

  std::unique_ptr<const bar> BP(fub());
  auto FP = cast_or_null<foo>(std::move(BP));
  EXPECT_EQ(FP.get(), null_foo);
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

// All these tests forward to dyn_cast_if_present, so they also provde an
// effective test for its use cases.
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
  // dyn_cast_if_present should have exactly the same behavior as
  // dyn_cast_or_null.
  const foo *F6 = dyn_cast_if_present<foo>(B2);
  EXPECT_EQ(F6, F2);
}

TEST(CastingTest, dyn_cast_value_types) {
  T1 t1;
  Optional<T2> t2 = dyn_cast<T2>(t1);
  EXPECT_TRUE(t2);

  T2 *t2ptr = dyn_cast<T2>(&t1);
  EXPECT_TRUE(t2ptr != nullptr);

  T3 t3 = dyn_cast<T3>(&t1);
  EXPECT_TRUE(t3.hasValue);
}

TEST(CastingTest, dyn_cast_if_present) {
  Optional<T1> empty{};
  Optional<T2> F1 = dyn_cast_if_present<T2>(empty);
  EXPECT_FALSE(F1.hasValue());

  T1 t1;
  Optional<T2> F2 = dyn_cast_if_present<T2>(t1);
  EXPECT_TRUE(F2.hasValue());

  T1 *t1Null = nullptr;

  // T3 should have hasValue == false because t1Null is nullptr.
  T3 t3 = dyn_cast_if_present<T3>(t1Null);
  EXPECT_FALSE(t3.hasValue);

  // Now because of that, T4 should receive the castFailed implementation of its
  // FallibleCastTraits, which default-constructs a T4, which has no value.
  T4 t4 = dyn_cast_if_present<T4>(t3);
  EXPECT_FALSE(t4.hasValue);
}

std::unique_ptr<derived> newd() { return std::make_unique<derived>(); }
std::unique_ptr<base> newb() { return std::make_unique<derived>(); }

TEST(CastingTest, unique_dyn_cast) {
  derived *OrigD = nullptr;
  auto D = std::make_unique<derived>();
  OrigD = D.get();

  // Converting from D to itself is valid, it should return a new unique_ptr
  // and the old one should become nullptr.
  auto NewD = unique_dyn_cast<derived>(D);
  ASSERT_EQ(OrigD, NewD.get());
  ASSERT_EQ(nullptr, D);

  // Converting from D to B is valid, B should have a value and D should be
  // nullptr.
  auto B = unique_dyn_cast<base>(NewD);
  ASSERT_EQ(OrigD, B.get());
  ASSERT_EQ(nullptr, NewD);

  // Converting from B to itself is valid, it should return a new unique_ptr
  // and the old one should become nullptr.
  auto NewB = unique_dyn_cast<base>(B);
  ASSERT_EQ(OrigD, NewB.get());
  ASSERT_EQ(nullptr, B);

  // Converting from B to D is valid, D should have a value and B should be
  // nullptr;
  D = unique_dyn_cast<derived>(NewB);
  ASSERT_EQ(OrigD, D.get());
  ASSERT_EQ(nullptr, NewB);

  // This is a very contrived test, casting between completely unrelated types
  // should generally fail to compile. See the classof shenanigans we have in
  // the definition of `foo` above.
  auto F = unique_dyn_cast<foo>(D);
  ASSERT_EQ(nullptr, F);
  ASSERT_EQ(OrigD, D.get());

  // All of the above should also hold for temporaries.
  auto D2 = unique_dyn_cast<derived>(newd());
  EXPECT_NE(nullptr, D2);

  auto B2 = unique_dyn_cast<derived>(newb());
  EXPECT_NE(nullptr, B2);

  auto B3 = unique_dyn_cast<base>(newb());
  EXPECT_NE(nullptr, B3);

  // This is a very contrived test, casting between completely unrelated types
  // should generally fail to compile. See the classof shenanigans we have in
  // the definition of `foo` above.
  auto F2 = unique_dyn_cast<foo>(newb());
  EXPECT_EQ(nullptr, F2);
}

// These lines are errors...
// foo *F20 = cast<foo>(B2);  // Yields const foo*
// foo &F21 = cast<foo>(B3);  // Yields const foo&
// foo *F22 = cast<foo>(B4);  // Yields const foo*
// foo &F23 = cast_or_null<foo>(B1);
// const foo &F24 = cast_or_null<foo>(B3);

const bar *B2 = &B;
} // anonymous namespace

bar *llvm::fub() { return nullptr; }

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
  EXPECT_NE(BP, nullptr);
}

// This test verifies that the inferred upcast takes precedence over an
// explicitly written one. This is important because it verifies that the
// dynamic check gets optimized away.
class UseInferredUpcast {
public:
  int Dummy;
  static bool classof(const UseInferredUpcast *) { return false; }
};

TEST(CastingTest, InferredUpcastTakesPrecedence) {
  UseInferredUpcast UIU;
  // Since the explicit classof() returns false, this will fail if the
  // explicit one is used.
  EXPECT_TRUE(isa<UseInferredUpcast>(&UIU));
}

} // end namespace inferred_upcasting
} // end anonymous namespace

namespace {
namespace pointer_wrappers {

struct Base {
  bool IsDerived;
  Base(bool IsDerived = false) : IsDerived(IsDerived) {}
};

struct Derived : Base {
  Derived() : Base(true) {}
  static bool classof(const Base *B) { return B->IsDerived; }
};

class PTy {
  Base *B;

public:
  PTy(Base *B) : B(B) {}
  explicit operator bool() const { return get(); }
  Base *get() const { return B; }
};

} // end namespace pointer_wrappers
} // end namespace

namespace llvm {

template <> struct ValueIsPresent<pointer_wrappers::PTy> {
  using UnwrappedType = pointer_wrappers::PTy;
  static inline bool isPresent(const pointer_wrappers::PTy &P) {
    return P.get() != nullptr;
  }
  static UnwrappedType &unwrapValue(pointer_wrappers::PTy &P) { return P; }
};

template <> struct ValueIsPresent<const pointer_wrappers::PTy> {
  using UnwrappedType = pointer_wrappers::PTy;
  static inline bool isPresent(const pointer_wrappers::PTy &P) {
    return P.get() != nullptr;
  }

  static UnwrappedType &unwrapValue(const pointer_wrappers::PTy &P) {
    return const_cast<UnwrappedType &>(P);
  }
};

template <> struct simplify_type<pointer_wrappers::PTy> {
  typedef pointer_wrappers::Base *SimpleType;
  static SimpleType getSimplifiedValue(pointer_wrappers::PTy &P) {
    return P.get();
  }
};
template <> struct simplify_type<const pointer_wrappers::PTy> {
  typedef pointer_wrappers::Base *SimpleType;
  static SimpleType getSimplifiedValue(const pointer_wrappers::PTy &P) {
    return P.get();
  }
};

} // end namespace llvm

namespace {
namespace pointer_wrappers {

// Some objects.
pointer_wrappers::Base B;
pointer_wrappers::Derived D;

// Mutable "smart" pointers.
pointer_wrappers::PTy MN(nullptr);
pointer_wrappers::PTy MB(&B);
pointer_wrappers::PTy MD(&D);

// Const "smart" pointers.
const pointer_wrappers::PTy CN(nullptr);
const pointer_wrappers::PTy CB(&B);
const pointer_wrappers::PTy CD(&D);

TEST(CastingTest, smart_isa) {
  EXPECT_TRUE(!isa<pointer_wrappers::Derived>(MB));
  EXPECT_TRUE(!isa<pointer_wrappers::Derived>(CB));
  EXPECT_TRUE(isa<pointer_wrappers::Derived>(MD));
  EXPECT_TRUE(isa<pointer_wrappers::Derived>(CD));
}

TEST(CastingTest, smart_cast) {
  EXPECT_EQ(cast<pointer_wrappers::Derived>(MD), &D);
  EXPECT_EQ(cast<pointer_wrappers::Derived>(CD), &D);
}

TEST(CastingTest, smart_cast_or_null) {
  EXPECT_EQ(cast_or_null<pointer_wrappers::Derived>(MN), nullptr);
  EXPECT_EQ(cast_or_null<pointer_wrappers::Derived>(CN), nullptr);
  EXPECT_EQ(cast_or_null<pointer_wrappers::Derived>(MD), &D);
  EXPECT_EQ(cast_or_null<pointer_wrappers::Derived>(CD), &D);
}

TEST(CastingTest, smart_dyn_cast) {
  EXPECT_EQ(dyn_cast<pointer_wrappers::Derived>(MB), nullptr);
  EXPECT_EQ(dyn_cast<pointer_wrappers::Derived>(CB), nullptr);
  EXPECT_EQ(dyn_cast<pointer_wrappers::Derived>(MD), &D);
  EXPECT_EQ(dyn_cast<pointer_wrappers::Derived>(CD), &D);
}

TEST(CastingTest, smart_dyn_cast_or_null) {
  EXPECT_EQ(dyn_cast_or_null<pointer_wrappers::Derived>(MN), nullptr);
  EXPECT_EQ(dyn_cast_or_null<pointer_wrappers::Derived>(CN), nullptr);
  EXPECT_EQ(dyn_cast_or_null<pointer_wrappers::Derived>(MB), nullptr);
  EXPECT_EQ(dyn_cast_or_null<pointer_wrappers::Derived>(CB), nullptr);
  EXPECT_EQ(dyn_cast_or_null<pointer_wrappers::Derived>(MD), &D);
  EXPECT_EQ(dyn_cast_or_null<pointer_wrappers::Derived>(CD), &D);
}

} // end namespace pointer_wrappers

#ifndef NDEBUG
namespace assertion_checks {
struct Base {
  virtual ~Base() {}
};

struct Derived : public Base {
  static bool classof(const Base *B) { return false; }
};

TEST(CastingTest, assertion_check_const_ref) {
  const Base B;
  EXPECT_DEATH((void)cast<Derived>(B), "argument of incompatible type")
      << "Invalid cast of const ref did not cause an abort()";
}

TEST(CastingTest, assertion_check_ref) {
  Base B;
  EXPECT_DEATH((void)cast<Derived>(B), "argument of incompatible type")
      << "Invalid cast of const ref did not cause an abort()";
}

TEST(CastingTest, assertion_check_ptr) {
  Base B;
  EXPECT_DEATH((void)cast<Derived>(&B), "argument of incompatible type")
      << "Invalid cast of const ref did not cause an abort()";
}

TEST(CastingTest, assertion_check_unique_ptr) {
  auto B = std::make_unique<Base>();
  EXPECT_DEATH((void)cast<Derived>(std::move(B)),
               "argument of incompatible type")
      << "Invalid cast of const ref did not cause an abort()";
}

} // end namespace assertion_checks
#endif
} // end namespace
