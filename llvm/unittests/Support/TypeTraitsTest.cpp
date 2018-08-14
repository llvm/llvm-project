//===- TypeTraitsTest.cpp -------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/type_traits.h"

namespace {

// Compile-time tests using static assert.
namespace triviality {

// Helper for compile time checking trivially copy constructible and trivially
// move constructible type traits.
template <typename T, bool IsTriviallyCopyConstructible,
          bool IsTriviallyMoveConstructible>
void TrivialityTester() {
  static_assert(llvm::is_trivially_copy_constructible<T>::value ==
                    IsTriviallyCopyConstructible,
                "Mismatch in expected trivial copy construction!");
  static_assert(llvm::is_trivially_move_constructible<T>::value ==
                    IsTriviallyMoveConstructible,
                "Mismatch in expected trivial move construction!");

#if defined(_LIBCPP_VERSION) || defined(_MSC_VER)
  // On compilers with support for the standard traits, make sure they agree.
  static_assert(std::is_trivially_copy_constructible<T>::value ==
                    IsTriviallyCopyConstructible,
                "Mismatch in expected trivial copy construction!");
  static_assert(std::is_trivially_move_constructible<T>::value ==
                    IsTriviallyMoveConstructible,
                "Mismatch in expected trivial move construction!");
#endif
}

template void TrivialityTester<int, true, true>();
template void TrivialityTester<void *, true, true>();
template void TrivialityTester<int &, true, true>();
template void TrivialityTester<int &&, false, true>();

struct X {};
struct Y {
  Y(const Y &);
};
struct Z {
  Z(const Z &);
  Z(Z &&);
};
struct A {
  A(const A &) = default;
  A(A &&);
};
struct B {
  B(const B &);
  B(B &&) = default;
};

template void TrivialityTester<X, true, true>();
template void TrivialityTester<Y, false, false>();
template void TrivialityTester<Z, false, false>();
template void TrivialityTester<A, true, false>();
template void TrivialityTester<B, false, true>();

template void TrivialityTester<Z &, true, true>();
template void TrivialityTester<A &, true, true>();
template void TrivialityTester<B &, true, true>();
template void TrivialityTester<Z &&, false, true>();
template void TrivialityTester<A &&, false, true>();
template void TrivialityTester<B &&, false, true>();

} // namespace triviality

} // end anonymous namespace
