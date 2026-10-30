#include <cstdint>

// Simulates the layouts of libstdc++'s std::atomic. The layout of some
// specializations depends on the libstdc++ version:
//
// FLOAT_LAYOUT:
//   0: std::atomic<float> and std::atomic<double> use the generic layout
//      (`_M_i`), as before the C++20 floating point specializations.
//   1: std::atomic<float> and std::atomic<double> are specialized and store
//      their value in `_M_fp`.
//
// REFCOUNT_LAYOUT:
//   0: The reference count of std::atomic<std::shared_ptr<T>> is a scalar
//      `_M_val`.
//   1: `_M_val` is a struct wrapping the scalar in `_M_i`.

#ifndef FLOAT_LAYOUT
#define FLOAT_LAYOUT 1
#endif
#ifndef REFCOUNT_LAYOUT
#define REFCOUNT_LAYOUT 1
#endif

namespace std {

struct _Sp_counted_base {
  int _M_use_count;
  int _M_weak_count;
};

struct __shared_count {
  _Sp_counted_base *_M_pi;
};

template <typename _Tp> struct shared_ptr {
  using element_type = _Tp;
  element_type *_M_ptr;
  __shared_count _M_refcount;
};

template <typename _Tp> struct weak_ptr {
  using element_type = _Tp;
  element_type *_M_ptr;
  __shared_count _M_refcount;
};

#if REFCOUNT_LAYOUT == 0
using _Atomic_val = uintptr_t;
#else
struct _Atomic_val {
  uintptr_t _M_i;
};
#endif

struct _Atomic_count {
  using pointer = _Sp_counted_base *;
  // Never called, it makes sure the debug info contains `pointer`.
  pointer get() const;

  _Atomic_val _M_val;
};

template <typename _Tp> struct _Sp_atomic {
  typename _Tp::element_type *_M_ptr;
  _Atomic_count _M_refcount;
};

template <typename _Tp> struct __atomic_base {
  _Tp _M_i;
};

template <typename _Tp> struct __atomic_base<_Tp *> {
  _Tp *_M_p;
};

#if FLOAT_LAYOUT == 1
template <typename _Fp> struct __atomic_float {
  _Fp _M_fp;
};
#endif

template <typename _Tp> struct atomic {
  _Tp _M_i;
};

template <> struct atomic<bool> {
  __atomic_base<bool> _M_base;
};

template <typename _Tp> struct atomic<_Tp *> {
  __atomic_base<_Tp *> _M_b;
};

#if FLOAT_LAYOUT == 1
template <> struct atomic<float> : __atomic_float<float> {};
template <> struct atomic<double> : __atomic_float<double> {};
#endif

template <typename _Tp> struct atomic<shared_ptr<_Tp>> {
  _Sp_atomic<shared_ptr<_Tp>> _M_impl;
};

template <typename _Tp> struct atomic<weak_ptr<_Tp>> {
  _Sp_atomic<weak_ptr<_Tp>> _M_impl;
};

} // namespace std

static std::_Atomic_count make_count(std::_Sp_counted_base *control) {
  std::_Atomic_count count{};
  // The least significant bit is used as a lock by libstdc++, it is not set
  // while the atomic is idle.
#if REFCOUNT_LAYOUT == 0
  count._M_val = reinterpret_cast<uintptr_t>(control);
#else
  count._M_val._M_i = reinterpret_cast<uintptr_t>(control);
#endif
  return count;
}

struct S {
  int x = 1;
  int y = 2;
};

typedef float vector4float __attribute__((__vector_size__(16)));

static _Complex float make_complex(float real, float imag) {
  _Complex float value;
  __real__ value = real;
  __imag__ value = imag;
  return value;
}

int main() {
  std::atomic<int> atomic_int{5};
  std::atomic<S> atomic_struct{S()};
  std::atomic<bool> atomic_bool{{true}};
  std::atomic<float> atomic_float{3.14f};
  std::atomic<double> atomic_double{6.28};
  std::atomic<_Complex float> atomic_complex_float{make_complex(1.5f, 2.5f)};
  std::atomic<vector4float> atomic_vector_float{vector4float{1, 2, 3, 4}};

  int pointee = 42;
  std::atomic<int *> atomic_pointer{{&pointee}};

  int shared_value = 300;
  // use_count = 1 and weak_count = 1, the weak count includes one reference
  // for all the strong references.
  std::_Sp_counted_base shared_control{1, 1};
  std::atomic<std::shared_ptr<int>> atomic_shared{
      {&shared_value, make_count(&shared_control)}};
  // use_count = 1 and weak_count = 2.
  std::_Sp_counted_base weak_control{1, 2};
  std::atomic<std::weak_ptr<int>> atomic_weak{
      {&shared_value, make_count(&weak_control)}};

  return 0; // Set break point at this line.
}
