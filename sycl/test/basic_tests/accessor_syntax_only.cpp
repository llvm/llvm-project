//==--- accessor_syntax_only.cpp - Syntax checks for SYCL accessors --------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This test is supposed to check that interface of sycl::accessor
// conforms to the specification. It checks that valid code can be
// compiled and invalid code causes compilation errors.

// RUN: %clang -std=c++11 -fsyntax-only -Xclang -verify %s

// Check that the test an be compiled with device compiler as well.
// RUN: %clang --sycl -fsyntax-only -Xclang -verify %s

#include <CL/sycl.hpp>

namespace sycl {
  using namespace cl::sycl;
  using namespace cl::sycl::access;
}

struct IdxSz {
  operator size_t() { return 1; }
};

struct IdxId1 {
  operator sycl::id<1>() { return sycl::id<1>(1); }
};

struct IdxId2 {
  operator sycl::id<2>() { return sycl::id<2>(1, 1); }
};

struct IdxId3 {
  operator sycl::id<3>() { return sycl::id<3>(1, 1, 1); }
};

struct IdxIdAny {
  operator sycl::id<1>() { return sycl::id<1>(1); }
  operator sycl::id<2>() { return sycl::id<2>(1, 1); }
  operator sycl::id<3>() { return sycl::id<3>(1, 1, 1); }
};

struct IdxIdSz {
  operator size_t() { return 1; }
  operator sycl::id<1>() { return sycl::id<1>(1); }
  operator sycl::id<2>() { return sycl::id<2>(1, 1); }
  operator sycl::id<3>() { return sycl::id<3>(1, 1, 1); }
};

template <int dimensions, sycl::mode accessMode, sycl::target accessTarget>
using acc_t = sycl::accessor<int, dimensions, accessMode, accessTarget,
                             sycl::placeholder::false_t>;

// Check that operator dataT is defined only if (dimensions == 0).
void test1() {
  int data = 5;
  sycl::buffer<int, 1> buf(&data, 1);
  auto acc = buf.get_access<sycl::access::mode::read>();
  (int) acc; // expected-error {{cannot convert}}
}

// Check that operator dataT returns by value in case of read-accessor
// and by reference in case of write-accessor.
void test2(acc_t<0, sycl::mode::read, sycl::target::host_buffer> acc0,
           acc_t<0, sycl::mode::write, sycl::target::global_buffer> acc1,
           acc_t<0, sycl::mode::read_write, sycl::target::constant_buffer> acc2,
           acc_t<0, sycl::mode::discard_write, sycl::target::local> acc3) {
  int val0 = acc0;
  int &val0_r = acc0; // expected-error {{cannot bind}}

  int val1 = acc1;
  int &val1_r = acc1;

  int val2 = acc2;
  int &val2_r = acc2;

  int val3 = acc3;
  int &val3_r = acc3;
}

// Check that operator[](size_t) is defined according to spec.
void test3(acc_t<0, sycl::mode::discard_read_write, sycl::target::host_buffer> acc0,
           acc_t<1, sycl::mode::write, sycl::target::global_buffer> acc1,
           acc_t<2, sycl::mode::read, sycl::target::constant_buffer> acc2,
           acc_t<3, sycl::mode::read_write, sycl::target::local> acc3) {
  IdxSz idx;
  acc0[idx]; // expected-error {{does not provide a subscript operator}}
  acc1[idx];
  acc1[idx] = 1;
  acc2[idx][idx];
  acc2[idx][idx] = 2; // expected-error {{expression is not assignable}}
  acc3[idx][idx][idx];
  acc3[idx][idx][idx] = 3;
}

// Check that operator[](id<n>) is not defined if (dimensions == 0 || dimensions != n).
void test4(acc_t<0, sycl::mode::read_write, sycl::target::local> acc0,
           acc_t<1, sycl::mode::read, sycl::target::host_buffer> acc1,
           acc_t<2, sycl::mode::write, sycl::target::global_buffer> acc2,
           acc_t<3, sycl::mode::discard_write, sycl::target::constant_buffer> acc3) {
  IdxIdAny idx;
  acc0[idx]; // expected-error {{does not provide a subscript operator}}
  acc1[idx];
  acc2[idx];
  acc3[idx];

  IdxId1 idx1;
  IdxId2 idx2;
  IdxId3 idx3;

  acc1[idx1];
  acc1[idx2]; // expected-error {{no viable overloaded operator[]}}
  // expected-note@* {{no known conversion from 'IdxId2' to 'id<1>'}}
  // expected-note@* {{no known conversion from 'IdxId2' to 'size_t'}}
  acc1[idx3]; // expected-error {{no viable overloaded operator[]}}
  // expected-note@* {{no known conversion from 'IdxId3' to 'id<1>'}}
  // expected-note@* {{no known conversion from 'IdxId3' to 'size_t'}}

  acc2[idx1]; // expected-error {{no viable overloaded operator[]}}
  // expected-note@* {{no known conversion from 'IdxId1' to 'id<2>'}}
  acc2[idx2];
  acc2[idx3]; // expected-error {{no viable overloaded operator[]}}
  // expected-note@* {{no known conversion from 'IdxId3' to 'id<2>'}}

  acc3[idx1]; // expected-error {{no viable overloaded operator[]}}
  // expected-note@* {{no known conversion from 'IdxId1' to 'id<3>'}}
  acc3[idx2]; // expected-error {{no viable overloaded operator[]}}
  // expected-note@* {{no known conversion from 'IdxId2' to 'id<3>'}}
  acc3[idx3];
}

// Check that operator[] returns values by value if accessMode == mode::read,
// and by reference otherwise.
void test5(acc_t<1, sycl::mode::read, sycl::target::global_buffer> acc1,
           acc_t<2, sycl::mode::write, sycl::target::host_buffer> acc2,
           acc_t<3, sycl::mode::read_write, sycl::target::local> acc3) {
  IdxIdAny idx;

  int val1 = acc1[idx];
  int &val1_r = acc1[idx]; // expected-error {{cannot bind}}

  int val2 = acc2[idx];
  int &val2_r = acc2[idx];

  int val3 = acc3[idx];
  int &val3_r = acc3[idx];
}

// Check get_pointer() method.
void test6(acc_t<1, sycl::mode::read, sycl::target::host_buffer> acc1,
           acc_t<2, sycl::mode::write, sycl::target::global_buffer> acc2,
           acc_t<3, sycl::mode::read_write, sycl::target::constant_buffer> acc3) {
  int *val = acc1.get_pointer();
  acc2.get_pointer();
  acc3.get_pointer();
}

// Check that there are two different versions of operator[] if
// (dimensions == 1) and only one if (dimensions > 1).
void test7(acc_t<1, sycl::mode::read_write, sycl::target::host_buffer> acc1,
           acc_t<2, sycl::mode::write, sycl::target::global_buffer> acc2,
           acc_t<3, sycl::mode::read, sycl::target::constant_buffer> acc3) {
  IdxIdSz idx;
  acc1[idx]; // expected-error {{use of overloaded operator '[]' is ambiguous}}
  // expected-note@* {{candidate function}}
  // expected-note@* {{candidate function}}
  // expected-note@* {{candidate function}}
  // expected-note@* {{candidate function}}
  acc2[idx][idx]; // expected-error {{use of overloaded operator '[]' is ambiguous}}
  // expected-note@* {{candidate function}}
  // expected-note@* {{candidate function}}
  // expected-note@* {{candidate function}}
  acc3[idx][idx][idx]; // expected-error {{use of overloaded operator '[]' is ambiguous}}
  // expected-note@* {{candidate function}}
  // expected-note@* {{candidate function}}
  // expected-note@* {{candidate function}}

}

// Check that there is no operator[] if (dimensions == 0).
struct A {
  int operator[](size_t x);
};
template <sycl::target Target>
struct X : acc_t<0, sycl::mode::read, Target>, A {};
void test8(X<sycl::target::host_buffer> acc1,
           X<sycl::target::global_buffer> acc2,
           X<sycl::target::constant_buffer> acc3,
           X<sycl::target::local> acc4) {
  acc1[42];
  acc2[42];
  acc3[42];
  acc4[42];
};
