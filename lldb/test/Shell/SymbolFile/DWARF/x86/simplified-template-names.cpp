// Test lldb is able to compute the fully qualified names on templates with
// -gsimple-template-names and -fdebug-types-section.

// REQUIRES: lld

// Test against logging to see if we print the fully qualified names correctly.
// RUN: %clangxx --target=x86_64-pc-linux -g -gsimple-template-names %s -c -o %t1.o
// RUN: ld.lld %t1.o -o %t1
// RUN: %lldb %t1 -o "log enable dwarf comp" -o "target variable v3" -o exit | FileCheck %s --check-prefix=LOG

// Test that we following DW_AT_signature correctly. If not, lldb might confuse the types of v1 and v2.
// RUN: %clangxx --target=x86_64-pc-linux -g -gsimple-template-names -fdebug-types-section %s -c -o %t2.o
// RUN: ld.lld %t2.o -o %t2
// RUN: %lldb %t2 -o "target variable v1 v2" -o exit | FileCheck %s --check-prefix=TYPE

// LOG: unique name: t3<t2<int> >::t4

// TYPE:      (t2<outer_struct1::t1<int> >) v1 = {}
// TYPE-NEXT: (t2<outer_struct2::t1<int> >) v2 = {}

struct outer_struct1 {
  template <typename> struct t1 {};
};

struct outer_struct2 {
  template <typename> struct t1 {};
};

template <typename> struct t2 {};
t2<outer_struct1::t1<int>> v1;
t2<outer_struct2::t1<int>> v2;

template <typename> struct t3 {
  struct t4 {};
};
t3<t2<int>>::t4 v3;
