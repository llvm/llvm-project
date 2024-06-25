// Test that we can correctly resolve forward declared types when they only
// differ in the template arguments of the surrounding context. The reproducer
// is sensitive to the order of declarations, so we test in both directions.

// REQUIRES: lld

// RUN: %clang --target=x86_64-pc-linux -c %s -o %t-a.o -g -gsimple-template-names -DFILE_A
// RUN: %clang --target=x86_64-pc-linux -c %s -o %t-b.o -g -gsimple-template-names -DFILE_B
// RUN: ld.lld %t-a.o %t-b.o -o %t
// RUN: %lldb %t -o "target variable --ptr-depth 1 --show-types both_a both_b" -o exit | FileCheck %s

// CHECK: (lldb) target variable
// CHECK-NEXT: (ReferencesBoth<'A'>) both_a = {
// CHECK-NEXT:   (Outer<'A'>::Inner *) a = 0x{{[0-9A-Fa-f]*}} {}
// CHECK-NEXT:   (Outer<'A'>::Inner *) b = 0x{{[0-9A-Fa-f]*}} {}
// CHECK-NEXT: }
// CHECK-NEXT: (ReferencesBoth<'B'>) both_b = {
// CHECK-NEXT:   (Outer<'A'>::Inner *) a = 0x{{[0-9A-Fa-f]*}} {}
// CHECK-NEXT:   (Outer<'B'>::Inner *) b = 0x{{[0-9A-Fa-f]*}} {}
// CHECK-NEXT: }

template<char C>
struct Outer {
  struct Inner {};
};

template<char C>
struct ReferencesBoth {
  Outer<'A'>::Inner *a;
  Outer<'B'>::Inner *b;
};

#ifdef FILE_A
Outer<'A'>::Inner inner_a;
extern Outer<'B'>::Inner inner_b;

ReferencesBoth<'A'> both_a{&inner_a, &inner_b};

#else
extern Outer<'A'>::Inner inner_a;
Outer<'B'>::Inner inner_b;

ReferencesBoth<'B'> both_b{&inner_a, &inner_b};
#endif
