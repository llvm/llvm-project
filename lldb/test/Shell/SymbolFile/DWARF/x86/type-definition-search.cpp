// Test that we can correctly resolve forward declared types when they only
// differ in the template arguments of the surrounding context. The reproducer
// is sensitive to the order of declarations, so we test in both directions.

// REQUIRES: lld

// RUN: %clang --target=x86_64-pc-linux -c %s -o %t-n-a.o -g -gsimple-template-names -DFILE_A
// RUN: %clang --target=x86_64-pc-linux -c %s -o %t-n-b.o -g -gsimple-template-names -DFILE_B
// RUN: ld.lld %t-n-a.o %t-n-b.o -o %t-n
// RUN: %lldb %t-n -o "target variable --ptr-depth 1 --show-types both_a both_b" -o exit | FileCheck %s

// RUN: %clang --target=x86_64-pc-linux -c %s -o %t-t-a.o -g -fdebug-types-section -DFILE_A
// RUN: %clang --target=x86_64-pc-linux -c %s -o %t-t-b.o -g -fdebug-types-section -DFILE_B
// RUN: ld.lld %t-t-a.o %t-t-b.o -o %t-t
// RUN: %lldb %t-t -o "target variable --ptr-depth 1 --show-types both_a both_b" -o exit | FileCheck %s

// RUN: %clang --target=x86_64-pc-linux -c %s -o %t-tn-a.o -g -fdebug-types-section -gsimple-template-names -DFILE_A
// RUN: %clang --target=x86_64-pc-linux -c %s -o %t-tn-b.o -g -fdebug-types-section -gsimple-template-names -DFILE_B
// RUN: ld.lld %t-tn-a.o %t-tn-b.o -o %t-tn
// RUN: %lldb %t-tn -o "target variable --ptr-depth 1 --show-types both_a both_b" -o exit | FileCheck %s

// CHECK: (lldb) target variable
// CHECK-NEXT: (ReferencesBoth<'A'>) both_a = {
// CHECK-NEXT:   (Outer<'A'>::Inner *) a = 0x{{[0-9A-Fa-f]*}} {}
// CHECK-NEXT:   (Outer<'B'>::Inner *) b = 0x{{[0-9A-Fa-f]*}} {}
// CHECK-NEXT: }
// CHECK-NEXT: (ReferencesBoth<'B'>) both_b = {
// CHECK-NEXT:   (Outer<'A'>::Inner *) a = 0x{{[0-9A-Fa-f]*}} {}
// CHECK-NEXT:   (Outer<'B'>::Inner *) b = 0x{{[0-9A-Fa-f]*}} {}
// CHECK-NEXT: }

template <char C> struct Outer {
  struct Inner {};
};

template <char C> struct ReferencesBoth {
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
