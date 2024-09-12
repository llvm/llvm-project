// RUN: %clang --target=x86_64-apple-macosx -c -gdwarf -o %t %s
// RUN: %lldb %t \
// RUN:   -o "target var global" \
// RUN:   -o "target var global2" \
// RUN:   -o "image dump ast" \
// RUN:   -o exit | FileCheck %s

// CHECK:      (lldb) image dump ast
// CHECK:      CXXRecordDecl {{.*}} struct Foo definition
// CHECK:      |-FieldDecl {{.*}} data 'char[5]'
// CHECK-NEXT: |-FieldDecl {{.*}} padding 'Empty'
// CHECK-NEXT: `-FieldDecl {{.*}} sloc> flag 'unsigned long'
// CHECK-NEXT:   `-IntegerLiteral {{.*}} 'int' 1

struct Empty {};
struct Empty2 {};
struct Empty3 {};

struct Foo {
  char data[5];
  [[no_unique_address]] Empty padding;
  unsigned long flag : 1;
};

Foo global;

// CHECK:      CXXRecordDecl {{.*}} struct ConsecutiveOverlap definition
// CHECK:      |-FieldDecl {{.*}} data 'char[5]'
// CHECK-NEXT: |-FieldDecl {{.*}} p1 'Empty'
// CHECK-NEXT: |-FieldDecl {{.*}} p2 'Empty2'
// CHECK-NEXT: |-FieldDecl {{.*}} p3 'Empty3'
// CHECK-NEXT: `-FieldDecl {{.*}} sloc> flag 'unsigned long'
// CHECK-NEXT:   `-IntegerLiteral {{.*}} 'int' 1

struct ConsecutiveOverlap {
  char data[5];
  [[no_unique_address]] Empty p1;
  [[no_unique_address]] Empty2 p2;
  [[no_unique_address]] Empty3 p3;
  unsigned long flag : 1;
};

ConsecutiveOverlap global2;
