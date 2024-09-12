// RUN: %clang --target=x86_64-apple-macosx -c -gdwarf -o %t %s
// RUN: %lldb %t \
// RUN:   -o "target var global" \
// RUN:   -o "image dump ast" \
// RUN:   -o exit | FileCheck %s

// CHECK:      (lldb) image dump ast
// CHECK:      CXXRecordDecl {{.*}} struct Foo definition
// CHECK:      |-FieldDecl {{.*}} data 'char[5]'
// CHECK-NEXT: |-FieldDecl {{.*}} padding 'Empty'
// CHECK-NEXT: `-FieldDecl {{.*}} sloc> flag 'unsigned long'
// CHECK-NEXT:   `-IntegerLiteral {{.*}} 'int' 1

struct Empty {};

struct Foo {
  char data[5];
  [[no_unique_address]] Empty padding;
  unsigned long flag : 1;
};

Foo global;
