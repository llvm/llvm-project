// RUN: %clang --target=x86_64-apple-macosx -c -gdwarf -o %t %s
// RUN: %lldb %t \
// RUN:   -o "target var global" \
// RUN:   -o "target var global2" \
// RUN:   -o "target var global3" \
// RUN:   -o "target var global4" \
// RUN:   -o "target var global5" \
// RUN:   -o "image dump ast" \
// RUN:   -o exit | FileCheck %s

// CHECK:      (lldb) image dump ast
// CHECK:      CXXRecordDecl {{.*}} struct Foo definition
// CHECK:      |-FieldDecl {{.*}} data 'char[5]'
// CHECK-NEXT: |-FieldDecl {{.*}} padding 'Empty'
// CHECK-NEXT: `-FieldDecl {{.*}} flag 'unsigned long'
// CHECK-NEXT:   `-ConstantExpr {{.*}}
// CHECK-NEXT:   |-value: Int 1
// CHECK-NEXT:     `-IntegerLiteral {{.*}} 'int' 1

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
// CHECK-NEXT: `-FieldDecl {{.*}} flag 'unsigned long'
// CHECK-NEXT:   `-ConstantExpr {{.*}}
// CHECK-NEXT:   |-value: Int 1
// CHECK-NEXT:     `-IntegerLiteral {{.*}} 'int' 1

struct ConsecutiveOverlap {
  char data[5];
  [[no_unique_address]] Empty p1;
  [[no_unique_address]] Empty2 p2;
  [[no_unique_address]] Empty3 p3;
  unsigned long flag : 1;
};

ConsecutiveOverlap global2;

// FIXME: we fail to deduce the unnamed bitfields here.
//
// CHECK:      CXXRecordDecl {{.*}} struct MultipleAtOffsetZero definition
// CHECK:      |-FieldDecl {{.*}} data 'char[5]'
// CHECK-NEXT: |-FieldDecl {{.*}} p1 'Empty'
// CHECK-NEXT: |-FieldDecl {{.*}} f1 'unsigned long'
// CHECK-NEXT: | `-ConstantExpr {{.*}}
// CHECK-NEXT: | |-value: Int 1
// CHECK-NEXT: |   `-IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: |-FieldDecl {{.*}} p2 'Empty2'
// CHECK-NEXT: `-FieldDecl {{.*}} f2 'unsigned long'
// CHECK-NEXT: | `-ConstantExpr {{.*}}
// CHECK-NEXT: | |-value: Int 1
// CHECK-NEXT: |   `-IntegerLiteral {{.*}} 'int' 1

struct MultipleAtOffsetZero {
  char data[5];
  [[no_unique_address]] Empty p1;
  int : 4;
  unsigned long f1 : 1;
  [[no_unique_address]] Empty2 p2;
  int : 4;
  unsigned long f2 : 1;
};

MultipleAtOffsetZero global3;

// FIXME: we fail to deduce the unnamed bitfields here.
//
// CHECK:      CXXRecordDecl {{.*}} struct MultipleEmpty definition
// CHECK:      |-FieldDecl {{.*}} data 'char[5]'
// CHECK-NEXT: |-FieldDecl {{.*}} p1 'Empty'
// CHECK-NEXT: |-FieldDecl {{.*}} f1 'unsigned long'
// CHECK-NEXT: | `-ConstantExpr {{.*}}
// CHECK-NEXT: | |-value: Int 1
// CHECK-NEXT: |   `-IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: |-FieldDecl {{.*}} p2 'Empty'
// CHECK-NEXT: `-FieldDecl {{.*}} f2 'unsigned long'
// CHECK-NEXT: | `-ConstantExpr {{.*}}
// CHECK-NEXT: | |-value: Int 1
// CHECK-NEXT: |   `-IntegerLiteral {{.*}} 'int' 1

struct MultipleEmpty {
  char data[5];
  [[no_unique_address]] Empty p1;
  int : 4;
  unsigned long f1 : 1;
  [[no_unique_address]] Empty p2;
  int : 4;
  unsigned long f2 : 1;
};

MultipleEmpty global4;

// CHECK:      CXXRecordDecl {{.*}} struct FieldBitfieldOverlap definition
// CHECK:      |-FieldDecl {{.*}} a 'int'
// CHECK-NEXT: | `-ConstantExpr {{.*}}
// CHECK-NEXT: | |-value: Int 3
// CHECK-NEXT: |   `-IntegerLiteral {{.*}} 'int' 3
// CHECK-NEXT: |-FieldDecl {{.*}} p1 'Empty'
// CHECK-NEXT: |-FieldDecl {{.*}} b 'int'
// CHECK-NEXT: | `-ConstantExpr {{.*}}
// CHECK-NEXT: | |-value: Int 6
// CHECK-NEXT: |   `-IntegerLiteral {{.*}} 'int' 6
// CHECK-NEXT: `-FieldDecl {{.*}} c 'int'
// CHECK-NEXT:   `-ConstantExpr {{.*}}
// CHECK-NEXT:   |-value: Int 1
// CHECK-NEXT:     `-IntegerLiteral {{.*}} 'int' 1

struct FieldBitfieldOverlap {
  int a : 3;
  [[no_unique_address]] Empty p1;
  int b : 6;
  int c : 1;
};

FieldBitfieldOverlap global5;
