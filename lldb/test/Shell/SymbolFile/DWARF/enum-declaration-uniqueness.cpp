// RUN: %clangxx_host -gdwarf -c -o %t_a.o %s -DFILE_A
// RUN: %clangxx_host -gdwarf -c -o %t_b.o %s -DFILE_B
// RUN: %clangxx_host -o %t %t_a.o %t_b.o
// RUN: %lldb %t \
// RUN:   -o "target variable my_enum my_enum_ref" -o "image dump ast" \
// RUN:   -o exit | FileCheck %s


// CHECK: (lldb) target variable
// CHECK: (MyEnum) my_enum = MyEnum_A
// CHECK: (MyEnum &) my_enum_ref =
// CHECK-SAME: &::my_enum_ref = MyEnum_A

// CHECK: (lldb) image dump ast
// CHECK: EnumDecl {{.*}} MyEnum
// CHECK-NEXT: EnumConstantDecl {{.*}} MyEnum_A 'MyEnum'
// CHECK-NOT: MyEnum

enum MyEnum : int;

extern MyEnum my_enum;

#ifdef FILE_A
enum MyEnum : int { MyEnum_A };

MyEnum my_enum = MyEnum_A;

int main() {}
#endif
#ifdef FILE_B
MyEnum &my_enum_ref = my_enum;
#endif
