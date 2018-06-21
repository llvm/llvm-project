/* Compile with:
   for FILE in `seq 3`; do
     clang -g -c  X86/odr-fwd-declaration2.cpp -DFILE$FILE -o Inputs/odr-fwd-declaration2/$FILE.o
   done
 */

// RUN: llvm-dsymutil -f -oso-prepend-path=%p/../Inputs/odr-fwd-declaration2 -y %p/dummy-debug-map.map -o - | llvm-dwarfdump -v -debug-info - | FileCheck %s

#ifdef FILE1
# 1 "Header.h" 1
struct A {
  struct B;
  B *bPtr;
  B &bRef;
  int B::*bPtrToField;
};
# 3 "Source1.cpp" 2
void foo() {
  A *ptr1 = 0;
}

// First we confirm that bPtr, bRef and bPtrToField reference the forward
// declaration of the struct B.
//
// CHECK: DW_TAG_member
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_name{{.*}} "bPtr"
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_type{{.*}}0x{{0*}}[[PTR1:[a-f0-9]*]]
//
// CHECK: [[STRUCT1:[a-f0-9]*]]:{{.*}}TAG_structure_type
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_name{{.*}} "B"
// CHECK-NOT: AT_byte_size
// CHECK: DW_AT_declaration
//
// CHECK: DW_TAG_member
// CHECK: AT_name{{.*}} "bRef"
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_type{{.*}}0x{{0*}}[[REF1:[a-f0-9]*]]
//
// CHECK: TAG_member
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_name{{.*}} "bPtrToField"
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_type{{.*}}0x{{0*}}[[PTRTOMEMBER1:[a-f0-9]*]]
//
// CHECK: [[PTR1]]:{{.*}}TAG_pointer_type
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_type{{.*}}0x{{0*}}[[STRUCT1]]
//
// CHECK: [[REF1]]:{{.*}}TAG_reference_type
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_type{{.*}}0x{{0*}}[[STRUCT1]]
//
// CHECK: [[PTRTOMEMBER1]]:{{.*}}TAG_ptr_to_member_type
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_containing_type{{.*}}0x{{0*}}[[STRUCT1]]

#elif defined(FILE2)
# 1 "Header.h" 1
struct A {
  struct B;
  B *bPtr;
  B &bRef;
  int B::*bPtrToField;
};
# 3 "Source2.cpp" 2
struct A::B {
  int x;
};
void bar() {
  A *ptr2 = 0;
}

// Next we confirm that bPtr, bRef and bPtrToField reference the definition of
// B, rather than its declaration.
//
// CHECK: [[STRUCTA:[a-f0-9]*]]:{{.*}}TAG_structure_type
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_name{{.*}} "A"
// CHECK-NOT: AT_byte_size
// CHECK: DW_AT_byte_size
//
// CHECK: DW_TAG_member
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_name{{.*}} "bPtr"
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_type{{.*}}0x{{0*}}[[PTR2:[a-f0-9]*]]
//
// CHECK: [[STRUCT2:[a-f0-9]*]]:{{.*}}TAG_structure_type
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_name{{.*}} "B"
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: DW_AT_byte_size
//
// CHECK: DW_TAG_member
// CHECK: AT_name{{.*}} "bRef"
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_type{{.*}}0x{{0*}}[[REF2:[a-f0-9]*]]
//
// CHECK: TAG_member
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_name{{.*}} "bPtrToField"
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_type{{.*}}0x{{0*}}[[PTRTOMEMBER2:[a-f0-9]*]]
//
// CHECK: [[PTR2]]:{{.*}}TAG_pointer_type
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_type{{.*}}0x{{0*}}[[STRUCT2]]
//
// CHECK: [[REF2]]:{{.*}}TAG_reference_type
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_type{{.*}}0x{{0*}}[[STRUCT2]]
//
// CHECK: [[PTRTOMEMBER2]]:{{.*}}TAG_ptr_to_member_type
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_containing_type{{.*}}0x{{0*}}[[STRUCT2:[a-f0-9]*]]

#elif defined(FILE3)
# 1 "Header.h" 1
struct A {
  struct B;
  B *bPtr;
  B &bRef;
  int B::*bPtrToField;
};
# 3 "Source2.cpp" 2
struct A::B {
  int x;
};
void bar() {
  A *ptr2 = 0;
}

// Finally we confirm that uniquing isn't broken by checking that further
// references to 'struct A' point to its now complete definition.
//
// CHECK: TAG_variable
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_name{{.*}} "ptr2"
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_type{{.*}}0x{{0*}}[[PTR3:[a-f0-9]*]]
//
// CHECK: [[PTR3]]:{{.*}}TAG_pointer_type
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_type{{.*}}0x{{0*}}[[STRUCTA]]

#else
#error "You must define which file you generate"
#endif
