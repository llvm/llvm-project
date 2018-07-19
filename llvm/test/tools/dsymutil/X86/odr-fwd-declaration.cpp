/* Compile with:
   for FILE in `seq 3`; do
     clang -g -c  X86/odr-fwd-declaration.cpp -DFILE$FILE -o Inputs/odr-fwd-declaration/$FILE.o
   done
 */

// RUN: llvm-dsymutil -f -oso-prepend-path=%p/../Inputs/odr-fwd-declaration -y %p/dummy-debug-map.map -o - | llvm-dwarfdump -v -debug-info - | FileCheck %s

#ifdef FILE1
# 1 "Header.h" 1
typedef struct S *Sptr;
typedef Sptr *Sptrptr;
# 3 "Source1.cpp" 2
void foo() {
  Sptrptr ptr1 = 0;
}

// First we confirm that the typedefs reference the forward declaration of the
// struct S.
//
// CHECK: TAG_variable
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_name{{.*}} "ptr1"
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_type{{.*}}0x{{0*}}[[TYPEDEF1:[a-f0-9]*]]
//
// CHECK: [[TYPEDEF1]]:{{.*}}TAG_typedef
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_type{{.*}}0x{{0*}}[[PTR1:[a-f0-9]*]]
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_name{{.*}} "Sptrptr"
//
// CHECK: [[PTR1]]:{{.*}}TAG_pointer_type
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_type{{.*}}0x{{0*}}[[TYPEDEF2:[a-f0-9]*]]
//
// CHECK: [[TYPEDEF2]]:{{.*}}TAG_typedef
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_type{{.*}}0x{{0*}}[[PTR2:[a-f0-9]*]]
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_name{{.*}} "Sptr"
//
// CHECK: [[PTR2]]:{{.*}}TAG_pointer_type
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_type{{.*}}0x{{0*}}[[FWDSTRUCT:[a-f0-9]*]]
//
// CHECK: [[FWDSTRUCT]]:{{.*}}TAG_structure_type
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_name{{.*}} "S"
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_declaration
// CHECK-NOT AT_byte_size

#elif defined(FILE2)
# 1 "Header.h" 1
typedef struct S *Sptr;
typedef Sptr *Sptrptr;
# 3 "Source2.cpp" 2
struct S {
  int field;
};
void bar() {
  Sptrptr ptr2 = 0;
}

// Next we confirm that the typedefs reference the definition rather than the
// previous declaration of S.
//
// CHECK: TAG_variable
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_name{{.*}} "ptr2"
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_type{{.*}}0x{{0*}}[[TYPEDEF3:[a-f0-9]*]]
//
// CHECK: [[TYPEDEF3]]:{{.*}}TAG_typedef
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_type{{.*}}0x{{0*}}[[PTR3:[a-f0-9]*]]
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_name{{.*}} "Sptrptr"
//
// CHECK: [[PTR3]]:{{.*}}TAG_pointer_type
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_type{{.*}}0x{{0*}}[[TYPEDEF4:[a-f0-9]*]]
//
// CHECK: [[TYPEDEF4]]:{{.*}}TAG_typedef
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_type{{.*}}0x{{0*}}[[PTR4:[a-f0-9]*]]
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_name{{.*}} "Sptr"
//
// CHECK: [[PTR4]]:{{.*}}TAG_pointer_type
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_type{{.*}}0x{{0*}}[[STRUCT:[a-f0-9]*]]
//
// CHECK: [[STRUCT]]:{{.*}}TAG_structure_type
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_name{{.*}} "S"
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK-NOT: AT_declaration
// CHECK: AT_byte_size
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: TAG_member

#elif defined(FILE3)
# 1 "Header.h" 1
typedef struct S *Sptr;
typedef Sptr *Sptrptr;
# 3 "Source1.cpp" 2
void foo() {
  Sptrptr ptr1 = 0;
}

// Finally we confirm that uniquing is not broken and the same typedef is
// referenced by ptr1.
//
// CHECK: TAG_variable
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_name{{.*}} "ptr1"
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_type{{.*}}0x{{0*}}[[TYPEDEF3]]
// CHECK-NOT: TAG_typedef
// CHECK-NOT: TAG_pointer
// CHECK-NOT: TAG_structure_type

#else
#error "You must define which file you generate"
#endif
