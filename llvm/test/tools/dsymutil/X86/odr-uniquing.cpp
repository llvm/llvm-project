/* Compile with:
   clang -g -c  odr-uniquing.cpp -o odr-uniquing/1.o
   cp odr-uniquing/1.o odr-uniquing/2.o
   The aim of these test is to check that all the 'type types' that
   should be uniqued through the ODR really are.
   
   The resulting object file is linked against itself using a fake
   debug map. The end result is:
    - with ODR uniquing: all types (expect for the union for now) in
   the second CU should point back to the types of the first CU.
    - without ODR uniquing: all types are re-emited in the second CU
 */

// RUN: dsymutil -f -oso-prepend-path=%p/../Inputs/odr-uniquing -y %p/dummy-debug-map.map -o - | llvm-dwarfdump -v -debug-info - | FileCheck -check-prefix=ODR -check-prefix=CHECK %s
// RUN: dsymutil -f -oso-prepend-path=%p/../Inputs/odr-uniquing -y %p/dummy-debug-map.map -no-odr -o - | llvm-dwarfdump -v -debug-info - | FileCheck -check-prefix=NOODR -check-prefix=CHECK %s

// The first compile unit contains all the types:
// CHECK: TAG_compile_unit
// CHECK-NOT: DW_TAG
// CHECK: AT_name{{.*}}"odr-uniquing.cpp"

struct S {
  struct Nested {};
};

// CHECK: 0x[[S:[0-9a-f]*]]:{{.*}}DW_TAG_structure_type
// CHECK-NEXT: DW_AT_name{{.*}}"S"
// CHECK-NOT: NULL
// CHECK: 0x[[NESTED:[0-9a-f]*]]:{{.*}}DW_TAG_structure_type
// CHECK-NOT: DW_TAG
// CHECK: DW_AT_name{{.*}}"Nested"
// CHECK: NULL

namespace N {
class C {};
}

// CHECK: DW_TAG_namespace
// CHECK-NEXT: DW_AT_name{{.*}}"N"
// CHECK-NOT: NULL
// CHECK: 0x[[NC:[0-9a-f]*]]:{{.*}}DW_TAG_class_type
// CHECK-NEXT: DW_AT_name{{.*}}"C"
// CHECK: NULL

union U {
  class C {} C;
  struct S {} S;
};

// CHECK:  0x[[U:[0-9a-f]*]]:{{.*}}DW_TAG_union_type
// CHECK-NEXT: DW_AT_name{{.*}}"U"
// CHECK-NOT: NULL
// CHECK:  0x[[UC:[0-9a-f]*]]:{{.*}}DW_TAG_class_type
// CHECK-NOT: NULL
// CHECK:  0x[[US:[0-9a-f]*]]:{{.*}}DW_TAG_structure_type
// CHECK: NULL

typedef S AliasForS;

// CHECK: 0x[[ALIASFORS:[0-9a-f]*]]:{{.*}}DW_TAG_typedef
// CHECK-NEXT: DW_AT_type{{.*}}[[S]]
// CHECK-NEXT: DW_AT_name{{.*}}"AliasForS"

namespace {
class AnonC {};
}

// CHECK: DW_TAG_namespace
// CHECK-NOT: {{DW_AT_name|NULL|DW_TAG}}
// CHECK: 0x[[ANONC:[0-9a-f]*]]:{{.*}}DW_TAG_class_type
// CHECK-NEXT: DW_AT_name{{.*}}"AnonC"

// This function is only here to hold objects that refer to the above types.
void foo() {
  AliasForS s;
  S::Nested n;
  N::C nc;
  AnonC ac;
  U u;
}

// The second CU contents depend on whether we disabled ODR uniquing or
// not.

// CHECK: TAG_compile_unit
// CHECK-NOT: DW_TAG
// CHECK: AT_name{{.*}}"odr-uniquing.cpp"

// The union itself is not uniqued for now (for dsymutil-compatibility),
// but the types defined inside it should be.
// ODR: DW_TAG_union_type
// ODR-NEXT: DW_AT_name{{.*}}"U"
// ODR: DW_TAG_member
// ODR-NEXT: DW_AT_name{{.*}}"C"
// ODR-NOT: DW_TAG
// ODR: DW_AT_type{{.*}}[[UC]]
// ODR: DW_TAG_member
// ODR-NEXT: DW_AT_name{{.*}}"S"
// ODR-NOT: DW_TAG
// ODR: DW_AT_type{{.*}}[[US]]

// Check that the variables point to the right type
// ODR: DW_TAG_subprogram
// ODR-NOT: DW_TAG
// ODR: DW_AT_name{{.*}}"foo"
// ODR-NOT: NULL
// ODR: DW_TAG_variable
// ODR-NOT: DW_TAG
// ODR: DW_AT_name{{.*}}"s"
// ODR-NOT: DW_TAG
// ODR: DW_AT_type{{.*}}[[ALIASFORS]]
// ODR: DW_AT_name{{.*}}"n"
// ODR-NOT: DW_TAG
// ODR: DW_AT_type{{.*}}[[NESTED]]
// ODR: DW_TAG_variable
// ODR-NOT: DW_TAG
// ODR: DW_AT_name{{.*}}"nc"
// ODR-NOT: DW_TAG
// ODR: DW_AT_type{{.*}}[[NC]]
// ODR: DW_TAG_variable
// ODR-NOT: DW_TAG
// ODR: DW_AT_name{{.*}}"ac"
// ODR-NOT: DW_TAG
// ODR: DW_AT_type{{.*}}[[ANONC]]

// With no ODR uniquing, we should get copies of all the types:

// This is "struct S"
// NOODR: 0x[[DUP_S:[0-9a-f]*]]:{{.*}}DW_TAG_structure_type
// NOODR-NEXT: DW_AT_name{{.*}}"S"
// NOODR-NOT: NULL
// NOODR: 0x[[DUP_NESTED:[0-9a-f]*]]:{{.*}}DW_TAG_structure_type
// NOODR-NOT: DW_TAG
// NOODR: DW_AT_name{{.*}}"Nested"

// This is "class N::C"
// NOODR: DW_TAG_namespace
// NOODR-NEXT: DW_AT_name{{.*}}"N"
// NOODR: 0x[[DUP_NC:[0-9a-f]*]]:{{.*}}DW_TAG_class_type
// NOODR-NEXT: DW_AT_name{{.*}}"C"

// This is "union U"
// NOODR:  0x[[DUP_U:[0-9a-f]*]]:{{.*}}DW_TAG_union_type
// NOODR-NEXT: DW_AT_name{{.*}}"U"
// NOODR-NOT: NULL
// NOODR:  0x[[DUP_UC:[0-9a-f]*]]:{{.*}}DW_TAG_class_type
// NOODR-NOT: NULL
// NOODR:  0x[[DUP_US:[0-9a-f]*]]:{{.*}}DW_TAG_structure_type
// NOODR: NULL

// Check that the variables point to the right type
// NOODR: DW_TAG_subprogram
// NOODR-NOT: DW_TAG
// NOODR: DW_AT_name{{.*}}"foo"
// NOODR-NOT: NULL
// NOODR: DW_TAG_variable
// NOODR-NOT: DW_TAG
// NOODR: DW_AT_name{{.*}}"s"
// NOODR-NOT: DW_TAG
// NOODR: DW_AT_type{{.*}}0x[[DUP_ALIASFORS:[0-9a-f]*]]
// NOODR: DW_TAG_variable
// NOODR-NOT: DW_TAG
// NOODR: DW_AT_name{{.*}}"n"
// NOODR-NOT: DW_TAG
// NOODR: DW_AT_type{{.*}}[[DUP_NESTED]]
// NOODR: DW_TAG_variable
// NOODR-NOT: DW_TAG
// NOODR: DW_AT_name{{.*}}"nc"
// NOODR-NOT: DW_TAG
// NOODR: DW_AT_type{{.*}}[[DUP_NC]]
// NOODR: DW_TAG_variable
// NOODR-NOT: DW_TAG
// NOODR: DW_AT_name{{.*}}"ac"
// NOODR-NOT: DW_TAG
// NOODR: DW_AT_type{{.*}}0x[[DUP_ANONC:[0-9a-f]*]]

// This is "AliasForS"
// NOODR: 0x[[DUP_ALIASFORS]]:{{.*}}DW_TAG_typedef
// NOODR-NOT: DW_TAG
// NOODR: DW_AT_name{{.*}}"AliasForS"

// This is "(anonymous namespace)::AnonC"
// NOODR: DW_TAG_namespace
// NOODR-NOT: {{DW_AT_name|NULL|DW_TAG}}
// NOODR: 0x[[DUP_ANONC]]:{{.*}}DW_TAG_class_type
// NOODR-NEXT: DW_AT_name{{.*}}"AnonC"

