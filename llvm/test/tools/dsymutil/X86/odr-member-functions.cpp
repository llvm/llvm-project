/* Compile with:
   for FILE in `seq 3`; do
     clang -g -c  odr-member-functions.cpp -DFILE$FILE -o odr-member-functions/$FILE.o
   done
 */

// RUN: dsymutil -f -oso-prepend-path=%p/../Inputs/odr-member-functions -y %p/dummy-debug-map.map -o - | llvm-dwarfdump -debug-info - | FileCheck %s

struct S {
  __attribute__((always_inline)) void foo() { bar(); }
  __attribute__((always_inline)) void foo(int i) { if (i) bar(); }
  void bar();

  template<typename T> void baz(T t) {}
};

#ifdef FILE1
void foo() {
  S s;
}

// CHECK: TAG_compile_unit
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_name{{.*}}"odr-member-functions.cpp"

// CHECK: 0x[[S:[0-9a-f]*]]:{{.*}}DW_TAG_structure_type
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: DW_AT_name{{.*}}"S"
// CHECK-NOT: NULL
// CHECK: 0x[[FOO:[0-9a-f]*]]:{{.*}}DW_TAG_subprogram
// CHECK-NEXT: DW_AT_MIPS_linkage_name{{.*}}"_ZN1S3fooEv"
// CHECK: NULL
// CHECK: 0x[[FOOI:[0-9a-f]*]]:{{.*}}DW_TAG_subprogram
// CHECK-NEXT: DW_AT_MIPS_linkage_name{{.*}}"_ZN1S3fooEi"

#elif defined(FILE2)
void foo() {
  S s;
  // Check that the overloaded member functions are resolved correctly
  s.foo();
  s.foo(1);
}

// CHECK: TAG_compile_unit
// CHECK-NOT: DW_TAG
// CHECK: AT_name{{.*}}"odr-member-functions.cpp"

// Normal member functions should be desribed by the type in the first
// CU, thus we should be able to reuse its definition and avoid
// reemiting it.
// CHECK-NOT: DW_TAG_structure_type

// CHECK: 0x[[FOO_SUB:[0-9a-f]*]]:{{.*}}DW_TAG_subprogram
// CHECK-NEXT: DW_AT_specification{{.*}}[[FOO]]
// CHECK-NOT: DW_TAG_structure_type
// CHECK: 0x[[FOOI_SUB:[0-9a-f]*]]:{{.*}}DW_TAG_subprogram
// CHECK-NEXT: DW_AT_specification{{.*}}[[FOOI]]
// CHECK-NOT: DW_TAG_structure_type

// CHECK: DW_TAG_variable
// CHECK-NOT: DW_TAG
// CHECK: DW_AT_name {{.*}}"s"
// CHECK-NOT: DW_TAG
// CHECK: DW_AT_type {{.*}}[[S]]
// CHECK: DW_TAG_inlined_subroutine
// CHECK-NEXT: DW_AT_abstract_origin{{.*}}[[FOO_SUB]]
// CHECK-NOT: DW_TAG
// CHECK: DW_AT_call_line{{.*}}40
// CHECK: DW_TAG_inlined_subroutine
// CHECK-NEXT: DW_AT_abstract_origin{{.*}}[[FOOI_SUB]]
// CHECK-NOT: DW_TAG
// CHECK: DW_AT_call_line{{.*}}41

#elif defined(FILE3)
void foo() {
  S s;
  s.baz<int>(42);
}

// CHECK: TAG_compile_unit
// CHECK-NOT: DW_TAG
// CHECK: AT_name{{.*}}"odr-member-functions.cpp"

// Template or other implicit members will be included in the type
// only if they are generated. Thus actually creating a new type.
// CHECK: DW_TAG_structure_type

// Skip 'normal' member functions
// CHECK: DW_TAG_subprogram
// CHECK: DW_TAG_subprogram
// CHECK: DW_TAG_subprogram

// This is the 'baz' member
// CHECK: 0x[[BAZ:[0-9a-f]*]]: DW_TAG_subprogram
// CHECK-NOT: DW_TAG
// CHECK: DW_AT_MIPS_linkage_name {{.*}}"_ZN1S3bazIiEEvT_"
// CHECK-NOT: DW_TAG
// CHECK: DW_AT_name {{.*}}"baz<int>"

// Skip foo3
// CHECK: DW_TAG_subprogram

// baz instanciation:
// CHECK: DW_TAG_subprogram
// CHECK-NOT: DW_TAG
// CHECK: DW_AT_specification {{.*}}[[BAZ]] "_ZN1S3bazIiEEvT_"
#else
#error "You must define which file you generate"
#endif
