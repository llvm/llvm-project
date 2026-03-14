/* Compile with:
   for FILE in `seq 3`; do
     clang -g -c  odr-member-functions.cpp -DFILE$FILE -o
   odr-member-functions/$FILE.o done
 */

// RUN: dsymutil --linker=parallel -f \
// RUN: -oso-prepend-path=%p/../../Inputs/odr-member-functions \
// RUN: -y %p/../dummy-debug-map.map -o %t1.out
// RUN: llvm-dwarfdump -debug-info %t1.out | FileCheck %s

struct S {
  __attribute__((always_inline)) void foo() { bar(); }
  __attribute__((always_inline)) void foo(int i) {
    if (i)
      bar();
  }
  void bar();

  template <typename T> void baz(T t) {}
};

#ifdef FILE1
void foo() { S s; }

// First chack that types are moved into the type table unit.

// CHECK: TAG_compile_unit
// CHECK: AT_name{{.*}}"__artificial_type_unit"

// CHECK: 0x[[INT_BASE:[0-9a-f]*]]: DW_TAG_base_type
// CHECK-NEXT: DW_AT_name{{.*}}"int"

// CHECK: 0x[[PTR_S:[0-9a-f]*]]:{{.*}}DW_TAG_pointer_type
// CHECK-NEXT: DW_AT_type{{.*}}0x[[STRUCT_S:[0-9a-f]*]] "S")

// CHECK: 0x[[STRUCT_S]]:{{.*}}DW_TAG_structure_type
// CHECK-NEXT: DW_AT_name{{.*}}"S"

// CHECK: DW_TAG_subprogram
// CHECK: DW_AT_MIPS_linkage_name{{.*}}"_ZN1S3barEv"
// CHECK: DW_AT_name{{.*}}"bar"

// CHECK: DW_TAG_formal_parameter
// CHECK-NEXT: DW_AT_type{{.*}}0x[[PTR_S]] "S *"

// CHECK: 0x[[BAZ_SUBPROGRAM:[0-9a-f]*]]:{{.*}}DW_TAG_subprogram
// CHECK: DW_AT_MIPS_linkage_name{{.*}}"_ZN1S3bazIiEEvT_"
// CHECK: DW_AT_name{{.*}}"baz<int>"

// CHECK: DW_TAG_formal_parameter
// CHECK-NEXT: DW_AT_type{{.*}}0x[[PTR_S]] "S *"

// CHECK: DW_TAG_formal_parameter
// CHECK-NEXT: DW_AT_type{{.*}}0x[[INT_BASE]] "int"

// CHECK: DW_TAG_template_type_parameter
// CHECK-NEXT: DW_AT_type{{.*}}0x[[INT_BASE]] "int"
// CHECK-NEXT: DW_AT_name{{.*}}"T"

// CHECK: 0x[[FOO_Ei:[0-9a-f]*]]:{{.*}}DW_TAG_subprogram
// CHECK: DW_AT_MIPS_linkage_name{{.*}}"_ZN1S3fooEi"
// CHECK: DW_AT_name{{.*}}"foo"

// CHECK: DW_TAG_formal_parameter
// CHECK-NEXT: DW_AT_type{{.*}}0x[[PTR_S]] "S *"

// CHECK: DW_TAG_formal_parameter
// CHECK-NEXT: DW_AT_type{{.*}}0x[[INT_BASE]] "int"

// CHECK: 0x[[FOO_Ev:[0-9a-f]*]]:{{.*}}DW_TAG_subprogram
// CHECK: DW_AT_MIPS_linkage_name{{.*}}"_ZN1S3fooEv"
// CHECK: DW_AT_name{{.*}}"foo"

// CHECK: DW_TAG_formal_parameter
// CHECK-NEXT: DW_AT_type{{.*}}0x[[PTR_S]] "S *"

// For the second unit check that it references structure "S"

// CHECK: TAG_compile_unit
// CHECK-NOT: {{DW_TAG|NULL}}
// CHECK: AT_name{{.*}}"odr-member-functions.cpp"

// CHECK: DW_TAG_subprogram
// CHECK: DW_AT_MIPS_linkage_name{{.*}}"_Z3foov"
// CHECK-NEXT: AT_name{{.*}}"foo"

// CHECK: DW_TAG_variable
// CHECK: DW_AT_name{{.*}}"s"
// CHECK: DW_AT_type{{.*}}(0x00000000[[STRUCT_S]] "S"

#elif defined(FILE2)
void foo() {
  S s;
  // Check that the overloaded member functions are resolved correctly
  s.foo();
  s.foo(1);
}

// For the third unit check that it references member functions of structure S.
// CHECK: TAG_compile_unit
// CHECK-NOT: DW_TAG
// CHECK: AT_name{{.*}}"odr-member-functions.cpp"

// CHECK: 0x[[ABASE_FOO_Ev:[0-9a-f]*]]:{{.*}}DW_TAG_subprogram
// CHECK-NEXT: DW_AT_specification {{.*}}(0x00000000[[FOO_Ev]] "_ZN1S3fooEv"
// CHECK-NEXT: DW_AT_inline    (DW_INL_inlined)
// CHECK-NEXT: DW_AT_object_pointer {{.*}}(0x[[ABASE_FOO_Ev_PARAM1:[0-9a-f]*]]

// CHECK: 0x[[ABASE_FOO_Ev_PARAM1]]:{{.*}}DW_TAG_formal_parameter
// CHECK-NEXT: DW_AT_name{{.*}}"this"
// CHECK-NEXT: DW_AT_type{{.*}}(0x00000000[[PTR_S]] "S *"

// CHECK: 0x[[ABASE_FOO_Ei:[0-9a-f]*]]:{{.*}}DW_TAG_subprogram
// CHECK-NEXT: DW_AT_specification {{.*}}(0x00000000[[FOO_Ei]] "_ZN1S3fooEi"
// CHECK-NEXT: DW_AT_inline    (DW_INL_inlined)
// CHECK-NEXT: DW_AT_object_pointer {{.*}}(0x[[ABASE_FOO_Ei_PARAM1:[0-9a-f]*]]

// CHECK: 0x[[ABASE_FOO_Ei_PARAM1]]:{{.*}}DW_TAG_formal_parameter
// CHECK-NEXT: DW_AT_name{{.*}}"this"
// CHECK-NEXT: DW_AT_type{{.*}}(0x00000000[[PTR_S]] "S *"

// CHECK: 0x[[ABASE_FOO_Ei_PARAM2:[0-9a-f]*]]:{{.*}}DW_TAG_formal_parameter
// CHECK-NEXT: DW_AT_name{{.*}}"i"
// CHECK: DW_AT_type    (0x00000000[[INT_BASE]] "int"

// CHECK: DW_TAG_subprogram
// CHECK: DW_AT_MIPS_linkage_name{{.*}}"_Z3foov"
// CHECK: DW_AT_name{{.*}}"foo"

// CHECK: DW_TAG_variable
// CHECK: DW_AT_name{{.*}}"s"
// CHECK: DW_AT_type{{.*}}(0x00000000[[STRUCT_S]] "S"

// CHECK: DW_TAG_inlined_subroutine
// CHECK: DW_AT_abstract_origin{{.*}}(0x[[ABASE_FOO_Ev]] "_ZN1S3fooEv"
// CHECK: DW_AT_low_pc
// CHECK: DW_AT_high_pc

// CHECK: DW_TAG_formal_parameter
// CHECK: DW_AT_abstract_origin{{.*}}(0x[[ABASE_FOO_Ev_PARAM1]] "this"

// CHECK: DW_TAG_inlined_subroutine
// CHECK: DW_AT_abstract_origin{{.*}}(0x[[ABASE_FOO_Ei]] "_ZN1S3fooEi"
// CHECK: DW_AT_low_pc
// CHECK: DW_AT_high_pc

// CHECK: DW_TAG_formal_parameter
// CHECK: DW_AT_abstract_origin{{.*}}(0x[[ABASE_FOO_Ei_PARAM1]] "this"

// CHECK: DW_TAG_formal_parameter
// CHECK: DW_AT_abstract_origin{{.*}}(0x[[ABASE_FOO_Ei_PARAM2]] "i"

#elif defined(FILE3)
void foo() {
  S s;
  s.baz<int>(42);
}

// For the fourth unit check that it references member functions of structure S.

// CHECK: TAG_compile_unit
// CHECK-NOT: DW_TAG
// CHECK: AT_name{{.*}}"odr-member-functions.cpp"

// CHECK: DW_TAG_subprogram
// CHECK: DW_AT_MIPS_linkage_name{{.*}}"_Z3foov"
// CHECK: DW_AT_name{{.*}}"foo"

// CHECK: DW_TAG_variable
// CHECK: DW_AT_name{{.*}}"s"
// CHECK: DW_AT_type{{.*}}(0x00000000[[STRUCT_S]] "S"

// CHECK: DW_TAG_subprogram
// CHECK: DW_AT_object_pointer{{.*}}(0x[[INST_PARAM2:[0-9a-f]*]]
// CHECK: DW_AT_specification{{.*}}(0x00000000[[BAZ_SUBPROGRAM]] "_ZN1S3bazIiEEvT_"

// CHECK: 0x[[INST_PARAM2]]:{{.*}}DW_TAG_formal_parameter
// CHECK: DW_AT_name{{.*}}"this"
// CHECK: DW_AT_type{{.*}}(0x00000000[[PTR_S]] "S *"

// CHECK: DW_TAG_formal_parameter
// CHECK: DW_AT_name{{.*}}"t"
// CHECK: DW_AT_type{{.*}}(0x00000000[[INT_BASE]] "int"

// CHECK: DW_TAG_template_type_parameter
// CHECK: DW_AT_type{{.*}}(0x00000000[[INT_BASE]] "int"
// CHECK: DW_AT_name{{.*}}"T"

#else
#error "You must define which file you generate"
#endif
