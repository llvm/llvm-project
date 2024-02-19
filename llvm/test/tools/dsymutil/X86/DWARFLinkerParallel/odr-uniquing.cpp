/* Compile with:
   clang -g -c  odr-uniquing.cpp -o odr-uniquing/1.o
   cp odr-uniquing/1.o odr-uniquing/2.o
   The aim of these test is to check that all the 'type types' that
   should be uniqued through the ODR really are.

   The resulting object file is linked against itself using a fake
   debug map. The end result is:
    - with ODR uniquing: all types in second and third CUs should point back
   to the types of the first CU(except types from anonymous namespace).
    - without ODR uniquing: all types are re-emited in the second CU.
 */

/* Check by llvm-dwarfdump --verify */
// RUN: dsymutil --linker=parallel -f -oso-prepend-path=%p/../../Inputs/odr-uniquing \
// RUN: -y %p/../dummy-debug-map.map -o - | llvm-dwarfdump --verify - | \
// RUN: FileCheck -check-prefixes=VERIFY %s
// RUN: dsymutil --linker=parallel -f -oso-prepend-path=%p/../../Inputs/odr-uniquing \
// RUN: -y %p/../dummy-debug-map.map -no-odr -o - | llvm-dwarfdump --verify - | \
// RUN: FileCheck -check-prefixes=VERIFY %s

/* Check for llvm-dwarfdump -a output */
// RUN: dsymutil --linker=parallel -f -oso-prepend-path=%p/../../Inputs/odr-uniquing \
// RUN: -y %p/../dummy-debug-map.map -o - | llvm-dwarfdump -v -a - | \
// RUN: FileCheck -check-prefixes=CHECK %s
// RUN: dsymutil --linker=parallel -f -oso-prepend-path=%p/../../Inputs/odr-uniquing \
// RUN: -y %p/../dummy-debug-map.map -no-odr -o - | llvm-dwarfdump -v -a - | \
// RUN: FileCheck -check-prefixes=CHECK-NOODR %s

struct S {
  struct Nested {};
};

namespace N {
class C {};
} // namespace N

union U {
  class C {
  } C;
  struct S {
  } S;
};

typedef S AliasForS;

namespace {
class AnonC {};
} // namespace

// This function is only here to hold objects that refer to the above types.
void foo() {
  AliasForS s;
  S::Nested n;
  N::C nc;
  AnonC ac;
  U u;
}

// VERIFY: Verifying .debug_abbrev...
// VERIFY: Verifying .debug_info Unit Header Chain...
// VERIFY: Verifying .debug_types Unit Header Chain...
// VERIFY: Verifying .apple_names...
// VERIFY: Verifying .apple_types...
// VERIFY: Verifying .apple_namespaces...
// VERIFY: Verifying .apple_objc...
// VERIFY: No errors.

// The first compile unit contains all the types:
// CHECK: .debug_info contents
// CHECK: DW_TAG_compile_unit
// CHECK: DW_AT_language{{.*}} (DW_LANG_C_plus_plus)
// CHECK: DW_AT_name{{.*}}"__artificial_type_unit")
// CHECK: DW_AT_stmt_list{{.*}}(0x[[LINE_TABLE_OFF1:[0-9a-f]*]])

// CHECK:0x[[N_NAMESPACE:[0-9a-f]*]]:{{.*}}DW_TAG_namespace
// CHECK:DW_AT_name{{.*}}"N"

// CHECK:0x[[C_CLASS:[0-9a-f]*]]:{{.*}}DW_TAG_class_type
// CHECK:DW_AT_name{{.*}}"C"
// CHECK:DW_AT_byte_size [DW_FORM_data1] (0x01)
// CHECK-DAG:DW_AT_decl_file [DW_FORM_data1] ("{{[\\/]}}tmp{{[\\/]}}odr-uniquing.cpp
// CHECK-DAG:DW_AT_decl_line [DW_FORM_data1] (35)

// CHECK:0x[[S_STRUCTURE:[0-9a-f]*]]:{{.*}}DW_TAG_structure_type
// CHECK:DW_AT_name{{.*}}"S"
// CHECK-DAG:DW_AT_decl_file [DW_FORM_data1] ("{{[\\/]}}tmp{{[\\/]}}odr-uniquing.cpp
// CHECK-DAG:DW_AT_decl_line [DW_FORM_data1] (22)

// CHECK:0x[[S_STRUCTURE_NESTED:[0-9a-f]*]]:{{.*}}DW_TAG_structure_type
// CHECK:DW_AT_name{{.*}}"Nested"
// CHECK-DAG:DW_AT_decl_file [DW_FORM_data1] ("{{[\\/]}}tmp{{[\\/]}}odr-uniquing.cpp")
// CHECK-DAG:DW_AT_decl_line [DW_FORM_data1]       (23)

// CHECK:0x[[TYPEDEF_ALIASFORS:[0-9a-f]*]]:{{.*}}DW_TAG_typedef
// CHECK:DW_AT_name{{.*}}"AliasForS"
// CHECK-DAG:DW_AT_decl_file [DW_FORM_data1] ("{{[\\/]}}tmp{{[\\/]}}odr-uniquing.cpp
// CHECK-DAG:DW_AT_decl_line [DW_FORM_data1] (58)

// CHECK:0x[[U_UNION:[0-9a-f]*]]:{{.*}}DW_TAG_union_type
// CHECK:DW_AT_name{{.*}}"U"
// CHECK-DAG:DW_AT_decl_file [DW_FORM_data1] ("{{[\\/]}}tmp{{[\\/]}}odr-uniquing.cpp
// CHECK-DAG:DW_AT_decl_line [DW_FORM_data1] (45)

// CHECK:0x[[U_C_CLASS:[0-9a-f]*]]:{{.*}}DW_TAG_class_type
// CHECK:DW_AT_name{{.*}}"C"
// CHECK-DAG:DW_AT_decl_file [DW_FORM_data1] ("{{[\\/]}}tmp{{[\\/]}}odr-uniquing.cpp
// CHECK-DAG:DW_AT_decl_line [DW_FORM_data1] (46)

// CHECK:0x[[U_C_MEMBER:[0-9a-f]*]]:{{.*}}DW_TAG_member
// CHECK:DW_AT_name{{.*}}"C"
// CHECK-DAG:DW_AT_decl_file [DW_FORM_data1] ("{{[\\/]}}tmp{{[\\/]}}odr-uniquing.cpp
// CHECK-DAG:DW_AT_decl_line [DW_FORM_data1] (46)

// CHECK:0x[[U_S_MEMBER:[0-9a-f]*]]:{{.*}}DW_TAG_member
// CHECK:DW_AT_name{{.*}}"S"
// CHECK-DAG:DW_AT_decl_file [DW_FORM_data1] ("{{[\\/]}}tmp{{[\\/]}}odr-uniquing.cpp
// CHECK-DAG:DW_AT_decl_line [DW_FORM_data1] (47)

// CHECK:0x[[U_S_STRUCT:[0-9a-f]*]]:{{.*}}DW_TAG_structure_type
// CHECK:DW_AT_name{{.*}}"S"
// CHECK-DAG:DW_AT_decl_file [DW_FORM_data1] ("{{[\\/]}}tmp{{[\\/]}}odr-uniquing.cpp
// CHECK-DAG:DW_AT_decl_line [DW_FORM_data1] (47)

// The second compile unit contains subprogram and its variables:
// CHECK:DW_TAG_compile_unit
// CHECK:DW_AT_name{{.*}}"odr-uniquing.cpp"
// CHECK-NEXT: DW_AT_stmt_list{{.*}}(0x[[LINE_TABLE_OFF2:[0-9a-f]*]])

// CHECK:DW_TAG_subprogram
// CHECK:DW_AT_low_pc
// CHECK:DW_AT_high_pc
// CHECK:DW_AT_frame_base
// CHECK:DW_AT_MIPS_linkage_name{{.*}}"_Z3foov"
// CHECK:DW_AT_name{{.*}}"foo"
// CHECK:DW_AT_decl_file [DW_FORM_data1] ("{{[\\/]}}tmp{{[\\/]}}odr-uniquing.cpp
// CHECK:DW_AT_decl_line [DW_FORM_data1] (74)
// CHECK:DW_AT_external

// CHECK:DW_TAG_variable
// CHECK:DW_AT_name{{.*}}"s"
// CHECK:DW_AT_decl_file [DW_FORM_data1] ("{{[\\/]}}tmp{{[\\/]}}odr-uniquing.cpp
// CHECK:DW_AT_decl_line [DW_FORM_data1] (75)
// CHECK:DW_AT_type [DW_FORM_ref_addr] (0x{{0*}}[[TYPEDEF_ALIASFORS]] "AliasForS

// CHECK:DW_TAG_variable
// CHECK:DW_AT_name{{.*}}"n"
// CHECK:DW_AT_decl_file [DW_FORM_data1] ("{{[\\/]}}tmp{{[\\/]}}odr-uniquing.cpp
// CHECK:DW_AT_decl_line [DW_FORM_data1] (76)
// CHECK:DW_AT_type [DW_FORM_ref_addr] (0x{{0*}}[[S_STRUCTURE_NESTED]] "S::Neste

// CHECK:DW_TAG_variable
// CHECK:DW_AT_name{{.*}}"nc"
// CHECK:DW_AT_decl_file [DW_FORM_data1] ("{{[\\/]}}tmp{{[\\/]}}odr-uniquing.cpp
// CHECK:DW_AT_decl_line [DW_FORM_data1] (77)
// CHECK:DW_AT_type [DW_FORM_ref_addr] (0x{{0*}}[[C_CLASS]] "N::C"

// CHECK:DW_TAG_variable
// CHECK:DW_AT_name{{.*}}"ac"
// CHECK:DW_AT_decl_file [DW_FORM_data1] ("{{[\\/]}}tmp{{[\\/]}}odr-uniquing.cpp
// CHECK:DW_AT_decl_line [DW_FORM_data1] (78)
// CHECK:DW_AT_type [DW_FORM_ref4]{{.*}} {0x[[ANON_CLASS1:[0-9a-f]*]]} "(anonymous namespace)::AnonC")

// CHECK:DW_TAG_variable
// CHECK:DW_AT_name{{.*}}"u"
// CHECK:DW_AT_decl_file [DW_FORM_data1] ("{{[\\/]}}tmp{{[\\/]}}odr-uniquing.cpp
// CHECK:DW_AT_decl_line [DW_FORM_data1] (79)
// CHECK:DW_AT_type [DW_FORM_ref_addr] (0x{{0*}}[[U_UNION]] "U"

// CHECK:0x[[ANON_NAMESPACE1:[0-9a-f]*]]:{{.*}}DW_TAG_namespace
// CHECK-NEXT:DW_AT_decl_file{{.*}}"{{[\\/]}}tmp{{[\\/]}}odr-uniquing.cpp"

// CHECK:0x[[ANON_CLASS1]]:{{.*}}DW_TAG_class_type
// CHECK:DW_AT_name{{.*}}"AnonC"
// CHECK:DW_AT_byte_size [DW_FORM_data1] (0x01)
// CHECK:DW_AT_decl_file [DW_FORM_data1] ("{{[\\/]}}tmp{{[\\/]}}odr-uniquing.cpp
// CHECK:DW_AT_decl_line [DW_FORM_data1] (65)

// The third compile unit contains subprogram and its variables:
// CHECK:DW_TAG_compile_unit
// CHECK:DW_AT_name{{.*}}"odr-uniquing.cpp"
// CHECK-NEXT:DW_AT_stmt_list{{.*}}(0x[[LINE_TABLE_OFF3:[0-9a-f]*]])

// CHECK:DW_TAG_subprogram
// CHECK:DW_AT_low_pc
// CHECK:DW_AT_high_pc
// CHECK:DW_AT_frame_base
// CHECK:DW_AT_MIPS_linkage_name{{.*}}"_Z3foov"
// CHECK:DW_AT_name{{.*}}"foo"
// CHECK:DW_AT_decl_file [DW_FORM_data1] ("{{[\\/]}}tmp{{[\\/]}}odr-uniquing.cpp
// CHECK:DW_AT_decl_line [DW_FORM_data1] (74)
// CHECK:DW_AT_external

// CHECK:DW_TAG_variable
// CHECK:DW_AT_name{{.*}}"s"
// CHECK:DW_AT_decl_file [DW_FORM_data1] ("{{[\\/]}}tmp{{[\\/]}}odr-uniquing.cpp
// CHECK:DW_AT_decl_line [DW_FORM_data1] (75)
// CHECK:DW_AT_type [DW_FORM_ref_addr] (0x{{0*}}[[TYPEDEF_ALIASFORS]] "AliasForS

// CHECK:DW_TAG_variable
// CHECK:DW_AT_name{{.*}}"n"
// CHECK:DW_AT_decl_file [DW_FORM_data1] ("{{[\\/]}}tmp{{[\\/]}}odr-uniquing.cpp
// CHECK:DW_AT_decl_line [DW_FORM_data1] (76)
// CHECK:DW_AT_type [DW_FORM_ref_addr] (0x{{0*}}[[S_STRUCTURE_NESTED]] "S::Neste

// CHECK:DW_TAG_variable
// CHECK:DW_AT_name{{.*}}"nc"
// CHECK:DW_AT_decl_file [DW_FORM_data1] ("{{[\\/]}}tmp{{[\\/]}}odr-uniquing.cpp
// CHECK:DW_AT_decl_line [DW_FORM_data1] (77)
// CHECK:DW_AT_type [DW_FORM_ref_addr] (0x{{0*}}[[C_CLASS]] "N::C"

// CHECK:DW_TAG_variable
// CHECK:DW_AT_name{{.*}}"ac"
// CHECK:DW_AT_decl_file [DW_FORM_data1] ("{{[\\/]}}tmp{{[\\/]}}odr-uniquing.cpp
// CHECK:DW_AT_decl_line [DW_FORM_data1] (78)
// CHECK:DW_AT_type [DW_FORM_ref4]{{.*}} {0x[[ANON_CLASS2:[0-9a-f]*]]} "(anonymous namespace)::AnonC")

// CHECK:DW_TAG_variable
// CHECK:DW_AT_name{{.*}}"u"
// CHECK:DW_AT_decl_file [DW_FORM_data1] ("{{[\\/]}}tmp{{[\\/]}}odr-uniquing.cpp
// CHECK:DW_AT_decl_line [DW_FORM_data1] (79)
// CHECK:DW_AT_type [DW_FORM_ref_addr] (0x{{0*}}[[U_UNION]] "U"

// CHECK:0x[[ANON_NAMESPACE2:[0-9a-f]*]]:{{.*}}DW_TAG_namespace
// CHECK-NEXT:DW_AT_decl_file{{.*}}"{{[\\/]}}tmp{{[\\/]}}odr-uniquing.cpp"

// CHECK:0x[[ANON_CLASS2]]:{{.*}}DW_TAG_class_type
// CHECK:DW_AT_name{{.*}}"AnonC"
// CHECK:DW_AT_byte_size [DW_FORM_data1] (0x01)
// CHECK:DW_AT_decl_file [DW_FORM_data1] ("{{[\\/]}}tmp{{[\\/]}}odr-uniquing.cpp
// CHECK:DW_AT_decl_line [DW_FORM_data1] (65)

// CHECK:.debug_aranges contents

// CHECK:debug_line[0x[[LINE_TABLE_OFF1]]]

// CHECK:debug_line[0x[[LINE_TABLE_OFF2]]]

// CHECK:debug_line[0x[[LINE_TABLE_OFF3]]]

// CHECK:.debug_str contents:
// CHECK:0x00000000: ""
// CHECK:0x00000001: "clang version 3.8.0 (trunk 244290) (llvm/trunk 244270)"
// CHECK:0x00000038: "odr-uniquing.cpp"
// CHECK:0x00000049: "/tmp"
// CHECK:0x0000004e: "_Z3foov"
// CHECK:0x00000056: "foo"
// CHECK:0x0000005a: "s"
// CHECK:0x0000005c: "n"
// CHECK:0x0000005e: "nc"
// CHECK:0x00000061: "ac"
// CHECK:0x00000064: "u"
// CHECK:0x00000066: "AnonC"
// CHECK:0x0000006c: "(anonymous namespace)"
// CHECK:0x00000082: "llvm DWARFLinkerParallel library version "
// CHECK:0x000000ac: "__artificial_type_unit"
// CHECK:0x000000c3: ""
// CHECK:0x000000c4: "AliasForS"
// CHECK:0x000000ce: "C"
// CHECK:0x000000d0: "N"
// CHECK:0x000000d2: "Nested"
// CHECK:0x000000d9: "S"
// CHECK:0x000000db: "U"


// CHECK:.apple_names
// CHECK: Bucket count: 2
// CHECK: String: {{.*}} "foo"
// CHECK: String: {{.*}} "_Z3foov"

// CHECK:.apple_types
// CHECK: Bucket count: 6
// CHECK: String: {{.*}} "AnonC"
// CHECK: String: {{.*}} "Nested"
// CHECK: String: {{.*}} "S"
// CHECK: String: {{.*}} "C"
// CHECK: String: {{.*}} "U"
// CHECK: String: {{.*}} "AliasForS"

// CHECK:.apple_namespaces
// CHECK: Bucket count: 2
// CHECK: String: {{.*}} "(anonymous namespace)"
// CHECK: String: {{.*}} "N"

// CHECK:.apple_objc
// CHECK:Bucket count: 1

// CHECK-NOODR: .debug_info contents

// CHECK-NOODR: DW_TAG_compile_unit
// CHECK-NOODR: DW_AT_name{{.*}}"odr-uniquing.cpp"
// CHECK-NOODR-NEXT: DW_AT_stmt_list{{.*}}(0x[[LINE_TABLE_OFF1:[0-9a-f]*]])
// CHECK-NOODR: DW_AT_low_pc{{.*}}(0x{{0*}}[[LOW_PC1:[0-9a-f]*]])
// CHECK-NOODR-NEXT: DW_AT_high_pc{{.*}}(0x{{0*}}[[HIGH_PC1:[0-9a-f]*]])

// CHECK-NOODR: DW_TAG_structure_type
// CHECK-NOODR-NEXT: DW_AT_name{{.*}}"S"

// CHECK-NOODR: DW_TAG_structure_type
// CHECK-NOODR-NEXT: DW_AT_name{{.*}}"Nested"

// CHECK-NOODR: DW_TAG_namespace
// CHECK-NOODR-NEXT: DW_AT_name{{.*}}"N"

// CHECK-NOODR: DW_TAG_class_type
// CHECK-NOODR-NEXT: DW_AT_name{{.*}}"C"

// CHECK-NOODR: DW_TAG_union_type
// CHECK-NOODR-NEXT: DW_AT_name{{.*}}"U"

// CHECK-NOODR: DW_TAG_member
// CHECK-NOODR-NEXT: DW_AT_name{{.*}}"C"

// CHECK-NOODR: DW_TAG_class_type
// CHECK-NOODR-NEXT: DW_AT_name{{.*}}"C"

// CHECK-NOODR: DW_TAG_member
// CHECK-NOODR-NEXT: DW_AT_name{{.*}}"S"

// CHECK-NOODR: DW_TAG_structure_type
// CHECK-NOODR-NEXT: DW_AT_name{{.*}}"S"

// CHECK-NOODR: DW_TAG_subprogram
// CHECK-NOODR-NEXT: DW_AT_low_pc
// CHECK-NOODR-NEXT: DW_AT_high_pc
// CHECK-NOODR: DW_AT_MIPS_linkage_name{{.*}}"_Z3foov"
// CHECK-NOODR-NEXT: DW_AT_name{{.*}}"foo"

// CHECK-NOODR: DW_TAG_variable
// CHECK-NOODR: DW_AT_name{{.*}}"s"
// CHECK-NOODR: DW_AT_type{{.*}}"AliasForS"

// CHECK-NOODR: DW_TAG_variable
// CHECK-NOODR: DW_AT_name{{.*}}"n"
// CHECK-NOODR: DW_AT_type{{.*}}"S::Nested"

// CHECK-NOODR: DW_TAG_variable
// CHECK-NOODR: DW_AT_name{{.*}}"nc"
// CHECK-NOODR: DW_AT_type{{.*}}"N::C"

// CHECK-NOODR: DW_TAG_variable
// CHECK-NOODR: DW_AT_name{{.*}}"ac"
// CHECK-NOODR: DW_AT_type{{.*}}"(anonymous namespace)::AnonC"

// CHECK-NOODR: DW_TAG_variable
// CHECK-NOODR: DW_AT_name{{.*}}"u"
// CHECK-NOODR: DW_AT_type{{.*}}"U"

// CHECK-NOODR: DW_TAG_typedef
// CHECK-NOODR: DW_AT_type{{.*}}"S"
// CHECK-NOODR: DW_AT_name{{.*}}"AliasForS"

// CHECK-NOODR: DW_TAG_namespace

// CHECK-NOODR: DW_TAG_class_type
// CHECK-NOODR: DW_AT_name{{.*}}"AnonC"

// CHECK-NOODR: DW_TAG_compile_unit
// CHECK-NOODR: DW_AT_name{{.*}}"odr-uniquing.cpp"
// CHECK-NOODR-NEXT: DW_AT_stmt_list{{.*}}(0x[[LINE_TABLE_OFF2:[0-9a-f]*]])
// CHECK-NOODR: DW_AT_low_pc
// CHECK-NOODR: DW_AT_high_pc

// CHECK-NOODR: DW_TAG_structure_type
// CHECK-NOODR: DW_AT_name{{.*}}"S"

// CHECK-NOODR: DW_TAG_structure_type
// CHECK-NOODR: DW_AT_name{{.*}}"Nested"

// CHECK-NOODR: DW_TAG_namespace
// CHECK-NOODR: DW_AT_name{{.*}}"N"

// CHECK-NOODR: DW_TAG_class_type
// CHECK-NOODR: DW_AT_name{{.*}}"C"

// CHECK-NOODR: DW_TAG_union_type
// CHECK-NOODR: DW_AT_name{{.*}}"U"

// CHECK-NOODR: DW_TAG_member
// CHECK-NOODR: DW_AT_name{{.*}}"C"
// CHECK-NOODR: DW_AT_type{{.*}}"U::C"

// CHECK-NOODR: DW_TAG_class_type
// CHECK-NOODR: DW_AT_name{{.*}}"C"

// CHECK-NOODR: DW_TAG_member
// CHECK-NOODR: DW_AT_name{{.*}}"S"
// CHECK-NOODR: DW_AT_type{{.*}}"U::S"

// CHECK-NOODR: DW_TAG_structure_type
// CHECK-NOODR: DW_AT_name{{.*}}"S"

// CHECK-NOODR: DW_TAG_subprogram
// CHECK-NOODR: DW_AT_low_pc
// CHECK-NOODR: DW_AT_high_pc
// CHECK-NOODR: DW_AT_MIPS_linkage_name{{.*}}"_Z3foov"
// CHECK-NOODR: DW_AT_name{{.*}}"foo"

// CHECK-NOODR: DW_TAG_variable
// CHECK-NOODR: DW_AT_name{{.*}}"s"
// CHECK-NOODR: DW_AT_type{{.*}}"AliasForS"

// CHECK-NOODR: DW_TAG_variable
// CHECK-NOODR: DW_AT_name{{.*}}"n"
// CHECK-NOODR: DW_AT_type{{.*}}"S::Nested"

// CHECK-NOODR: DW_TAG_variable
// CHECK-NOODR: DW_AT_name{{.*}}"nc"
// CHECK-NOODR: DW_AT_type{{.*}} "N::C"

// CHECK-NOODR: DW_TAG_variable
// CHECK-NOODR: DW_AT_name{{.*}}"ac"
// CHECK-NOODR: DW_AT_type{{.*}}"(anonymous namespace)::AnonC"

// CHECK-NOODR: DW_TAG_variable
// CHECK-NOODR: DW_AT_name{{.*}}"u"
// CHECK-NOODR: DW_AT_type{{.*}}"U"

// CHECK-NOODR: DW_TAG_typedef
// CHECK-NOODR: DW_AT_type{{.*}}"S"
// CHECK-NOODR: DW_AT_name{{.*}}"AliasForS"

// CHECK-NOODR: DW_TAG_namespace

// CHECK-NOODR: DW_TAG_class_type
// CHECK-NOODR: DW_AT_name{{.*}}"AnonC"

// CHECK-NOODR:.debug_aranges contents

// CHECK-NOODR:debug_line[0x[[LINE_TABLE_OFF1]]]

// CHECK-NOODR:debug_line[0x[[LINE_TABLE_OFF2]]]

// CHECK-NOODR:.debug_str contents:
// CHECK-NOODR:0x00000000: ""
// CHECK-NOODR:0x00000001: "clang version 3.8.0 (trunk 244290) (llvm/trunk 244270)"
// CHECK-NOODR:0x00000038: "odr-uniquing.cpp"
// CHECK-NOODR:0x00000049: "/tmp"
// CHECK-NOODR:0x0000004e: "S"
// CHECK-NOODR:0x00000050: "Nested"
// CHECK-NOODR:0x00000057: "N"
// CHECK-NOODR:0x00000059: "C"
// CHECK-NOODR:0x0000005b: "U"
// CHECK-NOODR:0x0000005d: "_Z3foov"
// CHECK-NOODR:0x00000065: "foo"
// CHECK-NOODR:0x00000069: "s"
// CHECK-NOODR:0x0000006b: "n"
// CHECK-NOODR:0x0000006d: "nc"
// CHECK-NOODR:0x00000070: "ac"
// CHECK-NOODR:0x00000073: "u"
// CHECK-NOODR:0x00000075: "AliasForS"
// CHECK-NOODR:0x0000007f: "AnonC"
// CHECK-NOODR:0x00000085: "(anonymous namespace)"

// CHECK-NOODR: .apple_names
// CHECK-NOODR: Bucket count: 2
// CHECK-NOODR: String: {{.*}} "foo"
// CHECK-NOODR: String: {{.*}} "_Z3foov"

// CHECK-NOODR: .apple_types
// CHECK-NOODR: Bucket count: 6
// CHECK-NOODR: String: {{.*}} "AnonC"
// CHECK-NOODR: String: {{.*}} "Nested"
// CHECK-NOODR: String: {{.*}} "S"
// CHECK-NOODR: String: {{.*}} "C"
// CHECK-NOODR: String: {{.*}} "U"
// CHECK-NOODR: String: {{.*}} "AliasForS"

// CHECK-NOODR: .apple_namespaces
// CHECK-NOODR: Bucket count: 2
// CHECK-NOODR: String: {{.*}} "(anonymous namespace)"
// CHECK-NOODR: String: {{.*}} "N"

// CHECK-NOODR: .apple_objc
// CHECK-NOODR:Bucket count: 1
