// RUN: llvm-mc -triple x86_64-unknown-linux-gnu -filetype obj -g -dwarf-version 4 -o %t %s
// RUN: llvm-dwarfdump -debug-info -debug-line %t | FileCheck %s --check-prefixes=CHECK,DWARF4

// RUN: llvm-mc -triple x86_64-unknown-linux-gnu -filetype obj -g -dwarf-version 5 -o %t %s
// RUN: llvm-dwarfdump -debug-info -debug-line %t | FileCheck %s --check-prefixes=CHECK,DWARF5

// CHECK: DW_TAG_compile_unit
// CHECK-NOT: DW_TAG_
// CHECK: DW_AT_name      ("/MyTest/Inputs{{(/|\\)+}}other.S")
// CHECK: DW_TAG_label
// CHECK-NOT: DW_TAG_
// CHECK: DW_AT_decl_file ("/MyTest/Inputs{{(/|\\)+}}other.S")

// DWARF4: include_directories[ 1] = "/MyTest/Inputs"
// DWARF4: file_names[ 1]:
// DWARF4-NEXT: name: "other.S"
// DWARF4-NEXT: dir_index: 1

// DWARF5:      include_directories[ 0] =
// DWARF5-NOT:  include_directories[ 1] =
// DWARF5:      file_names[ 0]:
// DWARF5-NEXT:           name: "/MyTest/Inputs/other.S"
// DWARF5-NEXT:      dir_index: 0
// DWARF5-NOT:  file_names[ 1]:

# 1 "/MyTest/Inputs/other.S"

foo:
  nop
  nop
  nop
