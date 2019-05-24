// RUN: llvm-mc -triple x86_64-unknown-linux-gnu -filetype obj -g -dwarf-version 4 -o %t %s
// RUN: llvm-dwarfdump -debug-info -debug-line %t | FileCheck %s

// CHECK: DW_TAG_compile_unit
// CHECK-NOT: DW_TAG_
// CHECK: DW_AT_name      ("/MyTest/Inputs{{(/|\\)+}}other.S")
// CHECK: DW_TAG_label
// CHECK-NOT: DW_TAG_
// CHECK: DW_AT_decl_file ("/MyTest/Inputs{{(/|\\)+}}other.S")

// CHECK: include_directories[ 1] = "/MyTest/Inputs"
// CHECK: file_names[ 1]:
// CHECK-NEXT: name: "other.S"
// CHECK-NEXT: dir_index: 1

# 1 "/MyTest/Inputs/other.S"

foo:
  nop
  nop
  nop
