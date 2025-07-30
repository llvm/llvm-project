// RUN: llvm-mc -triple x86_64-unknown-linux-gnu -filetype obj -g -dwarf-version 5 -o %t %s
// RUN: llvm-dwarfdump -debug-info -debug-line %t | FileCheck %s

// CHECK-NOT: DW_TAG_

// CHECK:      include_directories[ 0] =
// CHECK-NOT:  include_directories[ 1] =
// CHECK:      file_names[ 0]:
// CHECK-NEXT:           name: "/MyTest/Inputs/other.S"
// CHECK-NEXT:      dir_index: 0
// CHECK-NOT:  file_names[ 1]:

// RUN: llvm-mc -triple=x86_64 -filetype=obj -g -dwarf-version=5 -fdebug-prefix-map=/MyTest=/src_root %s -o %t.5.o
// RUN: llvm-dwarfdump -debug-info -debug-line %t.5.o | FileCheck %s --check-prefixes=MAP

// MAP-NOT: DW_TAG_

// MAP:      include_directories[  0] = "{{.*}}"
// MAP-NEXT: file_names[  0]:
// MAP-NEXT:            name: "/src_root/Inputs/other.S"
// MAP-NEXT:       dir_index: 0

# 1 "/MyTest/Inputs/other.S"

.section .data
.asciz "data"
