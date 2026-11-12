// XFAIL: target={{.*}}-aix{{.*}}
// UNSUPPORTED: target={{.*}}-zos{{.*}}
// RUN: llvm-mc --dwarf-version=5 %s --filetype=obj -o %t
// RUN: llvm-dwarfdump --debug-line %t | FileCheck %s

// CHECK:      include_directories[  0] = "/test"
// CHECK-NEXT: include_directories[  1] = "/include"
// CHECK:      file_names[  0]:
// CHECK-NEXT:            name: "main.c"
// CHECK-NEXT:       dir_index: 0
// CHECK-NEXT: file_names[  1]:
// CHECK-NEXT:            name: "foo.h"
// CHECK-NEXT:       dir_index: 1
// CHECK-NEXT: file_names[  2]:
// CHECK-NEXT:            name: "main.c"
// CHECK-NEXT:       dir_index: 0

// CHECK:      Address            Line   Column File
// CHECK-NEXT: ------------------ ------ ------ ------
// CHECK-NEXT: 0x{{[0-9a-f]*}}        10     20      0
// CHECK-NEXT: 0x{{[0-9a-f]*}}        30     40      1
// CHECK-NEXT: 0x{{[0-9a-f]*}}        50     60      2
// CHECK-NEXT: 0x{{[0-9a-f]*}}        50     60      2

.file 0 "/test" "main.c"
.loc 0 10 20
.byte 0
.file 1 "/include" "foo.h"
.loc 1 30 40
.byte 0
.file 2 "main.c"
.loc 2 50 60
.byte 0
