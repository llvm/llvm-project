# RUN: llvm-mc -triple i386-unknown-unknown -filetype=obj %s -o %t
# RUN: llvm-dwarfdump -debug-line %t | FileCheck %s

        .file 1 "test.c"
        .loc 1 2
        nop
        .loc 1 4 is_stmt 0
        nop
        .loc 1 6
        nop
        .loc 1 8 is_stmt 1
        nop
        .loc 1 10
        nop

# CHECK: .debug_line
# CHECK: file_names[ 1]:
# CHECK-NEXT: name: "test.c"
# CHECK-NEXT: dir_index: 0
# CHECK: 0x{{0+}}0 2 0 1 0 0 is_stmt
# CHECK: 0x{{0+}}1 4 0 1 0 0 {{$}}
# CHECK: 0x{{0+}}2 6 0 1 0 0 {{$}}
# CHECK: 0x{{0+}}3 8 0 1 0 0 is_stmt
# CHECK: 0x{{0+}}4 10 0 1 0 0 is_stmt
# CHECK: 0x{{0+}}5 10 0 1 0 0 is_stmt end_sequence
