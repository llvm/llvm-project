; RUN: llc -mtriple=aarch64-none-linux -O0 -filetype=obj - < %S/../Inputs/line.ll | llvm-dwarfdump -v - | FileCheck %s
; RUN: llc -mtriple=aarch64_be-none-linux -O0 -filetype=obj - < %S/../Inputs/line.ll | llvm-dwarfdump -v - | FileCheck %s

; check line table length is correctly calculated for both big and little endian
CHECK-LABEL: .debug_line contents:
CHECK: total_length: 0x0000003e
