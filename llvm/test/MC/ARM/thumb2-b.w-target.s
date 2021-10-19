// RUN: llvm-mc -triple=thumbv7 -filetype=obj %s | llvm-objdump --triple=thumbv7 -d - | FileCheck %s

.syntax unified

// CHECK-LABEL: start
// CHECK-NEXT:	b.w	{{.+}} @ imm = #16777208
// CHECK-NEXT:  b.w	{{.+}} @ imm = #2
start:
  b.w start - 1f + 0x1000000
1:
  b.w . + (2f - 1b + 2)
2:
