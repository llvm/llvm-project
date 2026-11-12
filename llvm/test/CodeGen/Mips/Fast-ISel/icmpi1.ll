; RUN: llc -mtriple=mipsel -relocation-model=pic -O0 -fast-isel=true -mcpu=mips32r2 \
; RUN:     < %s -verify-machineinstrs | FileCheck %s


define zeroext i1 @foo(ptr nocapture readonly) {
; CHECK-LABEL: foo
; CHECK:         lbu $[[REG0:[0-9]+]], 0($4)
; CHECK-NEXT:    xori $[[REG1:[0-9]+]], $[[REG0]], 1
; CHECK-NEXT:    andi $2, $[[REG1]], 1
  %2 = load i8, ptr %0, align 1
  %3 = trunc i8 %2 to i1
  %4 = icmp ne i1 %3, true
  ret i1 %4
}
