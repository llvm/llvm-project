; RUN: llc -o - %s | FileCheck %s
; The selection DAG select(select()) normalisation crashed for different types
; on the condition inputs.
target datalayout = "E-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64"
target triple = "mips--"

; CHECK-LABEL: foobar
; CHECK: sltiu ${{[0-9]*}}, ${{[0-9]*}}, 42
; CHECK: sltiu ${{[0-9]*}}, ${{[0-9]*}}, 23
; CHECK: and ${{[0-9]*}}, ${{[0-9]*}}, ${{[0-9]*}}
; CHECK: sltu ${{[0-9]*}}, ${{[0-9]*}}, ${{[0-9]*}}
; CHECK: addiu ${{[0-9]*}}, ${{[0-9]*}}, -1
; CHECK: movn ${{[0-9]*}}, ${{[0-9]*}}, ${{[0-9]*}}
; CHECK: jr $ra
; CHECK: move ${{[0-9]*}}, ${{[0-9]*}}
define i64 @foobar(i32 %arg) #0 {
entry:
  %cmp0 = icmp ult i32 %arg, 23
  %cmp1 = icmp ult i32 %arg, 42
  %and = and i1 %cmp0, %cmp1
  %cmp2 = icmp ugt i32 %arg, 0
  %sext = sext i1 %cmp1 to i64
  %retval.0 = select i1 %and, i64 %sext, i64 0
  ret i64 %retval.0
}
