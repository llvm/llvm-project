; RUN: llc -march=mips < %s | FileCheck %s

define i64 @add64(i64 %u, i64 %v) nounwind  {
entry:
; CHECK-LABEL: add64:
; CHECK: addu
; CHECK-DAG: sltu
; CHECK-DAG: addu
; CHECK: addu
  %tmp2 = add i64 %u, %v
  ret i64 %tmp2
}

define i64 @sub64(i64 %u, i64 %v) nounwind  {
entry:
; CHECK-LABEL: sub64
; CHECK-DAG: sltu
; CHECK-DAG: subu
; CHECK: subu
; CHECK: subu
  %tmp2 = sub i64 %u, %v
  ret i64 %tmp2
}
