; RUN: llc -verify-machineinstrs < %s -mtriple=x86_64-unknown-unknown | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=x86_64-unknown-unknown -O0 | FileCheck %s

define swifttailcc [4 x i64] @return_int() {
; CHECK-LABEL: return_int:
; CHECK-DAG: movl $1, %eax
; CHECK-DAG: movl $2, %edx
; CHECK-DAG: movl $3, %ecx
; CHECK-DAG: movl $4, %r8d

  ret [4 x i64] [i64 1, i64 2, i64 3, i64 4]
}


; CHECK: [[ONE:.LCPI.*]]:
; CHECK-NEXT: # double 1
; CHECK: [[TWO:.LCPI.*]]:
; CHECK-NEXT: # double 2
; CHECK: [[THREE:.LCPI.*]]:
; CHECK-NEXT: # double 3

define swifttailcc [4 x double] @return_float() {
; CHECK-LABEL: return_float:
; CHECK-DAG: movsd [[ONE]](%rip), %xmm1
; CHECK-DAG: movsd [[TWO]](%rip), %xmm2
; CHECK-DAG: movsd [[THREE]](%rip), %xmm3
; CHECK-DAG: xorps %xmm0, %xmm0
  ret [4 x double] [double 0.0, double 1.0, double 2.0, double 3.0]
}
