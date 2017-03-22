; RUN: opt %loadPolly -polly-codegen -polly-invariant-load-hoisting=true \
; RUN: -S < %s | FileCheck %s
;
; This test case checks whether Polly generates alias metadata in case of
; the ublas gemm kernel and polly-invariant-load-hoisting.
;
; CHECK: store float 4.200000e+01, float* %polly.access.A.load, !alias.scope !3, !noalias !4
;
; CHECK: !0 = distinct !{!0, !1, !"polly.alias.scope.MemRef_A"}
; CHECK-NEXT: !1 = distinct !{!1, !"polly.alias.scope.domain"}
; CHECK-NEXT: !2 = !{!3}
; CHECK-NEXT: !3 = distinct !{!3, !1, !"polly.alias.scope.MemRef_ptrA"}
; CHECK-NEXT: !4 = !{!0}
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @nometadata(float** %A) {
  entry:
    br label %for

  for:
    %indvar = phi i64 [0, %entry], [%indvar.next, %for]
    %indvar.next = add i64 %indvar, 1
    %ptrA = load float*, float** %A
    store float 42.0, float* %ptrA
    %icmp = icmp sle i64 %indvar, 1024
    br i1 %icmp, label %for, label %exit

  exit:
    ret void
}
