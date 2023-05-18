; RUN: opt %loadPolly -polly-codegen -polly-invariant-load-hoisting=true \
; RUN: -S < %s | FileCheck %s
;
; This test case checks whether Polly generates alias metadata in case of
; the ublas gemm kernel and polly-invariant-load-hoisting.
;
; CHECK: store float 4.200000e+01, ptr %polly.access.A.load, align 4, !alias.scope !3, !noalias !0
;
; CHECK: !0 = !{!1}
; CHECK-NEXT: !1 = distinct !{!1, !2, !"polly.alias.scope.MemRef_A"}
; CHECK-NEXT: !2 = distinct !{!2, !"polly.alias.scope.domain"}
; CHECK-NEXT: !3 = !{!4}
; CHECK-NEXT: !4 = distinct !{!4, !2, !"polly.alias.scope.MemRef_ptrA"}
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @nometadata(ptr %A) {
  entry:
    br label %for

  for:
    %indvar = phi i64 [0, %entry], [%indvar.next, %for]
    %indvar.next = add i64 %indvar, 1
    %ptrA = load ptr, ptr %A
    store float 42.0, ptr %ptrA
    %icmp = icmp sle i64 %indvar, 1024
    br i1 %icmp, label %for, label %exit

  exit:
    ret void
}
