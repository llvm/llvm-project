; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s

; CHECK:      polly.stmt.if.then81:                             ; preds = %polly.stmt.if.end75
; CHECK-NEXT:   %scevgep = getelementptr [2 x %S], [2 x %S]* %tmp, i64 0, i64 %.147
; CHECK-NEXT:   %scevgep1 = bitcast %S* %scevgep to float*
; CHECK-NEXT:   store float undef, float* %scevgep1, align 4, !alias.scope !0, !noalias !3
; CHECK-NEXT:   br label %polly.stmt.if.end87.region_exiting

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android"

%S = type { float, float }

define void @f() {
entry:
  %tmp = alloca [2 x %S], align 4
  %cmp52 = fcmp olt float undef, undef
  %not.cmp52 = xor i1 %cmp52, true
  %.147 = zext i1 %not.cmp52 to i64
  %fX64 = getelementptr inbounds [2 x %S], [2 x %S]* %tmp, i64 0, i64 %.147, i32 0
  br label %if.end75

if.end75:
  %cmp80 = fcmp olt float undef, undef
  br i1 %cmp80, label %if.then81, label %if.end87

if.then81:
  store float undef, float* %fX64, align 4
  br label %if.end87

if.end87:
  %0 = phi float [ undef, %if.then81 ], [ undef, %if.end75 ]
  ret void
}
