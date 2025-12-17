; RUN: opt -passes=loop-idiom -S < %s | FileCheck %s
; RUN: opt -passes='default<O2>,loop-idiom' -S < %s | FileCheck %s
; RUN: opt -passes='default<O3>,loop-idiom' -S < %s | FileCheck %s

; Verify that a canonical loop which accumulates bits using shift+mask is
; converted into a single llvm.ctpop intrinsic call in all optimization modes.

target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @popcount_naive
; CHECK-NOT: lshr
; CHECK: {{.*}}call{{.*}}i64 @llvm.ctpop.i64
; CHECK: ret i32
define i32 @popcount_naive(i64 %x) {
entry:
  br label %for.body

for.body:                                          ; preds = %for.body, %entry
  %i = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %acc = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %shr = lshr i64 %x, %i
  %and = and i64 %shr, 1
  %conv = trunc i64 %and to i32
  %add = add i32 %acc, %conv
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %inc, 64
  br i1 %cmp, label %exit, label %for.body

exit:                                              ; preds = %for.body
  ret i32 %add
}
