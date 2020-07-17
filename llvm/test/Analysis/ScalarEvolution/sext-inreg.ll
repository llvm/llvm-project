; RUN: opt < %s -analyze -enable-new-pm=0 -scalar-evolution | FileCheck %s
; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" 2>&1 | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.6"

define i64 @foo(i64* nocapture %x, i64 %n) nounwind {
; CHECK-LABEL: Classifying expressions for: @foo
entry:
	%t0 = icmp sgt i64 %n, 0
	br i1 %t0, label %bb, label %return

bb:
	%i.01 = phi i64 [ 0, %entry ], [ %indvar.next, %bb ]
	%t1 = shl i64 %i.01, 7
	%t2 = ashr i64 %t1, 7
; CHECK: %t2 = ashr i64 %t1, 7
; CHECK-NEXT: sext i57 {0,+,199}<%bb> to i64
; CHECK-SAME: Exits: (sext i57 (-199 + (trunc i64 %n to i57)) to i64)
; CHECK: %s2 = ashr i64 %s1, 5
; CHECK-NEXT: sext i59 {0,+,199}<%bb> to i64
; CHECK-SAME: Exits: (sext i59 (-199 + (trunc i64 %n to i59)) to i64)
	%s1 = shl i64 %i.01, 5
	%s2 = ashr i64 %s1, 5
	%t3 = getelementptr i64, i64* %x, i64 %i.01
	store i64 0, i64* %t3, align 1
	%indvar.next = add i64 %i.01, 199
	%exitcond = icmp eq i64 %indvar.next, %n
	br i1 %exitcond, label %return, label %bb

return:
        %p = phi i64 [ 0, %entry ], [ %t2, %bb ]
        %q = phi i64 [ 0, %entry ], [ %s2, %bb ]
        %v = xor i64 %p, %q
	ret i64 %v
}
