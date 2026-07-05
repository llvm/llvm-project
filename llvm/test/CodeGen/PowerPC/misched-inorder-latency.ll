; RUN: llc -verify-machineinstrs < %s -enable-misched -pre-RA-sched=source -scheditins=false \
; RUN:          -disable-ifcvt-triangle-false -disable-post-ra -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s
;
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"

; %val1 is a load live out of %entry. It should be hoisted
; above the add.
; CHECK-LABEL: testload:
; CHECK: %entry
; CHECK: lwz
; CHECK: addi
; CHECK: bne
; CHECK: %true
define i32 @testload(ptr %ptr, i32 %sumin) {
entry:
  %sum1 = add i32 %sumin, 1
  %val1 = load i32, ptr %ptr
  %p = icmp eq i32 %sumin, 0
  br i1 %p, label %true, label %end, !prof !1
true:
  %sum2 = add i32 %sum1, 1
  %ptr2 = getelementptr i32, ptr %ptr, i32 1
  %val = load i32, ptr %ptr2
  %val2 = add i32 %val1, %val
  br label %end
end:
  %valmerge = phi i32 [ %val1, %entry], [ %val2, %true ]
  %summerge = phi i32 [ %sum1, %entry], [ %sum2, %true ]
  %sumout = add i32 %valmerge, %summerge
  ret i32 %sumout
}

; The prefetch gets a default latency of 3 cycles and should be hoisted
; above the add.
;
; CHECK-LABEL: testprefetch:
; CHECK: %entry
; CHECK: dcbt
; CHECK: addi
; CHECK: blr
define i32 @testprefetch(ptr %ptr, i32 %i) {
entry:
  %val1 = add i32 %i, 1
  tail call void @llvm.prefetch( ptr %ptr, i32 0, i32 3, i32 1 )
  %p = icmp eq i32 %i, 0
  br i1 %p, label %true, label %end
true:
  %val2 = add i32 %val1, 1
  br label %end
end:
  %valmerge = phi i32 [ %val1, %entry], [ %val2, %true ]
  ret i32 %valmerge
}
declare void @llvm.prefetch(ptr, i32, i32, i32) nounwind

!1 = !{!"branch_weights", i32 2, i32 1}
