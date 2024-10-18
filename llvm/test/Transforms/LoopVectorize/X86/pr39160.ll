; RUN: opt -passes=loop-vectorize -S < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128-ni:1"
target triple = "x86_64-unknown-linux-gnu"

; Make sure that we can compile the test without crash.
define void @barney(ptr %dst) {

; CHECK-LABEL: @barney(
; CHECK:       middle.block:

bb:
  br label %bb2

bb2:                                              ; preds = %bb2, %bb
  %tmp4 = icmp slt i32 undef, 0
  br i1 %tmp4, label %bb2, label %bb5

bb5:                                              ; preds = %bb2
  br label %bb19

bb18:                                             ; preds = %bb33
  ret void

bb19:                                             ; preds = %bb36, %bb5
  %tmp21 = phi i64 [ undef, %bb36 ], [ 2, %bb5 ]
  %tmp22 = phi i32 [ %tmp65, %bb36 ], [ undef, %bb5 ]
  br label %bb50

bb33:                                             ; preds = %bb62
  br i1 undef, label %bb18, label %bb36

bb36:                                             ; preds = %bb33
  br label %bb19

bb46:                                             ; preds = %bb50
  br i1 undef, label %bb48, label %bb59

bb48:                                             ; preds = %bb46
  %tmp49 = add i32 %tmp52, 14
  ret void

bb50:                                             ; preds = %bb50, %bb19
  %tmp52 = phi i32 [ %tmp55, %bb50 ], [ %tmp22, %bb19 ]
  %tmp53 = phi i64 [ %tmp56, %bb50 ], [ 1, %bb19 ]
  %gep = getelementptr inbounds i8, ptr %dst, i64 %tmp53
  store i8 1, ptr %gep
  %tmp54 = add i32 %tmp52, 12
  %tmp55 = add i32 %tmp52, 13
  %tmp56 = add nuw nsw i64 %tmp53, 1
  %tmp58 = icmp ult i64 %tmp53, undef
  br i1 %tmp58, label %bb50, label %bb46

bb59:                                             ; preds = %bb46
  br label %bb62

bb62:                                             ; preds = %bb68, %bb59
  %tmp63 = phi i32 [ %tmp65, %bb68 ], [ %tmp55, %bb59 ]
  %tmp64 = phi i64 [ %tmp66, %bb68 ], [ %tmp56, %bb59 ]
  %tmp65 = add i32 %tmp63, 13
  %tmp66 = add nuw nsw i64 %tmp64, 1
  %tmp67 = icmp ult i64 %tmp66, %tmp21
  br i1 %tmp67, label %bb68, label %bb33

bb68:                                             ; preds = %bb62
  br label %bb62
}

define i32 @foo(ptr addrspace(1) %p) {

; CHECK-LABEL: foo
; CHECK:       middle.block:

entry:
  br label %outer

outer:                                            ; preds = %outer_latch, %entry
  %iv = phi i64 [ 2, %entry ], [ %iv.next, %outer_latch ]
  br label %inner

inner:                                            ; preds = %inner, %outer
  %0 = phi i32 [ %2, %inner ], [ 0, %outer ]
  %a = phi i32 [ %3, %inner ], [ 1, %outer ]
  %b = phi i32 [ %1, %inner ], [ 6, %outer ]
  %1 = add i32 %b, 2
  %2 = or i32 %0, %b
  %3 = add nuw nsw i32 %a, 1
  %4 = zext i32 %3 to i64
  %5 = icmp ugt i64 %iv, %4
  br i1 %5, label %inner, label %outer_latch

outer_latch:                                      ; preds = %inner
  store atomic i32 %2, ptr addrspace(1) %p unordered, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %6 = icmp ugt i64 %iv, 63
  br i1 %6, label %exit, label %outer

exit:                                             ; preds = %outer_latch
  ret i32 0
}
