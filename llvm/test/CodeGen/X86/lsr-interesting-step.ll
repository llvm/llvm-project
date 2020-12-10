; RUN: llc < %s -relocation-model=static -mtriple=x86_64-unknown-linux-gnu -asm-verbose=0 | FileCheck %s

; The inner loop should require only one add (and no leas either).
; rdar://8100380

; CHECK:      BB0_2:
; CHECK-NEXT:   movb    $0, flags(%rcx)
; CHECK-NEXT:   addq    %rax, %rcx
; CHECK-NEXT:   cmpq    $8192, %rcx
; CHECK-NEXT:   jl

@flags = external dso_local global [8192 x i8], align 16 ; <[8192 x i8]*> [#uses=1]

define void @foo() nounwind {
entry:
  br label %bb

bb:                                               ; preds = %entry
  br label %bb7

bb7:                                              ; preds = %bb, %bb17
  %tmp8 = phi i64 [ %tmp18, %bb17 ], [ 2, %bb ]   ; <i64> [#uses=2]
  br label %bb10

bb10:                                             ; preds = %bb7
  br label %bb11

bb11:                                             ; preds = %bb10, %bb11
  %tmp12 = phi i64 [ %tmp14, %bb11 ], [ 2, %bb10 ] ; <i64> [#uses=2]
  %tmp13 = getelementptr inbounds [8192 x i8], [8192 x i8]* @flags, i64 0, i64 %tmp12 ; <i8*> [#uses=1]
  store i8 0, i8* %tmp13, align 1
  %tmp14 = add nsw i64 %tmp12, %tmp8              ; <i64> [#uses=2]
  %tmp15 = icmp slt i64 %tmp14, 8192              ; <i1> [#uses=1]
  br i1 %tmp15, label %bb11, label %bb16

bb16:                                             ; preds = %bb11
  br label %bb17

bb17:                                             ; preds = %bb16, %bb7
  %tmp18 = add nsw i64 %tmp8, 1                   ; <i64> [#uses=2]
  %tmp19 = icmp slt i64 %tmp18, 8192              ; <i1> [#uses=1]
  br i1 %tmp19, label %bb7, label %bb20

bb20:                                             ; preds = %bb17
  br label %bb21

bb21:                                             ; preds = %bb20, %entry
  ret void
}
