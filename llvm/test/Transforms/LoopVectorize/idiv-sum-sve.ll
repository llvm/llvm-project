; NOTE: This test checks that the loop vectorizer produces different IR shapes
;       under fixed-width and scalable vectorization, but both should remain
;       valid and semantically equivalent.

; RUN: opt -S -mtriple=aarch64-linux-gnu -scalable-vectorization=off \
; RUN:   -passes=loop-vectorize < %s | FileCheck %s --check-prefix=FIXED
; RUN: opt -S -mtriple=aarch64-linux-gnu -mattr=+sve -scalable-vectorization=on \
; RUN:   -passes=loop-vectorize < %s | FileCheck %s --check-prefix=SVE

define i32 @idiv_sum(ptr nocapture readonly %a,
                     ptr nocapture readonly %b, i32 %n) {
entry:
  %s = alloca i32, align 4
  store i32 0, ptr %s, align 4
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop, label %exit

loop:                                             ; preds = %entry, %loop
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %sum, %loop ]
  %a.val = load i32, ptr %a
  %b.val = load i32, ptr %b
  %div = sdiv i32 %a.val, %b.val
  %sum = add i32 %acc, %div
  %i.next = add i32 %i, 1
  %cmp.loop = icmp slt i32 %i.next, %n
  br i1 %cmp.loop, label %loop, label %exit

exit:                                             ; preds = %loop, %entry
  %res = phi i32 [ 0, %entry ], [ %sum, %loop ]
  ret i32 %res
}

; FIXED-LABEL: vector.body:
; SVE-LABEL: vector.body:
; SVE: <vscale x 4 x i32>
