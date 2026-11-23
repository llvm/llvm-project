; RUN: split-file %s %t
; RUN: opt -S --dxil-translate-metadata %t/not-distinct.ll 2>&1 | FileCheck %t/not-distinct.ll
; RUN: opt -S --dxil-translate-metadata %t/not-md.ll 2>&1 | FileCheck %t/not-md.ll

; Test that DXIL incompatible loop metadata is stripped

;--- not-distinct.ll

; Ensure it is stripped because it is not provided a distinct loop parent
; CHECK-NOT: {!"llvm.loop.unroll.disable"}

target triple = "dxilv1.0-unknown-shadermodel6.0-library"

define void @example_loop(i32 %n) {
entry:
  br label %loop.header

loop.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop.body ]
  %cmp = icmp slt i32 %i, %n
  br i1 %cmp, label %loop.body, label %exit

loop.body:
  %i.next = add nsw i32 %i, 1
  br label %loop.header, !llvm.loop !1

exit:
  ret void
}

!1 = !{!"llvm.loop.unroll.disable"} ; first node must be a distinct self-reference


;--- not-md.ll

target triple = "dxilv1.0-unknown-shadermodel6.0-library"

define void @example_loop(i32 %n) {
entry:
  br label %loop.header

loop.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop.body ]
  %cmp = icmp slt i32 %i, %n
  br i1 %cmp, label %loop.body, label %exit

loop.body:
  %i.next = add nsw i32 %i, 1
  ; CHECK: br label %loop.header, !llvm.loop ![[#LOOP_MD:]]
  br label %loop.header, !llvm.loop !1

exit:
  ret void
}

; CHECK: ![[#LOOP_MD:]] = distinct !{![[#LOOP_MD]]}

!1 = !{!1, i32 0} ; second operand is not a metadata node
