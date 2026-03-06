; RUN: split-file %s %t
; RUN: opt -S --dxil-translate-metadata %t/count.ll | FileCheck %t/count.ll
; RUN: opt -S --dxil-translate-metadata %t/disable.ll | FileCheck %t/disable.ll
; RUN: opt -S --dxil-translate-metadata %t/full.ll | FileCheck %t/full.ll

;--- count.ll

; Test that we collapse a self-referential chain and allow a unroll.count hint

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
  br label %loop.header, !llvm.loop !0

exit:
  ret void
}

; CHECK: ![[#LOOP_MD]] = distinct !{![[#LOOP_MD]], ![[#COUNT:]]}
; CHECK: ![[#COUNT]] = !{!"llvm.loop.unroll.count", i6 4}

!0 = !{!0, !1}
!1 = !{!1, !2}
!2 = !{!"llvm.loop.unroll.count", i6 4}

;--- disable.ll

; Test that we allow a disable hint

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
  br label %loop.header, !llvm.loop !0

exit:
  ret void
}

; CHECK: ![[#LOOP_MD]] = distinct !{![[#LOOP_MD]], ![[#DISABLE:]]}
; CHECK: ![[#DISABLE]] = !{!"llvm.loop.unroll.disable"}

!0 = !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}

;--- full.ll

; Test that we allow a full hint

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
  br label %loop.header, !llvm.loop !0

exit:
  ret void
}

; CHECK: ![[#LOOP_MD]] = distinct !{![[#LOOP_MD]], ![[#FULL:]]}
; CHECK: ![[#FULL]] = !{!"llvm.loop.unroll.full"}

!0 = !{!0, !1}
!1 = !{!"llvm.loop.unroll.full"}
