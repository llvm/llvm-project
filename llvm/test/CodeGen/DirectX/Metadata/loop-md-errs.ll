; RUN: split-file %s %t
; RUN: not opt -S --dxil-translate-metadata %t/args.ll 2>&1 | FileCheck %t/args.ll
; RUN: not opt -S --dxil-translate-metadata %t/bad-count.ll 2>&1 | FileCheck %t/bad-count.ll
; RUN: not opt -S --dxil-translate-metadata %t/invalid-disable.ll 2>&1 | FileCheck %t/invalid-disable.ll
; RUN: not opt -S --dxil-translate-metadata %t/invalid-full.ll 2>&1 | FileCheck %t/invalid-full.ll

; Test that loop metadata is validated as with the DXIL validator

;--- args.ll

; CHECK: Invalid "llvm.loop" metadata: Provided conflicting hints

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

!1 = !{!1, !2, !3} ; conflicting args
!2 = !{!"llvm.loop.unroll.full"}
!3 = !{!"llvm.loop.unroll.disable"}

;--- bad-count.ll

; CHECK: "llvm.loop.unroll.count" must have 2 operands and the second must be a constant integer

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

!1 = !{!1, !2}
!2 = !{!"llvm.loop.unroll.count", !"not an int"} ; invalid count parameters

;--- invalid-disable.ll

; CHECK: Invalid "llvm.loop" metadata: "llvm.loop.unroll.disable" and "llvm.loop.unroll.full" must be provided as a single operand

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

!1 = !{!1, !2}
!2 = !{!"llvm.loop.unroll.disable", i32 0} ; invalid second operand


;--- invalid-full.ll

; CHECK: Invalid "llvm.loop" metadata: "llvm.loop.unroll.disable" and "llvm.loop.unroll.full" must be provided as a single operand

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

!1 = !{!1, !2}
!2 = !{!"llvm.loop.unroll.full", i32 0} ; invalid second operand
