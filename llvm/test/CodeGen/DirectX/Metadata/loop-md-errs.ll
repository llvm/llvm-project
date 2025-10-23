; RUN: split-file %s %t
; RUN: not opt -S --dxil-validate-metadata %t/args.ll 2>&1 | FileCheck %t/args.ll
; RUN: not opt -S --dxil-validate-metadata %t/not-ref.ll 2>&1 | FileCheck %t/not-ref.ll
; RUN: not opt -S --dxil-validate-metadata %t/not-md.ll 2>&1 | FileCheck %t/not-md.ll
; RUN: not opt -S --dxil-validate-metadata %t/not-str.ll 2>&1 | FileCheck %t/not-str.ll
; RUN: not opt -S --dxil-validate-metadata %t/bad-count.ll 2>&1 | FileCheck %t/bad-count.ll
; RUN: not opt -S --dxil-validate-metadata %t/invalid-disable.ll 2>&1 | FileCheck %t/invalid-disable.ll
; RUN: not opt -S --dxil-validate-metadata %t/invalid-full.ll 2>&1 | FileCheck %t/invalid-full.ll

; Test that loop metadata is validated as with the DXIL validator

;--- args.ll

; CHECK: Invalid "llvm.loop" metadata: Requires exactly 1 or 2 operands

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

!1 = !{!1, !1, !1} ; too many args

;--- not-ref.ll

; CHECK: Invalid "llvm.loop" metadata: First operand must be a self-reference

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

!1 = !{i32 0} ; not a self-reference

;--- not-md.ll

; CHECK: Invalid "llvm.loop" metadata: Second operand must be a metadata node

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

!1 = !{!1, i32 0} ; not a metadata node

;--- not-str.ll

; CHECK: Invalid "llvm.loop" metadata: First operand must be a valid "llvm.loop.unroll" hint

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
!2 = !{i32 0} ; not a hint name string

;--- bad-count.ll

; CHECK: Second operand of "llvm.loop.unroll.count" must be a constant integer

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

; CHECK: Invalid "llvm.loop" metadata: Can't have a second operand

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

; CHECK: Invalid "llvm.loop" metadata: Can't have a second operand

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
