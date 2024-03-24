; RUN: llc < %s -mtriple=x86_64-pc-linux-gnu | FileCheck %s -check-prefixes=CHECK

declare void @bar();

; CHECK-LABEL: foo:
; CHECK-LABEL: # %bb.0:
; CHECK: .p2align        5, 0x90
; CHECK-LABEL: # %bb.1:
; CHECK: .p2align        10, 0x90
; CHECK-LABEL: .LBB0_2:
; CHECK: .p2align        5, 0x90
; CHECK-LABEL: # %bb.3:
; CHECK: .p2align        5, 0x90

define i32 @foo(i1 %1) "align-basic-blocks"="32" {
  br i1 %1, label %7, label %3
3:
  %4 = phi i32 [ %5, %3 ], [ 0, %2 ]
  call void @bar()
  %5 = add nuw nsw i32 %4, 1
  %6 = icmp eq i32 %5, 90
  br i1 %6, label %7, label %3, !llvm.loop !0
7:
  %8 = phi i32 [ 2, %2 ], [ 3, %3 ]
  ret i32 %8
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.align", i32 1024}
