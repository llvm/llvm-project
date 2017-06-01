; RUN: llc -mtriple=sparc-none-linux-gnu < %s | FileCheck %s

; This test checks that LLVM can do basic stripping and reapplying of branches
; to basic blocks.

declare void @test_true()
declare void @test_false()

; !0 corresponds to a branch being taken, !1 to not being takne.
!0 = !{!"branch_weights", i32 64, i32 4}
!1 = !{!"branch_weights", i32 4, i32 64}

define void @test_Bcc_fallthrough_taken(i32 %in) nounwind {
; CHECK-LABEL: test_Bcc_fallthrough_taken:
  %tst = icmp eq i32 %in, 42
  br i1 %tst, label %true, label %false, !prof !0

; CHECK: cmp {{%[goli][0-9]+}}, 42
; CHECK: bne [[FALSE:.LBB[0-9]+_[0-9]+]]
; CHECK-NEXT: nop
; CHECK-NEXT: ! BB#
; CHECK-NEXT: call test_true

; CHECK: [[FALSE]]:
; CHECK: call test_false

true:
  call void @test_true()
  ret void

false:
  call void @test_false()
  ret void
}

define void @test_Bcc_fallthrough_nottaken(i32 %in) nounwind {
; CHECK-LABEL: test_Bcc_fallthrough_nottaken:
  %tst = icmp eq i32 %in, 42
  br i1 %tst, label %true, label %false, !prof !1

; CHECK: cmp {{%[goli][0-9]+}}, 42

; CHECK: be [[TRUE:.LBB[0-9]+_[0-9]+]]
; CHECK-NEXT: nop
; CHECK-NEXT: ! BB#
; CHECK-NEXT: call test_false

; CHECK: [[TRUE]]:
; CHECK: call test_true

true:
  call void @test_true()
  ret void

false:
  call void @test_false()
  ret void
}
