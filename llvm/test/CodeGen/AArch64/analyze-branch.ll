; RUN: llc -mtriple=aarch64-none-linux-gnu < %s | FileCheck %s

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

; CHECK: cmp {{w[0-9]+}}, #42

; CHECK: b.ne [[FALSE:.LBB[0-9]+_[0-9]+]]
; CHECK-NEXT: // %bb.
; CHECK-NEXT: bl test_true

; CHECK: [[FALSE]]:
; CHECK: bl test_false

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

; CHECK: cmp {{w[0-9]+}}, #42

; CHECK: b.eq [[TRUE:.LBB[0-9]+_[0-9]+]]
; CHECK-NEXT: // %bb.
; CHECK-NEXT: bl test_false

; CHECK: [[TRUE]]:
; CHECK: bl test_true

true:
  call void @test_true()
  ret void

false:
  call void @test_false()
  ret void
}

define void @test_CBZ_fallthrough_taken(i32 %in) nounwind {
; CHECK-LABEL: test_CBZ_fallthrough_taken:
  %tst = icmp eq i32 %in, 0
  br i1 %tst, label %true, label %false, !prof !0

; CHECK: cbnz {{w[0-9]+}}, [[FALSE:.LBB[0-9]+_[0-9]+]]
; CHECK-NEXT: // %bb.
; CHECK-NEXT: bl test_true

; CHECK: [[FALSE]]:
; CHECK: bl test_false

true:
  call void @test_true()
  ret void

false:
  call void @test_false()
  ret void
}

define void @test_CBZ_fallthrough_nottaken(i64 %in) nounwind {
; CHECK-LABEL: test_CBZ_fallthrough_nottaken:
  %tst = icmp eq i64 %in, 0
  br i1 %tst, label %true, label %false, !prof !1

; CHECK: cbz {{x[0-9]+}}, [[TRUE:.LBB[0-9]+_[0-9]+]]
; CHECK-NEXT: // %bb.
; CHECK-NEXT: bl test_false

; CHECK: [[TRUE]]:
; CHECK: bl test_true

true:
  call void @test_true()
  ret void

false:
  call void @test_false()
  ret void
}

define void @test_CBNZ_fallthrough_taken(i32 %in) nounwind {
; CHECK-LABEL: test_CBNZ_fallthrough_taken:
  %tst = icmp ne i32 %in, 0
  br i1 %tst, label %true, label %false, !prof !0

; CHECK: cbz {{w[0-9]+}}, [[FALSE:.LBB[0-9]+_[0-9]+]]
; CHECK-NEXT: // %bb.
; CHECK-NEXT: bl test_true

; CHECK: [[FALSE]]:
; CHECK: bl test_false

true:
  call void @test_true()
  ret void

false:
  call void @test_false()
  ret void
}

define void @test_CBNZ_fallthrough_nottaken(i64 %in) nounwind {
; CHECK-LABEL: test_CBNZ_fallthrough_nottaken:
  %tst = icmp ne i64 %in, 0
  br i1 %tst, label %true, label %false, !prof !1

; CHECK: cbnz {{x[0-9]+}}, [[TRUE:.LBB[0-9]+_[0-9]+]]
; CHECK-NEXT: // %bb.
; CHECK-NEXT: bl test_false

; CHECK: [[TRUE]]:
; CHECK: bl test_true

true:
  call void @test_true()
  ret void

false:
  call void @test_false()
  ret void
}

define void @test_TBZ_fallthrough_taken(i32 %in) nounwind {
; CHECK-LABEL: test_TBZ_fallthrough_taken:
  %bit = and i32 %in, 32768
  %tst = icmp eq i32 %bit, 0
  br i1 %tst, label %true, label %false, !prof !0

; CHECK: tbnz {{w[0-9]+}}, #15, [[FALSE:.LBB[0-9]+_[0-9]+]]
; CHECK-NEXT: // %bb.
; CHECK-NEXT: bl test_true

; CHECK: [[FALSE]]:
; CHECK: bl test_false

true:
  call void @test_true()
  ret void

false:
  call void @test_false()
  ret void
}

define void @test_TBZ_fallthrough_nottaken(i64 %in) nounwind {
; CHECK-LABEL: test_TBZ_fallthrough_nottaken:
  %bit = and i64 %in, 32768
  %tst = icmp eq i64 %bit, 0
  br i1 %tst, label %true, label %false, !prof !1

; CHECK: tbz {{[wx][0-9]+}}, #15, [[TRUE:.LBB[0-9]+_[0-9]+]]
; CHECK-NEXT: // %bb.
; CHECK-NEXT: bl test_false

; CHECK: [[TRUE]]:
; CHECK: bl test_true

true:
  call void @test_true()
  ret void

false:
  call void @test_false()
  ret void
}


define void @test_TBNZ_fallthrough_taken(i32 %in) nounwind {
; CHECK-LABEL: test_TBNZ_fallthrough_taken:
  %bit = and i32 %in, 32768
  %tst = icmp ne i32 %bit, 0
  br i1 %tst, label %true, label %false, !prof !0

; CHECK: tbz {{w[0-9]+}}, #15, [[FALSE:.LBB[0-9]+_[0-9]+]]
; CHECK-NEXT: // %bb.
; CHECK-NEXT: bl test_true

; CHECK: [[FALSE]]:
; CHECK: bl test_false

true:
  call void @test_true()
  ret void

false:
  call void @test_false()
  ret void
}

define void @test_TBNZ_fallthrough_nottaken(i64 %in) nounwind {
; CHECK-LABEL: test_TBNZ_fallthrough_nottaken:
  %bit = and i64 %in, 32768
  %tst = icmp ne i64 %bit, 0
  br i1 %tst, label %true, label %false, !prof !1

; CHECK: tbnz {{[wx][0-9]+}}, #15, [[TRUE:.LBB[0-9]+_[0-9]+]]
; CHECK-NEXT: // %bb.
; CHECK-NEXT: bl test_false

; CHECK: [[TRUE]]:
; CHECK: bl test_true

true:
  call void @test_true()
  ret void

false:
  call void @test_false()
  ret void
}

