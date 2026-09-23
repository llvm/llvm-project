; RUN: not llvm-as -disable-output < %s 2>&1 | FileCheck %s

; FIXME: The verifier does not yet reject the overlapping-region (@test_overlapping_regions)
; or null-tag (@test_tbaa_missing) nodes below.

define void @test_overlapping_regions(ptr %a1) {
  %ld = load i8, ptr %a1, align 1, !tbaa.struct !0
  ret void
}

define void @test_size_not_integer(ptr %a1) {
; CHECK-DAG: !tbaa.struct field size must be a constant integer
  store i8 1, ptr %a1, align 1, !tbaa.struct !5
  ret void
}

define void @test_offset_not_integer(ptr %a1, ptr %a2) {
; CHECK-DAG: !tbaa.struct field offset must be a constant integer
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 8 %a1, ptr align 8 %a2, i64 16, i1 false), !tbaa.struct !6
  ret void
}

define void @test_tbaa_missing(ptr %a1, ptr %a2) {
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 8 %a1, ptr align 8 %a2, i64 16, i1 false), !tbaa.struct !7
  ret void
}

define void @test_tbaa_invalid(ptr %a1) {
; CHECK-DAG: Offset must be constant integer
  store i8 1, ptr %a1, align 1, !tbaa.struct !8
  ret void
}

define void @test_offsets_not_increasing(ptr %a1) {
; CHECK-DAG: !tbaa.struct field offsets must be non-decreasing
  store i8 1, ptr %a1, align 1, !tbaa.struct !9
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1) nounwind

!0 = !{i64 0, i64 4, !1, i64 1, i64 4, !1}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C++ TBAA"}
!5 = !{i64 0, !2, !1}
!6 = !{!2, i64 0, !1}
!7 = !{i64 0, i64 4, null}
!8 = !{i64 0, i64 4, !10}
!9 = !{i64 4, i64 4, !1, i64 0, i64 4, !1}
; A struct-path-shaped access tag with a non-constant offset. Auto-upgrade
; leaves it unchanged (operand 0 is already an MDNode), so it stays rejected.
!10 = !{!2, !2, !3}
