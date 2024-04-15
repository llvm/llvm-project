; RUN: not llvm-as < %s 2>&1 | FileCheck %s

define void @test_overlapping_regions(ptr %a1) {
; CHECK: Overlapping tbaa.struct regions
; CHECK-NEXT:  %ld = load i8, ptr %a1, align 1, !tbaa.struct !0
  %ld = load i8, ptr %a1, align 1, !tbaa.struct !0
  ret void
}

define void @test_size_not_integer(ptr %a1) {
; CHECK: Size must be a constant integer
; CHECK-NEXT:  store i8 1, ptr %a1, align 1, !tbaa.struct !5
  store i8 1, ptr %a1, align 1, !tbaa.struct !5
  ret void
}

define void @test_offset_not_integer(ptr %a1, ptr %a2) {
; CHECK: Offset must be a constant integer
; CHECK-NEXT:  tail call void @llvm.memcpy.p0.p0.i64(ptr align 8 %a1, ptr align 8 %a2, i64 16, i1 false), !tbaa.struct !6
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 8 %a1, ptr align 8 %a2, i64 16, i1 false), !tbaa.struct !6
  ret void
}

define void @test_tbaa_missing(ptr %a1, ptr %a2) {
; CHECK: TBAA tag missing
; CHECK-NEXT:  tail call void @llvm.memcpy.p0.p0.i64(ptr align 8 %a1, ptr align 8 %a2, i64 16, i1 false), !tbaa.struct !7
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 8 %a1, ptr align 8 %a2, i64 16, i1 false), !tbaa.struct !7
  ret void
}

define void @test_tbaa_invalid(ptr %a1) {
; CHECK: Old-style TBAA is no longer allowed, use struct-path TBAA instead
; CHECK-NEXT:  store i8 1, ptr %a1, align 1, !tbaa.struct !8
  store i8 1, ptr %a1, align 1, !tbaa.struct !8
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
!8 = !{i64 0, i64 4, !2}
