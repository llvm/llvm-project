; RUN: llc %s -enable-machine-outliner=never -mtriple=arm64-linux-gnu -o - | \
; RUN:   FileCheck --check-prefixes=CHECK,CHECK-LINUX %s
; <rdar://problem/14199482> ARM64: Calls to bzero() replaced with calls to memset()

; CHECK-LABEL: fct1:
; For small size (<= 256), we do not change memset to bzero.
; CHECK-DARWIN: {{b|bl}} _memset
; CHECK-LINUX: {{b|bl}} memset
define void @fct1(ptr nocapture %ptr) !prof !14 {
entry:
  tail call void @llvm.memset.p0.i64(ptr %ptr, i8 0, i64 256, i1 false)
  ret void
}

declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1)

; CHECK-LABEL: fct2:
; When the size is bigger than 256, change into bzero.
; CHECK-DARWIN: {{b|bl}} _bzero
; CHECK-LINUX: {{b|bl}} memset
define void @fct2(ptr nocapture %ptr) !prof !14 {
entry:
  tail call void @llvm.memset.p0.i64(ptr %ptr, i8 0, i64 257, i1 false)
  ret void
}

; CHECK-LABEL: fct3:
; For unknown size, change to bzero.
; CHECK-DARWIN: {{b|bl}} _bzero
; CHECK-LINUX: {{b|bl}} memset
define void @fct3(ptr nocapture %ptr, i32 %unknown) !prof !14 {
entry:
  %conv = sext i32 %unknown to i64
  tail call void @llvm.memset.p0.i64(ptr %ptr, i8 0, i64 %conv, i1 false)
  ret void
}

; CHECK-LABEL: fct4:
; Size <= 256, no change.
; CHECK-DARWIN: {{b|bl}} _memset
; CHECK-LINUX: {{b|bl}} memset
define void @fct4(ptr %ptr) !prof !14 {
entry:
  %tmp = tail call i64 @llvm.objectsize.i64(ptr %ptr, i1 false)
  %call = tail call ptr @__memset_chk(ptr %ptr, i32 0, i64 256, i64 %tmp)
  ret void
}

declare ptr @__memset_chk(ptr, i32, i64, i64)

declare i64 @llvm.objectsize.i64(ptr, i1)

; CHECK-LABEL: fct5:
; Size > 256, change.
; CHECK-DARWIN: {{b|bl}} _bzero
; CHECK-LINUX: {{b|bl}} memset
define void @fct5(ptr %ptr) !prof !14 {
entry:
  %tmp = tail call i64 @llvm.objectsize.i64(ptr %ptr, i1 false)
  %call = tail call ptr @__memset_chk(ptr %ptr, i32 0, i64 257, i64 %tmp)
  ret void
}

; CHECK-LABEL: fct6:
; Size = unknown, change.
; CHECK-DARWIN: {{b|bl}} _bzero
; CHECK-LINUX: {{b|bl}} memset
define void @fct6(ptr %ptr, i32 %unknown) !prof !14 {
entry:
  %conv = sext i32 %unknown to i64
  %tmp = tail call i64 @llvm.objectsize.i64(ptr %ptr, i1 false)
  %call = tail call ptr @__memset_chk(ptr %ptr, i32 0, i64 %conv, i64 %tmp)
  ret void
}

; Next functions check that memset is not turned into bzero
; when the set constant is non-zero, whatever the given size.

; CHECK-LABEL: fct7:
; memset with something that is not a zero, no change.
; CHECK-DARWIN: {{b|bl}} _memset
; CHECK-LINUX: {{b|bl}} memset
define void @fct7(ptr %ptr) !prof !14 {
entry:
  %tmp = tail call i64 @llvm.objectsize.i64(ptr %ptr, i1 false)
  %call = tail call ptr @__memset_chk(ptr %ptr, i32 1, i64 256, i64 %tmp)
  ret void
}

; CHECK-LABEL: fct8:
; memset with something that is not a zero, no change.
; CHECK-DARWIN: {{b|bl}} _memset
; CHECK-LINUX: {{b|bl}} memset
define void @fct8(ptr %ptr) !prof !14 {
entry:
  %tmp = tail call i64 @llvm.objectsize.i64(ptr %ptr, i1 false)
  %call = tail call ptr @__memset_chk(ptr %ptr, i32 1, i64 257, i64 %tmp)
  ret void
}

; CHECK-LABEL: fct9:
; memset with something that is not a zero, no change.
; CHECK-DARWIN: {{b|bl}} _memset
; CHECK-LINUX: {{b|bl}} memset
define void @fct9(ptr %ptr, i32 %unknown) !prof !14 {
entry:
  %conv = sext i32 %unknown to i64
  %tmp = tail call i64 @llvm.objectsize.i64(ptr %ptr, i1 false)
  %call = tail call ptr @__memset_chk(ptr %ptr, i32 1, i64 %conv, i64 %tmp)
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 10000}
!4 = !{!"MaxCount", i64 10}
!5 = !{!"MaxInternalCount", i64 1}
!6 = !{!"MaxFunctionCount", i64 1000}
!7 = !{!"NumCounts", i64 3}
!8 = !{!"NumFunctions", i64 3}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13}
!11 = !{i32 10000, i64 100, i32 1}
!12 = !{i32 999000, i64 100, i32 1}
!13 = !{i32 999999, i64 1, i32 2}
!14 = !{!"function_entry_count", i64 0}
