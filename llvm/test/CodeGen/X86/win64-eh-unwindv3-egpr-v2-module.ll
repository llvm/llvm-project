; RUN: llc -mtriple=x86_64-unknown-windows-msvc -o - %s | FileCheck %s

; In a V2-default module, the egpr function is promoted to V3 (with no V2
; markers) while the non-egpr function stays on V2.

; The non-EGPR function stays on V2 with its V2 epilog markers.
; CHECK-LABEL:  v2_baseline:
; CHECK:        .seh_proc v2_baseline
; CHECK:        .seh_unwindversion 2
; CHECK-NOT:    .seh_unwindversion 3
; CHECK:        .seh_unwindv2start
; CHECK:        .seh_endproc

; The EGPR function is emitted as V3 with no V2 markers.
; CHECK-LABEL:  v2_egpr:
; CHECK:        .seh_proc v2_egpr
; CHECK:        .seh_unwindversion 3
; CHECK-NOT:    .seh_unwindversion 2
; CHECK-NOT:    .seh_unwindv2start
; CHECK:        .seh_endproc

define dso_local void @v2_baseline() uwtable {
entry:
  call void @ext()
  ret void
}

define dso_local void @v2_egpr() uwtable #0 {
entry:
  call void @ext()
  ret void
}

declare void @ext()
attributes #0 = { "target-features"="+egpr" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"winx64-eh-unwind", i32 2}
