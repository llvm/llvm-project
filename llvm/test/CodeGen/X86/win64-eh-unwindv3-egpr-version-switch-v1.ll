; RUN: llc -mtriple=x86_64-unknown-windows-msvc -o - %s | FileCheck %s

; Like win64-eh-unwindv3-egpr-version-switch.ll but in a V1-default
; module: each egpr function toggles to V3, each non-egpr function stays on V1
; (no marker).

; CHECK-LABEL:  a_egpr:
; CHECK:        .seh_proc a_egpr
; CHECK:        .seh_unwindversion 3
; CHECK:        .seh_endproc

; CHECK-LABEL:  b_base:
; CHECK:        .seh_proc b_base
; CHECK-NOT:    .seh_unwindversion
; CHECK:        .seh_endproc

; CHECK-LABEL:  c_egpr:
; CHECK:        .seh_proc c_egpr
; CHECK:        .seh_unwindversion 3
; CHECK:        .seh_endproc

; CHECK-LABEL:  d_base:
; CHECK:        .seh_proc d_base
; CHECK-NOT:    .seh_unwindversion
; CHECK:        .seh_endproc

define dso_local void @a_egpr() uwtable #0 {
entry:
  call void @ext()
  ret void
}

define dso_local void @b_base() uwtable {
entry:
  call void @ext()
  ret void
}

define dso_local void @c_egpr() uwtable #0 {
entry:
  call void @ext()
  ret void
}

define dso_local void @d_base() uwtable {
entry:
  call void @ext()
  ret void
}

declare void @ext()
attributes #0 = { "target-features"="+egpr" }
