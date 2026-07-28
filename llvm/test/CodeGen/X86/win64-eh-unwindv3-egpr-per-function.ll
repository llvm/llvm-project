; RUN: llc -mtriple=x86_64-unknown-windows-msvc -o - %s | FileCheck %s

; Models auto-dispatch: in a V1-default module, only the egpr clone
; is promoted to V3; the baseline clone stays on the module default.

; The baseline function must NOT get a per-function unwind version.
; CHECK-LABEL:  baseline:
; CHECK:        .seh_proc baseline
; CHECK-NOT:    .seh_unwindversion
; CHECK:        .seh_endproc

; The APX clone may use EGPR and is promoted to V3 individually.
; CHECK-LABEL:  apx_clone:
; CHECK:        .seh_proc apx_clone
; CHECK:        .seh_unwindversion 3
; CHECK:        .seh_endproc

define dso_local void @baseline() uwtable {
entry:
  call void @ext()
  ret void
}

define dso_local void @apx_clone() uwtable #0 {
entry:
  call void @ext()
  ret void
}

declare void @ext()
attributes #0 = { "target-features"="+egpr" }
