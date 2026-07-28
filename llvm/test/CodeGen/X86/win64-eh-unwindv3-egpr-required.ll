; RUN: llc -mtriple=x86_64-unknown-windows-msvc -mattr=+egpr -o - %s | FileCheck %s

; An egpr function in a V1-default module is emitted as V3 via a per-function
; .seh_unwindversion 3.

; CHECK-LABEL:  func:
; CHECK:        .seh_proc func
; CHECK:        .seh_unwindversion 3

define dso_local void @func() uwtable #0 {
entry:
  call void @ext()
  ret void
}

declare void @ext()
attributes #0 = { "target-features"="+egpr" }
