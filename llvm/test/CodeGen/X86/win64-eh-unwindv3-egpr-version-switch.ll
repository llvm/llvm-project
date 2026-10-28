; RUN: llc -mtriple=x86_64-unknown-windows-msvc -o - %s | FileCheck %s

; Interleaved egpr / non-egpr functions in a V2 module: the per-function
; version must alternate both ways, 3, 2, 3, 2.

; CHECK-LABEL:  a_egpr:
; CHECK:        .seh_unwindversion 3
; CHECK-NOT:    .seh_unwindversion
; CHECK-LABEL:  b_base:
; CHECK:        .seh_unwindversion 2
; CHECK-NOT:    .seh_unwindversion
; CHECK-LABEL:  c_egpr:
; CHECK:        .seh_unwindversion 3
; CHECK-NOT:    .seh_unwindversion
; CHECK-LABEL:  d_base:
; CHECK:        .seh_unwindversion 2

define dso_local void @a_egpr() #0 {
entry:
  call void @ext()
  ret void
}

define dso_local void @b_base() #1 {
entry:
  call void @ext()
  ret void
}

define dso_local void @c_egpr() #0 {
entry:
  call void @ext()
  ret void
}

define dso_local void @d_base() #1 {
entry:
  call void @ext()
  ret void
}

declare void @ext()
attributes #0 = { uwtable "target-features"="+egpr" }
attributes #1 = { uwtable }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"winx64-eh-unwind", i32 2}
