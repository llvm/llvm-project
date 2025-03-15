; RUN: llc < %s -mtriple=arm64ec-pc-windows-msvc | FileCheck %s

declare void @called()
declare void @escaped()
define void @f(ptr %dst) {
  call void @called()
  store ptr @escaped, ptr %dst
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"cfguard", i32 1}

; CHECK-LABEL: .section .gfids$y,"dr"
; CHECK-NEXT:  .symidx escaped
; CHECK-NOT:   .symidx
