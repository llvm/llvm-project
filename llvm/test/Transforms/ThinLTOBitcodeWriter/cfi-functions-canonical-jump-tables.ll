; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t %s
; RUN: llvm-modextract -b -n 1 -o - %t | llvm-dis | FileCheck %s

; Check that cfi.functions metadata has the expected contents.

; CHECK: !"f1", i8 1
; CHECK: !"f2", i8 1
; CHECK: !"f3", i8 0

declare !type !1 void @f1()

define void @f2() !type !1 {
  ret void
}

define void @f3() "cfi-canonical-jump-table" !type !1 {
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 4, !"CFI Canonical Jump Tables", i32 0}
!1 = !{i32 0, !"typeid1"}
