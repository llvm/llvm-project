; RUN: opt -passes='default<O2>' -S %s | FileCheck %s

;; Confirms cfi_type_hash and !riscv_lpad_label survive a standard -O2
;; pipeline on an ordinary function.

declare void @sink(i32)

define void @f(ptr %fptr, i32 %x) "cfi_type_hash"="778899" {
; CHECK-LABEL: define {{.*}} @f(
; CHECK-SAME: #[[ATTR:[0-9]+]]
  %1 = call i32 %fptr(i32 %x), !riscv_lpad_label !0
  call void @sink(i32 %1)
  ret void
; CHECK: call {{.*}}, !riscv_lpad_label ![[MD:[0-9]+]]
}

; CHECK: attributes #[[ATTR]] = { "cfi_type_hash"="778899" }
; CHECK: ![[MD]] = !{i32 778899}

!llvm.module.flags = !{!1, !2}
!0 = !{i32 778899}
!1 = !{i32 8, !"cf-protection-branch", i32 1}
!2 = !{i32 1, !"cf-branch-label-scheme", !"func-sig"}