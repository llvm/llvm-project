; RUN: llc %s -o - -emit-jump-table-sizes-section -verify-machineinstrs --relocation-model=pic | FileCheck --check-prefix=CHECK %s
; RUN: llc %s -o - -verify-machineinstrs --relocation-model=pic | FileCheck --check-prefix=NOFLAG %s

; This test verifies the jump table size section. Currently only enabled by default on the PS5 target.

$foo1 = comdat any

; Ensure proper comdat handling.
define void @foo1(i32 %x, ptr %to) comdat {

; CHECK-LABEL: foo1
; CHECK:      .section        .llvm_jump_table_sizes,"G",@llvm_jt_sizes,foo1,comdat
; CHECK-NEXT: .quad   .LJTI0_0
; CHECK-NEXT: .quad   6

; NOFLAG-LABEL: foo1
; NOFLAG-NOT: .section        .llvm_jump_table_sizes

entry:
  switch i32 %x, label %default [
    i32 0, label %bb0
    i32 1, label %bb1
    i32 2, label %bb2
    i32 3, label %bb3
    i32 4, label %bb4
    i32 5, label %bb4
  ]
bb0:
  store i32 0, ptr %to
  br label %exit
bb1:
  store i32 1, ptr %to
  br label %exit
bb2:
  store i32 2, ptr %to
  br label %exit
bb3:
  store i32 3, ptr %to
  br label %exit
bb4:
  store i32 4, ptr %to
  br label %exit
exit:
  ret void
default:
  unreachable
}

define void @foo2(i32 %x, ptr %to) {

; CHECK-LABEL: foo2
; CHECK:      .section        .llvm_jump_table_sizes
; CHECK-NEXT: .quad   .LJTI1_0
; CHECK-NEXT: .quad   5

; NOFLAG-LABEL: foo2
; NOFLAG-NOT: .section        .llvm_jump_table_sizes

entry:
  switch i32 %x, label %default [
    i32 0, label %bb0
    i32 1, label %bb1
    i32 2, label %bb2
    i32 3, label %bb3
    i32 4, label %bb4
  ]
bb0:
  store i32 0, ptr %to
  br label %exit
bb1:
  store i32 1, ptr %to
  br label %exit
bb2:
  store i32 2, ptr %to
  br label %exit
bb3:
  store i32 3, ptr %to
  br label %exit
bb4:
  store i32 4, ptr %to
  br label %exit
exit:
  ret void
default:
  unreachable
}

; Ensure that the section isn't produced if there is no jump table.

define void @foo3(i32 %x, ptr %to) {

; CHECK-LABEL:    foo3
; CHECK-NOT:      .section        .llvm_jump_table_sizes

exit:
  ret void
}
