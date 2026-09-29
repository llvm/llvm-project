; RUN: llc %s -o - -mtriple x86_64-sie-ps5 -emit-jump-table-sizes-section -verify-machineinstrs --relocation-model=pic | FileCheck --check-prefix=PS5-CHECK %s
; RUN: llc %s -o - -mtriple x86_64-sie-ps5 -verify-machineinstrs --relocation-model=pic | FileCheck --check-prefix=NOFLAG %s
; RUN: llc %s -o - -mtriple x86_64-sie-ps5 -verify-machineinstrs --relocation-model=pic | FileCheck --check-prefix=NOTABLE %s

; RUN: llc %s -o - -mtriple x86_64-unknown-linux-gnu -emit-jump-table-sizes-section -verify-machineinstrs --relocation-model=pic | FileCheck --check-prefix=ELF-CHECK %s
; RUN: llc %s -o - -mtriple x86_64-unknown-linux-gnu -verify-machineinstrs --relocation-model=pic | FileCheck --check-prefix=NOFLAG %s
; RUN: llc %s -o - -mtriple x86_64-unknown-linux-gnu -verify-machineinstrs --relocation-model=pic | FileCheck --check-prefix=NOTABLE %s

; RUN: llc %s -o - -mtriple x86_64-pc-windows-msvc -emit-jump-table-sizes-section -verify-machineinstrs --relocation-model=pic | FileCheck --check-prefix=COFF-CHECK %s
; RUN: llc %s -o - -mtriple x86_64-pc-windows-msvc -verify-machineinstrs --relocation-model=pic | FileCheck --check-prefix=NOFLAG %s
; RUN: llc %s -o - -mtriple x86_64-pc-windows-msvc -verify-machineinstrs --relocation-model=pic | FileCheck --check-prefix=NOTABLE %s

; This test verifies the jump table size section. Currently only enabled by default on the PS5 target.

$foo1 = comdat any

; Ensure proper comdat handling.
define void @foo1(i32 %x, ptr %to) comdat {

; PS5-CHECK-LABEL: foo1
; PS5-CHECK:       .section        .llvm_jump_table_sizes,"G",@llvm_jt_sizes,foo1,comdat
; PS5-CHECK-NEXT: .quad   .LJTI0_0
; PS5-CHECK-NEXT: .quad   6

; ELF-CHECK-LABEL: foo1
; ELF-CHECK:       .section        .llvm_jump_table_sizes,"G",@llvm_jt_sizes,foo1,comdat
; ELF-CHECK-NEXT: .quad   .LJTI0_0
; ELF-CHECK-NEXT: .quad   6

; COFF-CHECK-LABEL: foo1
; COFF-CHECK:      .section         .llvm_jump_table_sizes,"drD",associative,foo1
; COFF-CHECK-NEXT: .quad   .LJTI0_0
; COFF-CHECK-NEXT: .quad   6

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

; PS5-CHECK-LABEL:    foo2
; PS5-CHECK:          .section .llvm_jump_table_sizes,"",@llvm_jt_sizes
; PS5-CHECK-NEXT:     .quad .LJTI1_0
; PS5-CHECK-NEXT:     .quad   5

; ELF-CHECK-LABEL:    foo2
; ELF-CHECK:          .section .llvm_jump_table_sizes,"",@llvm_jt_sizes
; ELF-CHECK-NEXT:     .quad .LJTI1_0
; ELF-CHECK-NEXT:     .quad   5

; COFF-CHECK-LABEL:   foo2
; COFF-CHECK:         .section         .llvm_jump_table_sizes,"drD"
; COFF-CHECK-NEXT:    .quad .LJTI1_0
; COFF-CHECK-NEXT:    .quad   5

; NOFLAG-LABEL:       foo1
; NOFLAG-NOT:         .section        .llvm_jump_table_sizes

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

; NOTABLE-LABEL:    foo3
; NOTABLE-NOT:      .section        .llvm_jump_table_sizes

exit:
  ret void
}

; Ensure we can deal with nested jump tables.

define void @nested(i32 %x, i32 %y, ptr %to) {

; PS5-CHECK-LABEL:    nested
; PS5-CHECK:          .section .llvm_jump_table_sizes,"",@llvm_jt_sizes
; PS5-CHECK-NEXT:     .quad .LJTI3_0
; PS5-CHECK-NEXT:     .quad   5
; PS5-CHECK-NEXT:     .quad .LJTI3_1
; PS5-CHECK-NEXT:     .quad 6

; ELF-CHECK-LABEL:    nested
; ELF-CHECK:          .section .llvm_jump_table_sizes,"",@llvm_jt_sizes
; ELF-CHECK-NEXT:     .quad .LJTI3_0
; ELF-CHECK-NEXT:     .quad   5
; ELF-CHECK-NEXT:     .quad .LJTI3_1
; ELF-CHECK-NEXT:     .quad 6

; COFF-CHECK-LABEL:   nested
; COFF-CHECK:         .section         .llvm_jump_table_sizes,"drD"
; COFF-CHECK-NEXT:     .quad .LJTI3_0
; COFF-CHECK-NEXT:     .quad   5
; COFF-CHECK-NEXT:     .quad .LJTI3_1
; COFF-CHECK-NEXT:     .quad 6

; NOFLAG-LABEL:       nested
; NOFLAG-NOT:         .section        .llvm_jump_table_sizes

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
  switch i32 %y, label %default [
    i32 1, label %bb5
    i32 2, label %bb6
    i32 3, label %bb7
    i32 4, label %bb8
    i32 5, label %bb9
    i32 6, label %bb10
  ]
  br label %exit2
bb5:
  store i32 4, ptr %to
  br label %exit
bb6:
  store i32 4, ptr %to
  br label %exit
bb7:
  store i32 4, ptr %to
  br label %exit
bb8:
  store i32 4, ptr %to
  br label %exit
bb9:
  store i32 4, ptr %to
  br label %exit
bb10:
  store i32 4, ptr %to
  br label %exit
exit:
  ret void
exit2:
  ret void
default:
  unreachable
}