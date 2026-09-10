; RUN: llc < %s -mtriple=x86_64_lfi | FileCheck %s

; LFI masks indirect branch targets down to a 32-byte bundle boundary, so every
; address that may be reached indirectly has to be aligned to one.

declare void @f(i32)

; Function entry points are reachable through function pointers.
define void @entry_aligned() {
; CHECK:      .p2align 5
; CHECK-NEXT: .type entry_aligned,@function
; CHECK-NEXT: entry_aligned:
  ret void
}

; Jump table targets are not marked address-taken by LLVM, but are still
; reached by the indirect branch that dispatches through the table.
define void @jump_table(i32 %x) {
; CHECK-LABEL: jump_table:
; CHECK:      jmpq *.LJTI
; CHECK:      .p2align 5
; CHECK-NEXT: .LBB{{[0-9_]+}}:
; CHECK:      .p2align 5
; CHECK-NEXT: .LBB{{[0-9_]+}}:
; CHECK:      .p2align 5
; CHECK-NEXT: .LBB{{[0-9_]+}}:
; CHECK:      .p2align 5
; CHECK-NEXT: .LBB{{[0-9_]+}}:
; CHECK:      .p2align 5
; CHECK-NEXT: .LBB{{[0-9_]+}}:
entry:
  switch i32 %x, label %exit [ i32 0, label %a
                               i32 1, label %b
                               i32 2, label %c
                               i32 3, label %d
                               i32 4, label %e ]
a:
  call void @f(i32 0)
  br label %exit
b:
  call void @f(i32 1)
  br label %exit
c:
  call void @f(i32 2)
  br label %exit
d:
  call void @f(i32 3)
  br label %exit
e:
  call void @f(i32 4)
  br label %exit
exit:
  ret void
}

; Blocks whose address is taken may be the target of an indirect branch.
define ptr @block_address() {
; CHECK-LABEL: block_address:
; CHECK:      .p2align 5
; CHECK-NEXT: .Ltmp{{[0-9]+}}:
entry:
  br label %target
target:
  ret ptr blockaddress(@block_address, %target)
}
