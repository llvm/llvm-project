; RUN: llc -mtriple=riscv64 -code-model=small < %s \
; RUN:   | FileCheck %s -check-prefix=RODATA
; RUN: llc -mtriple=riscv64 -code-model=medium < %s \
; RUN:   | FileCheck %s -check-prefix=RODATA
; RUN: llc -mtriple=riscv64 -code-model=large < %s \
; RUN:   | FileCheck %s -check-prefix=LARGE
; RUN: llc -mtriple=riscv64 -code-model=large -function-sections < %s \
; RUN:   | FileCheck %s -check-prefix=LARGE-FS

; With the large code model, the jump table should be placed in the same
; section as the function instead of .rodata.

; RODATA-LABEL: jt:
; RODATA:         .section .rodata,"a",@progbits
; RODATA:       .LJTI0_0:

; LARGE:          .text
; LARGE-LABEL:  jt:
; LARGE-NOT:      .section
; LARGE:        .LJTI0_0:

; LARGE-FS:       .section .text.jt,"ax",@progbits
; LARGE-FS-LABEL: jt:
; LARGE-FS-NOT:   .section
; LARGE-FS:     .LJTI0_0:

define void @jt(i32 signext %in, ptr %out) nounwind {
entry:
  switch i32 %in, label %exit [
    i32 1, label %bb1
    i32 2, label %bb2
    i32 3, label %bb3
    i32 4, label %bb4
    i32 5, label %bb5
    i32 6, label %bb6
  ]
bb1:
  store i32 4, ptr %out
  br label %exit
bb2:
  store i32 3, ptr %out
  br label %exit
bb3:
  store i32 2, ptr %out
  br label %exit
bb4:
  store i32 1, ptr %out
  br label %exit
bb5:
  store i32 100, ptr %out
  br label %exit
bb6:
  store i32 200, ptr %out
  br label %exit
exit:
  ret void
}
