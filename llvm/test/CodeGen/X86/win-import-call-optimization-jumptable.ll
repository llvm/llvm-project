; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s --check-prefix ASM
; RUN: llc --fast-isel -mtriple=x86_64-pc-windows-msvc -o - %s | FileCheck %s --check-prefix ASM
; RUN: llc -mtriple=x86_64-pc-windows-msvc  --filetype=obj -o - %s | llvm-objdump - --disassemble \
; RUN:  | FileCheck %s --check-prefix OBJ

; ASM-LABEL:  uses_rax:
; ASM:        .Limpcall0:
; ASM-NEXT:     jmpq    *%rax
; ASM-NEXT:     int3
; ASM-NEXT:     int3
; ASM-NEXT:     int3
; ASM-NEXT:     int3

define void @uses_rax(i32 %x) {
entry:
  switch i32 %x, label %sw.epilog [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
    i32 2, label %sw.bb2
    i32 3, label %sw.bb3
  ]

sw.bb:
  tail call void @g(i32 0) #2
  br label %sw.epilog

sw.bb1:
  tail call void @g(i32 1) #2
  br label %sw.epilog

sw.bb2:
  tail call void @g(i32 2) #2
  br label %sw.epilog

sw.bb3:
  tail call void @g(i32 3) #2
  br label %sw.epilog

sw.epilog:
  tail call void @g(i32 10) #2
  ret void
}

; ASM-LABEL:  uses_rcx:
; ASM:        .Limpcall1:
; ASM-NEXT:     jmpq    *%rcx
; ASM-NEXT:     int3
; ASM-NEXT:     int3
; ASM-NEXT:     int3
; ASM-NEXT:     int3

define void @uses_rcx(i32 %x) {
entry:
  switch i32 %x, label %sw.epilog [
    i32 10, label %sw.bb
    i32 11, label %sw.bb1
    i32 12, label %sw.bb2
    i32 13, label %sw.bb3
  ]

sw.bb:
  tail call void @g(i32 0) #2
  br label %sw.epilog

sw.bb1:
  tail call void @g(i32 1) #2
  br label %sw.epilog

sw.bb2:
  tail call void @g(i32 2) #2
  br label %sw.epilog

sw.bb3:
  tail call void @g(i32 3) #2
  br label %sw.epilog

sw.epilog:
  tail call void @g(i32 10) #2
  ret void
}

declare void @g(i32)

; ASM-LABEL:  .section        .retplne,"yi"
; ASM-NEXT:   .asciz  "RetpolineV1"
; ASM-NEXT:   .long   24
; ASM-NEXT:   .secnum .text
; ASM-NEXT:   .long   16
; ASM-NEXT:   .secoffset      .Limpcall0
; ASM-NEXT:   .long   17
; ASM-NEXT:   .secoffset      .Limpcall1

; The loader assumes an exact sequence of instructions/bytes at each marked site since it may
; replace the instruction(s) with new instruction(s), and the MSVC linker validates these at link
; time.

; Kind = 16-31 (IMAGE_RETPOLINE_AMD64_SWITCHTABLE_*)
; OBJ-LABEL:  <uses_rax>:
; OBJ:        : ff e0                         jmpq    *%rax
; OBJ-NEXT:   : cc                            int3
; OBJ-NEXT:   : cc                            int3
; OBJ-NEXT:   : cc                            int3
; OBJ-NEXT:   : cc                            int3
; OBJ-LABEL:  <uses_rcx>:
; OBJ:        : ff e1                         jmpq    *%rcx
; OBJ-NEXT:   : cc                            int3
; OBJ-NEXT:   : cc                            int3
; OBJ-NEXT:   : cc                            int3
; OBJ-NEXT:   : cc                            int3

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"import-call-optimization", i32 1}
