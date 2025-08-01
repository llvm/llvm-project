; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s

; CHECK-LABEL:  uses_rax:
; CHECK:        .Limpcall0:
; CHECK-NEXT:     jmpq    *%rax

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

; CHECK-LABEL:  uses_rcx:
; CHECK:        .Limpcall1:
; CHECK-NEXT:     jmpq    *%rcx

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

; CHECK-LABEL:  .section        .retplne,"yi"
; CHECK-NEXT:   .asciz  "RetpolineV1"
; CHECK-NEXT:   .long   24
; CHECK-NEXT:   .secnum .text
; CHECK-NEXT:   .long   16
; CHECK-NEXT:   .secoffset      .Limpcall0
; CHECK-NEXT:   .long   17
; CHECK-NEXT:   .secoffset      .Limpcall1

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"import-call-optimization", i32 1}
