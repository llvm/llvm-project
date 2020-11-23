; RUN: llc < %s -relocation-model static | FileCheck %s

; FIXME: Remove '-relocation-model static' when it is no longer necessary to
; trigger the separate .rdata section.

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24215"

define void @f(i32 %x) {
entry:
  switch i32 %x, label %sw.epilog [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
    i32 2, label %sw.bb2
    i32 3, label %sw.bb3
  ]

sw.bb:                                            ; preds = %entry
  tail call void @g(i32 0) #2
  br label %sw.epilog

sw.bb1:                                           ; preds = %entry
  tail call void @g(i32 1) #2
  br label %sw.epilog

sw.bb2:                                           ; preds = %entry
  tail call void @g(i32 2) #2
  br label %sw.epilog

sw.bb3:                                           ; preds = %entry
  tail call void @g(i32 3) #2
  br label %sw.epilog

sw.epilog:                                        ; preds = %entry, %sw.bb3, %sw.bb2, %sw.bb1, %sw.bb
  tail call void @g(i32 10) #2
  ret void
}

declare void @g(i32)

; CHECK: .text
; CHECK: f:
; CHECK: .seh_proc f
; CHECK: jmpq    *.LJTI0_0
; CHECK: .LBB0_{{.*}}: # %sw.bb
; CHECK: .LBB0_{{.*}}: # %sw.bb1
; CHECK: .LBB0_{{.*}}: # %sw.bb2
; CHECK: .LBB0_{{.*}}: # %sw.bb3
; CHECK: callq g
; CHECK: jmp g # TAILCALL
; CHECK: .section        .rdata,"dr"
; CHECK: .quad .LBB0_
; CHECK: .quad .LBB0_
; CHECK: .quad .LBB0_
; CHECK: .quad .LBB0_

; It's important that we switch back to .text here, not .rdata.
; CHECK: .text
; CHECK: .seh_endproc
