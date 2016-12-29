; RUN: llc < %s -relocation-model static | FileCheck %s --check-prefix=CHECK --check-prefix=STATIC
; RUN: llc < %s -relocation-model pic | FileCheck %s --check-prefix=CHECK --check-prefix=PIC

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

; STATIC: movslq .LJTI0_0(,%{{.*}},4), %[[target:[^ ]*]]
; STATIC: leaq f(%[[target]]), %[[target]]
; STATIC: jmpq *%[[target]]

; PIC: leaq .LJTI0_0(%rip), %[[jt:[^ ]*]]
; PIC: movslq (%[[jt]],%{{.*}},4), %[[offset:[^ ]*]]
; PIC: leaq f(%rip), %[[base:[^ ]*]]
; PIC: addq %[[offset]], %[[base]]
; PIC: jmpq *%[[base]]

; CHECK: .LBB0_{{.*}}: # %sw.bb
; CHECK: .LBB0_{{.*}}: # %sw.bb1
; CHECK: .LBB0_{{.*}}: # %sw.bb2
; CHECK: .LBB0_{{.*}}: # %sw.bb3
; CHECK: callq g
; CHECK: jmp g # TAILCALL
; CHECK: .section        .rdata,"dr"
; CHECK: .long .LBB0_{{.*}}-f
; CHECK: .long .LBB0_{{.*}}-f
; CHECK: .long .LBB0_{{.*}}-f
; CHECK: .long .LBB0_{{.*}}-f
; CHECK: .seh_handlerdata

; It's important that we switch back to .text here, not .rdata.
; CHECK: .text
; CHECK: .seh_endproc
