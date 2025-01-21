; RUN: llc -mtriple=aarch64-pc-windows-msvc -aarch64-win-import-call-optimization < %s | FileCheck %s

define dso_local void @normal_call() local_unnamed_addr {
entry:
  call void @a()
  ret void
}
; CHECK-LABEL:  normal_call:
; CHECK:        bl a

declare void @a() local_unnamed_addr

; Even if there are no calls to imported functions, we still need to emit the
; .impcall section.

; CHECK-LABEL  .section   .impcall,"yi"
; CHECK-NEXT   .asciz  "Imp_Call_V1"
; CHECK-NOT    .secnum
