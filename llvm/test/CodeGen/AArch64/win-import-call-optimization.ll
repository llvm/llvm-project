; RUN: llc -mtriple=aarch64-pc-windows-msvc < %s | FileCheck %s --check-prefix=CHECK

define dso_local void @normal_call() local_unnamed_addr section "nc_sect" {
entry:
  call void @a()
  call void @a()
  ret void
}
; CHECK-LABEL:  normal_call:
; CHECK:        adrp    [[ADRPREG:x[0-9]+]], __imp_a
; CHECK-NEXT:   ldr     [[LDRREG:x[0-9]+]], [[[ADRPREG]], :lo12:__imp_a]
; CHECK-NEXT:   .Limpcall0:
; CHECK-NEXT:   blr     [[LDRREG]]
; CHECK-NEXT:   .Limpcall1:
; CHECK-NEXT:   blr     [[LDRREG]]

define dso_local void @tail_call() local_unnamed_addr section "tc_sect" {
entry:
  tail call void @b()
  ret void
}
; CHECK-LABEL:  tail_call:
; CHECK:        adrp    [[ADRPREG:x[0-9]+]], __imp_b
; CHECK-NEXT:   ldr     [[LDRREG:x[0-9]+]], [[[ADRPREG]], :lo12:__imp_b]
; CHECK-NEXT:   .Limpcall2:
; CHECK-NEXT:   br      [[LDRREG]]

declare dllimport void @a() local_unnamed_addr
declare dllimport void @b() local_unnamed_addr

; CHECK-LABEL  .section   .impcall,"yi"
; CHECK-NEXT   .asciz  "Imp_Call_V1"
; CHECK-NEXT   .word   32
; CHECK-NEXT   .secnum nc_sect
; CHECK-NEXT   .word   19
; CHECK-NEXT   .secoffset      .Limpcall0
; CHECK-NEXT   .symidx __imp_a
; CHECK-NEXT   .word   19
; CHECK-NEXT   .secoffset      .Limpcall1
; CHECK-NEXT   .symidx __imp_a
; CHECK-NEXT   .word   20
; CHECK-NEXT   .secnum tc_sect
; CHECK-NEXT   .word   19
; CHECK-NEXT   .secoffset      .Limpcall2
; CHECK-NEXT   .symidx __imp_b

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"import-call-optimization", i32 1}
