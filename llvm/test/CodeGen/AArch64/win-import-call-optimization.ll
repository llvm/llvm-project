; RUN: llc -mtriple=aarch64-pc-windows-msvc %s -o - | FileCheck %s

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

; Regression test: branch folding would notice that both of these calls become
; branch instructions to the same register and would fold them. However, import
; call optimization requires them to remain separate as they are for different
; functions.
define dso_local void @call_one_or_other(i32 %val) local_unnamed_addr {
  %is_zero = icmp eq i32 %val, 0
  br i1 %is_zero, label %call_a, label %call_b

call_a:
  call void @a()
  br label %finish

call_b:
  call void @b()
  br label %finish

finish:
  ret void
}
; CHECK-LABEL:  call_one_or_other:
; CHECK:        adrp    [[ADRPREG:x[0-9]+]], __imp_b
; CHECK-NEXT:   ldr     [[LDRREG:x[0-9]+]], [[[ADRPREG]], :lo12:__imp_b]
; CHECK-NEXT:   .Limpcall3:
; CHECK-NEXT:   blr     [[LDRREG]]
; CHECK:        adrp    [[ADRPREG:x[0-9]+]], __imp_a
; CHECK-NEXT:   ldr     [[LDRREG:x[0-9]+]], [[[ADRPREG]], :lo12:__imp_a]
; CHECK-NEXT:   .Limpcall4:
; CHECK-NEXT:   blr     [[LDRREG]]

declare dllimport void @a() local_unnamed_addr
declare dllimport void @b() local_unnamed_addr

; CHECK-LABEL:  .section   .impcall,"yi"
; CHECK-NEXT:   .asciz  "Imp_Call_V1"
; CHECK-NEXT:   .word   32
; CHECK-NEXT:   .secnum nc_sect
; CHECK-NEXT:   .word   19
; CHECK-NEXT:   .secoffset      .Limpcall0
; CHECK-NEXT:   .symidx __imp_a
; CHECK-NEXT:   .word   19
; CHECK-NEXT:   .secoffset      .Limpcall1
; CHECK-NEXT:   .symidx __imp_a
; CHECK-NEXT:   .word   20
; CHECK-NEXT:   .secnum tc_sect
; CHECK-NEXT:   .word   19
; CHECK-NEXT:   .secoffset      .Limpcall2
; CHECK-NEXT:   .symidx __imp_b
; CHECK-NEXT:   .word   32
; CHECK-NEXT:   .secnum .text
; CHECK-NEXT:   .word   19
; CHECK-NEXT:   .secoffset      .Limpcall3
; CHECK-NEXT:   .symidx __imp_b
; CHECK-NEXT:   .word   19
; CHECK-NEXT:   .secoffset      .Limpcall4
; CHECK-NEXT:   .symidx __imp_a

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"import-call-optimization", i32 1}
