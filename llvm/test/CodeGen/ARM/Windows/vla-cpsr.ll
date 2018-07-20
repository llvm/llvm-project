; RUN: llc -mtriple thumbv7-windows-itanium -filetype asm -o /dev/null %s -print-machineinstrs=expand-isel-pseudos 2>&1 | FileCheck %s

declare arm_aapcs_vfpcc void @g(i8*) local_unnamed_addr

define arm_aapcs_vfpcc void @f(i32 %i) local_unnamed_addr {
entry:
  %vla = alloca i8, i32 %i, align 1
  call arm_aapcs_vfpcc void @g(i8* nonnull %vla)
  ret void
}

; CHECK: tBL 14, $noreg, &__chkstk, implicit-def $lr, implicit $sp, implicit killed $r4, implicit-def $r4, implicit-def dead $r12, implicit-def dead $cpsr

