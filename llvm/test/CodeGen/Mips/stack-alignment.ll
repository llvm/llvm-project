; RUN: split-file %s %t
; RUN: cat %t/main.ll %t/_32.ll > %t/32.ll
; RUN: llc -mtriple=mipsel < %t/main.ll | FileCheck %s -check-prefix=32
; RUN: llc -mtriple=mipsel < %t/32.ll | FileCheck %s -check-prefix=A32-32
; RUN: llc -mtriple=mipsel -mattr=+fp64,+mips32r2 < %t/main.ll | FileCheck %s -check-prefix=32
; RUN: llc -mtriple=mips64el -mcpu=mips3 < %t/main.ll | FileCheck %s -check-prefix=64
; RUN: llc -mtriple=mips64el -mcpu=mips4 < %t/main.ll | FileCheck %s -check-prefix=64
; RUN: llc -mtriple=mips64el -mcpu=mips64 < %t/main.ll | FileCheck %s -check-prefix=64
; RUN: llc -mtriple=mips64el -mcpu=mips64 < %t/32.ll | FileCheck %s -check-prefix=A32-64

;--- main.ll
; 32:      addiu  $sp, $sp, -8
; 64:      daddiu  $sp, $sp, -16
; A32-32:  addiu  $sp, $sp, -32
; A32-64:  daddiu  $sp, $sp, -32

define i32 @foo1() #0 {
entry:
  ret i32 14
}

attributes #0 = { "frame-pointer"="all" }
;--- _32.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"override-stack-alignment", i32 32}
