; RUN: llc -march=mipsel < %s | FileCheck %s -check-prefix=32
; RUN: llc -march=mipsel -stack-alignment=32 < %s | FileCheck %s -check-prefix=A32-32
; RUN: llc -march=mipsel -mattr=+fp64 < %s | FileCheck %s -check-prefix=32
; RUN: llc -march=mips64el -mcpu=mips3 < %s | FileCheck %s -check-prefix=64
; RUN: llc -march=mips64el -mcpu=mips4 < %s | FileCheck %s -check-prefix=64
; RUN: llc -march=mips64el -mcpu=mips64 < %s | FileCheck %s -check-prefix=64
; RUN: llc -march=mips64el -mcpu=mips64 -stack-alignment=32 < %s | FileCheck %s -check-prefix=A32-64

; 32:      addiu  $sp, $sp, -8
; 64:      daddiu  $sp, $sp, -16
; A32-32:  addiu  $sp, $sp, -32
; A32-64:  daddiu  $sp, $sp, -32

define i32 @foo1() #0 {
entry:
  ret i32 14
}

attributes #0 = { "no-frame-pointer-elim"="true" }
