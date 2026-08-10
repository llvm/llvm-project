; RUN: llc -asm-show-inst -mtriple=mipsel -relocation-model=pic < %s | FileCheck %s -check-prefix=PIC
; RUN: llc -asm-show-inst -mtriple=mipsel -relocation-model=static < %s | FileCheck %s -check-prefix=STATIC
; RUN: llc -asm-show-inst -mtriple=mipsel -mattr=mips16 -relocation-model=pic < %s | FileCheck %s -check-prefix=PIC16
; RUN: llc -asm-show-inst -mtriple=mipsel -mattr=mips16 -relocation-model=static < %s | FileCheck %s -check-prefix=STATIC16
; RUN: llc -asm-show-inst -mtriple=mips -mattr=+micromips -relocation-model=static < %s | FileCheck %s -check-prefix=STATICMM
; RUN: llc -asm-show-inst -mtriple=mips -mattr=+micromips -relocation-model=pic < %s | FileCheck %s -check-prefix=PICMM
; RUN: llc -asm-show-inst -mtriple=mips -mcpu=mips32r6 -mattr=+micromips -relocation-model=static < %s | FileCheck %s -check-prefix=STATICMMR6
; RUN: llc -asm-show-inst -mtriple=mips -mcpu=mips32r6 -mattr=+micromips -relocation-model=pic < %s | FileCheck %s -check-prefix=PICMMR6



define void @count(i32 %x, i32 %y, i32 %z) noreturn nounwind readnone {
entry:
  br label %bosco

bosco:                                            ; preds = %bosco, %entry
  br label %bosco
}

; PIC:        b  $BB0_1 # <MCInst #{{.*}} BEQ
; PICMM:      b  $BB0_1 # <MCInst #{{.*}} BEQ_MM
; STATIC:     j  $BB0_1 # <MCInst #{{.*}} J
; STATICMM:   j  $BB0_1 # <MCInst #{{.*}} J_MM
; STATICMMR6: bc $BB0_1 # <MCInst #{{.*}} BC_MMR6
; PICMMR6:    bc $BB0_1 # <MCInst #{{.*}} BC_MMR6
; PIC16:      b  $BB0_1
; STATIC16:   b  $BB0_1
