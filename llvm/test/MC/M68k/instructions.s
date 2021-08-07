; RUN: llvm-mc -triple m68k -show-encoding -motorola-integers %s | FileCheck %s

; At the moment, all encoding tests for M68k live in llvm/test/CodeGen/M68k/.
; This is the first test included as part of the AsmMatcher and lacks encoding checks.
; The current migration plan is to consolidate all of the encoding tests in this
; directory along with AsmMatcher/ Disassembler tests like the other platforms.
; For more information and status updates see bug #49865.

.global ext_fn

; CHECK: move.l %a1, %a0
move.l %a1, %a0
; CHECK: adda.l %a0, %a1
adda.l %a0, %a1
; CHECK: addx.l %d1, %d2
addx.l %d1, %d2
; CHECK: sub.w #4, %d1
sub.w #4, %d1
; CHECK: cmp.w %a0, %d0
cmp.w %a0, %d0
; CHECK: neg.w %d0
neg.w %d0
; CHECK: btst #8, %d3
btst #$8, %d3
; CHECK: bra ext_fn
bra ext_fn
; CHECK: jsr ext_fn
jsr ext_fn
; CHECK: seq %d0
seq %d0
; CHECK: sgt %d0
sgt %d0
; CHECK: lea (80,%a0), %a1
lea $50(%a0), %a1
; CHECK: lsl.l #8, %d1
lsl.l #8, %d1
; CHECK: lsr.l #8, %d1
lsr.l #8, %d1
; CHECK: asr.l #8, %d1
asr.l #8, %d1
; CHECK: rol.l #8, %d1
rol.l #8, %d1
; CHECK: ror.l #8, %d1
ror.l #8, %d1
; CHECK: nop
nop
; CHECK: rts
rts
