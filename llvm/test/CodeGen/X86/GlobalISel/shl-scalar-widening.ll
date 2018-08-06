; RUN: llc -mtriple=x86_64-linux-gnu -global-isel -verify-machineinstrs < %s -o - | FileCheck %s --check-prefix=X64

define i16 @test_shl_i4(i16 %v, i16 %a, i16 %b) {
; Let's say the arguments are the following unsigned
; integers in two’s complement representation:
;
; %v: 77 (0000 0000  0100 1101)
; %a: 74 (0000 0000  0100 1010)
; %b: 72 (0000 0000  0100 1000)
  %v.t = trunc i16 %v to i4  ; %v.t: 13 (1101)
  %a.t = trunc i16 %a to i4  ; %a.t: 10 (1010)
  %b.t = trunc i16 %b to i4  ; %b.t:  8 (1000)
  %n.t = add i4 %a.t, %b.t   ; %n.t:  2 (0010)
  %r.t = shl i4 %v.t, %n.t   ; %r.t:  4 (0100)
  %r = zext i4 %r.t to i16
; %r:  4 (0000 0000 0000 0100)
  ret i16 %r

; X64-LABEL: test_shl_i4
;
; %di:  77 (0000 0000  0100 1101)
; %si:  74 (0000 0000  0100 1010)
; %dx:  72 (0000 0000  0100 1000)
;
; X64:       # %bb.0:
;
; X64-NEXT:    addb %sil, %dl
; %dx: 146 (0000 0000  1001 0010)
;
; X64-NEXT:    andb $15, %dl
; %dx:   2 (0000 0000  0000 0010)
;
; X64-NEXT:    movl %edx, %ecx
; %cx:   2 (0000 0000  0000 0010)
;
; X64-NEXT:    shlb %cl, %dil
; %di:  52 (0000 0000  0011 0100)
;
; X64-NEXT:    andw $15, %di
; %di:   4 (0000 0000  0000 0100)
;
; X64-NEXT:    movl %edi, %eax
; %ax:   4 (0000 0000  0000 0100)
;
; X64-NEXT:    retq
;
; Let's pretend that legalizing G_SHL by widening its second
; source operand is done via G_ANYEXT rather than G_ZEXT and
; see what happens:
;
;              addb %sil, %dl
; %dx: 146 (0000 0000  1001 0010)
;
;              movl %edx, %ecx
; %cx: 146 (0000 0000  1001 0010)
;
;              shlb %cl, %dil
; %di:   0 (0000 0000  0000 0000)
;
;              andw $15, %di
; %di:   0 (0000 0000  0000 0000)
;
;              movl %edi, %eax
; %ax:   0 (0000 0000  0000 0000)
;
;              retq
}
