; RUN: llc -mtriple=xtensa  -mcpu=esp32 -verify-machineinstrs < %s \
; RUN:   | FileCheck -check-prefix=Xtensa %s


@gf = external global float

define float @constraint_f_float(float %a) nounwind {
; Xtensa-LABEL: constraint_f_float:
; Xtensa:       # %bb.0:
; Xtensa-NEXT:    entry a1, 32
; Xtensa-NEXT:    wfr f8, a2
; Xtensa-NEXT:    l32r a8, .LCPI0_0
; Xtensa-NEXT:    lsi f9, a8, 0
; Xtensa-NEXT:    #APP
; Xtensa-NEXT:    add.s f8, f8, f9
; Xtensa-NEXT:    #NO_APP
; Xtensa-NEXT:    rfr a2, f8
; Xtensa-NEXT:    retw
  %gf = load float, float* @gf
  %res = tail call float asm "add.s $0, $1, $2", "=f,f,f"(float %a, float %gf)
  ret float %res
}
