; RUN: not llc < %s -mtriple=mipsel 2>&1 \
; RUN:   | FileCheck --check-prefix=NO-RESERVE %s
; RUN: llc < %s -mtriple=mipsel -mattr=+reserve-gpr24,+noabicalls \
; RUN:   | FileCheck --check-prefix=RESERVE %s

define i32 @get_reg_t8() nounwind {
; NO-RESERVE: Trying to obtain non-reserved register "$24".
; RESERVE-LABEL: get_reg_t8:
; RESERVE: move $2, $24
  %t8 = call i32 @llvm.read_register.i32(metadata !0)
  ret i32 %t8
}

; Static MIPS code uses -mno-abicalls, which inherently reserves $gp.
define i32 @get_reg_gp() nounwind {
; RESERVE-LABEL: get_reg_gp:
; RESERVE: move $2, $gp
  %gp = call i32 @llvm.read_register.i32(metadata !1)
  ret i32 %gp
}

!0 = !{!"$24\00"}
!1 = !{!"$gp\00"}
