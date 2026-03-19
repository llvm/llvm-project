; RUN: llc -mtriple=mipsel-unknown-linux-gnu -mcpu=mips32r2 -verify-machineinstrs < %s | FileCheck %s -check-prefix=MIPS32

define i32 @test_sign_bit(double %a) nounwind {
; MIPS32-LABEL: test_sign_bit:
; MIPS32:       # %bb.0: # %entry
; MIPS32-NEXT:    mfhc1	$1, $f12
; MIPS32-NEXT:    slti $1, $1, 0
; MIPS32-NEXT:    addiu	$2, $zero, -4
; MIPS32-NEXT:    jr $ra
; MIPS32-NEXT:    or $2, $1, $2

entry:
  %conv = fptrunc double %a to float
  %i_val = bitcast float %conv to i32
  %1 = icmp slt i32 %i_val, 0
  %2 = select i1 %1, i32 -3, i32 -4
  ret i32 %2
}
